# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Trains the hurdle model on a split and measures its predictions.

Trainer.train writes, for one case and split, the two models, the
predictions of the test rows and of every candidate station of the test
quakes, and the record. The record is written last and holds the inputs, so
that complete tells a finished training from a stopped or outdated one. load
reads the predictions back and checks that they are of the split, and
measure turns them into the measures of training.evaluate.
"""

from collections.abc import Mapping
import dataclasses
import json
import logging
import pathlib
import subprocess
import time
from typing import Any, Self

import lightgbm
import numpy as np
import numpy.typing as npt

import shindo_est
from training import dataset
from training import evaluate
from training import features
from training import hurdle
from training import selection

FILES = (
    "classifier.txt",
    "regressor.txt",
    "predictions.npz",
    "training.json",
)
RECORD = "training.json"
PREDICTIONS = "predictions.npz"
GROUPS = ("trained", "held_out", "all")
# The keys of a difference: the RMSE difference and the 2.5th and 97.5th
# percentiles of its bootstrap.
INTERVAL = ("difference", "low", "high")

# The pairs predicted at a time, which bounds the memory of their features.
_PREDICTION_BATCH = 1 << 20

_logger = logging.getLogger(__name__)

# The quake and the displayed minus observed intensity of each felt row with
# an instrumental intensity.
type Errors = tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]


@dataclasses.dataclass(frozen=True)
class Predictions:
    """The predictions of one case on one split.

    Attributes:
        test_rows: The index of each test row in the dataset.
        test_probability: The felt probability of each test row.
        test_regression: The regression value of each test row.
        candidate_quake: The quake of each candidate.
        candidate_station: The station of each candidate.
        candidate_probability: The felt probability of each candidate.
        candidate_regression: The regression value of each candidate.
    """

    test_rows: npt.NDArray[np.intp]
    test_probability: npt.NDArray[np.float64]
    test_regression: npt.NDArray[np.float64]
    candidate_quake: npt.NDArray[np.int32]
    candidate_station: npt.NDArray[np.int32]
    candidate_probability: npt.NDArray[np.float64]
    candidate_regression: npt.NDArray[np.float64]

    def save(self, path: pathlib.Path) -> None:
        """Writes the arrays to an .npz file."""
        np.savez(
            path,
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
            },
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        """Reads the arrays written by save."""
        names = [field.name for field in dataclasses.fields(cls)]
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**{name: arrays[name] for name in names})


@dataclasses.dataclass(frozen=True)
class Trainer:
    """Trains and predicts the cases on the splits of one dataset.

    Attributes:
        data: The dataset.
        table: The ARV table.
        slabs: The slab geometry.
        threads: The number of threads computing features.
        eligible: Whether each station may be a candidate, as
            selection.candidates takes it.
    """

    data: dataset.Dataset
    table: selection.ArvTable
    slabs: shindo_est.SlabGeometry
    threads: int
    eligible: npt.NDArray[np.bool_]

    def train(
        self,
        case: str,
        split: selection.Split,
        settings: hurdle.Settings,
        directory: pathlib.Path,
        inputs: Mapping[str, object],
    ) -> None:
        """Trains one case on one split and writes FILES, the record last.

        Args:
            case: One of selection.CASES.
            split: The split to train and predict.
            settings: The training settings.
            directory: The directory to write FILES to.
            inputs: What the files are made from, written to the record for
                complete to compare.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECORD).unlink(missing_ok=True)
        started = time.monotonic()
        data = self.data
        train = np.flatnonzero(split.train)
        values, names = self._features(
            data.row_quake[train], data.row_station[train], case
        )
        model = hurdle.fit(
            values,
            names,
            data.row_quake[train],
            data.felt[train],
            data.observed[train],
            data.weight[train],
            settings=settings,
        )
        # Frees the largest array before the predictions take memory.
        del values
        test = np.flatnonzero(split.test)
        test_probability, test_regression = self.predict(
            model, data.row_quake[test], data.row_station[test], case
        )
        quakes, stations = selection.candidates(
            data, split.test, self.eligible
        )
        candidate_probability, candidate_regression = self.predict(
            model, quakes, stations, case
        )
        model.classifier.save_model(directory / "classifier.txt")
        model.regressor.save_model(directory / "regressor.txt")
        Predictions(
            test_rows=test,
            test_probability=test_probability,
            test_regression=test_regression,
            candidate_quake=quakes,
            candidate_station=stations,
            candidate_probability=candidate_probability,
            candidate_regression=candidate_regression,
        ).save(directory / PREDICTIONS)
        seconds = round(time.monotonic() - started, 1)
        record = {
            "train_rows": len(train),
            "test_rows": len(test),
            "candidates": len(quakes),
            "trees": {
                "classifier": model.classifier.num_trees(),
                "regressor": model.regressor.num_trees(),
            },
            "seconds": seconds,
            "gain": {
                "classifier": gain(model.classifier),
                "regressor": gain(model.regressor),
            },
            "settings": settings.parameters(),
            "inputs": inputs,
        }
        (directory / RECORD).write_text(
            json.dumps(record, indent=2) + "\n", encoding="ascii"
        )
        _logger.info("%s/%s: trained in %.1f s", case, split.name, seconds)

    def predict(
        self,
        model: hurdle.Hurdle,
        quakes: npt.NDArray[np.int32],
        stations: npt.NDArray[np.int32],
        case: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predicts pairs of a quake and a station a batch at a time.

        Args:
            model: The trained models.
            quakes: The quake of each pair.
            stations: The station of each pair.
            case: The case the model was trained on.

        Returns:
            A tuple (probability, regression) of the felt probability and the
            regression value of each pair.
        """
        probability = np.empty(len(quakes))
        regression = np.empty(len(quakes))
        for start in range(0, len(quakes), _PREDICTION_BATCH):
            batch = slice(start, start + _PREDICTION_BATCH)
            values, _ = self._features(quakes[batch], stations[batch], case)
            probability[batch], regression[batch] = model.predict(values)
        return probability, regression

    def _features(
        self,
        quakes: npt.NDArray[np.int32],
        stations: npt.NDArray[np.int32],
        case: str,
    ) -> tuple[npt.NDArray[np.float32], tuple[str, ...]]:
        return selection.feature_matrix(
            self.data,
            quakes,
            stations,
            self.table,
            case,
            self.slabs,
            threads=self.threads,
        )


def revision() -> tuple[str, bool]:
    """Returns the commit of the repository and whether it has changes.

    Raises:
        subprocess.CalledProcessError: git cannot tell the commit.
    """
    head = _git("rev-parse", "HEAD").strip()
    return head, bool(_git("status", "--porcelain").strip())


def complete(directory: pathlib.Path, inputs: Mapping[str, object]) -> bool:
    """Returns whether a case and split was trained from the same inputs.

    A record that cannot be read, as after a stop while writing it, counts
    as not trained.

    Args:
        directory: The directory Trainer.train wrote to.
        inputs: The inputs the training would be made from now.
    """
    if not all((directory / name).exists() for name in FILES):
        return False
    try:
        record = json.loads((directory / RECORD).read_text(encoding="ascii"))
    except ValueError:
        return False
    return isinstance(record, dict) and record.get("inputs") == inputs


def gain(booster: lightgbm.Booster) -> dict[str, float | None]:
    """Returns each feature's share of the total gain, None without any."""
    gains = booster.feature_importance(importance_type="gain")
    total = float(gains.sum())
    return {
        name: None if total == 0 else float(value) / total
        for name, value in zip(booster.feature_name(), gains)
    }


def load(
    directory: pathlib.Path,
    digests: Mapping[str, str],
    split: selection.Split,
    candidates: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]],
) -> Predictions:
    """Reads the predictions of a case on a split and checks what they are of.

    Args:
        directory: The directory Trainer.train wrote to.
        digests: The SHA-256 of each input file, by its name in the record.
        split: The split the predictions should be of.
        candidates: The candidate quakes and stations of the split.

    Returns:
        The predictions.

    Raises:
        ValueError: The record names another input file, or the predictions
            are not of the test rows and the candidates.
        OSError: A file cannot be read.
    """
    record = json.loads((directory / RECORD).read_text(encoding="ascii"))
    for name, digest in digests.items():
        if record["inputs"][name] != digest:
            raise ValueError(f"{directory}: trained on another {name}")
    predicted = Predictions.load(directory / PREDICTIONS)
    for name, actual, expected in (
        ("test rows", predicted.test_rows, np.flatnonzero(split.test)),
        ("candidate quakes", predicted.candidate_quake, candidates[0]),
        ("candidate stations", predicted.candidate_station, candidates[1]),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"{directory}: the {name} are not of the split")
    return predicted


def groups(data: dataset.Dataset) -> dict[str, npt.NDArray[np.bool_]]:
    """Returns whether each station is in each of GROUPS.

    The groups are the stations trained on, the held-out stations, and all
    stations.
    """
    held_out = data.station_held_out
    return {
        "trained": ~held_out,
        "held_out": held_out,
        "all": np.ones_like(held_out),
    }


def measure(
    data: dataset.Dataset,
    predicted: Predictions,
    chosen: npt.NDArray[np.bool_],
    values: evaluate.HurdleValues = evaluate.PREVIOUS,
) -> tuple[dict[str, Any], Errors]:
    """Measures one case on some stations.

    Args:
        data: The dataset.
        predicted: The predictions of the case.
        chosen: Whether each station is measured.
        values: The hurdle values that turn the predictions into displays.

    Returns:
        A tuple (measured, errors) of the summary, maxima, and calibration,
        and of the Errors of the rows.
    """
    inside = chosen[data.row_station[predicted.test_rows]]
    rows = predicted.test_rows[inside]
    probability = predicted.test_probability[inside]
    regression = predicted.test_regression[inside]
    row_quake = data.row_quake[rows]
    felt = data.felt[rows]
    observed = data.observed[rows]
    weight = data.weight[rows]
    candidate = chosen[predicted.candidate_station]
    candidate_display = evaluate.display(
        predicted.candidate_probability[candidate],
        predicted.candidate_regression[candidate],
        values,
    )
    measured = {
        "summary": evaluate.summarize(
            row_quake, felt, observed, weight, probability, regression, values
        ),
        "max": evaluate.max_per_quake(
            predicted.candidate_quake[candidate],
            candidate_display,
            row_quake,
            observed,
        ),
        "calibration": [
            dataclasses.asdict(calibration_bin)
            for calibration_bin in evaluate.calibration(
                felt, weight, probability
            )
        ],
    }
    known = felt & ~np.isnan(observed)
    shown = evaluate.display(probability, regression, values)
    return measured, (row_quake[known], shown[known] - observed[known])


def difference(errors: Errors, first_errors: Errors) -> dict[str, float] | None:
    """Returns the RMSE difference from the first case and its interval.

    Args:
        errors: The errors of a case, as measure returns them.
        first_errors: The errors of the first case on the same rows.

    Returns:
        The difference and its interval by INTERVAL, None without rows.
    """
    row_quake, error = errors
    if row_quake.size == 0:
        return None
    interval = evaluate.bootstrap_rmse_difference(
        row_quake, error, first_errors[1]
    )
    return dict(zip(INTERVAL, interval))


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(features.REPOSITORY), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

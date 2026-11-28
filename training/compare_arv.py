# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Trains the hurdle model on each ARV case and split, then compares them.

run trains every case on every split of selection.splits and writes, for
each, the two models, the predictions of the test rows and of every
candidate station of the test quakes, and training.json. training.json is
written last and records the inputs, so that a stopped run picks up where it
left off: a case and split is trained again unless its four files are there
and its inputs are the same, and always while the repository has changes.

report checks that the predictions still belong to the dataset and the ARV
table, then measures each case on the stations trained on, the held-out
stations, and all stations, and compares the RMSE of the other cases with
that of the first.

Usage:
    python -m training.compare_arv run [--dataset PATH] [--arv PATH]
        [--out DIR] [--case NAME ...] [--split NAME ...]
    python -m training.compare_arv report [--dataset PATH] [--arv PATH]
        [--out DIR]
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any, Self

import lightgbm
import numpy as np
import numpy.typing as npt

import shindo_est
from training import arv
from training import dataset
from training import evaluate
from training import features
from training import hurdle
from training import selection
from training import sources

DEFAULT_OUT = features.REPOSITORY / "training" / "data" / "compare-arv"
FILES = (
    "classifier.txt",
    "regressor.txt",
    "predictions.npz",
    "training.json",
)
GROUPS = ("trained", "held_out", "all")
RESULTS_JSON = "results.json"
RESULTS_MARKDOWN = "results.md"

_RECORD = "training.json"
_PREDICTIONS = "predictions.npz"
# The measures in the Markdown table of a split, and where each one is.
_COLUMNS = (
    ("summary", "felt_rows"),
    ("summary", "felt_bias"),
    ("summary", "felt_rmse"),
    ("summary", "strong4_rmse"),
    ("summary", "strong5_capture"),
    ("summary", "ets"),
    ("max", "max_bias"),
    ("max", "max_rmse"),
)
_INTERVAL = ("difference", "low", "high")
# The pairs predicted at a time, which bounds the memory of their features.
_PREDICTION_BATCH = 1 << 20

_logger = logging.getLogger(__name__)

type _Json = dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _Predictions:
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
class _Trainer:
    """Trains and predicts the cases on the splits of one dataset.

    Attributes:
        data: The dataset.
        table: The ARV table.
        slabs: The slab geometry.
        threads: The number of threads computing features.
        settings: The training settings.
    """

    data: dataset.Dataset
    table: selection.ArvTable
    slabs: shindo_est.SlabGeometry
    threads: int
    settings: hurdle.Settings

    def train(
        self,
        case: str,
        split: selection.Split,
        directory: pathlib.Path,
        inputs: Mapping[str, object],
    ) -> None:
        """Trains one case on one split and writes FILES, the record last.

        Args:
            case: One of selection.CASES.
            split: The split to train and predict.
            directory: The directory to write FILES to.
            inputs: What the files are made from, written to the record for
                _complete to compare.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _RECORD).unlink(missing_ok=True)
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
            settings=self.settings,
        )
        # Frees the largest array before the predictions take memory.
        del values
        test = np.flatnonzero(split.test)
        test_probability, test_regression = self._predict(
            model, data.row_quake[test], data.row_station[test], case
        )
        quakes, stations = selection.candidates(
            data,
            split.test,
            selection.stations_with_arv(
                data, self.table, selection.COMPARED_SOURCES
            ),
        )
        candidate_probability, candidate_regression = self._predict(
            model, quakes, stations, case
        )
        model.classifier.save_model(directory / "classifier.txt")
        model.regressor.save_model(directory / "regressor.txt")
        _Predictions(
            test_rows=test,
            test_probability=test_probability,
            test_regression=test_regression,
            candidate_quake=quakes,
            candidate_station=stations,
            candidate_probability=candidate_probability,
            candidate_regression=candidate_regression,
        ).save(directory / _PREDICTIONS)
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
                "classifier": _gain(model.classifier),
                "regressor": _gain(model.regressor),
            },
            "inputs": inputs,
        }
        (directory / _RECORD).write_text(
            json.dumps(record, indent=2) + "\n", encoding="ascii"
        )
        _logger.info("%s/%s: trained in %.1f s", case, split.name, seconds)

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

    def _predict(
        self,
        model: hurdle.Hurdle,
        quakes: npt.NDArray[np.int32],
        stations: npt.NDArray[np.int32],
        case: str,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        probability = np.empty(len(quakes))
        regression = np.empty(len(quakes))
        for start in range(0, len(quakes), _PREDICTION_BATCH):
            batch = slice(start, start + _PREDICTION_BATCH)
            values, _ = self._features(quakes[batch], stations[batch], case)
            probability[batch], regression[batch] = model.predict(values)
        return probability, regression


def run(
    dataset_path: pathlib.Path,
    arv_path: pathlib.Path,
    out_directory: pathlib.Path,
    *,
    cases: Sequence[str] = selection.CASES,
    split_names: Sequence[str] | None = None,
    threads: int = 16,
    max_rounds: int = hurdle.MAX_ROUNDS,
) -> None:
    """Trains and predicts each case on each split.

    Writes FILES to out_directory/<case>/<split>.

    Args:
        dataset_path: The dataset written by training.dataset.
        arv_path: The ARV table written by training.arv.
        out_directory: The directory to write to.
        cases: The cases to train, from selection.CASES.
        split_names: The names of the splits to train, None for all.
        threads: The number of threads computing features.
        max_rounds: The most boosting rounds of each model.

    Raises:
        features.StaleExtensionError: The installed extension is out of
            date. Nothing is written then.
        ValueError: A case or a split is unknown, or a file cannot be read
            as a dataset or an ARV table.
        subprocess.CalledProcessError: git cannot tell the commit of the
            repository.
        OSError: A file cannot be read or written.
    """
    features.check_extension()
    unknown = [case for case in cases if case not in selection.CASES]
    if unknown:
        raise ValueError(f"unknown cases {unknown}; known: {selection.CASES}")
    data = dataset.Dataset.load(dataset_path)
    table = arv.read_table(arv_path)
    splits = selection.splits(data, selection.rows(data, table))
    if split_names is not None:
        splits = _choose(splits, split_names)
    settings = hurdle.Settings(max_rounds=max_rounds)
    trainer = _Trainer(data, table, features.read_slabs(), threads, settings)
    head, dirty = _revision()
    common = {
        "dataset": sources.file_sha256(dataset_path),
        "arv": sources.file_sha256(arv_path),
        "slabs": [sources.file_sha256(path) for path in features.SLAB_GRIDS],
        "shindo_est": shindo_est.__version__,
        "head": head,
        "dirty": dirty,
        "parameters": settings.parameters(),
    }
    for case in cases:
        for split in splits:
            directory = out_directory / case / split.name
            inputs = {**common, "case": case, "split": split.name}
            if not dirty and _complete(directory, inputs):
                _logger.info("%s/%s: trained already", case, split.name)
                continue
            trainer.train(case, split, directory, inputs)


def report(
    out_directory: pathlib.Path,
    dataset_path: pathlib.Path,
    arv_path: pathlib.Path,
) -> str:
    """Measures and compares the trained cases.

    In each split, every trained case is measured on each of GROUPS of
    stations, with the test rows, the candidates, and the observations all
    limited to the stations of the group. The RMSE over the felt rows with an
    instrumental intensity of each other case is compared with that of the
    first of selection.CASES.

    Args:
        out_directory: The directory run wrote to. RESULTS_JSON and
            RESULTS_MARKDOWN are written there.
        dataset_path: The dataset given to run.
        arv_path: The ARV table given to run.

    Returns:
        The Markdown written to RESULTS_MARKDOWN.

    Raises:
        ValueError: Nothing has been trained, a file cannot be read as a
            dataset or an ARV table, a case was trained on another dataset or
            ARV table, or its predictions are not of the test rows and the
            candidates of its split.
        OSError: A file cannot be read or written.
    """
    data = dataset.Dataset.load(dataset_path)
    table = arv.read_table(arv_path)
    digests = {
        "dataset": sources.file_sha256(dataset_path),
        "arv": sources.file_sha256(arv_path),
    }
    groups = {
        "trained": ~data.station_held_out,
        "held_out": data.station_held_out,
        "all": np.ones_like(data.station_held_out),
    }
    results: _Json = {}
    for split in selection.splits(data, selection.rows(data, table)):
        directories = {
            case: out_directory / case / split.name
            for case in selection.CASES
            if (out_directory / case / split.name / _RECORD).exists()
        }
        if not directories:
            continue
        candidates = selection.candidates(
            data,
            split.test,
            selection.stations_with_arv(
                data, table, selection.COMPARED_SOURCES
            ),
        )
        predictions = {
            case: _load(directory, digests, split, candidates)
            for case, directory in directories.items()
        }
        results[split.name] = _compare(data, predictions, groups)
    if not results:
        raise ValueError(f"{out_directory}: no case has been trained")
    markdown = _markdown(results)
    (out_directory / RESULTS_JSON).write_text(
        json.dumps(results, indent=2) + "\n", encoding="ascii"
    )
    (out_directory / RESULTS_MARKDOWN).write_text(markdown, encoding="ascii")
    return markdown


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: The arguments after the program name. None reads sys.argv.

    Returns:
        The exit status: 0 on success, 1 when run or report raises one of the
        errors listed in their Raises sections.
    """
    parser = argparse.ArgumentParser(prog="python -m training.compare_arv")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="train and predict")
    report_parser = commands.add_parser("report", help="compare the cases")
    for command in (run_parser, report_parser):
        command.add_argument(
            "--dataset", type=pathlib.Path, default=dataset.DEFAULT_DATASET
        )
        command.add_argument(
            "--arv", type=pathlib.Path, default=arv.DEFAULT_TABLE
        )
        command.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    run_parser.add_argument("--case", action="append", choices=selection.CASES)
    run_parser.add_argument("--split", action="append")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if arguments.command == "run":
            run(
                arguments.dataset,
                arguments.arv,
                arguments.out,
                cases=arguments.case or selection.CASES,
                split_names=arguments.split,
            )
        else:
            markdown = report(arguments.out, arguments.dataset, arguments.arv)
            print(markdown, end="")
    except (
        features.StaleExtensionError,
        ValueError,
        subprocess.CalledProcessError,
        OSError,
    ) as error:
        _logger.error("%s", error)
        return 1
    return 0


def _choose(
    splits: list[selection.Split], names: Sequence[str]
) -> list[selection.Split]:
    """Returns the splits with the given names."""
    known = [split.name for split in splits]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"unknown splits {unknown}; known: {known}")
    return [split for split in splits if split.name in names]


def _revision() -> tuple[str, bool]:
    """Returns the commit of the repository and whether it has changes."""
    head = _git("rev-parse", "HEAD").strip()
    return head, bool(_git("status", "--porcelain").strip())


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(features.REPOSITORY), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _complete(directory: pathlib.Path, inputs: Mapping[str, object]) -> bool:
    """Returns whether a case and split was trained from the same inputs.

    A record that cannot be read, as after a stop while writing it, counts
    as not trained.
    """
    if not all((directory / name).exists() for name in FILES):
        return False
    try:
        record = json.loads((directory / _RECORD).read_text(encoding="ascii"))
    except ValueError:
        return False
    return isinstance(record, dict) and record.get("inputs") == inputs


def _gain(booster: lightgbm.Booster) -> dict[str, float | None]:
    """Returns each feature's share of the total gain, None without any."""
    gains = booster.feature_importance(importance_type="gain")
    total = float(gains.sum())
    return {
        name: None if total == 0 else float(gain) / total
        for name, gain in zip(booster.feature_name(), gains)
    }


def _load(
    directory: pathlib.Path,
    digests: Mapping[str, str],
    split: selection.Split,
    candidates: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]],
) -> _Predictions:
    """Reads the predictions of a case on a split and checks what they are of.

    Raises:
        ValueError: The record names another dataset or ARV table, or the
            predictions are not of the test rows and the candidates.
    """
    record = json.loads((directory / _RECORD).read_text(encoding="ascii"))
    for name, digest in digests.items():
        if record["inputs"][name] != digest:
            raise ValueError(f"{directory}: trained on another {name}")
    predicted = _Predictions.load(directory / _PREDICTIONS)
    for name, actual, expected in (
        ("test rows", predicted.test_rows, np.flatnonzero(split.test)),
        ("candidate quakes", predicted.candidate_quake, candidates[0]),
        ("candidate stations", predicted.candidate_station, candidates[1]),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"{directory}: the {name} are not of the split")
    return predicted


def _compare(
    data: dataset.Dataset,
    predictions: Mapping[str, _Predictions],
    groups: Mapping[str, npt.NDArray[np.bool_]],
) -> _Json:
    """Measures the cases of one split on each group and compares them."""
    first = selection.CASES[0]
    cases: _Json = {case: {} for case in predictions}
    differences: _Json = {}
    for group, chosen in groups.items():
        errors = {}
        for case, predicted in predictions.items():
            cases[case][group], errors[case] = _measure(data, predicted, chosen)
        differences[group] = {
            f"{case} - {first}": _difference(errors[case], errors[first])
            for case in predictions
            if case != first and first in predictions
        }
    return {"cases": cases, "differences": differences}


def _measure(
    data: dataset.Dataset,
    predicted: _Predictions,
    chosen: npt.NDArray[np.bool_],
) -> tuple[_Json, tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]]:
    """Measures one case on the stations of one group.

    Returns:
        A tuple (measured, errors) of the summary, maxima, and calibration,
        and of the quake and the displayed minus observed intensity of each
        felt row with an instrumental intensity.
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
        evaluate.PREVIOUS,
    )
    measured = {
        "summary": evaluate.summarize(
            row_quake, felt, observed, weight, probability, regression
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
    shown = evaluate.display(probability, regression, evaluate.PREVIOUS)
    return measured, (row_quake[known], shown[known] - observed[known])


def _difference(
    errors: tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]],
    first_errors: tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]],
) -> dict[str, float] | None:
    """Returns the RMSE difference and its interval, None without rows."""
    row_quake, error = errors
    if row_quake.size == 0:
        return None
    difference, low, high = evaluate.bootstrap_rmse_difference(
        row_quake, error, first_errors[1]
    )
    return dict(zip(_INTERVAL, (difference, low, high)))


def _markdown(results: _Json) -> str:
    """Returns the measures and the differences as Markdown tables."""
    header = " | ".join(name for _, name in _COLUMNS)
    lines = ["# ARV comparison", ""]
    for split, compared in results.items():
        lines += [
            f"## Split {split}",
            "",
            f"| Case | Stations | {header} |",
            "|---|---|" + "---|" * len(_COLUMNS),
        ]
        for case, measured in compared["cases"].items():
            for group in GROUPS:
                cells = " | ".join(
                    _cell(measured[group][part][name])
                    for part, name in _COLUMNS
                )
                lines.append(f"| {case} | {group} | {cells} |")
        lines += [
            "",
            "| Comparison | Stations | RMSE difference | 2.5% | 97.5% |",
            "|---|---|---|---|---|",
        ]
        for group in GROUPS:
            for name, difference in compared["differences"][group].items():
                values = [
                    None if difference is None else difference[key]
                    for key in _INTERVAL
                ]
                cells = " | ".join(_cell(value) for value in values)
                lines.append(f"| {name} | {group} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def _cell(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())

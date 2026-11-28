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
import json
import logging
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

import shindo_est
from training import arv
from training import dataset
from training import features
from training import hurdle
from training import runs
from training import selection
from training import sources

DEFAULT_OUT = features.REPOSITORY / "training" / "data" / "compare-arv"
FILES = runs.FILES
GROUPS = runs.GROUPS
RESULTS_JSON = "results.json"
RESULTS_MARKDOWN = "results.md"

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

_logger = logging.getLogger(__name__)

type _Json = dict[str, Any]


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
    trainer = runs.Trainer(
        data,
        table,
        features.read_slabs(),
        threads,
        selection.stations_with_arv(data, table, selection.COMPARED_SOURCES),
    )
    head, dirty = runs.revision()
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
            if not dirty and runs.complete(directory, inputs):
                _logger.info("%s/%s: trained already", case, split.name)
                continue
            trainer.train(case, split, settings, directory, inputs)


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
    groups = runs.groups(data)
    results: _Json = {}
    for split in selection.splits(data, selection.rows(data, table)):
        directories = {
            case: out_directory / case / split.name
            for case in selection.CASES
            if (out_directory / case / split.name / runs.RECORD).exists()
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
            case: runs.load(directory, digests, split, candidates)
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


def _compare(
    data: dataset.Dataset,
    predictions: Mapping[str, runs.Predictions],
    groups: Mapping[str, npt.NDArray[np.bool_]],
) -> _Json:
    """Measures the cases of one split on each group and compares them."""
    first = selection.CASES[0]
    cases: _Json = {case: {} for case in predictions}
    differences: _Json = {}
    for group, chosen in groups.items():
        errors = {}
        for case, predicted in predictions.items():
            cases[case][group], errors[case] = runs.measure(
                data, predicted, chosen
            )
        differences[group] = {
            f"{case} - {first}": runs.difference(errors[case], errors[first])
            for case in predictions
            if case != first and first in predictions
        }
    return {"cases": cases, "differences": differences}


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
                    for key in runs.INTERVAL
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

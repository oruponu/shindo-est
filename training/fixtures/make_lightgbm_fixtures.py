#!/usr/bin/env python3
# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Writes LightGBM reference fixtures from synthetic data.

Usage: python3 training/fixtures/make_lightgbm_fixtures.py <output directory>

Trains two small binary models on random data, one with the default missing
values (NaN) and one with zero_as_missing, and makes every other threshold a
float32 value. For each model, writes the model text, inputs that reach split
nodes with one feature at and around the node's threshold, and LightGBM's own
raw scores and leaf indices for them. The Rust tests compare their tree
evaluation with these files. No real data is used.
"""

from collections.abc import Iterable, Sequence
import dataclasses
import pathlib
import sys
from typing import Any

import lightgbm as lgb
import numpy as np

FEATURE_COUNT = 18
TRAINING_ROWS = 20_000
TEST_ROWS = 300
TREE_COUNT = 40
# Split nodes that get rows of their own, at most.
MAX_SPLITS = 100
# Column of small integers from -2 to 2, most often 0.
INTEGER_COLUMN = 2
NAN_COLUMNS = (5, 7)
NAN_RATE = 0.1
# LightGBM's kZeroThreshold, a float32. Inputs no larger in magnitude are read
# as zero.
ZERO_THRESHOLD = 1e-35

PARAMETERS = {
    "objective": "binary",
    "num_leaves": 15,
    "learning_rate": 0.1,
    "min_data_in_leaf": 20,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "seed": 0,
    "verbose": -1,
}
# Extra parameters and the (missing type, default left) pairs that the picked
# split nodes must cover. Missing values never reach the default direction of
# a None split, so only one None pair is needed.
VARIANTS = {
    "nan": ({}, {("None", True), ("NaN", False), ("NaN", True)}),
    "zero": ({"zero_as_missing": True}, {("Zero", False), ("Zero", True)}),
}


@dataclasses.dataclass(frozen=True)
class Split:
    """A split node of a model.

    Attributes:
        tree: The index of the tree that holds the node.
        feature: The feature the node compares with its threshold.
        threshold: The threshold.
        missing_type: "None", "Zero", or "NaN".
        default_left: Whether missing values go to the left child.
        leaves: The indices of the leaves under the node.
    """

    tree: int
    feature: int
    threshold: float
    missing_type: str
    default_left: bool
    leaves: frozenset[int]


def make_rows(rng: np.random.Generator, count: int) -> np.ndarray:
    """Returns a (count, FEATURE_COUNT) float64 array of synthetic features."""
    rows = rng.normal(size=(count, FEATURE_COUNT))
    rows[:, INTEGER_COLUMN] = np.clip(
        np.round(rng.normal(scale=0.8, size=count)), -2, 2
    )
    for column in NAN_COLUMNS:
        rows[rng.random(count) < NAN_RATE, column] = np.nan
    return rows


def make_labels(rng: np.random.Generator, rows: np.ndarray) -> np.ndarray:
    """Returns binary labels that depend on columns 0, 2, and 5 of the rows."""
    noise = rng.normal(size=len(rows))
    score = (
        rows[:, 0]
        + 0.5 * np.nan_to_num(rows[:, 5])
        + (rows[:, INTEGER_COLUMN] == 0)
        + 0.3 * noise
    )
    return (score > 0).astype(np.float64)


def round_thresholds(booster: lgb.Booster) -> lgb.Booster:
    """Returns a copy of a model with every other threshold made float32.

    In each tree, the thresholds of splits 0, 2, 4, and so on are replaced with
    the nearest float32 values, so that float32 inputs can equal them. The
    other thresholds keep their values, most of which cannot be represented
    exactly as float32.

    Args:
        booster: A trained LightGBM booster.

    Returns:
        A booster loaded from the changed model text.
    """
    lines = []
    for line in booster.model_to_string().splitlines():
        key, _, value = line.partition("=")
        if key == "tree_sizes":
            # The sizes no longer match the changed trees. Without them,
            # LightGBM reads the trees one after another.
            continue
        if key == "threshold":
            thresholds = value.split()
            for index in range(0, len(thresholds), 2):
                rounded = np.float32(float(thresholds[index]))
                thresholds[index] = repr(float(rounded))
            line = "threshold=" + " ".join(thresholds)
        lines.append(line)
    return lgb.Booster(model_str="\n".join(lines) + "\n")


def split_nodes(booster: lgb.Booster) -> list[Split]:
    """Returns the split nodes of a model, tree by tree."""
    found = []

    def visit(tree: int, node: dict[str, Any]) -> frozenset[int]:
        """Adds the splits under `node` to `found` and returns its leaves."""
        if "split_index" not in node:
            return frozenset([node["leaf_index"]])
        left = visit(tree, node["left_child"])
        right = visit(tree, node["right_child"])
        leaves = left | right
        found.append(
            Split(
                tree=tree,
                feature=node["split_feature"],
                threshold=node["threshold"],
                missing_type=node["missing_type"],
                default_left=node["default_left"],
                leaves=leaves,
            )
        )
        return leaves

    for tree in booster.dump_model()["tree_info"]:
        visit(tree["tree_index"], tree["tree_structure"])
    return found


def boundary_values(threshold: float) -> list[float]:
    """Returns the values that a feature takes in the rows near a threshold.

    These are the threshold itself, the threshold rounded to float32, the
    float32 values just below and above it, 0, NaN, +-ZERO_THRESHOLD, and the
    float32 values just outside +-ZERO_THRESHOLD. The threshold itself tells
    whether inputs are rounded to float32 before the comparison.

    Args:
        threshold: A split threshold.

    Returns:
        A list of ten values.
    """
    threshold32 = np.float32(threshold)
    zero32 = np.float32(ZERO_THRESHOLD)
    above_zero = np.nextafter(zero32, np.float32(np.inf), dtype=np.float32)
    values = [
        threshold,
        threshold32,
        np.nextafter(threshold32, np.float32(-np.inf), dtype=np.float32),
        np.nextafter(threshold32, np.float32(np.inf), dtype=np.float32),
        0.0,
        np.nan,
        zero32,
        -zero32,
        above_zero,
        -above_zero,
    ]
    return [float(value) for value in values]


def reaching_rows(
    split: Split, bases: np.ndarray, booster: lgb.Booster
) -> list[np.ndarray] | None:
    """Returns rows that reach a split node with its feature near the threshold.

    For each value from boundary_values, takes the first row of `bases` that
    reaches the node after the node's feature is set to the value.

    Args:
        split: The split node to reach.
        bases: The float64 rows to copy.
        booster: The booster that holds the node.

    Returns:
        One row per boundary value, or None if no row reaches the node with
        one of the values.
    """
    rows = []
    for value in boundary_values(split.threshold):
        candidates = bases.copy()
        candidates[:, split.feature] = value
        leaves = booster.predict(
            candidates.astype(np.float32), pred_leaf=True, num_threads=1
        )
        reached = np.isin(leaves[:, split.tree], sorted(split.leaves))
        if not reached.any():
            return None
        rows.append(candidates[np.argmax(reached)])
    return rows


def threshold_rows(
    splits: Sequence[Split],
    bases: np.ndarray,
    booster: lgb.Booster,
    rng: np.random.Generator,
    required: set[tuple[str, bool]],
) -> np.ndarray:
    """Returns rows that reach split nodes with one feature near the threshold.

    Goes through the split nodes in random order and, for up to MAX_SPLITS of
    them, adds the rows from reaching_rows. Nodes that some boundary value
    cannot reach are skipped.

    Args:
        splits: The split nodes of the model.
        bases: The float64 rows to copy.
        booster: The booster that holds the nodes.
        rng: The random generator that orders the nodes.
        required: The (missing type, default left) pairs that the picked nodes
            must cover besides those of the model.

    Returns:
        A float64 array with one row per boundary value of each picked node.

    Raises:
        RuntimeError: The picked nodes miss a pair of missing type and default
            direction that the model has or that `required` lists.
    """
    picked = []
    rows = []
    for index in rng.permutation(len(splits)):
        if len(picked) == MAX_SPLITS:
            break
        split = splits[index]
        reaching = reaching_rows(split, bases, booster)
        if reaching is not None:
            picked.append(split)
            rows.extend(reaching)
    kinds = {(split.missing_type, split.default_left) for split in splits}
    kinds |= required
    covered = {(split.missing_type, split.default_left) for split in picked}
    if covered != kinds:
        raise RuntimeError(
            f"picked splits cover {sorted(covered)} of {sorted(kinds)}"
        )
    return np.array(rows)


def write_csv(
    path: pathlib.Path,
    header: Sequence[str],
    rows: Iterable[Iterable[float | np.integer]],
) -> None:
    """Writes rows as CSV with LF line endings and UTF-8 on every OS.

    Args:
        path: The file to write.
        header: The column names.
        rows: The rows, each a sequence of numbers.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(",".join(header) + "\n")
        for row in rows:
            file.write(",".join(format_number(value) for value in row) + "\n")


def format_number(value: float | np.integer) -> str:
    """Returns a number as text that reads back to the same value."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.17g}"


def write_variant(
    directory: pathlib.Path, booster: lgb.Booster, inputs: np.ndarray
) -> None:
    """Writes the model, inputs, and LightGBM's predictions for one model.

    Args:
        directory: The directory to write into.
        booster: The booster to write and predict with.
        inputs: The float64 input rows.
    """
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(directory / "model.txt")
    write_csv(
        directory / "inputs.csv",
        [f"f{index}" for index in range(FEATURE_COUNT)],
        inputs,
    )
    inputs32 = inputs.astype(np.float32)
    raw = booster.predict(inputs32, raw_score=True, num_threads=1)
    leaves = booster.predict(inputs32, pred_leaf=True, num_threads=1)
    header = ["raw"] + [f"leaf_{index}" for index in range(leaves.shape[1])]
    rows = [[score, *leaf_row] for score, leaf_row in zip(raw, leaves)]
    write_csv(directory / "expected.csv", header, rows)


def main(argv: Sequence[str]) -> None:
    """Writes the fixtures into the directory named by argv[1].

    Args:
        argv: The program name and the output directory.

    Raises:
        RuntimeError: The rows near thresholds do not cover a kind of split
            that the model has or that its variant requires.
    """
    if len(argv) != 2:
        sys.exit(f"usage: {argv[0]} <output directory>")
    output = pathlib.Path(argv[1])
    rng = np.random.default_rng(0)
    training = make_rows(rng, TRAINING_ROWS)
    labels = make_labels(rng, training)
    tests = make_rows(rng, TEST_ROWS)

    for name, (extra, required) in VARIANTS.items():
        trained = lgb.train(
            {**PARAMETERS, **extra},
            lgb.Dataset(training, label=labels),
            num_boost_round=TREE_COUNT,
        )
        booster = round_thresholds(trained)
        splits = split_nodes(booster)
        near_thresholds = threshold_rows(splits, tests, booster, rng, required)
        inputs = np.concatenate([tests, near_thresholds])
        write_variant(output / name, booster, inputs)
        print(f"{name}: {len(inputs)} rows")


if __name__ == "__main__":
    main(sys.argv)

# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Reads `expected.csv`, the core's reference values for the fixture model.

The core's tests write and watch the file, so the bindings compare their
results with the same values bit for bit.
"""

from collections.abc import Mapping, Sequence
import csv
import dataclasses
import itertools
import pathlib
import unittest

import numpy as np
import numpy.typing as npt

import shindo_est

REPOSITORY = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = REPOSITORY / "tests" / "fixtures" / "bundle"

# The float64 arrays of a prediction, which expected.csv holds for each site.
FLOAT_ARRAYS = ("instrumental", "probability", "regression")

# The columns that hold the inputs of a call, the same for each of its sites.
_CALL_COLUMNS = (
    "hypocenter_latitude",
    "hypocenter_longitude",
    "hypocenter_depth",
    "hypocenter_magnitude",
    "offset",
)
_FEATURE_PREFIX = "feature_"


@dataclasses.dataclass(frozen=True)
class Call:
    """One call to the fixture model and what the core returned for it.

    Attributes:
        hypocenter_latitude: The latitude of the hypocenter in degrees.
        hypocenter_longitude: The longitude of the hypocenter in degrees.
        hypocenter_depth: The depth of the hypocenter in km, NaN if unknown.
        hypocenter_magnitude: The JMA magnitude of the hypocenter.
        offset: The offset that the call added to the regressor's
            instrumental intensity.
        site_latitude: The latitude of each site in degrees.
        site_longitude: The longitude of each site in degrees.
        site_arv: The ARV of each site.
        features: The 14 features of each site, in the order of
            `shindo_est.FEATURE_NAMES`.
        status: The status of each site.
        intensity_class: The intensity class of each site, -1 for none.
        instrumental: The instrumental intensity of each site.
        probability: The probability that each site is felt.
        regression: The regressor's intensity at each site.
    """

    hypocenter_latitude: float
    hypocenter_longitude: float
    hypocenter_depth: float
    hypocenter_magnitude: float
    offset: float
    site_latitude: list[float]
    site_longitude: list[float]
    site_arv: list[float]
    features: list[list[float]]
    status: list[int]
    intensity_class: list[int]
    instrumental: list[float]
    probability: list[float]
    regression: list[float]


def _call_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in _CALL_COLUMNS)


def _call(rows: Sequence[Mapping[str, str]]) -> Call:
    """Returns the call that a group of rows of the file describes."""
    first = rows[0]

    def floats(column: str) -> list[float]:
        return [float(row[column]) for row in rows]

    def ints(column: str) -> list[int]:
        return [int(row[column]) for row in rows]

    feature_columns = [
        name for name in first if name.startswith(_FEATURE_PREFIX)
    ]
    return Call(
        hypocenter_latitude=float(first["hypocenter_latitude"]),
        hypocenter_longitude=float(first["hypocenter_longitude"]),
        hypocenter_depth=float(first["hypocenter_depth"]),
        hypocenter_magnitude=float(first["hypocenter_magnitude"]),
        offset=float(first["offset"]),
        site_latitude=floats("site_latitude"),
        site_longitude=floats("site_longitude"),
        site_arv=floats("site_arv"),
        features=[
            [float(row[column]) for column in feature_columns] for row in rows
        ],
        status=ints("status"),
        intensity_class=ints("intensity_class"),
        instrumental=floats("instrumental"),
        probability=floats("probability"),
        regression=floats("regression"),
    )


def calls() -> list[Call]:
    """Returns the calls in `expected.csv`, in the order of the file.

    Consecutive rows whose first five columns have the same text belong to
    one call. The text is compared rather than the numbers, so that rows with
    a NaN depth still group.

    Returns:
        The calls, each with its sites in the order of the file.
    """
    path = FIXTURES / "expected.csv"
    with path.open(encoding="utf-8", newline="") as file:
        groups = itertools.groupby(csv.DictReader(file), key=_call_key)
        return [_call(list(rows)) for _, rows in groups]


def hypocenter(call: Call) -> shindo_est.Hypocenter:
    """Returns the hypocenter of a call."""
    return shindo_est.Hypocenter(
        call.hypocenter_latitude,
        call.hypocenter_longitude,
        call.hypocenter_depth,
        call.hypocenter_magnitude,
    )


def assert_same_bits(
    test: unittest.TestCase,
    actual: npt.ArrayLike,
    expected: Sequence[float],
    context: str,
) -> None:
    """Fails the test unless the values are the same float64 bits.

    Two values match when both are NaN or when their float64 bits are the
    same, so 0.0 and -0.0 differ.

    Args:
        test: The test to fail.
        actual: The values to check.
        expected: The values they should be.
        context: What the values are, for the failure message.
    """
    actual_values = np.asarray(actual, dtype=np.float64)
    expected_values = np.asarray(expected, dtype=np.float64)
    test.assertEqual(
        actual_values.shape, expected_values.shape, f"{context}: shapes"
    )
    actual_bits = actual_values.view(np.uint64)
    expected_bits = expected_values.view(np.uint64)
    for index, (value, expected_value) in enumerate(
        zip(actual_values, expected_values)
    ):
        both_nan = np.isnan(value) and np.isnan(expected_value)
        if not both_nan and actual_bits[index] != expected_bits[index]:
            test.fail(
                f"{context}: index {index} is {value!r}, "
                f"expected {expected_value!r}"
            )

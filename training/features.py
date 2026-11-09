# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Computes model input features with the installed shindo_est extension.

The extension is built from crates/shindo-est-py and has to be reinstalled
after the core changes. check_extension finds an out-of-date build before it
computes features for training.
"""

import concurrent.futures
import csv
import math
import pathlib
import struct

import numpy as np
import numpy.typing as npt

import shindo_est

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY / "tests" / "fixtures" / "bundle"
EXPECTED = FIXTURES / "expected.csv"
FEATURE_NAMES = (
    "mj",
    "depth",
    "qlat",
    "qlon",
    "slat",
    "slon",
    "log_hypo",
    "epi",
    "az_sin",
    "az_cos",
    "log_arv",
    "slab_top",
    "updip",
    "path_frac",
)
FEATURE_VERSION = 1

_INPUT_COLUMNS = (
    "hypocenter_latitude",
    "hypocenter_longitude",
    "hypocenter_depth",
    "hypocenter_magnitude",
    "site_latitude",
    "site_longitude",
    "site_arv",
)
_CHUNK_ROWS = 262_144
_REINSTALL = (
    "the installed shindo_est is out of date; reinstall it with"
    " `pip install ./crates/shindo-est-py`"
)


class StaleExtensionError(RuntimeError):
    """The installed extension does not compute the committed features."""


def check_extension(expected: pathlib.Path = EXPECTED) -> None:
    """Checks the installed extension against the committed features.

    Computes the features of every row of expected.csv with the fixture's
    slab grids and compares them with the file bit for bit, where two NaN
    values are equal.

    Args:
        expected: The CSV file of reference values.

    Raises:
        StaleExtensionError: The feature version, the feature names, or a
            feature differs from what this package expects.
    """
    if shindo_est.FEATURE_VERSION != FEATURE_VERSION:
        raise StaleExtensionError(
            f"the extension has feature version {shindo_est.FEATURE_VERSION},"
            f" not {FEATURE_VERSION}; {_REINSTALL}"
        )
    if tuple(shindo_est.FEATURE_NAMES) != FEATURE_NAMES:
        raise StaleExtensionError(
            f"the extension has features {shindo_est.FEATURE_NAMES};"
            f" {_REINSTALL}"
        )
    with expected.open(newline="", encoding="ascii") as file:
        rows = list(csv.DictReader(file))
    slabs = shindo_est.SlabGeometry.read(
        FIXTURES / "kuril.csv", FIXTURES / "izu_bonin.csv"
    )
    computed = slabs.features(
        **{
            name: [float(row[name]) for row in rows]
            for name in _INPUT_COLUMNS
        }
    )
    for index, row in enumerate(rows):
        for column, name in enumerate(FEATURE_NAMES):
            actual = float(computed[index, column])
            committed = float(row[f"feature_{name}"])
            if not _same_bits(actual, committed):
                raise StaleExtensionError(
                    f"row {index + 1}, feature {name}: the extension computes"
                    f" {actual!r}, {expected.name} has {committed!r};"
                    f" {_REINSTALL}"
                )


def read_slabs() -> shindo_est.SlabGeometry:
    """Reads the repository's Pacific slab grids.

    Returns:
        The slab geometry of data/slab2.

    Raises:
        OSError: A grid file cannot be read.
        shindo_est.SlabGridError: A grid file is not a valid grid.
    """
    directory = REPOSITORY / "data" / "slab2"
    return shindo_est.SlabGeometry.read(
        directory / "kur.csv", directory / "izu.csv"
    )


def compute(
    slabs: shindo_est.SlabGeometry,
    *,
    hypocenter_latitude: npt.NDArray[np.float64],
    hypocenter_longitude: npt.NDArray[np.float64],
    hypocenter_depth: npt.NDArray[np.float64],
    hypocenter_magnitude: npt.NDArray[np.float64],
    site_latitude: npt.NDArray[np.float64],
    site_longitude: npt.NDArray[np.float64],
    site_arv: npt.NDArray[np.float64],
    threads: int,
) -> npt.NDArray[np.float32]:
    """Computes the features of many rows in chunks on several threads.

    Each row pairs a hypocenter with a site. The features are rounded to
    float32, as LightGBM compares them. Calls check_extension first, so an
    out-of-date extension never computes features for training.

    Args:
        slabs: The slab geometry.
        hypocenter_latitude: The hypocenter latitude of each row in degrees.
        hypocenter_longitude: The hypocenter longitude of each row in degrees.
        hypocenter_depth: The hypocenter depth of each row in kilometers.
        hypocenter_magnitude: The JMA magnitude of each row.
        site_latitude: The site latitude of each row in degrees.
        site_longitude: The site longitude of each row in degrees.
        site_arv: The site ARV of each row.
        threads: The number of threads.

    Returns:
        A C-ordered float32 array of shape (rows, 14) whose columns follow
        FEATURE_NAMES.

    Raises:
        StaleExtensionError: The installed extension is out of date.
        ValueError: The arrays have different lengths.
    """
    check_extension()
    inputs = {
        "hypocenter_latitude": hypocenter_latitude,
        "hypocenter_longitude": hypocenter_longitude,
        "hypocenter_depth": hypocenter_depth,
        "hypocenter_magnitude": hypocenter_magnitude,
        "site_latitude": site_latitude,
        "site_longitude": site_longitude,
        "site_arv": site_arv,
    }
    lengths = {name: len(values) for name, values in inputs.items()}
    if len(set(lengths.values())) > 1:
        listed = ", ".join(
            f"{name} {length}" for name, length in lengths.items()
        )
        raise ValueError(f"the arrays must have the same length: {listed}")
    count = len(site_arv)
    result = np.empty((count, len(FEATURE_NAMES)), np.float32)

    def fill(start: int) -> None:
        stop = min(start + _CHUNK_ROWS, count)
        result[start:stop] = slabs.features(
            **{name: values[start:stop] for name, values in inputs.items()}
        )

    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        futures = [
            executor.submit(fill, start)
            for start in range(0, count, _CHUNK_ROWS)
        ]
        for future in futures:
            future.result()
    return result


def _same_bits(actual: float, committed: float) -> bool:
    if math.isnan(actual) and math.isnan(committed):
        return True
    return struct.pack("<d", actual) == struct.pack("<d", committed)

# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Chooses the rows, splits, and features for comparing the ARV sources.

Every row given to training or evaluation while choosing passes rows(): a
quake from FIRST_YEAR to LAST_YEAR, so the later quakes stay unseen for the
final report, and a station with an ARV from both sources, so that every
case learns and is measured on the same rows. Training never uses a
held-out station; the tests use every station, so that held-out stations
can be measured apart.
"""

from collections.abc import Mapping
import dataclasses
import math

import numpy as np
import numpy.typing as npt

import shindo_est
from training import dataset
from training import features

FIRST_YEAR = 2004
LAST_YEAR = 2019
TRAINING_YEARS = (FIRST_YEAR, 2016)
TEST_YEARS = (2017, LAST_YEAR)
CASES = ("cabinet_office", "jshis", "none")

# The column of the ARV table that each case takes.
_SOURCES = {"cabinet_office": 0, "jshis": 1, "none": 0}
_ARV_FEATURE = "log_arv"

type ArvTable = Mapping[str, tuple[float | None, float | None]]


@dataclasses.dataclass(frozen=True)
class Split:
    """A division of the rows into training and test.

    Attributes:
        name: "year", or "region:" and the region left out of training.
        train: Whether each row is trained on.
        test: Whether each row is measured.
    """

    name: str
    train: npt.NDArray[np.bool_]
    test: npt.NDArray[np.bool_]


def rows(data: dataset.Dataset, table: ArvTable) -> npt.NDArray[np.bool_]:
    """Returns the rows that may be used while choosing.

    Args:
        data: The dataset.
        table: The ARVs from the Cabinet Office and from J-SHIS of each
            station number, None where a source has none.

    Returns:
        Whether each row has a quake from FIRST_YEAR to LAST_YEAR and a
        station with an ARV from both sources.
    """
    year = data.quake_year[data.row_quake]
    both = _both_arv(data, table)[data.row_station]
    return _between(year, (FIRST_YEAR, LAST_YEAR)) & both


def splits(
    data: dataset.Dataset, usable: npt.NDArray[np.bool_]
) -> list[Split]:
    """Divides the usable rows by year and by region.

    Args:
        data: The dataset.
        usable: The rows from rows().

    Returns:
        The year split, trained on TRAINING_YEARS and measured on
        TEST_YEARS, then one split for each of dataset.REGIONS, measured on
        the quakes of the region and trained on the others. No split trains
        on a held-out station.
    """
    year = data.quake_year[data.row_quake]
    trainable = usable & ~data.station_held_out[data.row_station]
    result = [
        Split(
            name="year",
            train=trainable & _between(year, TRAINING_YEARS),
            test=usable & _between(year, TEST_YEARS),
        )
    ]
    for name in dataset.REGIONS:
        inside = (data.quake_region == name)[data.row_quake]
        result.append(
            Split(
                name=f"region:{name}",
                train=trainable & ~inside,
                test=usable & inside,
            )
        )
    return result


def candidates(
    data: dataset.Dataset, test: npt.NDArray[np.bool_], table: ArvTable
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Returns every station that could have recorded the test quakes.

    These include the unfelt stations that sampling left out, so that the
    largest prediction of a quake can be taken over all of them.

    Args:
        data: The dataset.
        test: The test rows of a split.
        table: The ARV table given to rows().

    Returns:
        A tuple (quakes, stations) of the quake and the station of each
        candidate, in the order of the quakes and then of the stations: for
        each quake of the test rows, the stations with an ARV from both
        sources that were in operation at its origin time or recorded it.
    """
    both = _both_arv(data, table)
    quakes: list[npt.NDArray[np.int32]] = []
    stations: list[npt.NDArray[np.int32]] = []
    for quake in np.unique(data.row_quake[test]):
        time = data.quake_time[quake]
        chosen = (data.station_start <= time) & (
            np.isnat(data.station_end) | (time < data.station_end)
        )
        start, stop = np.searchsorted(data.row_quake, [quake, quake + 1])
        recorded = data.row_station[start:stop][data.felt[start:stop]]
        chosen[recorded] = True
        found = np.flatnonzero(chosen & both).astype(np.int32)
        quakes.append(np.full(len(found), quake, dtype=np.int32))
        stations.append(found)
    return _join(quakes), _join(stations)


def feature_matrix(
    data: dataset.Dataset,
    quakes: npt.NDArray[np.int32],
    stations: npt.NDArray[np.int32],
    table: ArvTable,
    case: str,
    slabs: shindo_est.SlabGeometry,
    *,
    threads: int,
) -> tuple[npt.NDArray[np.float32], tuple[str, ...]]:
    """Computes the features of quake and station pairs for one case.

    Args:
        data: The dataset.
        quakes: The quake of each pair.
        stations: The station of each pair.
        table: The ARV table given to rows().
        case: One of CASES. cabinet_office and jshis take the ARV of that
            source; none takes the Cabinet Office's and drops log_arv.
        slabs: The slab geometry.
        threads: The number of threads.

    Returns:
        A tuple (values, names) of the features of each pair and the name of
        each column.

    Raises:
        ValueError: case is not one of CASES, or a station has no ARV from
            the source of the case.
        features.StaleExtensionError: The installed extension is out of
            date.
    """
    if case not in _SOURCES:
        raise ValueError(f"the case {case!r} is not one of {CASES}")
    source = _SOURCES[case]
    station_arv = np.full(len(data.station_number), math.nan)
    for at, number in enumerate(data.station_number):
        value = table.get(str(number), (None, None))[source]
        if value is not None:
            station_arv[at] = value
    arv = station_arv[stations]
    missing = np.isnan(arv)
    if missing.any():
        number = data.station_number[stations[np.argmax(missing)]]
        raise ValueError(f"station {number} has no ARV for {case!r}")
    values = features.compute(
        slabs,
        hypocenter_latitude=data.quake_latitude[quakes],
        hypocenter_longitude=data.quake_longitude[quakes],
        hypocenter_depth=data.quake_depth[quakes],
        hypocenter_magnitude=data.quake_magnitude[quakes],
        site_latitude=data.station_latitude[stations],
        site_longitude=data.station_longitude[stations],
        site_arv=arv,
        threads=threads,
    )
    if case != "none":
        return values, features.FEATURE_NAMES
    column = features.FEATURE_NAMES.index(_ARV_FEATURE)
    names = tuple(
        name for name in features.FEATURE_NAMES if name != _ARV_FEATURE
    )
    return np.delete(values, column, axis=1), names


def _both_arv(
    data: dataset.Dataset, table: ArvTable
) -> npt.NDArray[np.bool_]:
    """Returns whether each station has an ARV from both sources."""
    return np.array(
        [
            None not in table.get(str(number), (None, None))
            for number in data.station_number
        ],
        dtype=np.bool_,
    )


def _between(
    year: npt.NDArray[np.int16], years: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    inside: npt.NDArray[np.bool_] = (years[0] <= year) & (year <= years[1])
    return inside


def _join(arrays: list[npt.NDArray[np.int32]]) -> npt.NDArray[np.int32]:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)

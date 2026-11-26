# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Chooses the rows, splits, and features for training and evaluation.

Every row given to training or evaluation while choosing passes rows(): a
quake from FIRST_YEAR to LAST_YEAR, so the later quakes stay unseen for the
final report, and a station with an ARV from the given sources. The ARV
comparison asks for both sources, so that every case learns and is measured
on the same rows; later choices ask for the chosen source only, and may
leave out the suspicious unfelt rows of shallow quakes. Training never
uses a held-out station; the tests use every station, so that held-out
stations can be measured apart.
"""

from collections.abc import Mapping
from collections.abc import Sequence
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
# The depth in kilometers from which a quake keeps its suspicious unfelt
# rows: a deep quake can leave a station unshaken next to one that
# recorded 2.5, so its rows are kept as likely real.
DEEP_QUAKE_DEPTH = 150.0
CASES = ("cabinet_office", "jshis", "none")
COMPARED_SOURCES = ("cabinet_office", "jshis")

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


def stations_with_arv(
    data: dataset.Dataset, table: ArvTable, sources: Sequence[str]
) -> npt.NDArray[np.bool_]:
    """Returns whether each station has an ARV from every given source.

    Args:
        data: The dataset.
        table: The ARVs from the Cabinet Office and from J-SHIS of each
            station number, None where a source has none.
        sources: Some of COMPARED_SOURCES.

    Raises:
        ValueError: A source is not one of COMPARED_SOURCES.
    """
    unknown = [name for name in sources if name not in COMPARED_SOURCES]
    if unknown:
        raise ValueError(
            f"unknown sources {unknown}; known: {COMPARED_SOURCES}"
        )
    columns = [_SOURCES[name] for name in sources]
    return np.array(
        [
            all(
                table.get(str(number), (None, None))[column] is not None
                for column in columns
            )
            for number in data.station_number
        ],
        dtype=np.bool_,
    )


def usable(
    data: dataset.Dataset,
    table: ArvTable,
    *,
    sources: Sequence[str] = COMPARED_SOURCES,
    keep_suspicious: bool = True,
) -> npt.NDArray[np.bool_]:
    """Returns the rows of any year that training or evaluation may use.

    Args:
        data: The dataset.
        table: The ARV table given to stations_with_arv().
        sources: The sources each station needs an ARV from.
        keep_suspicious: Whether to keep the suspicious unfelt rows of
            quakes shallower than DEEP_QUAKE_DEPTH.

    Returns:
        Whether each row has a station with an ARV from every source and,
        unless keep_suspicious, is not a suspicious row of a quake
        shallower than DEEP_QUAKE_DEPTH.

    Raises:
        ValueError: A source is not one of COMPARED_SOURCES.
    """
    chosen = stations_with_arv(data, table, sources)[data.row_station]
    if keep_suspicious:
        return chosen
    shallow = data.quake_depth[data.row_quake] < DEEP_QUAKE_DEPTH
    kept: npt.NDArray[np.bool_] = chosen & ~(data.suspicious & shallow)
    return kept


def rows(
    data: dataset.Dataset,
    table: ArvTable,
    *,
    sources: Sequence[str] = COMPARED_SOURCES,
    keep_suspicious: bool = True,
) -> npt.NDArray[np.bool_]:
    """Returns the rows that may be used while choosing.

    Args:
        data: The dataset.
        table: The ARV table given to stations_with_arv().
        sources: The sources each station needs an ARV from.
        keep_suspicious: The keep_suspicious given to usable().

    Returns:
        The rows of usable() with a quake from FIRST_YEAR to LAST_YEAR.

    Raises:
        ValueError: A source is not one of COMPARED_SOURCES.
    """
    year = data.quake_year[data.row_quake]
    chosen = usable(
        data, table, sources=sources, keep_suspicious=keep_suspicious
    )
    return _between(year, (FIRST_YEAR, LAST_YEAR)) & chosen


def splits(
    data: dataset.Dataset, allowed: npt.NDArray[np.bool_]
) -> list[Split]:
    """Divides the allowed rows by year and by region.

    Args:
        data: The dataset.
        allowed: The rows from rows().

    Returns:
        The year split, trained on TRAINING_YEARS and measured on
        TEST_YEARS, then one split for each of dataset.REGIONS, measured on
        the quakes of the region and trained on the others. No split trains
        on a held-out station.
    """
    year = data.quake_year[data.row_quake]
    trainable = allowed & ~data.station_held_out[data.row_station]
    result = [
        Split(
            name="year",
            train=trainable & _between(year, TRAINING_YEARS),
            test=allowed & _between(year, TEST_YEARS),
        )
    ]
    for name in dataset.REGIONS:
        inside = (data.quake_region == name)[data.row_quake]
        result.append(
            Split(
                name=f"region:{name}",
                train=trainable & ~inside,
                test=allowed & inside,
            )
        )
    return result


def candidates(
    data: dataset.Dataset,
    test: npt.NDArray[np.bool_],
    eligible: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Returns every station that could have recorded the test quakes.

    These include the unfelt stations that sampling left out, so that the
    largest prediction of a quake can be taken over all of them.

    Args:
        data: The dataset.
        test: The test rows of a split.
        eligible: Whether each station may be a candidate.

    Returns:
        A tuple (quakes, stations) of the quake and the station of each
        candidate, in the order of the quakes and then of the stations: for
        each quake of the test rows, the eligible stations that were in
        operation at its origin time or recorded it.
    """
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
        found = np.flatnonzero(chosen & eligible).astype(np.int32)
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
        table: The ARV table given to stations_with_arv().
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


def _between(
    year: npt.NDArray[np.int16], years: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    inside: npt.NDArray[np.bool_] = (years[0] <= year) & (year <= years[1])
    return inside


def _join(arrays: list[npt.NDArray[np.int32]]) -> npt.NDArray[np.int32]:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)

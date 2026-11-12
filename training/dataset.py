# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Builds rows of training data from the JMA bulletin and its stations.

Each row pairs a quake with a station. A station that recorded the quake
gives a felt row with its instrumental intensity. A station in operation that
did not record it gives an unfelt row. Unfelt rows within
max(2 x the farthest felt station, 50 km) of the epicenter are all kept; the
farther ones are sampled at 3% and weighted by 1 / 0.03, using a uniform
number drawn from the quake and the station, so that the sample does not
depend on the order of either. An unfelt row is suspicious when a station
within 20 km recorded 2.5 or more, which suggests the station was out of
operation rather than unshaken.

Usage:
    python -m training.dataset [--sums PATH] [--jma DIR] [--out PATH]
"""

import argparse
from collections.abc import Iterable, Sequence
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import pathlib
import sys
from typing import Self
import zipfile
import zlib

import numpy as np
import numpy.typing as npt

from training import bulletin
from training import features
from training import sources

SAMPLE_RATE = 0.03
KEEP_FACTOR = 2.0
MIN_KEEP_DISTANCE = 50.0
SUSPICIOUS_DISTANCE = 20.0
SUSPICIOUS_INSTRUMENTAL = 2.5
EARTH_RADIUS = 6371.008
REGIONS = ("hokkaido", "tohoku", "kanto", "izu", "west", "kyushu")
YEARS = tuple(range(2004, 2025))
DEFAULT_DATASET = features.REPOSITORY / "training" / "data" / "dataset.npz"
SUMMARY_NAME = "dataset-summary.json"

_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_NEIGHBOR_BLOCK = 512

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Quakes, stations, and the rows that pair them.

    Times are in Japan Standard Time, truncated to the minute. A station
    period that cannot be read has both ends NaT, and a station in operation
    has only its end NaT. The rows are in the order of the quakes and, within
    a quake, of the station numbers.

    Attributes:
        quake_id: Columns 2 to 17 of each adopted hypocenter record.
        quake_year: The year of each origin time.
        quake_latitude: The latitude of each epicenter in degrees.
        quake_longitude: The longitude of each epicenter in degrees.
        quake_depth: The depth of each hypocenter in kilometers.
        quake_magnitude: The JMA magnitude of each quake.
        quake_region: The region of each epicenter, one of REGIONS.
        quake_time: The origin time of each quake.
        station_number: The number of each station, in ascending order.
        station_latitude: The latitude of each station in degrees.
        station_longitude: The longitude of each station in degrees.
        station_held_out: Whether each station is left out of training.
        station_start: The start of each station's period.
        station_end: The end of each station's period.
        row_quake: The index of the quake of each row.
        row_station: The index of the station of each row.
        felt: Whether the station recorded the quake.
        observed: The instrumental intensity, NaN when it is not known.
        weight: 1, or 1 / SAMPLE_RATE for a sampled unfelt row.
        suspicious: Whether an unfelt row has a station within 20 km that
            recorded 2.5 or more.
    """

    quake_id: npt.NDArray[np.str_]
    quake_year: npt.NDArray[np.int16]
    quake_latitude: npt.NDArray[np.float64]
    quake_longitude: npt.NDArray[np.float64]
    quake_depth: npt.NDArray[np.float64]
    quake_magnitude: npt.NDArray[np.float64]
    quake_region: npt.NDArray[np.str_]
    quake_time: npt.NDArray[np.datetime64]
    station_number: npt.NDArray[np.str_]
    station_latitude: npt.NDArray[np.float64]
    station_longitude: npt.NDArray[np.float64]
    station_held_out: npt.NDArray[np.bool_]
    station_start: npt.NDArray[np.datetime64]
    station_end: npt.NDArray[np.datetime64]
    row_quake: npt.NDArray[np.int32]
    row_station: npt.NDArray[np.int32]
    felt: npt.NDArray[np.bool_]
    observed: npt.NDArray[np.float32]
    weight: npt.NDArray[np.float32]
    suspicious: npt.NDArray[np.bool_]

    def save(self, path: pathlib.Path) -> None:
        """Writes the arrays to an .npz file.

        Args:
            path: The file.

        Raises:
            OSError: The file cannot be written.
        """
        np.savez(
            path,
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
            },
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        """Reads the arrays written by save.

        Args:
            path: The file.

        Returns:
            The dataset.

        Raises:
            ValueError: The file does not hold exactly the arrays of a
                dataset, or holds objects that need pickling.
            OSError: The file cannot be read.
        """
        names = [field.name for field in dataclasses.fields(cls)]
        with np.load(path, allow_pickle=False) as arrays:
            if sorted(arrays.files) != sorted(names):
                raise ValueError(f"{path}: not the arrays of a dataset")
            return cls(**{name: arrays[name] for name in names})


@dataclasses.dataclass(frozen=True)
class Summary:
    """Counts of what build used and left out.

    Attributes:
        quakes: The quakes in the dataset: those with a JMA magnitude above
            0, including any that gave no rows.
        nonpositive_magnitude: The quakes left out for a JMA magnitude of 0
            or less.
        felt_rows: The rows of stations that recorded a quake.
        felt_rows_with_instrumental: The felt rows with an intensity.
        unknown_station_records: The records of numbers not in code_p.dat,
            which are left out.
        conflicting_duplicates: The stations with differing records for the
            same quake, which become one record without an intensity.
        unfelt_candidates: The stations in operation that did not record a
            quake, before sampling.
        unfelt_kept: The unfelt rows kept for being near the epicenter.
        unfelt_sampled: The farther unfelt rows kept by sampling.
        suspicious_by_year: For each year, the suspicious unfelt candidates
            and all unfelt candidates, before sampling.
        suspicious_by_station: The same for each station number.
    """

    quakes: int
    nonpositive_magnitude: int
    felt_rows: int
    felt_rows_with_instrumental: int
    unknown_station_records: int
    conflicting_duplicates: int
    unfelt_candidates: int
    unfelt_kept: int
    unfelt_sampled: int
    suspicious_by_year: dict[int, tuple[int, int]]
    suspicious_by_station: dict[str, tuple[int, int]]

    def to_json(self) -> str:
        """Returns the counts as JSON, with the years as strings."""
        return json.dumps(dataclasses.asdict(self), indent=2)


def region(latitude: float, longitude: float) -> str:
    """Returns the region of an epicenter, one of REGIONS.

    Args:
        latitude: The latitude in degrees.
        longitude: The longitude in degrees.
    """
    if latitude >= 41.0:
        return "hokkaido"
    if latitude >= 37.0:
        return "tohoku"
    if longitude >= 138.0:
        return "izu" if latitude < 34.0 else "kanto"
    if latitude < 34.0 and longitude < 132.6:
        return "kyushu"
    return "west"


def is_held_out(number: str) -> bool:
    """Returns whether a station is left out of training, about 20%."""
    return zlib.crc32(number.encode("ascii")) % 5 == 0


def splitmix64(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Returns the SplitMix64 output for each state, wrapping at 2 ** 64."""
    mixed = values + _GOLDEN_GAMMA
    mixed = (mixed ^ (mixed >> np.uint64(30))) * _MIX_1
    mixed = (mixed ^ (mixed >> np.uint64(27))) * _MIX_2
    return mixed ^ (mixed >> np.uint64(31))


def sample_uniform(
    quake_id: int, numbers: npt.NDArray[np.uint64]
) -> npt.NDArray[np.float64]:
    """Returns a uniform number in [0, 1) for a quake and each station.

    Args:
        quake_id: The quake, as an integer below 2 ** 64.
        numbers: The stations, as integers.

    Returns:
        The top 53 bits of splitmix64(splitmix64(quake_id) ^ number),
        divided by 2 ** 53.
    """
    seed = splitmix64(np.array([quake_id], dtype=np.uint64))
    bits = splitmix64(seed ^ numbers) >> np.uint64(11)
    return bits.astype(np.float64) * 2.0**-53


def build(
    stations: Sequence[bulletin.Station], quakes: Iterable[bulletin.Quake]
) -> tuple[Dataset, Summary]:
    """Pairs the quakes with the stations.

    Args:
        stations: Every station in code_p.dat.
        quakes: The quakes, in the order of the rows.

    Returns:
        The rows and the counts.

    Raises:
        ValueError: Two stations have the same number, or a quake with a
            JMA magnitude above 0 has a position, depth, or magnitude that is
            not finite.
    """
    table = _StationTable(stations)
    rows = _Rows(len(table.numbers))
    for quake in quakes:
        if quake.magnitude <= 0:
            rows.nonpositive_magnitude += 1
            continue
        values = (quake.latitude, quake.longitude, quake.depth, quake.magnitude)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"quake {quake.id}: {values} are not all finite")
        rows.add(quake, table)
    return rows.dataset(table), rows.summary(table)


def write(
    jma: pathlib.Path,
    out: pathlib.Path,
    *,
    sums_path: pathlib.Path = sources.SHA256SUMS,
    years: Iterable[int] = YEARS,
) -> Summary:
    """Checks the JMA files, then writes the dataset and its summary.

    Args:
        jma: The directory holding code_p.zip and i<year>.zip.
        out: The .npz file to write. The summary goes next to it as
            dataset-summary.json.
        sums_path: The SHA256SUMS file.
        years: The years of the bulletin to read.

    Returns:
        The counts.

    Raises:
        sources.SourceError: A file fails its check.
        ValueError: A file cannot be read as expected.
        zipfile.BadZipFile: A file is not a ZIP file.
        OSError: A file cannot be read or written.
    """
    sums = sources.read_sums(sums_path)
    station_list = jma / "code_p.zip"
    bulletins = [jma / f"i{year}.zip" for year in years]
    for path in (station_list, *bulletins):
        sources.verify(path, sums)
    stations = bulletin.read_stations(bulletin.read_zip(station_list))
    quakes: list[bulletin.Quake] = []
    for path in bulletins:
        kept, counts = bulletin.read_quakes(bulletin.read_zip(path))
        _logger.info("%s: %s", path.name, counts)
        quakes.extend(kept)
    data, summary = build(stations, quakes)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.save(out)
    (out.parent / SUMMARY_NAME).write_text(
        summary.to_json() + "\n", encoding="ascii"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: The arguments after the program name. None reads sys.argv.

    Returns:
        The exit status: 0 on success, 1 when a JMA file fails a check or
        cannot be read, or the dataset cannot be written.
    """
    parser = argparse.ArgumentParser(prog="python -m training.dataset")
    parser.add_argument(
        "--sums", type=pathlib.Path, default=sources.SHA256SUMS
    )
    parser.add_argument(
        "--jma", type=pathlib.Path, default=sources.DEFAULT_JMA_DIRECTORY
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_DATASET)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        summary = write(arguments.jma, arguments.out, sums_path=arguments.sums)
    except (
        sources.SourceError,
        ValueError,
        zipfile.BadZipFile,
        OSError,
    ) as error:
        _logger.error("%s", error)
        return 1
    _logger.info(
        "%d quakes, %d felt rows, %d unfelt rows",
        summary.quakes,
        summary.felt_rows,
        summary.unfelt_kept + summary.unfelt_sampled,
    )
    return 0


class _StationTable:
    """The stations in the order of their numbers, as arrays."""

    def __init__(self, stations: Sequence[bulletin.Station]) -> None:
        ordered = sorted(stations, key=lambda station: station.number)
        self.numbers = [station.number for station in ordered]
        for first, second in zip(self.numbers, self.numbers[1:]):
            if first == second:
                raise ValueError(f"station {first} appears twice")
        self.index = {number: at for at, number in enumerate(self.numbers)}
        self.latitude = np.array(
            [station.latitude for station in ordered], dtype=np.float64
        )
        self.longitude = np.array(
            [station.longitude for station in ordered], dtype=np.float64
        )
        periods = [station.period for station in ordered]
        self.start = np.array(
            [_minute(None if p is None else p.start) for p in periods],
            dtype="datetime64[m]",
        )
        self.end = np.array(
            [_minute(None if p is None else p.end) for p in periods],
            dtype="datetime64[m]",
        )
        self.keys = np.array(
            [_key(number) for number in self.numbers], dtype=np.uint64
        )
        self.neighbors = _neighbors(self.latitude, self.longitude)

    def operating(
        self, time: datetime.datetime, recorded: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Returns the stations in operation at time, except recorded."""
        minute = _minute(time)
        operating = (self.start <= minute) & (
            np.isnat(self.end) | (minute < self.end)
        )
        operating[recorded] = False
        return np.flatnonzero(operating)

    def distance(
        self, quake: bulletin.Quake, stations: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Returns the distance in kilometers from the epicenter."""
        return _distance(
            quake.latitude,
            quake.longitude,
            self.latitude[stations],
            self.longitude[stations],
        )

    def near_strong(
        self,
        recorded: npt.NDArray[np.int64],
        observed: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        """Returns whether each station is near one that recorded 2.5 up."""
        near = np.zeros(len(self.numbers), dtype=np.bool_)
        for station in recorded[observed >= SUSPICIOUS_INSTRUMENTAL]:
            near[self.neighbors[station]] = True
        return near


class _Rows:
    """The rows and counts gathered quake by quake."""

    def __init__(self, stations: int) -> None:
        self.quakes: list[bulletin.Quake] = []
        self.stations: list[npt.NDArray[np.int32]] = []
        self.felt: list[npt.NDArray[np.bool_]] = []
        self.observed: list[npt.NDArray[np.float32]] = []
        self.weight: list[npt.NDArray[np.float32]] = []
        self.suspicious: list[npt.NDArray[np.bool_]] = []
        self.nonpositive_magnitude = 0
        self.felt_rows = 0
        self.felt_rows_with_instrumental = 0
        self.unknown_station_records = 0
        self.conflicting_duplicates = 0
        self.unfelt_kept = 0
        self.unfelt_sampled = 0
        self.by_year: dict[int, tuple[int, int]] = {}
        self.suspicious_count = np.zeros(stations, dtype=np.int64)
        self.candidate_count = np.zeros(stations, dtype=np.int64)

    def add(self, quake: bulletin.Quake, table: _StationTable) -> None:
        """Adds the rows of a quake."""
        felt_stations, observed = self._felt(quake, table)
        candidates = table.operating(quake.origin_time, felt_stations)
        farthest = table.distance(quake, felt_stations).max(initial=0.0)
        near = table.distance(quake, candidates) <= max(
            KEEP_FACTOR * farthest, MIN_KEEP_DISTANCE
        )
        suspicious = table.near_strong(felt_stations, observed)[candidates]
        self._count(quake.origin_time.year, candidates, suspicious)
        sampled = ~near & (
            sample_uniform(_key(quake.id), table.keys[candidates])
            < SAMPLE_RATE
        )
        kept = near | sampled
        self.unfelt_kept += int(near.sum())
        self.unfelt_sampled += int(sampled.sum())
        unfelt = int(kept.sum())
        self._append(
            quake,
            stations=np.concatenate([felt_stations, candidates[kept]]),
            felt=np.arange(len(felt_stations) + unfelt) < len(felt_stations),
            observed=np.concatenate(
                [observed, np.full(unfelt, np.nan, dtype=np.float32)]
            ),
            weight=np.concatenate([
                np.ones(len(felt_stations), dtype=np.float32),
                np.where(near[kept], 1.0, 1.0 / SAMPLE_RATE).astype(
                    np.float32
                ),
            ]),
            suspicious=np.concatenate(
                [np.zeros(len(felt_stations), dtype=np.bool_), suspicious[kept]]
            ),
        )

    def _append(
        self,
        quake: bulletin.Quake,
        *,
        stations: npt.NDArray[np.int64],
        felt: npt.NDArray[np.bool_],
        observed: npt.NDArray[np.float32],
        weight: npt.NDArray[np.float32],
        suspicious: npt.NDArray[np.bool_],
    ) -> None:
        """Appends the rows of a quake in the order of the stations."""
        order = np.argsort(stations, kind="stable")
        self.quakes.append(quake)
        self.stations.append(stations[order].astype(np.int32))
        self.felt.append(felt[order])
        self.observed.append(observed[order])
        self.weight.append(weight[order])
        self.suspicious.append(suspicious[order])

    def _felt(
        self, quake: bulletin.Quake, table: _StationTable
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Counts the records of a quake, then returns the felt stations.

        Returns:
            The stations that recorded the quake, in ascending order, and
            the instrumental intensity of each, NaN when it is not known.
        """
        records: dict[str, list[bulletin.Record]] = {}
        for record in quake.records:
            records.setdefault(record.station, []).append(record)
        values: dict[int, float] = {}
        for number, same in records.items():
            if number not in table.index:
                self.unknown_station_records += len(same)
                continue
            if any(record != same[0] for record in same[1:]):
                self.conflicting_duplicates += 1
                values[table.index[number]] = math.nan
                continue
            instrumental = same[0].instrumental
            values[table.index[number]] = (
                math.nan if instrumental is None else instrumental
            )
        stations = np.array(sorted(values), dtype=np.int64)
        observed = np.array(
            [values[station] for station in stations], dtype=np.float32
        )
        self.felt_rows += len(stations)
        self.felt_rows_with_instrumental += int(
            np.count_nonzero(~np.isnan(observed))
        )
        return stations, observed

    def _count(
        self,
        year: int,
        candidates: npt.NDArray[np.int64],
        suspicious: npt.NDArray[np.bool_],
    ) -> None:
        flagged, total = self.by_year.get(year, (0, 0))
        self.by_year[year] = (
            flagged + int(suspicious.sum()),
            total + len(candidates),
        )
        self.suspicious_count[candidates] += suspicious
        self.candidate_count[candidates] += 1

    def dataset(self, table: _StationTable) -> Dataset:
        """Returns the rows gathered so far as a dataset."""
        quakes = self.quakes
        return Dataset(
            quake_id=np.array([quake.id for quake in quakes], dtype="U16"),
            quake_year=np.array(
                [quake.origin_time.year for quake in quakes], dtype=np.int16
            ),
            quake_latitude=_floats(quake.latitude for quake in quakes),
            quake_longitude=_floats(quake.longitude for quake in quakes),
            quake_depth=_floats(quake.depth for quake in quakes),
            quake_magnitude=_floats(quake.magnitude for quake in quakes),
            quake_region=np.array(
                [region(quake.latitude, quake.longitude) for quake in quakes],
                dtype="U8",
            ),
            quake_time=np.array(
                [_minute(quake.origin_time) for quake in quakes],
                dtype="datetime64[m]",
            ),
            station_number=np.array(table.numbers, dtype=str),
            station_latitude=table.latitude,
            station_longitude=table.longitude,
            station_held_out=np.array(
                [is_held_out(number) for number in table.numbers],
                dtype=np.bool_,
            ),
            station_start=table.start,
            station_end=table.end,
            row_quake=_join(
                [
                    np.full(len(stations), at, dtype=np.int32)
                    for at, stations in enumerate(self.stations)
                ],
                np.int32,
            ),
            row_station=_join(self.stations, np.int32),
            felt=_join(self.felt, np.bool_),
            observed=_join(self.observed, np.float32),
            weight=_join(self.weight, np.float32),
            suspicious=_join(self.suspicious, np.bool_),
        )

    def summary(self, table: _StationTable) -> Summary:
        """Returns the counts gathered so far."""
        counted = np.flatnonzero(self.candidate_count)
        return Summary(
            quakes=len(self.quakes),
            nonpositive_magnitude=self.nonpositive_magnitude,
            felt_rows=self.felt_rows,
            felt_rows_with_instrumental=self.felt_rows_with_instrumental,
            unknown_station_records=self.unknown_station_records,
            conflicting_duplicates=self.conflicting_duplicates,
            unfelt_candidates=int(self.candidate_count.sum()),
            unfelt_kept=self.unfelt_kept,
            unfelt_sampled=self.unfelt_sampled,
            suspicious_by_year=dict(sorted(self.by_year.items())),
            suspicious_by_station={
                table.numbers[station]: (
                    int(self.suspicious_count[station]),
                    int(self.candidate_count[station]),
                )
                for station in counted
            },
        )


def _neighbors(
    latitude: npt.NDArray[np.float64], longitude: npt.NDArray[np.float64]
) -> list[npt.NDArray[np.int64]]:
    """Returns the other stations within SUSPICIOUS_DISTANCE of each one."""
    neighbors = []
    for first in range(0, len(latitude), _NEIGHBOR_BLOCK):
        block = slice(first, first + _NEIGHBOR_BLOCK)
        near = _distance(
            latitude[block, np.newaxis],
            longitude[block, np.newaxis],
            latitude[np.newaxis, :],
            longitude[np.newaxis, :],
        ) <= SUSPICIOUS_DISTANCE
        for offset, row in enumerate(near):
            row[first + offset] = False
            neighbors.append(np.flatnonzero(row))
    return neighbors


def _distance(
    latitude: float | npt.NDArray[np.float64],
    longitude: float | npt.NDArray[np.float64],
    other_latitude: npt.NDArray[np.float64],
    other_longitude: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Returns the great-circle distance in kilometers, by haversine."""
    phi = np.radians(latitude)
    other_phi = np.radians(other_latitude)
    half_sine = (
        np.sin((other_phi - phi) / 2) ** 2
        + np.cos(phi)
        * np.cos(other_phi)
        * np.sin(np.radians(other_longitude - longitude) / 2) ** 2
    )
    distance: npt.NDArray[np.float64] = (
        2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.minimum(half_sine, 1.0)))
    )
    return distance


def _minute(time: datetime.datetime | None) -> np.datetime64:
    if time is None:
        return np.datetime64("NaT", "m")
    return np.datetime64(time.replace(second=0, microsecond=0), "m")


def _key(text: str) -> int:
    """Returns a 64-bit integer that stands for a quake id or a number."""
    digest = hashlib.blake2b(text.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _floats(values: Iterable[float]) -> npt.NDArray[np.float64]:
    return np.array(list(values), dtype=np.float64)


def _join[Scalar: np.generic](
    arrays: Sequence[npt.NDArray[Scalar]], dtype: type[Scalar]
) -> npt.NDArray[Scalar]:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)


if __name__ == "__main__":
    sys.exit(main())

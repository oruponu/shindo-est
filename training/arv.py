# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Looks up the ARV of each station in the two sources of ground data.

The ARV is the amplification of peak ground velocity from the engineering
bedrock to the surface. The Cabinet Office's nationwide intensity increments
give it as 10 ** (increment / 1.72), rounded to three decimals, from the
increment one sigma below the mean. J-SHIS gives it directly for its surface
ground model. Both are read from the ZIP files as distributed, and each
station takes the value of the 250 m mesh holding its position. A value that
is missing is left out instead of being filled in.

Usage:
    python -m training.arv [--sums PATH] [--jma DIR] [--ground DIR]
        [--out PATH]
"""

import argparse
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import io
import itertools
import logging
import math
import pathlib
import sys
from typing import IO
import zipfile

from training import bulletin
from training import features
from training import mesh
from training import sources

CABINET_OFFICE = "to250mr3japan.zip"
JSHIS = "Z-V4-JAPAN-AMP-VS400_M250.zip"
DEFAULT_GROUND_DIRECTORY = features.REPOSITORY / "training" / "data" / "ground"
DEFAULT_TABLE = features.REPOSITORY / "training" / "data" / "arv.csv"

_INCREMENT_PER_DECADE = 1.72
_DECIMALS = 3
_CODE_LENGTH = 10
_JSHIS_COLUMNS = ["CODE", "JCODE", "AVS", "ARV"]
_JSHIS_ARV = 3
_TABLE_HEADER = "number,cabinet_office,jshis"

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Coverage:
    """How many stations have an ARV from each source.

    Attributes:
        stations: Every station in code_p.dat.
        cabinet_office: The stations with an ARV from the Cabinet Office.
        jshis: The stations with an ARV from J-SHIS.
        both: The stations with an ARV from both.
    """

    stations: int
    cabinet_office: int
    jshis: int
    both: int


def read_cabinet_office(
    path: pathlib.Path, meshes: Collection[str]
) -> dict[str, float]:
    """Reads the ARV of some meshes from the Cabinet Office's increments.

    The CSV file holds the quoted mesh code, the increment one sigma below
    the mean, the mean increment, the longitude, and the latitude.

    Args:
        path: The ZIP file holding the CSV file.
        meshes: The mesh codes to read.

    Returns:
        The ARV of each mesh in meshes whose increment is a finite number.

    Raises:
        ValueError: The ZIP file does not hold exactly one CSV file, the
            file is not UTF-8, or a mesh in meshes appears twice.
        zipfile.BadZipFile: The file is not a ZIP file.
        OSError: The file cannot be read.
    """
    wanted = set(meshes)
    table: dict[str, float] = {}
    seen: set[str] = set()
    with (
        _open_csv(path) as file,
        io.TextIOWrapper(file, encoding="utf-8", newline="") as text,
    ):
        for line in text:
            if not line.startswith('"'):
                continue
            code = line[1 : 1 + _CODE_LENGTH]
            if code not in wanted:
                continue
            columns = line.rstrip("\r\n").split(",")
            if columns[0] != f'"{code}"':
                continue
            _see(seen, code, path)
            increment = _finite(columns[1]) if len(columns) > 1 else None
            if increment is not None:
                table[code] = round(
                    10 ** (increment / _INCREMENT_PER_DECADE), _DECIMALS
                )
    return table


def read_jshis(
    path: pathlib.Path, meshes: Collection[str]
) -> dict[str, float]:
    """Reads the ARV of some meshes from J-SHIS's surface ground model.

    The CSV file starts with comment lines, the last of which names the
    columns, and the ARV is the fourth column.

    Args:
        path: The ZIP file holding the CSV file.
        meshes: The mesh codes to read.

    Returns:
        The ARV of each mesh in meshes whose ARV is a positive number.

    Raises:
        ValueError: The ZIP file does not hold exactly one CSV file, the
            columns do not start with CODE, JCODE, AVS, and ARV, or a mesh in
            meshes appears twice.
        zipfile.BadZipFile: The file is not a ZIP file.
        OSError: The file cannot be read.
    """
    wanted = {code.encode("ascii") for code in meshes}
    table: dict[str, float] = {}
    seen: set[str] = set()
    with _open_csv(path) as file:
        first_row = file.readline()
        comment = b""
        while first_row.startswith(b"#"):
            comment = first_row
            first_row = file.readline()
        _check_jshis_columns(comment, path)
        for row in itertools.chain([first_row], file):
            columns = row.split(b",")
            if columns[0] not in wanted or len(columns) <= _JSHIS_ARV:
                continue
            code = columns[0].decode("ascii")
            _see(seen, code, path)
            value = _finite(columns[_JSHIS_ARV])
            if value is not None and value > 0:
                table[code] = value
    return table


def station_arv(
    stations: Iterable[bulletin.Station], table: Mapping[str, float]
) -> dict[str, float]:
    """Looks up the ARV of each station at its mesh.

    Args:
        stations: The stations.
        table: The ARV of each mesh code.

    Returns:
        The ARV of each station number whose mesh is in table.

    Raises:
        ValueError: Two stations have the same number, or a station is
            outside the first-order meshes.
    """
    values: dict[str, float] = {}
    numbers: set[str] = set()
    for station in stations:
        if station.number in numbers:
            raise ValueError(f"station {station.number} appears twice")
        numbers.add(station.number)
        value = table.get(mesh.station_code(station))
        if value is not None:
            values[station.number] = value
    return values


def write_table(
    path: pathlib.Path, table: Mapping[str, tuple[float | None, float | None]]
) -> None:
    """Writes the ARVs of the stations as CSV in the order of the numbers.

    Args:
        path: The CSV file.
        table: The ARVs from the Cabinet Office and from J-SHIS of each
            station number, None where a source has none.

    Raises:
        OSError: The file cannot be written.
    """
    lines = [_TABLE_HEADER]
    for number in sorted(table):
        cabinet_office, jshis = table[number]
        lines.append(f"{number},{_text(cabinet_office)},{_text(jshis)}")
    path.write_text(
        "".join(f"{line}\n" for line in lines), encoding="ascii", newline=""
    )


def read_table(
    path: pathlib.Path,
) -> dict[str, tuple[float | None, float | None]]:
    """Reads a table written by write_table.

    Args:
        path: The CSV file.

    Returns:
        The ARVs from the Cabinet Office and from J-SHIS of each station
        number, None where a source has none.

    Raises:
        ValueError: The file is not a table of ARVs. The message names the
            line.
        OSError: The file cannot be read.
    """
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != _TABLE_HEADER:
        raise ValueError(f"{path}: the header is not {_TABLE_HEADER}")
    table: dict[str, tuple[float | None, float | None]] = {}
    for number, line in enumerate(lines[1:], start=2):
        columns = line.split(",")
        if len(columns) != 3 or columns[0] in table:
            raise ValueError(f"{path}:{number}: not a new station and 2 ARVs")
        try:
            table[columns[0]] = (_value(columns[1]), _value(columns[2]))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return table


def build(
    jma: pathlib.Path,
    ground: pathlib.Path,
    out: pathlib.Path,
    sums_path: pathlib.Path = sources.SHA256SUMS,
) -> Coverage:
    """Checks the sources, then writes the ARVs of every station.

    Args:
        jma: The directory holding code_p.zip.
        ground: The directory holding the Cabinet Office and J-SHIS files.
        out: The CSV file to write.
        sums_path: The SHA256SUMS file.

    Returns:
        How many stations have an ARV from each source.

    Raises:
        sources.SourceError: A source fails its check.
        ValueError: A source cannot be read as expected.
        zipfile.BadZipFile: A source is not a ZIP file.
        OSError: A file cannot be read or written.
    """
    sums = sources.read_sums(sums_path)
    station_list = jma / "code_p.zip"
    cabinet_office_path = ground / CABINET_OFFICE
    jshis_path = ground / JSHIS
    for path in (station_list, cabinet_office_path, jshis_path):
        sources.verify(path, sums)
    stations = bulletin.read_stations(bulletin.read_zip(station_list))
    meshes = {mesh.station_code(station) for station in stations}
    cabinet_office = station_arv(
        stations, read_cabinet_office(cabinet_office_path, meshes)
    )
    jshis = station_arv(stations, read_jshis(jshis_path, meshes))
    write_table(
        out,
        {
            station.number: (
                cabinet_office.get(station.number),
                jshis.get(station.number),
            )
            for station in stations
        },
    )
    return Coverage(
        stations=len(stations),
        cabinet_office=len(cabinet_office),
        jshis=len(jshis),
        both=len(cabinet_office.keys() & jshis.keys()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line.

    Args:
        argv: The arguments after the program name. None reads sys.argv.

    Returns:
        The exit status: 0 on success, 1 when a source fails a check or
        cannot be read, or the table cannot be written.
    """
    parser = argparse.ArgumentParser(prog="python -m training.arv")
    parser.add_argument(
        "--sums", type=pathlib.Path, default=sources.SHA256SUMS
    )
    parser.add_argument(
        "--jma", type=pathlib.Path, default=sources.DEFAULT_JMA_DIRECTORY
    )
    parser.add_argument(
        "--ground", type=pathlib.Path, default=DEFAULT_GROUND_DIRECTORY
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_TABLE)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        coverage = build(
            arguments.jma, arguments.ground, arguments.out, arguments.sums
        )
    except (
        sources.SourceError,
        ValueError,
        zipfile.BadZipFile,
        OSError,
    ) as error:
        _logger.error("%s", error)
        return 1
    print(coverage)
    return 0


@contextlib.contextmanager
def _open_csv(path: pathlib.Path) -> Iterator[IO[bytes]]:
    """Opens the one CSV file in a ZIP file.

    Returns:
        A context manager providing the CSV file, open for reading bytes.
    """
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]
        if len(names) != 1:
            raise ValueError(f"{path}: {len(names)} CSV files instead of one")
        with archive.open(names[0]) as file:
            yield file


def _check_jshis_columns(comment: bytes, path: pathlib.Path) -> None:
    text = comment.removeprefix(b"#").decode("ascii", "replace")
    names = [name.strip() for name in text.split(",")]
    if names[: len(_JSHIS_COLUMNS)] != _JSHIS_COLUMNS:
        raise ValueError(
            f"{path}: the columns are {text.strip()!r}, not starting with"
            f" {", ".join(_JSHIS_COLUMNS)}"
        )


def _see(seen: set[str], code: str, path: pathlib.Path) -> None:
    if code in seen:
        raise ValueError(f"{path}: mesh {code} appears twice")
    seen.add(code)


def _finite(text: str | bytes) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _text(value: float | None) -> str:
    return "" if value is None else repr(value)


def _value(text: str) -> float | None:
    return None if text == "" else float(text)


if __name__ == "__main__":
    sys.exit(main())

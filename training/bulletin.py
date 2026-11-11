# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Reads the JMA bulletin of seismic intensities and its station list.

Each i<year>.zip holds one i<year>.dat of 96-byte records: a quake starts with
hypocenter records, the first of which holds the adopted values, followed by
one record for each station that felt it. code_p.zip holds code_p.dat, a
tab-separated list of every station with the period it was in operation.
Columns are numbered from 1, as in the JMA format description.
"""

import codecs
import dataclasses
import datetime
import pathlib
import re
import zipfile

RECORD_LENGTH = 96
FELT_STRONGLY = frozenset("34ABCD7")
JMA_MAGNITUDE_TYPES = frozenset("JDdVv")

_CLASSES = frozenset("1234ABCD7")
_UNKNOWN_CLASS = "9"
_DIGITS = re.compile(rb"[0-9]+")
_DEGREES_MINUTES = re.compile(r"([0-9]+)([0-9]{2})")
_PERIOD = re.compile(r"[0-9]{12}")
_UNKNOWN_YEAR = "9999"
_UNKNOWN_FIELD = 99
_MAGNITUDE_SIGNS = {"-": 0, "A": 10, "B": 20, "C": 30}
_MAGNITUDE_COLUMNS = (53, 56)
_TIME_COLUMNS = ((2, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 17))


@dataclasses.dataclass(frozen=True)
class Period:
    """The time a station was in operation, in naive Japan Standard Time.

    Attributes:
        start: The first time the station was surely in operation.
        end: The time it stopped, or None while it is in operation.
    """

    start: datetime.datetime
    end: datetime.datetime | None

    def contains(self, time: datetime.datetime) -> bool:
        """Returns whether the station was in operation at time."""
        return self.start <= time and (self.end is None or time < self.end)


@dataclasses.dataclass(frozen=True)
class Station:
    """A station in code_p.dat.

    Attributes:
        number: The station number the records refer to.
        name: The name, as published.
        latitude_minutes: The latitude in whole minutes of arc.
        longitude_minutes: The longitude in whole minutes of arc.
        period: The time in operation, or None when it cannot be read.
    """

    number: str
    name: str
    latitude_minutes: int
    longitude_minutes: int
    period: Period | None

    @property
    def latitude(self) -> float:
        """The latitude in degrees."""
        return self.latitude_minutes / 60

    @property
    def longitude(self) -> float:
        """The longitude in degrees."""
        return self.longitude_minutes / 60


@dataclasses.dataclass(frozen=True)
class Record:
    """The shaking of one station in one quake.

    Attributes:
        station: The station number.
        intensity_class: The symbol of the intensity class (1 to 4, A to D
            for 5 lower to 6 upper, and 7), or None when it is not known.
        instrumental: The instrumental intensity, or None when it is missing.
    """

    station: str
    intensity_class: str | None
    instrumental: float | None


@dataclasses.dataclass(frozen=True)
class Quake:
    """A quake with its adopted hypocenter and its records.

    Attributes:
        id: Columns 2 to 17 of the hypocenter record, the origin time as
            written.
        origin_time: The origin time in naive Japan Standard Time.
        latitude: The latitude of the epicenter in degrees.
        longitude: The longitude of the epicenter in degrees.
        depth: The depth in kilometers.
        magnitude: The JMA magnitude.
        records: The records of the stations that felt the quake.
    """

    id: str
    origin_time: datetime.datetime
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    records: tuple[Record, ...]


@dataclasses.dataclass(frozen=True)
class QuakeCounts:
    """How many quakes were read, and why the others were left out.

    Attributes:
        read: Every quake in the file.
        not_adopted_a: The adopted hypocenter is not of kind A.
        weak: The largest intensity is less than 3.
        no_jma_magnitude: Neither magnitude is of a JMA type, or the first
            one that is cannot be read.
        unreadable: The time, latitude, longitude, or depth cannot be read.
    """

    read: int
    not_adopted_a: int
    weak: int
    no_jma_magnitude: int
    unreadable: int


def read_stations(data: bytes) -> list[Station]:
    """Reads code_p.dat.

    A leading byte order mark and empty lines are skipped.

    Args:
        data: The file, UTF-8 text of lines with the number, the name, the
            latitude and the longitude in degrees and minutes (DDMM), the
            start (YYYYMMDDhhmm), and the end, which is empty for a station
            in operation.

    Returns:
        The stations in the order of the file.

    Raises:
        ValueError: A line is not UTF-8, has fewer than five columns, or has
            a coordinate that is not degrees and minutes. The message names
            the line.
    """
    lines = _lines(data.removeprefix(codecs.BOM_UTF8))
    stations = []
    for number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            stations.append(_station(line.decode("utf-8")))
        except ValueError as error:
            raise ValueError(f"line {number}: {error}") from error
    return stations


def read_quakes(data: bytes) -> tuple[list[Quake], QuakeCounts]:
    """Reads i<year>.dat and keeps the quakes used for training.

    A quake is kept when its adopted hypocenter is of kind A, its largest
    intensity is 3 or more, it has a JMA magnitude (the first of the two
    magnitudes whose type is one), and its time, latitude, longitude, and
    depth can be read.

    Args:
        data: The file, 96-byte records with CRLF or LF line endings.

    Returns:
        The kept quakes in the order of the file, and the counts.

    Raises:
        ValueError: A line is not 96 bytes, a station record comes before
            any hypocenter, a station record has a symbol that is not an
            intensity class, or two kept quakes have the same id. The
            message names the line.
    """
    lines = _lines(data)
    for index, line in enumerate(lines):
        if len(line) != RECORD_LENGTH:
            raise ValueError(
                f"line {index + 1}: {len(line)} bytes instead of"
                f" {RECORD_LENGTH}"
            )
    quakes: list[Quake] = []
    ids: set[str] = set()
    counts = dict.fromkeys(
        (field.name for field in dataclasses.fields(QuakeCounts)), 0
    )
    index = 0
    while index < len(lines):
        if not _is_hypocenter(lines[index]):
            raise ValueError(
                f"line {index + 1}: a station record before any hypocenter"
            )
        adopted = index
        while index < len(lines) and _is_hypocenter(lines[index]):
            index += 1
        first_record = index
        while index < len(lines) and not _is_hypocenter(lines[index]):
            index += 1
        records = tuple(
            _record(lines[number], number + 1)
            for number in range(first_record, index)
        )
        counts["read"] += 1
        quake = _quake(lines[adopted], records)
        if isinstance(quake, str):
            counts[quake] += 1
            continue
        if quake.id in ids:
            raise ValueError(f"line {adopted + 1}: quake {quake.id} repeats")
        ids.add(quake.id)
        quakes.append(quake)
    return quakes, QuakeCounts(**counts)


def read_zip(path: pathlib.Path) -> bytes:
    """Reads the one .dat file in a ZIP file.

    Args:
        path: The ZIP file.

    Returns:
        The contents of the .dat file.

    Raises:
        ValueError: The ZIP file does not hold exactly one .dat file.
        zipfile.BadZipFile: The file is not a ZIP file.
        OSError: The file cannot be read.
    """
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".dat")
        ]
        if len(names) != 1:
            raise ValueError(f"{path}: {len(names)} .dat files instead of one")
        return archive.read(names[0])


def _lines(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return [line.removesuffix(b"\r") for line in lines]


def _station(line: str) -> Station:
    columns = line.split("\t")
    if len(columns) < 5:
        raise ValueError(f"{len(columns)} columns instead of at least 5")
    end = columns[5].strip() if len(columns) > 5 else ""
    return Station(
        number=columns[0].strip(),
        name=columns[1],
        latitude_minutes=_minutes(columns[2]),
        longitude_minutes=_minutes(columns[3]),
        period=_period(columns[4].strip(), end),
    )


def _minutes(text: str) -> int:
    match = _DEGREES_MINUTES.fullmatch(text.strip())
    if match is None or int(match[2]) >= 60:
        raise ValueError(f"{text!r} is not degrees and minutes")
    return int(match[1]) * 60 + int(match[2])


def _period(start: str, end: str) -> Period | None:
    begins = _period_time(start, is_start=True)
    if begins is None:
        return None
    if not end:
        return Period(begins, None)
    ends = _period_time(end, is_start=False)
    if ends is None or ends <= begins:
        return None
    return Period(begins, ends)


def _period_time(text: str, *, is_start: bool) -> datetime.datetime | None:
    """Reads a YYYYMMDDhhmm time in which 99 marks unknown fields.

    The fields from the first 99 on are unknown. A start is taken at the
    beginning of the next unit, when the station surely operated, and an
    end at the beginning of the unit.

    Args:
        text: The time as written.
        is_start: Whether the time starts a period.

    Returns:
        The time, or None when the text is not 12 digits, its year is 9999,
        or it is not a date.
    """
    if not _PERIOD.fullmatch(text) or text[:4] == _UNKNOWN_YEAR:
        return None
    fields = [int(text[at : at + 2]) for at in range(4, 12, 2)]
    known = (
        fields.index(_UNKNOWN_FIELD)
        if _UNKNOWN_FIELD in fields
        else len(fields)
    )
    month, day, hour, minute = [*fields[:known], *(1, 1, 0, 0)[known:]]
    try:
        time = datetime.datetime(int(text[:4]), month, day, hour, minute)
    except ValueError:
        return None
    if not is_start or known == len(fields):
        return time
    if known == 0:
        return time.replace(year=time.year + 1)
    if known == 1:
        return (time + datetime.timedelta(days=31)).replace(day=1)
    if known == 2:
        return time + datetime.timedelta(days=1)
    return time + datetime.timedelta(hours=1)


def _is_hypocenter(line: bytes) -> bool:
    return "A" <= _column(line, 1) <= "Z"


def _record(line: bytes, number: int) -> Record:
    symbol = _column(line, 19)
    if symbol != _UNKNOWN_CLASS and symbol not in _CLASSES:
        raise ValueError(
            f"line {number}: {symbol!r} is not an intensity class"
        )
    tenths = _integer(_field(line, 21, 22))
    return Record(
        station=_field(line, 1, 7).decode("ascii").strip(),
        intensity_class=None if symbol == _UNKNOWN_CLASS else symbol,
        instrumental=None if tenths is None else tenths / 10,
    )


def _quake(line: bytes, records: tuple[Record, ...]) -> Quake | str:
    """Returns the quake, or the QuakeCounts field of why it is left out."""
    if _column(line, 1) != "A":
        return "not_adopted_a"
    if _column(line, 62) not in FELT_STRONGLY:
        return "weak"
    magnitude = _magnitude(line)
    if magnitude is None:
        return "no_jma_magnitude"
    time = _time(line)
    latitude = _angle(_field(line, 22, 24), _field(line, 25, 28))
    longitude = _angle(_field(line, 33, 36), _field(line, 37, 40))
    depth = _depth(_field(line, 45, 49))
    if time is None or latitude is None or longitude is None or depth is None:
        return "unreadable"
    return Quake(
        id=_field(line, 2, 17).decode("ascii"),
        origin_time=time,
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        magnitude=magnitude,
        records=records,
    )


def _magnitude(line: bytes) -> float | None:
    for first in _MAGNITUDE_COLUMNS:
        if _column(line, first + 2) in JMA_MAGNITUDE_TYPES:
            return _magnitude_value(_field(line, first, first + 1))
    return None


def _magnitude_value(field: bytes) -> float | None:
    """Reads a magnitude of two characters, with letters for -1 to -3."""
    tenth = _integer(field[1:])
    if tenth is None:
        return None
    lead = _column(field, 1)
    if lead in _MAGNITUDE_SIGNS:
        return -(_MAGNITUDE_SIGNS[lead] + tenth) / 10
    tenths = _integer(field)
    return None if tenths is None else tenths / 10


def _time(line: bytes) -> datetime.datetime | None:
    values = _integers(line, _TIME_COLUMNS)
    if values is None:
        return None
    year, month, day, hour, minute, centiseconds = values
    try:
        time = datetime.datetime(year, month, day, hour, minute)
    except ValueError:
        return None
    return time + datetime.timedelta(milliseconds=centiseconds * 10)


def _integers(
    line: bytes, columns: tuple[tuple[int, int], ...]
) -> tuple[int, ...] | None:
    values = []
    for first, last in columns:
        value = _integer(_field(line, first, last))
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _angle(degrees: bytes, minutes: bytes) -> float | None:
    whole = _integer(degrees)
    hundredths = _integer(minutes)
    if whole is None or hundredths is None:
        return None
    return whole + hundredths / 100 / 60


def _depth(field: bytes) -> float | None:
    """Reads a depth, F5.2 when free and I3 then two spaces when fixed."""
    if field[3:].strip():
        hundredths = _integer(field)
        return None if hundredths is None else hundredths / 100
    kilometers = _integer(field[:3])
    return None if kilometers is None else float(kilometers)


def _field(line: bytes, first: int, last: int) -> bytes:
    return line[first - 1 : last]


def _column(line: bytes, column: int) -> str:
    return chr(line[column - 1])


def _integer(field: bytes) -> int | None:
    digits = field.strip(b" ")
    return int(digits) if _DIGITS.fullmatch(digits) else None

# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the reading of the JMA bulletin and its station list."""

import datetime
import pathlib
import tempfile
import unittest
import zipfile

from training import bulletin

_ISHIKARI = chr(0x77F3) + chr(0x72E9)


def _fields(fields):
    """Returns a 96-byte record with each text at its 1-based column."""
    line = bytearray(b" " * bulletin.RECORD_LENGTH)
    for column, text in fields.items():
        line[column - 1 : column - 1 + len(text)] = text.encode("ascii")
    return bytes(line)


def _hypocenter(
    *,
    kind="A",
    time="200401020304 512",
    latitude=(" 35", "4012"),
    longitude=(" 139", "3045"),
    depth=" 50  ",
    m1=("65", "J"),
    m2=("  ", " "),
    maximum="4",
):
    return _fields({
        1: kind,
        2: time,
        22: latitude[0],
        25: latitude[1],
        33: longitude[0],
        37: longitude[1],
        45: depth,
        53: m1[0],
        55: m1[1],
        56: m2[0],
        58: m2[1],
        62: maximum,
    })


def _record(station="7654321", symbol="4", instrumental="38"):
    return _fields({1: station, 19: symbol, 21: instrumental})


def _quakes(*lines):
    return bulletin.read_quakes(b"\r\n".join(lines) + b"\r\n")


def _counts(**counts):
    values = {
        "read": 0,
        "not_adopted_a": 0,
        "weak": 0,
        "no_jma_magnitude": 0,
        "unreadable": 0,
    }
    values.update(counts)
    return bulletin.QuakeCounts(**values)


def _station(start, end=""):
    line = f"7654321\tTest\t4310\t14119\t{start}\t{end}\n"
    return bulletin.read_stations(line.encode("ascii"))[0]


class StationsTest(unittest.TestCase):

    def test_reads_stations(self):
        stations = bulletin.read_stations(
            b"7654321\tTest\t4310\t14119\t199604011200\t\n"
            b"\n"
            b"7654322 \tOther\t3500\t13900\t200001010000\t201001010000\n"
        )

        self.assertEqual(len(stations), 2)
        station = stations[0]
        self.assertEqual(station.number, "7654321")
        self.assertEqual(station.name, "Test")
        self.assertEqual(station.latitude_minutes, 2590)
        self.assertEqual(station.longitude_minutes, 8479)
        self.assertEqual(station.latitude, 2590 / 60)
        self.assertEqual(station.longitude, 8479 / 60)
        self.assertEqual(
            station.period,
            bulletin.Period(datetime.datetime(1996, 4, 1, 12, 0), None),
        )
        self.assertEqual(stations[1].number, "7654322")

    def test_a_byte_order_mark_is_skipped(self):
        line = "7654321\tTest\t4310\t14119\t199604011200\t\n"

        stations = bulletin.read_stations(
            b"\xef\xbb\xbf" + line.encode("ascii")
        )

        self.assertEqual([station.number for station in stations], ["7654321"])

    def test_names_are_utf8(self):
        line = f"7654321\t{_ISHIKARI}\t4310\t14119\t199604011200\t\n"

        stations = bulletin.read_stations(line.encode("utf-8"))

        self.assertEqual([station.name for station in stations], [_ISHIKARI])
        with self.assertRaisesRegex(ValueError, "line 1"):
            bulletin.read_stations(line.encode("euc_jp"))

    def test_periods(self):
        self.assertEqual(
            _station("200405991200").period,
            bulletin.Period(datetime.datetime(2004, 6, 1), None),
        )
        self.assertEqual(
            _station("200499999999").period,
            bulletin.Period(datetime.datetime(2005, 1, 1), None),
        )
        self.assertEqual(
            _station("200405019999").period,
            bulletin.Period(datetime.datetime(2004, 5, 2), None),
        )
        self.assertEqual(
            _station("200405011299").period,
            bulletin.Period(datetime.datetime(2004, 5, 1, 13), None),
        )
        self.assertIsNone(_station("999901010000").period)
        self.assertIsNone(_station("2004059912").period)
        self.assertIsNone(_station("200413011200").period)
        self.assertIsNone(
            _station("201203311200", "201203311200").period
        )
        self.assertIsNone(
            _station("201203311200", "201001010000").period
        )
        self.assertEqual(
            _station("200405011200", "201203311200").period,
            bulletin.Period(
                datetime.datetime(2004, 5, 1, 12),
                datetime.datetime(2012, 3, 31, 12),
            ),
        )
        self.assertEqual(
            _station("200405011200", "201203999999").period,
            bulletin.Period(
                datetime.datetime(2004, 5, 1, 12),
                datetime.datetime(2012, 3, 1),
            ),
        )
        self.assertIsNone(_station("200405011200", "2012").period)

    def test_period_contains(self):
        period = bulletin.Period(
            datetime.datetime(2004, 5, 1), datetime.datetime(2012, 3, 1)
        )
        open_period = bulletin.Period(datetime.datetime(2004, 5, 1), None)

        self.assertTrue(period.contains(datetime.datetime(2004, 5, 1)))
        self.assertFalse(period.contains(datetime.datetime(2004, 4, 30)))
        self.assertTrue(period.contains(datetime.datetime(2012, 2, 29)))
        self.assertFalse(period.contains(datetime.datetime(2012, 3, 1)))
        self.assertTrue(open_period.contains(datetime.datetime(2030, 1, 1)))

    def test_rejects_malformed_station_lines(self):
        good = "7654321\tTest\t4310\t14119\t199604011200\t\n"
        for bad in (
            "7654322\tTest\t4310\t14119\n",
            "7654322\tTest\t43a0\t14119\t199604011200\t\n",
            "7654322\tTest\t4310\t41\t199604011200\t\n",
            "7654322\tTest\t4360\t14119\t199604011200\t\n",
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "line 2"):
                    bulletin.read_stations((good + bad).encode("ascii"))


class QuakesTest(unittest.TestCase):

    def test_reads_an_adopted_quake(self):
        quakes, counts = _quakes(
            _hypocenter(),
            _record("7654321", "4", "38"),
            _record("  12345", "1", " 7"),
        )

        self.assertEqual(len(quakes), 1)
        quake = quakes[0]
        self.assertEqual(quake.id, "200401020304 512")
        self.assertEqual(
            quake.origin_time,
            datetime.datetime(2004, 1, 2, 3, 4, 5, 120000),
        )
        self.assertEqual(quake.latitude, 35 + 40.12 / 60)
        self.assertEqual(quake.longitude, 139 + 30.45 / 60)
        self.assertEqual(quake.depth, 50.0)
        self.assertEqual(quake.magnitude, 6.5)
        self.assertEqual(
            quake.records,
            (
                bulletin.Record("7654321", "4", 3.8),
                bulletin.Record("12345", "1", 0.7),
            ),
        )
        self.assertEqual(counts, _counts(read=1))

    def test_reads_free_depths(self):
        quakes, _ = _quakes(_hypocenter(depth=" 5012"), _record())

        self.assertEqual(quakes[0].depth, 50.12)

    def test_picks_the_jma_magnitude(self):
        quakes, counts = _quakes(
            _hypocenter(m1=("71", "W"), m2=("68", "J")),
            _record(),
            _hypocenter(
                time="200401020305 000", m1=("71", "W"), m2=("70", "W")
            ),
            _record(),
        )

        self.assertEqual([quake.magnitude for quake in quakes], [6.8])
        self.assertEqual(counts, _counts(read=2, no_jma_magnitude=1))

    def test_the_first_jma_magnitude_decides(self):
        quakes, counts = _quakes(
            _hypocenter(m1=("65", "J"), m2=("68", "J")),
            _record(),
            _hypocenter(
                time="200401020305 000", m1=("  ", "J"), m2=("68", "J")
            ),
            _record(),
        )

        self.assertEqual([quake.magnitude for quake in quakes], [6.5])
        self.assertEqual(counts, _counts(read=2, no_jma_magnitude=1))

    def test_non_ascii_magnitudes_are_unreadable(self):
        lines = []
        for column, minute in ((53, "01"), (54, "02")):
            line = bytearray(_hypocenter(time=f"2004010203{minute} 000"))
            line[column - 1] = 0xB2
            lines += [bytes(line), _record()]

        quakes, counts = _quakes(*lines, _hypocenter(), _record())

        self.assertEqual(len(quakes), 1)
        self.assertEqual(counts, _counts(read=3, no_jma_magnitude=2))

    def test_magnitude_types(self):
        for kind in "JDdVv":
            with self.subTest(kind=kind):
                quakes, _ = _quakes(_hypocenter(m1=("50", kind)), _record())
                self.assertEqual(quakes[0].magnitude, 5.0)

    def test_negative_magnitudes(self):
        for field, magnitude in (
            (" 5", 0.5),
            ("-5", -0.5),
            ("A3", -1.3),
            ("B0", -2.0),
            ("C2", -3.2),
        ):
            with self.subTest(field=field):
                quakes, _ = _quakes(_hypocenter(m1=(field, "V")), _record())
                self.assertEqual(quakes[0].magnitude, magnitude)

    def test_skips_quakes_that_are_not_used(self):
        quakes, counts = _quakes(
            _hypocenter(kind="B", time="200401020301 000"),
            _record(),
            _hypocenter(maximum="2", time="200401020302 000"),
            _record(),
            _hypocenter(depth="     ", time="200401020303 000"),
            _record(),
            _hypocenter(latitude=(" 35", "    "), time="200401020304 000"),
            _record(),
            _hypocenter(m1=("6 ", "J"), time="200401020305 000"),
            _record(),
            _hypocenter(time="200401020306 000"),
            _record(),
        )

        self.assertEqual([quake.id for quake in quakes], ["200401020306 000"])
        self.assertEqual(
            counts,
            _counts(
                read=6,
                not_adopted_a=1,
                weak=1,
                no_jma_magnitude=1,
                unreadable=2,
            ),
        )

    def test_felt_strongly(self):
        for maximum in "1234ABCD7":
            with self.subTest(maximum=maximum):
                quakes, _ = _quakes(_hypocenter(maximum=maximum), _record())
                self.assertEqual(len(quakes), int(maximum in "34ABCD7"))

    def test_unreadable_times(self):
        for time in (
            "200402300304 000",
            "20040102  04 000",
            "200401020304 5x0",
        ):
            with self.subTest(time=time):
                quakes, counts = _quakes(_hypocenter(time=time), _record())
                self.assertEqual(quakes, [])
                self.assertEqual(counts, _counts(read=1, unreadable=1))

    def test_only_the_first_hypocenter_is_adopted(self):
        quakes, counts = _quakes(
            _hypocenter(depth=" 50  "),
            _hypocenter(kind="D", depth=" 10  "),
            _record("7654321"),
            _record("7654322"),
        )

        self.assertEqual(len(quakes), 1)
        quake = quakes[0]
        self.assertEqual(quake.depth, 50.0)
        self.assertEqual(
            [record.station for record in quake.records],
            ["7654321", "7654322"],
        )
        self.assertEqual(counts, _counts(read=1))

    def test_class_9_and_missing_instrumental(self):
        quakes, _ = _quakes(
            _hypocenter(), _record(symbol="9"), _record(instrumental="//")
        )

        self.assertEqual(
            quakes[0].records,
            (
                bulletin.Record("7654321", None, 3.8),
                bulletin.Record("7654321", "4", None),
            ),
        )

    def test_crlf_and_lf_give_the_same_quakes(self):
        lines = [_hypocenter(), _record(), _record("7654322", "3", "29")]

        self.assertEqual(
            bulletin.read_quakes(b"\r\n".join(lines) + b"\r\n"),
            bulletin.read_quakes(b"\n".join(lines)),
        )

    def test_rejects_malformed_bulletins(self):
        good = [_hypocenter(), _record()]
        for bad in (
            [_record()[:-1]],
            [_record(symbol="5")],
            [b""],
            [_hypocenter(), _record()],
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, r"line 3\b"):
                    _quakes(*good, *bad)
        with self.assertRaisesRegex(ValueError, r"line 1\b"):
            _quakes(_record(), *good)


class ReadZipTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.directory = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )

    def _zip(self, entries):
        path = self.directory / "i2004.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        return path

    def test_reads_the_one_dat(self):
        path = self._zip({"readme.txt": b"text", "i2004.dat": b"records"})

        self.assertEqual(bulletin.read_zip(path), b"records")

    def test_needs_one_dat(self):
        for entries in (
            {"readme.txt": b"text"},
            {"a.dat": b"records", "b.DAT": b"records"},
        ):
            with self.subTest(entries=list(entries)):
                with self.assertRaisesRegex(ValueError, "i2004.zip"):
                    bulletin.read_zip(self._zip(entries))


if __name__ == "__main__":
    unittest.main()

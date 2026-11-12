# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests that quakes and stations become rows of training data."""

import dataclasses
import datetime
import hashlib
import json
import math
import pathlib
import tempfile
import unittest
import zipfile

import numpy as np

from training import bulletin
from training import dataset
from training import sources

_ALWAYS = bulletin.Period(datetime.datetime(1990, 1, 1), None)
_TIME = datetime.datetime(2010, 6, 1, 12, 0)
# The epicenter of most test quakes, in minutes of arc.
_LATITUDE = 35 * 60
_LONGITUDE = 139 * 60
_FIELDS = [field.name for field in dataclasses.fields(dataset.Dataset)]


def _station(number, north=0, east=0, period=_ALWAYS):
    """Returns a station some minutes of arc north and east of the epicenter."""
    return bulletin.Station(
        number=number,
        name="Test",
        latitude_minutes=_LATITUDE + north,
        longitude_minutes=_LONGITUDE + east,
        period=period,
    )


def _quake(records=(), time=_TIME, depth=10.0, magnitude=5.0, north=0):
    return bulletin.Quake(
        id=f"{time:%Y%m%d%H%M}0000",
        origin_time=time,
        latitude=(_LATITUDE + north) / 60,
        longitude=_LONGITUDE / 60,
        depth=depth,
        magnitude=magnitude,
        records=tuple(records),
    )


def _record(number, instrumental=1.0):
    return bulletin.Record(number, "1", instrumental)


def _rows(data, quake=0):
    """Returns the rows of a quake as {number: (felt, observed, weight)}.

    An observed value of NaN is returned as None, which compares equal.
    """
    rows = {}
    for index in np.flatnonzero(data.row_quake == quake):
        number = str(data.station_number[data.row_station[index]])
        observed = float(data.observed[index])
        rows[number] = (
            bool(data.felt[index]),
            None if math.isnan(observed) else observed,
            float(data.weight[index]),
        )
    return rows


class RegionTest(unittest.TestCase):

    def test_regions_follow_the_boundaries(self):
        for latitude, longitude, region in (
            (41.0, 141, "hokkaido"),
            (40.99, 141, "tohoku"),
            (37.0, 140, "tohoku"),
            (33.9, 138.0, "izu"),
            (35, 138.0, "kanto"),
            (33.9, 132.59, "kyushu"),
            (33.9, 132.6, "west"),
            (35, 137.9, "west"),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assertEqual(dataset.region(latitude, longitude), region)
        self.assertEqual(
            dataset.REGIONS,
            ("hokkaido", "tohoku", "kanto", "izu", "west", "kyushu"),
        )

    def test_holdout_uses_crc32(self):
        self.assertTrue(dataset.is_held_out("7654322"))
        self.assertFalse(dataset.is_held_out("7654321"))
        numbers = [f"{number:07}" for number in range(10_000)]
        share = sum(map(dataset.is_held_out, numbers)) / len(numbers)
        self.assertTrue(0.18 <= share <= 0.22, share)


class SampleTest(unittest.TestCase):

    def test_splitmix64(self):
        self.assertEqual(
            dataset.splitmix64(np.array([0], dtype=np.uint64)).tolist(),
            [0xE220A8397B1DCDAF],
        )

    def test_uniform_samples_follow_the_formula(self):
        def splitmix64(value):
            mask = (1 << 64) - 1
            value = (value + 0x9E3779B97F4A7C15) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            return value ^ (value >> 31)

        quake_id = (1 << 64) - 5
        numbers = [0, 7, (1 << 64) - 1]

        samples = dataset.sample_uniform(
            quake_id, np.array(numbers, dtype=np.uint64)
        )

        self.assertEqual(
            samples.tolist(),
            [
                (splitmix64(splitmix64(quake_id) ^ number) >> 11) * 2.0**-53
                for number in numbers
            ],
        )

    def test_uniform_samples(self):
        numbers = np.arange(10_000, dtype=np.uint64)

        samples = dataset.sample_uniform(12345, numbers)

        self.assertEqual(samples.dtype, np.float64)
        self.assertTrue(np.all((samples >= 0) & (samples < 1)))
        self.assertAlmostEqual(float(samples.mean()), 0.5, delta=0.02)
        np.testing.assert_array_equal(
            samples[::-1], dataset.sample_uniform(12345, numbers[::-1])
        )
        self.assertFalse(
            np.array_equal(samples, dataset.sample_uniform(12346, numbers))
        )


class BuildTest(unittest.TestCase):

    def test_felt_rows_carry_the_records(self):
        stations = [_station("7654321", north=5), _station("7654322", east=5)]
        quake = _quake([_record("7654321", 3.8), _record("7654322", None)])

        data, summary = dataset.build(stations, [quake])

        self.assertEqual(
            _rows(data),
            {
                "7654321": (True, np.float32(3.8).item(), 1.0),
                "7654322": (True, None, 1.0),
            },
        )
        self.assertEqual(data.observed.dtype, np.float32)
        self.assertEqual(data.weight.dtype, np.float32)
        self.assertEqual(summary.felt_rows, 2)
        self.assertEqual(summary.felt_rows_with_instrumental, 1)

    def test_unfelt_rows_follow_the_operating_period(self):
        stations = [
            _station("7654321"),
            _station("7654322", 1, period=bulletin.Period(_TIME, None)),
            _station(
                "7654323",
                2,
                period=bulletin.Period(datetime.datetime(2000, 1, 1), _TIME),
            ),
            _station("7654324", 3, period=None),
            _station(
                "7654325",
                4,
                period=bulletin.Period(
                    datetime.datetime(2000, 1, 1),
                    _TIME + datetime.timedelta(minutes=1),
                ),
            ),
        ]

        data, summary = dataset.build(stations, [_quake([_record("7654321")])])

        self.assertEqual(
            sorted(_rows(data)), ["7654321", "7654322", "7654325"]
        )
        self.assertEqual(summary.unfelt_candidates, 2)

    def test_near_unfelt_rows_are_all_kept(self):
        for felt_north, kept_north in ((16, 31), (5, 26)):
            with self.subTest(felt_north=felt_north):
                stations = [
                    _station("7654321", felt_north),
                    _station("7654322", kept_north),
                ]
                quake = _quake([_record("7654321")])

                data, summary = dataset.build(stations, [quake])

                self.assertEqual(_rows(data)["7654322"], (False, None, 1.0))
                self.assertEqual(summary.unfelt_kept, 1)

    def test_keep_distance_is_on_the_sphere(self):
        # 26 and 27 minutes of latitude are 48.18 and 50.04 km.
        stations = [
            _station("7654321", 5),
            _station("7654322", 26),
            _station("7654323", 27),
        ]

        data, summary = dataset.build(stations, [_quake([_record("7654321")])])

        self.assertEqual(_rows(data)["7654322"], (False, None, 1.0))
        self.assertEqual(summary.unfelt_kept, 1)

    def test_far_unfelt_rows_are_sampled(self):
        stations = [_station("0000001", 1)] + [
            _station(f"{2 + row * 50 + column:07}", 300 + row, 300 + column)
            for row in range(40)
            for column in range(50)
        ]
        quake = _quake([_record("0000001")])

        data, summary = dataset.build(stations, [quake])
        again, _ = dataset.build(stations, [quake])

        unfelt = ~data.felt
        self.assertTrue(40 <= unfelt.sum() <= 80, unfelt.sum())
        np.testing.assert_array_equal(
            data.weight[unfelt], np.float32(1 / 0.03)
        )
        self.assertEqual(data.weight.dtype, np.float32)
        np.testing.assert_array_equal(data.row_station, again.row_station)
        self.assertEqual(summary.unfelt_candidates, 2000)
        self.assertEqual(summary.unfelt_kept, 0)
        self.assertEqual(summary.unfelt_sampled, unfelt.sum())

    def test_samples_do_not_depend_on_order(self):
        stations = [_station("0000001", 1)] + [
            _station(f"{2 + index:07}", 300 + index // 50, 300 + index % 50)
            for index in range(500)
        ]
        quakes = [
            _quake(
                [_record("0000001")],
                time=_TIME + datetime.timedelta(days=day),
            )
            for day in range(3)
        ]
        unrelated = _station("0000000", 400, 400, period=None)

        data, _ = dataset.build(stations, quakes)
        reordered, _ = dataset.build(
            [unrelated, *stations[::-1]], quakes[::-1]
        )

        for at, quake in enumerate(quakes):
            with self.subTest(quake=quake.id):
                self.assertEqual(
                    _rows(data, at), _rows(reordered, len(quakes) - 1 - at)
                )
                self.assertGreater(len(_rows(data, at)), 1)
        self.assertNotEqual(_rows(data, 0), _rows(data, 1))

    def test_duplicate_records(self):
        stations = [_station("7654321"), _station("7654322", 1)]
        quake = _quake([
            _record("7654321", 3.1),
            _record("7654321", 3.1),
            _record("7654322", 2.0),
            bulletin.Record("7654322", "3", 2.6),
        ])

        data, summary = dataset.build(stations, [quake])

        rows = _rows(data)
        self.assertEqual(rows["7654321"], (True, np.float32(3.1).item(), 1.0))
        self.assertEqual(rows["7654322"], (True, None, 1.0))
        self.assertEqual(len(data.felt), 2)
        self.assertEqual(summary.conflicting_duplicates, 1)

    def test_unknown_stations_are_counted(self):
        stations = [_station("7654321")]
        quake = _quake([
            _record("7654321"),
            _record("1234567"),
            _record("1234567", 2.0),
        ])

        data, summary = dataset.build(stations, [quake])

        self.assertEqual(list(_rows(data)), ["7654321"])
        self.assertEqual(summary.unknown_station_records, 2)

    def test_suspicious_rows(self):
        stations = [
            _station("7654321"),
            _station("7654322", 8),
            _station("7654323", -14),
        ]
        strong = _quake([_record("7654321", 2.5)])
        weak = _quake(
            [_record("7654321", 2.4)],
            time=_TIME + datetime.timedelta(days=1),
        )

        data, summary = dataset.build(stations, [strong, weak])

        suspicious = {
            (int(quake), str(data.station_number[station]))
            for quake, station, flag in zip(
                data.row_quake, data.row_station, data.suspicious
            )
            if flag
        }
        self.assertEqual(suspicious, {(0, "7654322")})
        self.assertEqual(summary.suspicious_by_year, {2010: (1, 4)})
        self.assertEqual(
            summary.suspicious_by_station,
            {"7654322": (1, 2), "7654323": (0, 2)},
        )

    def test_suspicious_counts_include_unfelt_rows_that_are_left_out(self):
        stations = [_station("0000001", 1)] + [
            _station(f"{2 + index:07}", 300 + index // 50, 300 + index % 50)
            for index in range(500)
        ]
        quake = _quake([_record("0000001", 2.5)])

        data, summary = dataset.build(stations, [quake])

        self.assertLess((~data.felt).sum(), 500)
        self.assertEqual(summary.suspicious_by_year, {2010: (0, 500)})
        self.assertEqual(len(summary.suspicious_by_station), 500)

    def test_rows_are_ordered_by_quake_then_station(self):
        stations = [
            _station("7654323", 2),
            _station("7654321"),
            _station("7654322", 1),
        ]
        first = _quake([_record("7654322")])
        second = _quake(
            [_record("7654323"), _record("7654321")],
            time=_TIME + datetime.timedelta(days=1),
        )

        data, _ = dataset.build(stations, [first, second])

        order = [
            (int(quake), str(data.station_number[station]))
            for quake, station in zip(data.row_quake, data.row_station)
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(len(order), 6)
        self.assertEqual(list(data.quake_id), [first.id, second.id])
        self.assertEqual(data.row_quake.dtype, np.int32)
        self.assertEqual(data.row_station.dtype, np.int32)

    def test_quake_and_station_arrays(self):
        stations = [
            _station(
                "7654322",
                period=bulletin.Period(
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2020, 1, 1, 12, 30),
                ),
            ),
            _station("7654321", 1),
            _station("7654323", 2, period=None),
        ]
        quake = _quake(
            [_record("7654321")],
            time=datetime.datetime(2010, 6, 1, 12, 0, 30, 500000),
        )

        data, summary = dataset.build(stations, [quake])

        self.assertEqual(
            list(data.station_number), ["7654321", "7654322", "7654323"]
        )
        np.testing.assert_array_equal(
            data.station_start,
            np.array(
                ["1990-01-01T00:00", "2000-01-01T00:00", "NaT"],
                dtype="datetime64[m]",
            ),
        )
        np.testing.assert_array_equal(
            data.station_end,
            np.array(["NaT", "2020-01-01T12:30", "NaT"], dtype="datetime64[m]"),
        )
        np.testing.assert_array_equal(
            data.quake_time,
            np.array(["2010-06-01T12:00"], dtype="datetime64[m]"),
        )
        self.assertEqual(data.station_latitude[0], (_LATITUDE + 1) / 60)
        self.assertEqual(data.station_longitude[0], _LONGITUDE / 60)
        np.testing.assert_array_equal(
            data.station_held_out,
            [dataset.is_held_out(number) for number in data.station_number],
        )
        self.assertEqual(data.quake_year.tolist(), [2010])
        self.assertEqual(data.quake_year.dtype, np.int16)
        self.assertEqual(data.quake_region.tolist(), ["kanto"])
        self.assertEqual(data.quake_magnitude.tolist(), [5.0])
        self.assertEqual(data.quake_depth.tolist(), [10.0])
        self.assertEqual(data.quake_latitude.tolist(), [35.0])
        self.assertEqual(data.quake_longitude.tolist(), [139.0])
        self.assertEqual(summary.quakes, 1)

    def test_quakes_without_felt_stations(self):
        # 26 minutes of latitude are 48.18 km, within the 50 km floor.
        quake = _quake([_record("1234567")])

        data, summary = dataset.build([_station("7654321", 26)], [quake])
        empty, empty_summary = dataset.build(
            [_station("7654321", period=None)], [quake]
        )

        self.assertEqual(_rows(data), {"7654321": (False, None, 1.0)})
        self.assertEqual(summary.felt_rows, 0)
        self.assertEqual(summary.unfelt_kept, 1)
        self.assertEqual(list(empty.quake_id), [quake.id])
        self.assertEqual(len(empty.row_quake), 0)
        self.assertEqual(empty_summary.quakes, 1)

    def test_rejects_quakes_with_values_that_are_not_finite(self):
        quake = _quake([_record("7654321")])
        for field in ("latitude", "longitude", "depth", "magnitude"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    dataset.build(
                        [_station("7654321")],
                        [dataclasses.replace(quake, **{field: math.nan})],
                    )

    def test_nonpositive_magnitudes_are_left_out(self):
        stations = [_station("7654321")]
        quakes = [
            _quake([_record("7654321")], magnitude=0.0),
            _quake(
                [_record("7654321")],
                time=_TIME + datetime.timedelta(days=1),
                magnitude=0.1,
            ),
        ]

        data, summary = dataset.build(stations, quakes)

        self.assertEqual(list(data.quake_id), [quakes[1].id])
        self.assertEqual(data.row_quake.tolist(), [0])
        self.assertEqual(summary.nonpositive_magnitude, 1)
        self.assertEqual(summary.quakes, 1)

    def test_rejects_repeated_station_numbers(self):
        with self.assertRaisesRegex(ValueError, "7654321"):
            dataset.build([_station("7654321"), _station("7654321", 1)], [])

    def test_summary_to_json(self):
        stations = [_station("7654321"), _station("7654322", 8)]

        _, summary = dataset.build(stations, [_quake([_record("7654321", 3)])])

        self.assertEqual(
            json.loads(summary.to_json()),
            {
                "quakes": 1,
                "nonpositive_magnitude": 0,
                "felt_rows": 1,
                "felt_rows_with_instrumental": 1,
                "unknown_station_records": 0,
                "conflicting_duplicates": 0,
                "unfelt_candidates": 1,
                "unfelt_kept": 1,
                "unfelt_sampled": 0,
                "suspicious_by_year": {"2010": [1, 1]},
                "suspicious_by_station": {"7654322": [1, 1]},
            },
        )


class FileTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )

    def test_save_and_load_round_trip(self):
        stations = [_station("7654321"), _station("7654322", 8, period=None)]
        data, _ = dataset.build(stations, [_quake([_record("7654321", None)])])
        path = self.root / "dataset.npz"

        data.save(path)
        loaded = dataset.Dataset.load(path)

        for name in _FIELDS:
            with self.subTest(name=name):
                expected = getattr(data, name)
                actual = getattr(loaded, name)
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(actual, expected)
        with np.load(path, allow_pickle=False) as arrays:
            self.assertEqual(
                sorted(arrays.files),
                sorted(_FIELDS),
            )

    def test_load_needs_the_arrays_of_a_dataset(self):
        stations = [_station("7654321")]
        data, _ = dataset.build(stations, [_quake([_record("7654321")])])
        arrays = {
            name: getattr(data, name)
            for name in _FIELDS
        }
        path = self.root / "dataset.npz"
        for changed in (
            {**arrays, "extra": np.zeros(1)},
            {name: array for name, array in arrays.items() if name != "felt"},
        ):
            with self.subTest(names=sorted(changed)):
                np.savez(path, **changed)
                with self.assertRaisesRegex(ValueError, "dataset"):
                    dataset.Dataset.load(path)

    def _bulletin(self, directory):
        """Writes code_p.zip and i2010.zip with one felt station."""
        record = bytearray(b" " * bulletin.RECORD_LENGTH)
        hypocenter = bytearray(record)
        for column, text in (
            (1, "A"),
            (2, "201006011200 000"),
            (22, " 35"),
            (25, "0000"),
            (33, " 139"),
            (37, "0000"),
            (45, " 10  "),
            (53, "50J"),
            (62, "4"),
        ):
            hypocenter[column - 1 : column - 1 + len(text)] = text.encode()
        for column, text in ((1, "7654321"), (19, "4"), (21, "38")):
            record[column - 1 : column - 1 + len(text)] = text.encode()
        with zipfile.ZipFile(directory / "i2010.zip", "w") as archive:
            archive.writestr(
                "i2010.dat", b"".join([hypocenter, b"\r\n", record, b"\r\n"])
            )
        with zipfile.ZipFile(directory / "code_p.zip", "w") as archive:
            archive.writestr(
                "code_p.dat",
                b"7654321\tTest\t3501\t13900\t199001010000\t\n"
                b"7654322\tOther\t3508\t13900\t199001010000\t\n",
            )
        return [directory / "code_p.zip", directory / "i2010.zip"]

    def test_write_checks_the_sources_and_writes_the_summary(self):
        jma = self.root / "jma"
        jma.mkdir()
        paths = self._bulletin(jma)
        sums = self.root / "SHA256SUMS"
        lines = [
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in paths
        ]
        sums.write_text("".join(lines), encoding="ascii")
        out = self.root / "out" / "dataset.npz"

        summary = dataset.write(jma, out, sums_path=sums, years=[2010])

        self.assertEqual(summary.felt_rows, 1)
        self.assertEqual(len(dataset.Dataset.load(out).felt), 2)
        self.assertEqual(
            json.loads((out.parent / "dataset-summary.json").read_text()),
            json.loads(summary.to_json()),
        )

        (jma / "i2010.zip").write_bytes(b"replaced")
        with self.assertRaisesRegex(sources.SourceError, "i2010.zip"):
            dataset.write(jma, out, sums_path=sums, years=[2010])
        with self.assertLogs(level="ERROR") as logs:
            status = dataset.main([
                "--sums", str(sums), "--jma", str(jma), "--out", str(out)
            ])
        self.assertEqual(status, 1)
        self.assertIn("SHA256SUMS", logs.output[0])


if __name__ == "__main__":
    unittest.main()

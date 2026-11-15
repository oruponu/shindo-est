# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the rows, splits, and features used to choose the ARV source."""

import datetime
import unittest

import numpy as np

import shindo_est
from training import bulletin
from training import dataset
from training import features
from training import selection

_ALWAYS = bulletin.Period(datetime.datetime(1990, 1, 1), None)
# The epicenter of the kanto quakes, in minutes of arc.
_LATITUDE = 35 * 60
_LONGITUDE = 139 * 60
# 7654322 is held out, the other numbers are not.
_TRAINED = "7654321"
_HELD_OUT = "7654322"
_ONE_ARV = "7654323"
_RETIRED = "7654325"
_ENDS_AT_2017 = "7654335"
_FAR = "7654328"
_UNLISTED_PERIOD = "7654333"
_NO_ARV = "7654334"
_TABLE = {
    _TRAINED: (1.2, 1.5),
    _HELD_OUT: (1.0, 0.9),
    _ONE_ARV: (1.1, None),
    _RETIRED: (1.3, 1.4),
    _ENDS_AT_2017: (1.6, 1.7),
    _FAR: (2.0, 1.8),
    _UNLISTED_PERIOD: (0.8, 0.7),
}


def _station(number, north, period=_ALWAYS):
    return bulletin.Station(
        number=number,
        name="Test",
        latitude_minutes=_LATITUDE + north,
        longitude_minutes=_LONGITUDE,
        period=period,
    )


def _quake(year, numbers, north=0):
    """Returns a quake on June 1 of a year recorded at some stations."""
    time = datetime.datetime(year, 6, 1, 12, 0)
    return bulletin.Quake(
        id=f"{time:%Y%m%d%H%M}0000",
        origin_time=time,
        latitude=(_LATITUDE + north) / 60,
        longitude=_LONGITUDE / 60,
        depth=10.0,
        magnitude=5.0,
        records=tuple(
            bulletin.Record(number, "3", 3.0) for number in numbers
        ),
    )


def _dataset():
    stations = [
        _station(_TRAINED, 0),
        _station(_HELD_OUT, 1),
        _station(_ONE_ARV, 2),
        _station(
            _RETIRED,
            3,
            bulletin.Period(
                datetime.datetime(2000, 1, 1), datetime.datetime(2015, 1, 1)
            ),
        ),
        _station(_NO_ARV, 4),
        _station(
            _ENDS_AT_2017,
            6,
            bulletin.Period(
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2017, 6, 1, 12, 0),
            ),
        ),
        _station(_UNLISTED_PERIOD, 5, period=None),
        _station(_FAR, 300),
    ]
    quakes = [
        _quake(2003, [_TRAINED]),
        _quake(2010, [_TRAINED, _UNLISTED_PERIOD]),
        _quake(2012, [_HELD_OUT], north=180),
        _quake(2017, [_TRAINED, _HELD_OUT]),
        _quake(2019, [_TRAINED]),
        _quake(2020, [_TRAINED, _HELD_OUT]),
    ]
    data, _ = dataset.build(stations, quakes)
    return data


class SelectionTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.data = _dataset()
        self.rows = selection.rows(self.data, _TABLE)
        self.year = self.data.quake_year[self.data.row_quake]
        self.number = self.data.station_number[self.data.row_station]
        self.held_out = self.data.station_held_out[self.data.row_station]
        self.region = self.data.quake_region[self.data.row_quake]
        self.splits = {
            split.name: split
            for split in selection.splits(self.data, self.rows)
        }

    def test_fixture(self):
        self.assertEqual(
            self.data.quake_region.tolist(),
            ["kanto", "kanto", "tohoku", "kanto", "kanto", "kanto"],
        )
        self.assertEqual(
            {str(number) for number in self.number[self.held_out]},
            {_HELD_OUT},
        )

    def test_selection_rows_leave_out_quakes_after_2019(self):
        self.assertTrue(np.any(self.year == 2020))
        self.assertFalse(np.any(self.rows[self.year == 2020]))
        trained_2019 = (self.year == 2019) & (self.number == _TRAINED)
        self.assertTrue(np.all(self.rows[trained_2019]))
        for split in self.splits.values():
            with self.subTest(split=split.name):
                self.assertFalse(np.any(split.train[self.year > 2019]))
                self.assertFalse(np.any(split.test[self.year > 2019]))
        self.assertEqual(selection.LAST_YEAR, 2019)

    def test_selection_rows_leave_out_quakes_before_2004(self):
        self.assertTrue(np.any(self.year == 2003))
        self.assertFalse(np.any(self.rows[self.year == 2003]))
        for split in self.splits.values():
            with self.subTest(split=split.name):
                self.assertFalse(np.any(split.train[self.year < 2004]))
                self.assertFalse(np.any(split.test[self.year < 2004]))
        self.assertEqual(selection.FIRST_YEAR, 2004)

    def test_selection_rows_need_both_arv(self):
        both = np.isin(
            self.number,
            [
                _TRAINED,
                _HELD_OUT,
                _RETIRED,
                _FAR,
                _UNLISTED_PERIOD,
                _ENDS_AT_2017,
            ],
        )

        np.testing.assert_array_equal(
            self.rows, both & (self.year >= 2004) & (self.year <= 2019)
        )
        self.assertTrue(np.any(self.number == _ONE_ARV))
        self.assertTrue(np.any(self.number == _NO_ARV))

    def test_training_rows_leave_out_the_held_out_stations(self):
        for split in self.splits.values():
            with self.subTest(split=split.name):
                self.assertFalse(np.any(split.train & self.held_out))
        self.assertTrue(np.any(self.splits["year"].test & self.held_out))

    def test_year_split(self):
        split = self.splits["year"]

        np.testing.assert_array_equal(
            split.train, self.rows & (self.year <= 2016) & ~self.held_out
        )
        np.testing.assert_array_equal(
            split.test, self.rows & (self.year >= 2017)
        )
        self.assertTrue(np.any(split.train))
        self.assertEqual(
            (selection.TRAINING_YEARS, selection.TEST_YEARS),
            ((2004, 2016), (2017, 2019)),
        )

    def test_region_splits(self):
        self.assertEqual(
            list(self.splits),
            ["year"] + [f"region:{name}" for name in dataset.REGIONS],
        )
        for name in ("kanto", "tohoku"):
            with self.subTest(region=name):
                split = self.splits[f"region:{name}"]
                np.testing.assert_array_equal(
                    split.test, self.rows & (self.region == name)
                )
                np.testing.assert_array_equal(
                    split.train,
                    self.rows & (self.region != name) & ~self.held_out,
                )
                self.assertTrue(np.any(split.test))
        self.assertFalse(np.any(self.splits["region:izu"].test))

    def test_candidates_of_an_empty_split(self):
        quakes, stations = selection.candidates(
            self.data, self.splits["region:izu"].test, _TABLE
        )

        self.assertEqual((len(quakes), len(stations)), (0, 0))
        self.assertEqual((quakes.dtype, stations.dtype), (np.int32, np.int32))

    def test_candidates_include_sampled_out_sites(self):
        sampled_out = (self.year == 2017) & (self.number == _FAR)
        self.assertFalse(np.any(sampled_out))

        quakes, stations = selection.candidates(
            self.data, self.splits["year"].test, _TABLE
        )
        kanto = selection.candidates(
            self.data, self.splits["region:kanto"].test, _TABLE
        )

        pairs = [
            (
                int(self.data.quake_year[quake]),
                str(self.data.station_number[station]),
            )
            for quake, station in zip(quakes, stations)
        ]
        self.assertEqual(
            pairs,
            [
                (year, number)
                for year in (2017, 2019)
                for number in (_TRAINED, _HELD_OUT, _FAR)
            ],
        )
        self.assertEqual((quakes.dtype, stations.dtype), (np.int32, np.int32))
        recorded = [
            str(self.data.station_number[station])
            for quake, station in zip(*kanto)
            if self.data.quake_year[quake] == 2010
        ]
        self.assertEqual(
            recorded,
            [
                _TRAINED,
                _HELD_OUT,
                _RETIRED,
                _FAR,
                _UNLISTED_PERIOD,
                _ENDS_AT_2017,
            ],
        )


class FeatureMatrixTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.data = _dataset()
        rows = selection.rows(self.data, _TABLE)
        self.quakes = self.data.row_quake[rows]
        self.stations = self.data.row_station[rows]
        self.slabs = shindo_est.SlabGeometry.read(
            features.FIXTURES / "kuril.csv", features.FIXTURES / "izu_bonin.csv"
        )

    def _expected(self, source):
        """Returns the 14 features with the ARV of a source, 0 or 1."""
        numbers = self.data.station_number[self.stations]
        return features.compute(
            self.slabs,
            hypocenter_latitude=self.data.quake_latitude[self.quakes],
            hypocenter_longitude=self.data.quake_longitude[self.quakes],
            hypocenter_depth=self.data.quake_depth[self.quakes],
            hypocenter_magnitude=self.data.quake_magnitude[self.quakes],
            site_latitude=self.data.station_latitude[self.stations],
            site_longitude=self.data.station_longitude[self.stations],
            site_arv=np.array([_TABLE[str(n)][source] for n in numbers]),
            threads=2,
        )

    def _matrix(self, case, stations=None, table=None):
        return selection.feature_matrix(
            self.data,
            self.quakes,
            self.stations if stations is None else stations,
            _TABLE if table is None else table,
            case,
            self.slabs,
            threads=2,
        )

    def test_cases_use_their_arv(self):
        for case, source in (("cabinet_office", 0), ("jshis", 1)):
            with self.subTest(case=case):
                values, names = self._matrix(case)

                self.assertEqual(names, features.FEATURE_NAMES)
                np.testing.assert_array_equal(
                    values.view(np.uint32),
                    self._expected(source).view(np.uint32),
                )
        self.assertEqual(
            selection.CASES, ("cabinet_office", "jshis", "none")
        )

    def test_case_none_drops_log_arv(self):
        values, names = self._matrix("none")

        column = features.FEATURE_NAMES.index("log_arv")
        self.assertEqual(
            names, tuple(n for n in features.FEATURE_NAMES if n != "log_arv")
        )
        np.testing.assert_array_equal(
            values.view(np.uint32),
            np.delete(self._expected(0), column, axis=1).view(np.uint32),
        )
        self.assertEqual(values.dtype, np.float32)

    def test_missing_arv_is_refused(self):
        one_arv = int(np.flatnonzero(self.data.station_number == _ONE_ARV)[0])
        stations = self.stations.copy()
        stations[3] = one_arv
        jshis_only = {**_TABLE, _ONE_ARV: (None, 1.1)}

        self._matrix("cabinet_office", stations)
        self._matrix("none", stations)
        self._matrix("jshis", stations, jshis_only)
        for case, table in (
            ("jshis", _TABLE),
            ("cabinet_office", jshis_only),
            ("none", jshis_only),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, _ONE_ARV):
                    self._matrix(case, stations, table)
        with self.assertRaisesRegex(ValueError, "case"):
            self._matrix("other")

    def test_no_pairs(self):
        values, names = selection.feature_matrix(
            self.data,
            self.quakes[:0],
            self.stations[:0],
            _TABLE,
            "none",
            self.slabs,
            threads=2,
        )

        self.assertEqual(values.shape, (0, len(names)))
        self.assertEqual(len(names), 13)


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks SlabGeometry.features against expected.csv bit for bit.

Also checks the arrays it accepts and rejects, the values it passes through
unchecked, reading the grids and their errors, and threads.
"""

import concurrent.futures
import math
import pathlib
import pickle
import tempfile
import unittest

import numpy as np

import shindo_est
from tests import expected
from tests import gil

_KURIL = expected.FIXTURES / "kuril.csv"
_IZU_BONIN = expected.FIXTURES / "izu_bonin.csv"

# Values that float32 holds exactly, so float32 inputs give the same features.
_ROWS = {
    "hypocenter_latitude": (35.5, 35.5, 35.5),
    "hypocenter_longitude": (140.25, 140.25, 140.25),
    "hypocenter_depth": (40.0, 40.0, 40.0),
    "hypocenter_magnitude": (6.5, 6.5, 6.5),
    "site_latitude": (35.0, 35.125, 34.875),
    "site_longitude": (140.0, 140.125, 140.0625),
    "site_arv": (1.0, 2.0, 0.5),
}


def _rows_of(calls):
    """Returns the arguments of features for the sites of calls, in order."""
    rows = {name: [] for name in _ROWS}
    for call in calls:
        sites = len(call.site_latitude)
        rows["hypocenter_latitude"] += [call.hypocenter_latitude] * sites
        rows["hypocenter_longitude"] += [call.hypocenter_longitude] * sites
        rows["hypocenter_depth"] += [call.hypocenter_depth] * sites
        rows["hypocenter_magnitude"] += [call.hypocenter_magnitude] * sites
        rows["site_latitude"] += call.site_latitude
        rows["site_longitude"] += call.site_longitude
        rows["site_arv"] += call.site_arv
    return rows


def _many_rows():
    """Returns `_ROWS` repeated to 10,000 rows, enough to keep a call busy."""
    return {
        name: np.resize(np.array(values), 10_000)
        for name, values in _ROWS.items()
    }


def _read_only(values):
    """Returns the values as an array that cannot be written to."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _column(name):
    return shindo_est.FEATURE_NAMES.index(name)


class FeaturesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.slabs = shindo_est.SlabGeometry.read(_KURIL, _IZU_BONIN)
        cls.calls = expected.calls()

    def setUp(self):
        super().setUp()
        self.dir = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def assert_same_features(self, actual, wanted, context):
        self.assertEqual(actual.shape, wanted.shape, context)
        for row, values in enumerate(wanted):
            expected.assert_same_bits(
                self, actual[row], values, f"{context}, row {row}"
            )

    def test_matches_the_core_bit_for_bit(self):
        rows = _rows_of(self.calls)
        self.assertEqual(len(rows["site_latitude"]), 30)
        # The calls differ in their hypocenters, so each row must use its own.
        self.assertGreater(len(set(rows["hypocenter_latitude"])), 1)
        features = self.slabs.features(**rows)
        wanted = []
        for call in self.calls:
            wanted += call.features
        self.assertEqual(features.shape, (30, len(shindo_est.FEATURE_NAMES)))
        for row, values in enumerate(wanted):
            expected.assert_same_bits(
                self, features[row], values, f"row {row}"
            )

    def test_shape_and_type(self):
        features = self.slabs.features(**_rows_of(self.calls[:1]))
        self.assertEqual(features.shape, (6, 14))
        self.assertEqual(features.dtype, np.float64)
        self.assertTrue(features.flags.c_contiguous)
        empty = self.slabs.features(**{name: [] for name in _ROWS})
        self.assertEqual(empty.shape, (0, 14))
        self.assertEqual(empty.dtype, np.float64)

    def test_columns_follow_the_feature_names(self):
        features = self.slabs.features(**_ROWS)
        self.assertEqual(
            features[:, _column("mj")].tolist(),
            list(_ROWS["hypocenter_magnitude"]),
        )
        self.assertEqual(
            features[:, _column("slat")].tolist(), list(_ROWS["site_latitude"])
        )
        for row, arv in enumerate(_ROWS["site_arv"]):
            self.assertAlmostEqual(
                features[row, _column("log_arv")], math.log10(arv), delta=1e-15
            )

    def test_accepts_lists_float32_read_only_and_strided_arrays(self):
        forms = {
            "list": list,
            "float32": lambda values: np.array(values, dtype=np.float32),
            "read-only": _read_only,
            "strided": lambda values: np.repeat(np.array(values), 2)[::2],
        }
        wanted = self.slabs.features(
            **{name: np.array(values) for name, values in _ROWS.items()}
        )
        for form_name, form in forms.items():
            with self.subTest(form=form_name):
                features = self.slabs.features(
                    **{name: form(values) for name, values in _ROWS.items()}
                )
                self.assert_same_features(features, wanted, form_name)

    def test_accepts_integer_arrays(self):
        rows = {
            "hypocenter_latitude": [35, 35],
            "hypocenter_longitude": [140, 140],
            "hypocenter_depth": [40, 40],
            "hypocenter_magnitude": [7, 7],
            "site_latitude": [35, 36],
            "site_longitude": [141, 140],
            "site_arv": [1, 2],
        }
        integers = {
            name: np.array(values, dtype=np.int64)
            for name, values in rows.items()
        }
        floats = {
            name: values.astype(np.float64)
            for name, values in integers.items()
        }
        self.assert_same_features(
            self.slabs.features(**integers),
            self.slabs.features(**floats),
            "int64",
        )

    def test_rejects_arrays_of_different_lengths(self):
        rows = dict(_ROWS, site_arv=(1.0, 2.0))
        with self.assertRaises(ValueError) as context:
            self.slabs.features(**rows)
        self.assertIn("site_arv 2", str(context.exception))

    def test_rejects_arrays_that_are_not_one_dimensional(self):
        rows = dict(_ROWS, site_latitude=np.full((3, 2), 35.0))
        with self.assertRaises(TypeError):
            self.slabs.features(**rows)

    def test_arguments_are_keyword_only(self):
        with self.assertRaises(TypeError):
            self.slabs.features(*_ROWS.values())

    def test_unknown_values_pass_through(self):
        rows = dict(
            _ROWS,
            hypocenter_depth=(math.nan, 40.0, 40.0),
            site_latitude=(35.0, 91.0, 34.875),
        )
        features = self.slabs.features(**rows)
        self.assertTrue(math.isnan(features[0, _column("depth")]))
        self.assertEqual(features[1, _column("slat")], 91.0)

    def test_path_fraction_is_0_for_an_unknown_location_or_depth(self):
        rows = _rows_of(self.calls)
        path_fractions = self.slabs.features(**rows)[:, _column("path_frac")]
        row = int(np.argmax(path_fractions > 0.0))
        self.assertGreater(path_fractions[row], 0.0)
        known = {name: [values[row]] for name, values in rows.items()}
        unknowns = (
            "hypocenter_latitude",
            "hypocenter_longitude",
            "hypocenter_depth",
            "site_latitude",
            "site_longitude",
        )
        for name in unknowns:
            with self.subTest(unknown=name):
                features = self.slabs.features(**{**known, name: [math.nan]})
                self.assertTrue(math.isnan(features[0, _column("log_hypo")]))
                self.assertEqual(features[0, _column("path_frac")], 0.0)

    def test_reads_the_repository_grids(self):
        slab2 = expected.REPOSITORY / "data" / "slab2"
        slabs = shindo_est.SlabGeometry.read(
            slab2 / "kur.csv", slab2 / "izu.csv"
        )
        features = slabs.features(
            hypocenter_latitude=[38.0],
            hypocenter_longitude=[142.5],
            hypocenter_depth=[40.0],
            hypocenter_magnitude=[7.0],
            site_latitude=[36.0],
            site_longitude=[140.0],
            site_arv=[1.0],
        )
        slab_top = features[0, _column("slab_top")]
        self.assertTrue(math.isfinite(slab_top), slab_top)
        self.assertGreater(slab_top, 0.0)
        self.assertLess(slab_top, 700.0)

    def test_missing_grid_raises_file_not_found_error(self):
        path = self.dir / "missing.csv"
        with self.assertRaises(FileNotFoundError) as context:
            shindo_est.SlabGeometry.read(path, _IZU_BONIN)
        self.assertIn(str(path), str(context.exception))

    def test_broken_grid_raises_slab_grid_error(self):
        broken = self.write("broken.csv", b"140,35,100\n")
        not_text = self.write("not-text.csv", b"\xff")
        cases = [
            ("kuril", (broken, _IZU_BONIN), broken, None),
            ("izu_bonin", (_KURIL, broken), broken, None),
            ("not UTF-8", (not_text, _IZU_BONIN), not_text, "UTF-8"),
        ]
        for name, paths, path, words in cases:
            with self.subTest(name=name):
                with self.assertRaises(shindo_est.SlabGridError) as context:
                    shindo_est.SlabGeometry.read(*paths)
                message = str(context.exception)
                self.assertTrue(message.startswith(f"{path}: "), message)
                if words is not None:
                    self.assertIn(words, message)

    def test_cannot_be_constructed_directly(self):
        with self.assertRaises(TypeError):
            shindo_est.SlabGeometry()

    def test_is_not_picklable(self):
        with self.assertRaises(TypeError):
            pickle.dumps(self.slabs)

    @unittest.skipUnless(gil.is_enabled(), "the GIL is disabled")
    def test_releases_the_gil(self):
        rows = _many_rows()
        self.assertTrue(
            gil.is_released_during(lambda: self.slabs.features(**rows))
        )

    def test_threads_can_share_slabs(self):
        rows = _many_rows()
        wanted = self.slabs.features(**rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.slabs.features, **rows) for _ in range(8)
            ]
            results = [future.result() for future in futures]
        for number, features in enumerate(results):
            with self.subTest(result=number):
                self.assertEqual(
                    features.tobytes(), wanted.tobytes(), f"thread {number}"
                )


if __name__ == "__main__":
    unittest.main()

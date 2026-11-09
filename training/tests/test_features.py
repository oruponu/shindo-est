# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the stale extension check and the chunked feature computation."""

import csv
import math
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import numpy.typing as npt

import shindo_est
from training import features


def _columns(
    rows: list[dict[str, str]],
) -> dict[str, npt.NDArray[np.float64]]:
    names = (
        "hypocenter_latitude",
        "hypocenter_longitude",
        "hypocenter_depth",
        "hypocenter_magnitude",
        "site_latitude",
        "site_longitude",
        "site_arv",
    )
    return {
        name: np.array([float(row[name]) for row in rows], np.float64)
        for name in names
    }


class CheckExtensionTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        with features.EXPECTED.open(newline="", encoding="ascii") as file:
            self.committed = list(csv.reader(file))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.copy = pathlib.Path(directory) / "expected.csv"

    def _check_rows(self, rows):
        with self.copy.open("w", newline="", encoding="ascii") as file:
            csv.writer(file, lineterminator="\n").writerows(rows)
        features.check_extension(self.copy)

    def _changed(self, column, value):
        rows = [list(row) for row in self.committed]
        rows[1][column] = value
        return rows

    def test_extension_matches_the_committed_features(self):
        features.check_extension()

    def test_feature_names_match_the_extension(self):
        self.assertEqual(features.FEATURE_NAMES, shindo_est.FEATURE_NAMES)

    def test_stale_features_are_reported(self):
        for name in features.FEATURE_NAMES:
            with self.subTest(name=name):
                column = self.committed[0].index(f"feature_{name}")
                with self.assertRaises(features.StaleExtensionError) as context:
                    self._check_rows(self._changed(column, "1e-300"))
                message = str(context.exception)
                self.assertIn(f"feature {name}:", message)
                self.assertIn("row 1", message)
                self.assertIn("pip install", message)

    def test_features_are_compared_bit_for_bit(self):
        mj_column = self.committed[0].index("feature_mj")
        epi_column = self.committed[0].index("feature_epi")
        mj = float(self.committed[1][mj_column])
        self.assertEqual(self.committed[1][epi_column], "0.0")
        changes = {
            "one ulp": (mj_column, repr(math.nextafter(mj, math.inf))),
            "negative zero": (epi_column, "-0.0"),
            "NaN for a number": (mj_column, "NaN"),
        }
        for change, (column, value) in changes.items():
            with self.subTest(change=change):
                with self.assertRaises(features.StaleExtensionError):
                    self._check_rows(self._changed(column, value))

    def test_nan_values_are_equal_whatever_their_bits(self):
        rows = [
            ["-nan" if value == "NaN" else value for value in row]
            for row in self.committed
        ]
        self.assertNotEqual(rows, self.committed)
        self._check_rows(rows)

    def test_other_feature_versions_are_reported(self):
        with mock.patch.object(shindo_est, "FEATURE_VERSION", 2):
            with self.assertRaisesRegex(
                features.StaleExtensionError, "feature version 2"
            ):
                features.check_extension()

    def test_other_feature_names_are_reported(self):
        names = tuple(reversed(features.FEATURE_NAMES))
        with mock.patch.object(shindo_est, "FEATURE_NAMES", names):
            with self.assertRaisesRegex(
                features.StaleExtensionError, "pip install"
            ):
                features.check_extension()


class ComputeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.slabs = features.read_slabs()
        with features.EXPECTED.open(newline="", encoding="ascii") as file:
            cls.rows = list(csv.DictReader(file))

    def test_compute_matches_one_call(self):
        columns = {
            name: np.tile(values, 20_000)
            for name, values in _columns(self.rows).items()
        }
        computed = features.compute(self.slabs, threads=4, **columns)
        expected = self.slabs.features(**columns).astype(np.float32)
        self.assertEqual(computed.shape, (600_000, 14))
        self.assertEqual(computed.dtype, np.float32)
        self.assertTrue(computed.flags.c_contiguous)
        np.testing.assert_array_equal(
            computed.view(np.uint32), expected.view(np.uint32)
        )

    def test_compute_refuses_a_stale_extension(self):
        columns = _columns(self.rows)
        with mock.patch.object(shindo_est, "FEATURE_VERSION", 2):
            with self.assertRaises(features.StaleExtensionError):
                features.compute(self.slabs, threads=4, **columns)

    def test_compute_rejects_arrays_of_different_lengths(self):
        columns = _columns(self.rows)
        columns["site_latitude"] = np.append(columns["site_latitude"], 35.0)
        with self.assertRaisesRegex(ValueError, "site_latitude 31"):
            features.compute(self.slabs, threads=4, **columns)

    def test_compute_handles_no_rows(self):
        columns = {
            name: values[:0] for name, values in _columns(self.rows).items()
        }
        computed = features.compute(self.slabs, threads=4, **columns)
        self.assertEqual(computed.shape, (0, 14))
        self.assertEqual(computed.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

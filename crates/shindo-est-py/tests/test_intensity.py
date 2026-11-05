# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks the class boundaries, NaN, the accepted arrays, and the labels."""

import math
import unittest

import numpy as np

import shindo_est


class IntensityClassTest(unittest.TestCase):

    def test_boundaries_belong_to_the_upper_class(self):
        classes = shindo_est.intensity_class(
            [0.49999, 0.5, 4.5, 4.99999, 5.0, 6.5, -1.0, 9.0]
        )
        self.assertEqual(classes.tolist(), [0, 1, 5, 5, 6, 9, 0, 9])

    def test_nan_has_no_class(self):
        self.assertEqual(shindo_est.intensity_class([math.nan]).tolist(), [-1])

    def test_returns_a_one_dimensional_int32_array(self):
        classes = shindo_est.intensity_class([1.0, 2.0, 3.0])
        self.assertEqual(classes.dtype, np.int32)
        self.assertEqual(classes.shape, (3,))
        self.assertEqual(shindo_est.intensity_class([]).shape, (0,))

    def test_accepts_lists_float32_and_strided_arrays(self):
        strided = np.arange(8.0)[::2]
        cases = [
            ([0.5, 4.5], np.array([0.5, 4.5])),
            (np.array([0.5, 4.5], dtype=np.float32), np.array([0.5, 4.5])),
            (strided, np.ascontiguousarray(strided)),
        ]
        for values, contiguous in cases:
            with self.subTest(values=values):
                self.assertEqual(
                    shindo_est.intensity_class(values).tolist(),
                    shindo_est.intensity_class(contiguous).tolist(),
                )

    def test_rejects_arrays_that_are_not_one_dimensional(self):
        for values in [np.zeros((2, 2)), 3.0]:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    shindo_est.intensity_class(values)


class IntensityLabelTest(unittest.TestCase):

    def test_labels(self):
        labels = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        for value, label in enumerate(labels):
            with self.subTest(value=value):
                self.assertEqual(shindo_est.intensity_label(value), label)
        for value in [-1, 10]:
            with self.subTest(value=value):
                self.assertIsNone(shindo_est.intensity_label(value))
        self.assertEqual(shindo_est.intensity_label(np.int32(5)), "5-")

    def test_integers_of_any_size_outside_0_to_9_have_no_label(self):
        for value in [2**100, -(2**100), 2**63, np.uint64(2**64 - 1)]:
            with self.subTest(value=value):
                self.assertIsNone(shindo_est.intensity_label(value))

    def test_rejects_values_that_are_not_integers(self):
        for value in [5.0, "5", None]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    shindo_est.intensity_label(value)


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the displayed intensity and the measures of a prediction."""

import math
import unittest

import numpy as np

from training import evaluate


def _summary(felt, observed, display, weight=None, row_quake=None):
    """Summarizes rows whose displayed intensity is given directly.

    A probability of 0.9 keeps a regression value of 0.5 or more, and 0.1
    keeps one below 0.5, so the displayed intensity is the regression value.
    """
    if weight is None:
        weight = [1] * len(display)
    if row_quake is None:
        row_quake = [0] * len(display)
    shown = np.array(display, dtype=np.float64)
    return evaluate.summarize(
        row_quake=np.array(row_quake, dtype=np.int32),
        felt=np.array(felt, dtype=np.bool_),
        observed=np.array(observed, dtype=np.float32),
        weight=np.array(weight, dtype=np.float32),
        probability=np.where(shown >= 0.5, 0.9, 0.1),
        regression=shown,
    )


class DisplayTest(unittest.TestCase):

    def test_display_follows_the_hurdle(self):
        shown = evaluate.display(
            np.array([0.35, 0.3499, 0.9, 0.3499, 0.9]),
            np.array([0.2, 0.7, 4.2, 0.3, 0.1]),
            evaluate.PREVIOUS,
        )

        self.assertEqual(shown.tolist(), [0.5, 0.49, 4.2, 0.3, 0.5])

    def test_display_uses_the_values(self):
        values = evaluate.HurdleValues(
            felt_probability=0.6, felt_floor=1.0, unfelt_ceiling=0.2
        )

        shown = evaluate.display(
            np.array([0.6, 0.5]), np.array([0.7, 0.7]), values
        )

        self.assertEqual(shown.tolist(), [1.0, 0.2])
        self.assertEqual(
            evaluate.PREVIOUS, evaluate.HurdleValues(0.35, 0.5, 0.49)
        )


class SummarizeTest(unittest.TestCase):

    def test_felt_errors(self):
        summary = _summary(
            felt=[True, True, True, True, False],
            observed=[1.0, 2.0, math.nan, 3.0, math.nan],
            display=[1.5, 2.5, 9.0, 4.0, 0.1],
            row_quake=[0, 0, 1, 1, 2],
        )

        self.assertEqual(summary["quakes"], 3)
        self.assertEqual(summary["felt_rows"], 3)
        self.assertAlmostEqual(summary["felt_bias"], 2 / 3)
        self.assertAlmostEqual(summary["felt_rmse"], math.sqrt(0.5))
        self.assertEqual(summary["strong4_rows"], 0)
        self.assertIsNone(summary["strong4_bias"])
        self.assertIsNone(summary["strong5_capture"])

    def test_weak_and_strong_metrics(self):
        summary = _summary(
            felt=[True] * 4,
            observed=[1.0, 2.0, 3.6, 4.6],
            display=[1.4, 2.6, 3.6, 4.4],
        )

        self.assertEqual(summary["weak_rows"], 2)
        self.assertAlmostEqual(summary["weak_bias"], 0.5)
        self.assertEqual(summary["weak_over_class"], 0.5)
        self.assertEqual(summary["strong4_rows"], 2)
        self.assertAlmostEqual(summary["strong4_bias"], -0.1, places=6)
        self.assertAlmostEqual(
            summary["strong4_rmse"], math.sqrt(0.02), places=6
        )
        self.assertEqual(summary["strong4_under_class"], 0.5)
        self.assertEqual(summary["strong5_rows"], 1)
        self.assertAlmostEqual(summary["strong5_bias"], -0.2, places=6)
        self.assertEqual(summary["strong5_capture"], 0.0)

    def test_classes_follow_the_edges(self):
        # A value on an edge belongs to the class above: 1.5 against 1.49 is
        # a class over, 2.5 is not weak, and 3.5 and 4.5 are strong.
        summary = _summary(
            felt=[True] * 5,
            observed=[4.5, 1.49, 2.0, 2.5, 3.5],
            display=[4.5, 1.5, 2.4, 2.5, 3.5],
        )

        self.assertEqual(summary["weak_rows"], 2)
        self.assertEqual(summary["weak_over_class"], 0.5)
        self.assertEqual(summary["strong4_rows"], 2)
        self.assertEqual(summary["strong5_rows"], 1)
        self.assertEqual(summary["strong5_capture"], 1.0)
        self.assertEqual(
            evaluate.CLASS_EDGES, (0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5)
        )

    def test_weighted_counts(self):
        weight = np.float32(33.3).item()
        summary = _summary(
            felt=[False, False, True, True, True, True],
            observed=[math.nan, math.nan, 4.0, 5.0, math.nan, 2.0],
            display=[3.6, 4.9, 4.6, 4.7, 4.8, 3.5],
            weight=[weight, weight, 1, 1, 1, 1],
        )

        self.assertAlmostEqual(summary["predicted5"], 3 + weight, places=5)
        self.assertAlmostEqual(
            summary["predicted5_wrong"], 1 + weight, places=5
        )
        self.assertAlmostEqual(
            summary["predicted4_observed2"], 1 + 2 * weight, places=5
        )

    def test_ets(self):
        # The hits are displayed at the felt floor, 0.5, and weigh 6.
        summary = _summary(
            felt=[True] * 5 + [False, True, False],
            observed=[1.0] * 5 + [math.nan, 1.0, math.nan],
            display=[0.5] * 5 + [1.0, 0.3, 0.0],
            weight=[1, 1, 1, 1, 2] + [2, 1, 91],
        )

        chance = (6 + 2) * (6 + 1) / 100
        self.assertAlmostEqual(
            summary["ets"], (6 - chance) / (6 + 2 + 1 - chance)
        )
        self.assertAlmostEqual(summary["pod"], 6 / 7)
        self.assertAlmostEqual(summary["far"], 2 / 8)

    def test_calibration(self):
        felt = np.array([False, True, False, True, True])
        weight = np.array([10, 2, 1, 1, 1], dtype=np.float32)
        probability = np.array([0.05, 0.1, 0.15, 1.0, 0.95])

        bins = evaluate.calibration(felt, weight, probability)
        summary = evaluate.summarize(
            row_quake=np.zeros(5, dtype=np.int32),
            felt=felt,
            observed=np.full(5, math.nan, dtype=np.float32),
            weight=weight,
            probability=probability,
            regression=np.zeros(5),
        )

        self.assertEqual(len(bins), 10)
        self.assertEqual((bins[1].low, bins[1].high), (0.1, 0.2))
        self.assertEqual((bins[0].weight, bins[0].felt), (10.0, 0.0))
        self.assertAlmostEqual(bins[0].probability, 0.05)
        self.assertEqual(bins[1].weight, 3.0)
        self.assertAlmostEqual(bins[1].probability, 0.35 / 3)
        self.assertAlmostEqual(bins[1].felt, 2 / 3)
        self.assertEqual((bins[9].weight, bins[9].felt), (2.0, 1.0))
        self.assertAlmostEqual(bins[9].probability, 0.975)
        self.assertEqual((bins[5].weight, bins[5].probability), (0.0, None))
        self.assertAlmostEqual(summary["calibration_total"], 2.8 / 4)


class EmptyTest(unittest.TestCase):

    def test_no_rows(self):
        nothing = np.empty(0)

        summary = _summary(felt=[], observed=[], display=[])
        bins = evaluate.calibration(
            np.empty(0, dtype=np.bool_), nothing.astype(np.float32), nothing
        )
        maxima = evaluate.max_per_quake(
            np.empty(0, dtype=np.int32),
            nothing,
            np.empty(0, dtype=np.int32),
            nothing.astype(np.float32),
        )

        counts = (
            "quakes", "felt_rows", "weak_rows", "strong4_rows", "strong5_rows"
        )
        weights = ("predicted5", "predicted5_wrong", "predicted4_observed2")
        for name, value in summary.items():
            with self.subTest(name=name):
                if name in counts or name in weights:
                    self.assertEqual(value, 0)
                else:
                    self.assertIsNone(value)
        self.assertEqual([b.weight for b in bins], [0.0] * 10)
        self.assertEqual(maxima["max_quakes"], 0)
        with self.assertRaisesRegex(ValueError, "same length"):
            evaluate.bootstrap_rmse_difference(
                np.empty(0, dtype=np.int32), nothing, nothing
            )


class MaxPerQuakeTest(unittest.TestCase):

    def test_max_per_quake(self):
        maxima = evaluate.max_per_quake(
            candidate_quake=np.array([0, 0, 0, 1, 1, 2], dtype=np.int32),
            candidate_display=np.array([3.0, 4.8, 1.0, 2.0, 2.4, 5.0]),
            observed_quake=np.array([0, 0, 1, 1, 3], dtype=np.int32),
            observed=np.array([4.6, math.nan, 2.7, 1.0, 3.0], dtype=np.float32),
        )

        self.assertEqual(maxima["max_quakes"], 2)
        self.assertAlmostEqual(maxima["max_bias"], -0.05, places=6)
        self.assertAlmostEqual(maxima["max_rmse"], math.sqrt(0.065), places=6)
        self.assertEqual(maxima["max_under_class"], 0.5)

    def test_max_per_quake_without_pairs(self):
        maxima = evaluate.max_per_quake(
            candidate_quake=np.array([0], dtype=np.int32),
            candidate_display=np.array([1.0]),
            observed_quake=np.array([1], dtype=np.int32),
            observed=np.array([1.0], dtype=np.float32),
        )

        self.assertEqual(
            maxima,
            {
                "max_quakes": 0,
                "max_bias": None,
                "max_rmse": None,
                "max_under_class": None,
            },
        )


class BootstrapTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(1)
        self.row_quake = np.repeat(np.arange(20, dtype=np.int32), 5)
        self.error_a = rng.normal(scale=0.5, size=100)
        self.error_b = rng.normal(scale=0.4, size=100)

    def test_bootstrap_is_reproducible(self):
        first = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_b
        )
        again = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_b
        )
        other = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_b, seed=1
        )
        same = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_a, samples=50
        )

        self.assertEqual(first, again)
        self.assertNotEqual(first[1:], other[1:])
        self.assertEqual(first[0], other[0])
        self.assertEqual(same, (0.0, 0.0, 0.0))

    def test_bootstrap_difference_and_interval(self):
        difference, low, high = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_b
        )

        self.assertAlmostEqual(
            difference,
            math.sqrt(np.mean(self.error_a**2))
            - math.sqrt(np.mean(self.error_b**2)),
        )
        self.assertLess(low, difference)
        self.assertLess(difference, high)

    def test_bootstrap_interval_follows_the_formula(self):
        draws = np.random.default_rng(0).integers(0, 20, size=(1000, 20))
        rows = np.arange(100).reshape(20, 5)
        differences = [
            math.sqrt(np.mean(self.error_a[rows[draw]] ** 2))
            - math.sqrt(np.mean(self.error_b[rows[draw]] ** 2))
            for draw in draws
        ]

        _, low, high = evaluate.bootstrap_rmse_difference(
            self.row_quake, self.error_a, self.error_b
        )

        expected = np.percentile(differences, [2.5, 97.5])
        self.assertAlmostEqual(low, expected[0])
        self.assertAlmostEqual(high, expected[1])

    def test_bootstrap_resamples_whole_quakes(self):
        # One quake holds every error of a, so a resample of quakes either
        # has it or not; the interval of a resample of rows would be narrow.
        error_a = np.zeros(100)
        error_a[:5] = 1.0

        _, low, high = evaluate.bootstrap_rmse_difference(
            self.row_quake, error_a, np.zeros(100)
        )

        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.3)


if __name__ == "__main__":
    unittest.main()

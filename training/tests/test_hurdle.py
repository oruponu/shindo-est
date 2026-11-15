# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the hurdle model of felt probability and intensity."""

import unittest
from unittest import mock

import lightgbm
import numpy as np

from training import features
from training import hurdle

_ROUNDS = 30


def _synthetic(rows=5000, quakes=50, columns=14):
    """Returns rows whose intensity grows with mj and falls with epi.

    The rows are the keyword arguments of hurdle.fit other than the names.
    """
    rng = np.random.default_rng(1)
    row_quake = np.sort(rng.integers(0, quakes, rows)).astype(np.int32)
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    signal = values[:, 0] - values[:, 7]
    felt = signal + rng.normal(scale=0.5, size=rows) > 0
    observed = np.where(felt, 2.0 + signal, np.nan).astype(np.float32)
    observed[felt & (rng.random(rows) < 0.2)] = np.nan
    weight = np.where(felt | (rng.random(rows) < 0.5), 1.0, 1 / 0.03)
    return {
        "features": values,
        "row_quake": row_quake,
        "felt": felt,
        "observed": observed,
        "weight": weight.astype(np.float32),
    }


def _fit(rows, names=features.FEATURE_NAMES, max_rounds=_ROUNDS):
    return hurdle.fit(feature_names=names, max_rounds=max_rounds, **rows)


class ValidationTest(unittest.TestCase):

    def test_validation_quakes_are_whole_quakes(self):
        row_quake = np.repeat(np.arange(50, dtype=np.int32), 7)

        validation = hurdle.validation_quakes(row_quake)

        chosen = np.unique(row_quake[validation])
        self.assertEqual(len(chosen), 5)
        np.testing.assert_array_equal(
            validation, np.isin(row_quake, chosen)
        )
        np.testing.assert_array_equal(
            validation, hurdle.validation_quakes(row_quake)
        )
        other = hurdle.validation_quakes(row_quake, seed=1)
        self.assertFalse(np.array_equal(validation, other))

    def test_validation_needs_rows(self):
        with self.assertRaisesRegex(ValueError, "no quakes"):
            hurdle.validation_quakes(np.empty(0, dtype=np.int32))

    def test_validation_takes_at_least_one_quake(self):
        row_quake = np.array([3, 3, 8, 9], dtype=np.int32)

        validation = hurdle.validation_quakes(row_quake)

        self.assertEqual(len(np.unique(row_quake[validation])), 1)


class FitTest(unittest.TestCase):

    def test_fit_is_deterministic(self):
        rows = _synthetic()

        first = _fit(rows)
        again = _fit(rows)

        self.assertEqual(
            first.classifier.model_to_string(),
            again.classifier.model_to_string(),
        )
        self.assertEqual(
            first.regressor.model_to_string(),
            again.regressor.model_to_string(),
        )
        self.assertIs(first.classifier.params["deterministic"], True)

    def test_monotone_constraints_follow_the_names(self):
        rows = _synthetic(columns=13)
        without_arv = tuple(n for n in features.FEATURE_NAMES if n != "log_arv")
        for names in (without_arv, without_arv[::-1]):
            with self.subTest(names=names[0]):
                model = _fit(rows, names)

                expected = [hurdle.MONOTONE.get(name, 0) for name in names]
                self.assertEqual(expected.count(0), 10)
                for booster in (model.classifier, model.regressor):
                    self.assertEqual(
                        list(booster.params["monotone_constraints"]), expected
                    )
                    self.assertEqual(booster.feature_name(), list(names))
        self.assertEqual(hurdle.MONOTONE, {"mj": 1, "log_hypo": -1, "epi": -1})

    def test_training_rows(self):
        rows = _synthetic()
        validation = hurdle.validation_quakes(rows["row_quake"])
        known = ~np.isnan(rows["observed"])

        with mock.patch.object(
            lightgbm, "train", wraps=lightgbm.train
        ) as train:
            _fit(rows)

        classifier, regressor = train.call_args_list
        self.assertEqual(classifier.kwargs["params"]["objective"], "binary")
        self.assertEqual(
            classifier.kwargs["train_set"].num_data(), (~validation).sum()
        )
        np.testing.assert_array_equal(
            classifier.kwargs["train_set"].get_weight(),
            rows["weight"][~validation],
        )
        self.assertEqual(
            classifier.kwargs["valid_sets"][0].num_data(), validation.sum()
        )
        self.assertEqual(regressor.kwargs["params"]["objective"], "regression")
        self.assertEqual(
            regressor.kwargs["train_set"].num_data(),
            (known & ~validation).sum(),
        )
        self.assertIsNone(regressor.kwargs["train_set"].get_weight())
        self.assertEqual(
            regressor.kwargs["valid_sets"][0].num_data(),
            (known & validation).sum(),
        )

    def test_models_stop_at_the_best_iteration(self):
        # Targets that do not depend on the features stop early.
        rows = _synthetic()
        count = len(rows["felt"])
        rng = np.random.default_rng(2)
        rows["felt"] = rng.random(count) < 0.5
        rows["observed"] = np.where(
            rows["felt"], rng.normal(size=count), np.nan
        ).astype(np.float32)
        model = _fit(rows, max_rounds=1000)
        sample = rows["features"][:100]

        probability, regression = model.predict(sample)

        for booster, predicted in (
            (model.classifier, probability),
            (model.regressor, regression),
        ):
            self.assertGreater(booster.best_iteration, 0)
            self.assertEqual(booster.num_trees(), booster.best_iteration)
            np.testing.assert_array_equal(predicted, booster.predict(sample))
            self.assertEqual(predicted.dtype, np.float64)
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))
        self.assertEqual(
            [len(values) for values in model.predict(sample[:0])], [0, 0]
        )

    def test_fit_needs_rows_on_both_sides(self):
        rows = _synthetic()
        validation = hurdle.validation_quakes(rows["row_quake"])
        observed = rows["observed"]
        one_quake = {**rows, "row_quake": np.zeros_like(rows["row_quake"])}
        no_observation = {**rows, "observed": np.full_like(observed, np.nan)}
        observed_in_validation = {
            **rows, "observed": np.where(validation, observed, np.nan)
        }
        observed_in_training = {
            **rows, "observed": np.where(validation, np.nan, observed)
        }
        for changed, message in (
            (one_quake, "classifier has no training"),
            (no_observation, "regressor has no training"),
            (observed_in_validation, "regressor has no training"),
            (observed_in_training, "regressor has no validation"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _fit(changed)

    def test_parameters(self):
        self.assertEqual(
            dict(hurdle.PARAMETERS),
            {
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "monotone_constraints_method": "advanced",
                "seed": 0,
                "deterministic": True,
                "force_row_wise": True,
                "num_threads": 16,
                "verbose": -1,
            },
        )
        self.assertEqual(
            (
                hurdle.MAX_ROUNDS,
                hurdle.EARLY_STOPPING_ROUNDS,
                hurdle.VALIDATION_FRACTION,
            ),
            (3000, 100, 0.1),
        )


if __name__ == "__main__":
    unittest.main()

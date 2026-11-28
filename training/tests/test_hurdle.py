# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests the hurdle model of felt probability and intensity."""

import unittest
from unittest import mock

import lightgbm
import numpy as np

from training import evaluate
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


def _fit(rows, names=features.FEATURE_NAMES, **settings):
    return hurdle.fit(
        feature_names=names,
        settings=hurdle.Settings(**{"max_rounds": _ROUNDS, **settings}),
        **rows,
    )


def _largest_leaf(booster):
    """Returns the largest leaf magnitude of the trees after the first.

    The first tree is skipped because it also holds the initial score.
    """
    trees = booster.dump_model()["tree_info"][1:]
    return max(
        abs(value)
        for tree in trees
        for value in _leaf_values(tree["tree_structure"])
    )


def _leaf_values(node):
    """Yields the leaf values under a node of dump_model()."""
    if "leaf_value" in node:
        yield node["leaf_value"]
        return
    for child in ("left_child", "right_child"):
        yield from _leaf_values(node[child])


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

    def test_max_delta_step_bounds_the_classifier(self):
        rows = _synthetic()
        bound = 0.05 * 1.0 + 1e-12

        free = _fit(rows)
        capped = _fit(rows, max_delta_step=1.0)

        self.assertGreater(_largest_leaf(free.classifier), bound)
        self.assertLessEqual(_largest_leaf(capped.classifier), bound)
        self.assertEqual(capped.classifier.params["max_delta_step"], 1.0)
        self.assertNotIn("max_delta_step", capped.regressor.params)

    def test_weighted_regression_changes_only_the_regressor(self):
        rows = _synthetic()
        validation = hurdle.validation_quakes(rows["row_quake"])
        known = ~np.isnan(rows["observed"])
        plain = _fit(rows)

        with mock.patch.object(
            lightgbm, "train", wraps=lightgbm.train
        ) as train:
            weighted = _fit(rows, weighted_regression=True)

        self.assertEqual(
            plain.classifier.model_to_string(),
            weighted.classifier.model_to_string(),
        )
        self.assertNotEqual(
            plain.regressor.model_to_string(),
            weighted.regressor.model_to_string(),
        )
        regressor = train.call_args_list[1].kwargs
        for dataset, chosen in (
            (regressor["train_set"], known & ~validation),
            (regressor["valid_sets"][0], known & validation),
        ):
            np.testing.assert_array_equal(
                dataset.get_weight(),
                hurdle.class_weight(rows["observed"][chosen]),
            )

    def test_settings_reach_lightgbm(self):
        model = _fit(_synthetic(), learning_rate=0.1, max_rounds=7)

        for booster in (model.classifier, model.regressor):
            self.assertEqual(booster.params["learning_rate"], 0.1)
            self.assertLessEqual(booster.num_trees(), 7)

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


class ClassWeightTest(unittest.TestCase):

    def test_class_weight(self):
        observed = np.array([0.5, 1.4, 1.5, 4.5, 5.0, 6.6, 0.4], np.float32)

        weight = hurdle.class_weight(observed)

        np.testing.assert_array_equal(weight, [2, 2, 3, 6, 7, 10, 1])
        self.assertEqual(weight.dtype, np.float64)
        with self.assertRaisesRegex(ValueError, "NaN"):
            hurdle.class_weight(np.array([1.0, np.nan], np.float32))

    def test_class_weight_at_every_edge(self):
        edges = np.array(evaluate.CLASS_EDGES, np.float32)
        below = np.nextafter(edges, np.float32(0))

        np.testing.assert_array_equal(
            hurdle.class_weight(edges), np.arange(2, 11)
        )
        np.testing.assert_array_equal(
            hurdle.class_weight(below), np.arange(1, 10)
        )


class RefitTest(unittest.TestCase):

    def test_refit_trains_the_given_rounds(self):
        rows = _synthetic()
        known = ~np.isnan(rows["observed"])
        refit_rows = {
            name: rows[name]
            for name in ("features", "felt", "observed", "weight")
        }

        with mock.patch.object(
            lightgbm, "train", wraps=lightgbm.train
        ) as train:
            model = hurdle.refit(
                feature_names=features.FEATURE_NAMES,
                settings=hurdle.Settings(weighted_regression=True),
                rounds=hurdle.Rounds(classifier=13, regressor=11),
                **refit_rows,
            )

        self.assertEqual(model.classifier.num_trees(), 13)
        self.assertEqual(model.regressor.num_trees(), 11)
        classifier, regressor = train.call_args_list
        for call, rounds in ((classifier, 13), (regressor, 11)):
            self.assertEqual(call.kwargs["num_boost_round"], rounds)
            self.assertNotIn("valid_sets", call.kwargs)
            self.assertNotIn("callbacks", call.kwargs)
        self.assertEqual(
            classifier.kwargs["train_set"].num_data(), len(rows["felt"])
        )
        np.testing.assert_array_equal(
            classifier.kwargs["train_set"].get_weight(), rows["weight"]
        )
        regression_set = regressor.kwargs["train_set"]
        np.testing.assert_array_equal(
            regression_set.get_label(), rows["observed"][known]
        )
        np.testing.assert_array_equal(
            regression_set.get_weight(),
            hurdle.class_weight(rows["observed"][known]),
        )

    def test_refit_rounds(self):
        for best, expected in (
            ((2999, 2308), hurdle.Rounds(3299, 2539)),
            ((1, 1), hurdle.Rounds(1, 1)),
        ):
            with self.subTest(best=best):
                model = hurdle.Hurdle(
                    classifier=mock.Mock(best_iteration=best[0]),
                    regressor=mock.Mock(best_iteration=best[1]),
                )

                self.assertEqual(hurdle.refit_rounds(model), expected)


class SettingsTest(unittest.TestCase):

    def test_parameters_record_the_settings(self):
        self.assertEqual(
            hurdle.Settings().parameters(),
            {
                **hurdle.PARAMETERS,
                "learning_rate": 0.05,
                "max_rounds": 3000,
                "max_delta_step": 0.0,
                "weighted_regression": False,
            },
        )


if __name__ == "__main__":
    unittest.main()

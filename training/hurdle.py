# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Trains the hurdle model: a felt classifier and an intensity regressor.

The classifier learns from every row with its weight, so that sampled unfelt
rows stand for the rows they replace. The regressor learns the instrumental
intensity from the felt rows that have one, without weights unless the
settings weigh each row by its intensity class. fit stops both early on the
same tenth of the quakes, held out whole; refit trains on every row for given
rounds. Both models are monotone in the magnitude and the two distances.

Only the classifier takes max_delta_step: on a group of outlying rows whose
probability sits at 0 or 1, the Newton step of the log loss grows without
bound, and a positive max_delta_step caps it. The class weight of an
intensity is the number of its JMA intensity class counted from 1 below
intensity 1, so that the few strong rows weigh more.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math

import lightgbm
import numpy as np
import numpy.typing as npt

from training import evaluate

PARAMETERS: Mapping[str, object] = {
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
}
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
VALIDATION_FRACTION = 0.1
MONOTONE: Mapping[str, int] = {"mj": 1, "log_hypo": -1, "epi": -1}
REFIT_FACTOR = 1.1


@dataclasses.dataclass(frozen=True)
class Settings:
    """The training settings that are chosen rather than fixed.

    Attributes:
        learning_rate: The learning rate of both models.
        max_rounds: The most boosting rounds of each model in fit.
        max_delta_step: The largest leaf output of the classifier before
            the learning rate, 0 for no limit.
        weighted_regression: Whether the regressor weighs each row by
            class_weight.
    """

    learning_rate: float = 0.05
    max_rounds: int = MAX_ROUNDS
    max_delta_step: float = 0.0
    weighted_regression: bool = False

    def parameters(self) -> dict[str, object]:
        """Returns PARAMETERS and the settings, as a training record holds."""
        return {**PARAMETERS, **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class Rounds:
    """The boosting rounds of each model in refit.

    Attributes:
        classifier: The rounds of the classifier.
        regressor: The rounds of the regressor.
    """

    classifier: int
    regressor: int


@dataclasses.dataclass(frozen=True)
class Hurdle:
    """A trained felt classifier and intensity regressor.

    Each model from fit holds the trees up to its best iteration on the
    validation quakes, as lightgbm.train drops the later ones; each model
    from refit holds every round it was given.

    Attributes:
        classifier: The model of the probability that a station is felt.
        regressor: The model of the instrumental intensity.
    """

    classifier: lightgbm.Booster
    regressor: lightgbm.Booster

    def predict(
        self, features: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predicts rows.

        Args:
            features: The features of each row, in the order of training.

        Returns:
            A tuple (probability, regression) of the felt probability and the
            regression value of each row.
        """
        return (
            _predict(self.classifier, features),
            _predict(self.regressor, features),
        )


def class_weight(
    observed: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    """Returns the weight of each instrumental intensity by its class.

    Args:
        observed: Instrumental intensities.

    Returns:
        1 below evaluate.CLASS_EDGES[0], and one more for each edge at or
        below the intensity: 2 for intensity 1 up to 10 for intensity 7.

    Raises:
        ValueError: An intensity is NaN.
    """
    if np.isnan(observed).any():
        raise ValueError("an instrumental intensity is NaN")
    classes = np.digitize(observed, evaluate.CLASS_EDGES)
    return (classes + 1).astype(np.float64)


def validation_quakes(
    row_quake: npt.NDArray[np.int32], *, seed: int = 0
) -> npt.NDArray[np.bool_]:
    """Chooses whole quakes for early stopping.

    Args:
        row_quake: The quake of each row.
        seed: The seed of the choice.

    Returns:
        Whether each row belongs to one of VALIDATION_FRACTION of the quakes,
        rounded down but at least one, chosen at random.

    Raises:
        ValueError: There are no rows.
    """
    quakes = np.unique(row_quake)
    if quakes.size == 0:
        raise ValueError("there are no quakes to choose from")
    count = max(1, math.floor(len(quakes) * VALIDATION_FRACTION))
    chosen = np.random.default_rng(seed).choice(quakes, count, replace=False)
    return np.isin(row_quake, chosen)


def fit(
    features: npt.NDArray[np.float32],
    feature_names: Sequence[str],
    row_quake: npt.NDArray[np.int32],
    felt: npt.NDArray[np.bool_],
    observed: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    *,
    settings: Settings = Settings(),
) -> Hurdle:
    """Trains the classifier and the regressor, stopping each early.

    Args:
        features: The features of each row.
        feature_names: The name of each column of features, which places the
            monotone constraints.
        row_quake: The quake of each row.
        felt: Whether each station recorded the quake.
        observed: The instrumental intensity of each row, NaN when it is not
            known.
        weight: The weight of each row.
        settings: The training settings.

    Returns:
        The trained models.

    Raises:
        ValueError: There are no rows, or the validation quakes leave the
            classifier or the regressor without training or validation rows.
    """
    validation = validation_quakes(row_quake)
    known = ~np.isnan(observed)
    for model, part, chosen in (
        ("classifier", "training", ~validation),
        ("classifier", "validation", validation),
        ("regressor", "training", known & ~validation),
        ("regressor", "validation", known & validation),
    ):
        if not chosen.any():
            raise ValueError(f"the {model} has no {part} rows")
    classifier, regressor = _models(
        feature_names, felt, observed, weight, settings
    )
    return Hurdle(
        classifier=classifier.train(
            features, ~validation, validation, settings.max_rounds
        ),
        regressor=regressor.train(
            features,
            known & ~validation,
            known & validation,
            settings.max_rounds,
        ),
    )


def refit(
    features: npt.NDArray[np.float32],
    feature_names: Sequence[str],
    felt: npt.NDArray[np.bool_],
    observed: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    *,
    settings: Settings,
    rounds: Rounds,
) -> Hurdle:
    """Trains the classifier on every row and the regressor on every known one.

    Nothing is held out and nothing stops early: each model trains for its
    rounds.

    Args:
        features: The features of each row.
        feature_names: The name of each column of features.
        felt: Whether each station recorded the quake.
        observed: The instrumental intensity of each row, NaN when it is not
            known.
        weight: The weight of each row.
        settings: The training settings; max_rounds is not used.
        rounds: The rounds of each model.

    Returns:
        The trained models.
    """
    classifier, regressor = _models(
        feature_names, felt, observed, weight, settings
    )
    every = np.ones(len(felt), dtype=np.bool_)
    return Hurdle(
        classifier=classifier.train(features, every, None, rounds.classifier),
        regressor=regressor.train(
            features, ~np.isnan(observed), None, rounds.regressor
        ),
    )


def refit_rounds(model: Hurdle) -> Rounds:
    """Returns REFIT_FACTOR times the rounds each model of fit kept, rounded.

    Args:
        model: The models from fit.
    """
    return Rounds(
        classifier=round(REFIT_FACTOR * model.classifier.best_iteration),
        regressor=round(REFIT_FACTOR * model.regressor.best_iteration),
    )


@dataclasses.dataclass(frozen=True)
class _Model:
    """How to train one model.

    Attributes:
        parameters: The LightGBM parameters.
        names: The name of each column of features.
        label: The label of each row.
        weight: The weight of each row, None for no weights.
    """

    parameters: dict[str, object]
    names: list[str]
    label: npt.NDArray[np.float64]
    weight: npt.NDArray[np.float64] | None

    def train(
        self,
        features: npt.NDArray[np.float32],
        train: npt.NDArray[np.bool_],
        valid: npt.NDArray[np.bool_] | None,
        rounds: int,
    ) -> lightgbm.Booster:
        """Trains on the train rows, stopping early on the valid ones if any."""
        train_set = lightgbm.Dataset(
            features[train],
            label=self.label[train],
            weight=None if self.weight is None else self.weight[train],
            feature_name=self.names,
        )
        if valid is None:
            return lightgbm.train(
                params=self.parameters,
                train_set=train_set,
                num_boost_round=rounds,
            )
        valid_set = lightgbm.Dataset(
            features[valid],
            label=self.label[valid],
            weight=None if self.weight is None else self.weight[valid],
            reference=train_set,
        )
        return lightgbm.train(
            params=self.parameters,
            train_set=train_set,
            num_boost_round=rounds,
            valid_sets=[valid_set],
            callbacks=[
                lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
            ],
        )


def _models(
    feature_names: Sequence[str],
    felt: npt.NDArray[np.bool_],
    observed: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    settings: Settings,
) -> tuple[_Model, _Model]:
    """Returns how to train the classifier and the regressor."""
    names = list(feature_names)
    common = {
        **PARAMETERS,
        "learning_rate": settings.learning_rate,
        "monotone_constraints": [MONOTONE.get(name, 0) for name in names],
    }
    regression_weight = None
    if settings.weighted_regression:
        known = ~np.isnan(observed)
        regression_weight = np.zeros(len(observed))
        regression_weight[known] = class_weight(observed[known])
    classifier = _Model(
        parameters={
            **common,
            "objective": "binary",
            "max_delta_step": settings.max_delta_step,
        },
        names=names,
        label=felt.astype(np.float64),
        weight=weight.astype(np.float64),
    )
    regressor = _Model(
        parameters={**common, "objective": "regression"},
        names=names,
        label=observed.astype(np.float64),
        weight=regression_weight,
    )
    return classifier, regressor


def _predict(
    booster: lightgbm.Booster, features: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    return np.asarray(booster.predict(features), dtype=np.float64)

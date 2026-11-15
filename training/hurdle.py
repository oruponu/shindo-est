# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Trains the hurdle model: a felt classifier and an intensity regressor.

The classifier learns from every row with its weight, so that sampled unfelt
rows stand for the rows they replace. The regressor learns the instrumental
intensity from the felt rows that have one, without weights. Both stop early
on the same tenth of the quakes, held out whole, and are monotone in the
magnitude and the two distances.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math

import lightgbm
import numpy as np
import numpy.typing as npt

PARAMETERS: Mapping[str, object] = {
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
}
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
VALIDATION_FRACTION = 0.1
MONOTONE: Mapping[str, int] = {"mj": 1, "log_hypo": -1, "epi": -1}


@dataclasses.dataclass(frozen=True)
class Hurdle:
    """A trained felt classifier and intensity regressor.

    Each model holds the trees up to its best iteration on the validation
    quakes; lightgbm.train drops the later ones.

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
    max_rounds: int = MAX_ROUNDS,
) -> Hurdle:
    """Trains the classifier and the regressor.

    Args:
        features: The features of each row.
        feature_names: The name of each column of features, which places the
            monotone constraints.
        row_quake: The quake of each row.
        felt: Whether each station recorded the quake.
        observed: The instrumental intensity of each row, NaN when it is not
            known.
        weight: The weight of each row.
        max_rounds: The most boosting rounds of each model.

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
    names = list(feature_names)
    parameters = {
        **PARAMETERS,
        "monotone_constraints": [MONOTONE.get(name, 0) for name in names],
    }
    classifier = _train(
        {**parameters, "objective": "binary"},
        names,
        features,
        label=felt.astype(np.float64),
        weight=weight,
        train=~validation,
        valid=validation,
        max_rounds=max_rounds,
    )
    regressor = _train(
        {**parameters, "objective": "regression"},
        names,
        features,
        label=observed.astype(np.float64),
        weight=None,
        train=known & ~validation,
        valid=known & validation,
        max_rounds=max_rounds,
    )
    return Hurdle(classifier=classifier, regressor=regressor)


def _train(
    parameters: dict[str, object],
    names: list[str],
    features: npt.NDArray[np.float32],
    *,
    label: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float32] | None,
    train: npt.NDArray[np.bool_],
    valid: npt.NDArray[np.bool_],
    max_rounds: int,
) -> lightgbm.Booster:
    """Trains one model on the train rows, stopping early on the valid ones."""
    train_set = lightgbm.Dataset(
        features[train],
        label=label[train],
        weight=None if weight is None else weight[train],
        feature_name=names,
    )
    valid_set = lightgbm.Dataset(
        features[valid],
        label=label[valid],
        weight=None if weight is None else weight[valid],
        reference=train_set,
    )
    return lightgbm.train(
        params=parameters,
        train_set=train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        callbacks=[
            lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    )


def _predict(
    booster: lightgbm.Booster, features: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    return np.asarray(booster.predict(features), dtype=np.float64)

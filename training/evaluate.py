# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Measures predicted intensities against the observed ones.

Every measure uses the displayed intensity, which is the regression value
passed through the hurdle: a station whose felt probability reaches the
threshold shows at least the felt floor, and any other at most the unfelt
ceiling. Counts of rows are not weighted; counts of predictions and the
contingency table are weighted, so that sampled unfelt rows stand for the
rows they replace. A measure without any row to measure is None.
"""

import dataclasses
import math

import numpy as np
import numpy.typing as npt

CLASS_EDGES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5)
CALIBRATION_BINS = 10

_FELT = 0.5
_WEAK = 2.5
_STRONG4 = 3.5
_STRONG5 = 4.5

type Measures = dict[str, float | None]


@dataclasses.dataclass(frozen=True)
class HurdleValues:
    """The values that turn a probability and a regression into a display.

    Attributes:
        felt_probability: The probability from which a station is shown as
            felt.
        felt_floor: The lowest intensity shown at a felt station.
        unfelt_ceiling: The highest intensity shown at any other station.
    """

    felt_probability: float = 0.35
    felt_floor: float = 0.5
    unfelt_ceiling: float = 0.49


PREVIOUS = HurdleValues()


@dataclasses.dataclass(frozen=True)
class CalibrationBin:
    """The rows whose felt probability falls in [low, high).

    The last bin also holds a probability of exactly 1.

    Attributes:
        low: The lowest probability in the bin.
        high: The probability above the bin.
        weight: The total weight of the rows.
        probability: The weighted mean probability, None without rows.
        felt: The weighted share of felt rows, None without rows.
    """

    low: float
    high: float
    weight: float
    probability: float | None
    felt: float | None


def display(
    probability: npt.NDArray[np.float64],
    regression: npt.NDArray[np.float64],
    values: HurdleValues,
) -> npt.NDArray[np.float64]:
    """Returns the displayed intensity of each row.

    Args:
        probability: The felt probability of each row.
        regression: The regression value of each row.
        values: The hurdle values.
    """
    shown: npt.NDArray[np.float64] = np.where(
        probability >= values.felt_probability,
        np.maximum(regression, values.felt_floor),
        np.minimum(regression, values.unfelt_ceiling),
    )
    return shown


def summarize(
    row_quake: npt.NDArray[np.int32],
    felt: npt.NDArray[np.bool_],
    observed: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    probability: npt.NDArray[np.float64],
    regression: npt.NDArray[np.float64],
    values: HurdleValues = PREVIOUS,
) -> Measures:
    """Measures the displayed intensity of some rows.

    Args:
        row_quake: The quake of each row.
        felt: Whether each station recorded the quake.
        observed: The instrumental intensity of each row, NaN when it is not
            known.
        weight: The weight of each row.
        probability: The felt probability of each row.
        regression: The regression value of each row.
        values: The hurdle values.

    Returns:
        The measures by name. The errors are displayed minus observed, over
        the felt rows with an instrumental intensity: all of them (felt_*),
        those observed below 2.5 (weak_*), 3.5 or more (strong4_*), and 4.5
        or more (strong5_*). A class is over or under when the JMA class of
        the displayed intensity is above or below the observed one, and
        strong5_capture is the share displayed at 4.5 or more. predicted5 is
        the weight of the rows displayed at 4.5 or more and predicted5_wrong
        the part of it that was unfelt or observed below 4.5;
        predicted4_observed2 is the weight of the rows displayed at 3.5 or
        more that were unfelt or observed below 2.5. ets, pod, and far come
        from the weighted table of felt rows against rows displayed at 0.5
        or more, and calibration_total is the weighted sum of the
        probabilities over the weighted number of felt rows.
    """
    shown = display(probability, regression, values)
    known = felt & ~np.isnan(observed)
    error = shown - observed
    weights = weight.astype(np.float64)
    weak = known & (observed < _WEAK)
    strong4 = known & (observed >= _STRONG4)
    strong5 = known & (observed >= _STRONG5)
    predicted5 = shown >= _STRONG5
    predicted_felt = shown >= _FELT
    measures: Measures = {
        "quakes": len(np.unique(row_quake)),
        "felt_rows": int(known.sum()),
        "felt_bias": _mean(error[known]),
        "felt_rmse": _rmse(error[known]),
        "weak_rows": int(weak.sum()),
        "weak_bias": _mean(error[weak]),
        "weak_over_class": _share(
            _classes(shown[weak]) > _classes(observed[weak])
        ),
        "strong4_rows": int(strong4.sum()),
        "strong4_bias": _mean(error[strong4]),
        "strong4_rmse": _rmse(error[strong4]),
        "strong4_under_class": _share(
            _classes(shown[strong4]) < _classes(observed[strong4])
        ),
        "strong5_rows": int(strong5.sum()),
        "strong5_bias": _mean(error[strong5]),
        "strong5_capture": _share(shown[strong5] >= _STRONG5),
        "predicted5": float(weights[predicted5].sum()),
        "predicted5_wrong": float(
            weights[predicted5 & (~felt | (observed < _STRONG5))].sum()
        ),
        "predicted4_observed2": float(
            weights[(shown >= _STRONG4) & (~felt | (observed < _WEAK))].sum()
        ),
    }
    measures.update(_contingency(felt, predicted_felt, weights))
    measures["calibration_total"] = _ratio(
        float((weights * probability).sum()), float(weights[felt].sum())
    )
    return measures


def calibration(
    felt: npt.NDArray[np.bool_],
    weight: npt.NDArray[np.float32],
    probability: npt.NDArray[np.float64],
) -> list[CalibrationBin]:
    """Compares the felt probability with the share of felt rows.

    Args:
        felt: Whether each station recorded the quake.
        weight: The weight of each row.
        probability: The felt probability of each row.

    Returns:
        CALIBRATION_BINS bins of equal width from 0 to 1.
    """
    edges = np.arange(1, CALIBRATION_BINS) / CALIBRATION_BINS
    index = np.digitize(probability, edges)
    weights = weight.astype(np.float64)
    totals = np.bincount(index, weights, CALIBRATION_BINS)
    probabilities = np.bincount(index, weights * probability, CALIBRATION_BINS)
    felts = np.bincount(index, weights * felt, CALIBRATION_BINS)
    return [
        CalibrationBin(
            low=at / CALIBRATION_BINS,
            high=(at + 1) / CALIBRATION_BINS,
            weight=float(totals[at]),
            probability=_ratio(float(probabilities[at]), float(totals[at])),
            felt=_ratio(float(felts[at]), float(totals[at])),
        )
        for at in range(CALIBRATION_BINS)
    ]


def max_per_quake(
    candidate_quake: npt.NDArray[np.int32],
    candidate_display: npt.NDArray[np.float64],
    observed_quake: npt.NDArray[np.int32],
    observed: npt.NDArray[np.float32],
) -> Measures:
    """Measures the largest displayed intensity of each quake.

    The largest prediction has to come from every candidate station, not
    only from the rows kept by sampling, and weights cannot restore a
    maximum, so the candidates are passed apart from the observations.

    Args:
        candidate_quake: The quake of each candidate station.
        candidate_display: The displayed intensity of each candidate.
        observed_quake: The quake of each observation.
        observed: The instrumental intensity of each observation; NaN is
            left out.

    Returns:
        max_quakes, the quakes with both a candidate and an observation, and
        over them the bias and RMSE of the largest displayed minus the
        largest observed intensity (max_bias, max_rmse) and the share whose
        largest displayed intensity is a class under (max_under_class).
    """
    known = ~np.isnan(observed)
    count = 1 + max(
        int(candidate_quake.max(initial=-1)),
        int(observed_quake.max(initial=-1)),
    )
    predicted = np.full(count, -np.inf)
    np.maximum.at(predicted, candidate_quake, candidate_display)
    actual = np.full(count, -np.inf)
    np.maximum.at(actual, observed_quake[known], observed[known])
    both = np.isfinite(predicted) & np.isfinite(actual)
    error = predicted[both] - actual[both]
    return {
        "max_quakes": int(both.sum()),
        "max_bias": _mean(error),
        "max_rmse": _rmse(error),
        "max_under_class": _share(
            _classes(predicted[both]) < _classes(actual[both])
        ),
    }


def bootstrap_rmse_difference(
    row_quake: npt.NDArray[np.int32],
    error_a: npt.NDArray[np.float64],
    error_b: npt.NDArray[np.float64],
    *,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compares the RMSE of two predictions of the same rows.

    The quakes are resampled with replacement, keeping all rows of a quake
    together, since the errors of one quake are not independent.

    Args:
        row_quake: The quake of each row.
        error_a: The error of the first prediction in each row.
        error_b: The error of the second prediction in each row.
        samples: The number of resamples.
        seed: The seed of the resampling.

    Returns:
        A tuple (difference, low, high) of RMSE(a) - RMSE(b) over all rows
        and its 2.5th and 97.5th percentiles over the resamples.

    Raises:
        ValueError: The arrays have different lengths or no rows.
    """
    lengths = {len(row_quake), len(error_a), len(error_b)}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("the arrays must have the same length above 0")
    _, inverse = np.unique(row_quake, return_inverse=True)
    rows = np.bincount(inverse).astype(np.float64)
    squares_a = np.bincount(inverse, error_a**2)
    squares_b = np.bincount(inverse, error_b**2)
    draws = np.random.default_rng(seed).integers(
        0, len(rows), size=(samples, len(rows))
    )
    drawn = rows[draws].sum(axis=1)
    resampled_a = np.sqrt(squares_a[draws].sum(axis=1) / drawn)
    resampled_b = np.sqrt(squares_b[draws].sum(axis=1) / drawn)
    low, high = np.percentile(resampled_a - resampled_b, [2.5, 97.5])
    rmse_a = math.sqrt(squares_a.sum() / rows.sum())
    rmse_b = math.sqrt(squares_b.sum() / rows.sum())
    return rmse_a - rmse_b, float(low), float(high)


def _classes(values: npt.NDArray[np.floating]) -> npt.NDArray[np.intp]:
    """Returns the JMA class of each intensity, 0 below 0.5."""
    classes: npt.NDArray[np.intp] = np.digitize(values, CLASS_EDGES)
    return classes


def _contingency(
    felt: npt.NDArray[np.bool_],
    predicted: npt.NDArray[np.bool_],
    weights: npt.NDArray[np.float64],
) -> Measures:
    """Returns the ETS, POD, and FAR of a weighted 2 x 2 table."""
    hits = float(weights[felt & predicted].sum())
    false_alarms = float(weights[~felt & predicted].sum())
    misses = float(weights[felt & ~predicted].sum())
    total = float(weights.sum())
    chance = (hits + false_alarms) * (hits + misses) / total if total else 0.0
    return {
        "ets": _ratio(hits - chance, hits + false_alarms + misses - chance),
        "pod": _ratio(hits, hits + misses),
        "far": _ratio(false_alarms, hits + false_alarms),
    }


def _mean(values: npt.NDArray[np.floating]) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _rmse(values: npt.NDArray[np.floating]) -> float | None:
    return None if values.size == 0 else math.sqrt(float(np.mean(values**2)))


def _share(flags: npt.NDArray[np.bool_]) -> float | None:
    return None if flags.size == 0 else float(flags.mean())


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None

//! The hurdle rule that combines a felt classifier with an intensity regressor.

/// Settings of the hurdle rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HurdleParams {
    /// Probability at or above which a site counts as felt.
    pub felt_probability: f64,
    /// Lowest instrumental intensity of a felt site.
    pub felt_floor: f64,
    /// Highest instrumental intensity of an unfelt site.
    pub unfelt_ceiling: f64,
    /// Amount the classifier's logit rises for each unit of offset, as
    /// lowering the felt boundary by one intensity unit makes more sites felt.
    pub logit_per_intensity: f64,
}

impl HurdleParams {
    /// Returns whether all four values are finite, `felt_probability` lies
    /// strictly between 0 and 1, `unfelt_ceiling` is below `felt_floor`, and
    /// `logit_per_intensity` is not negative.
    ///
    /// [`combine`] does not check this; callers check the settings once when
    /// they load them.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let finite = self.felt_probability.is_finite()
            && self.felt_floor.is_finite()
            && self.unfelt_ceiling.is_finite()
            && self.logit_per_intensity.is_finite();
        finite
            && 0.0 < self.felt_probability
            && self.felt_probability < 1.0
            && self.unfelt_ceiling < self.felt_floor
            && self.logit_per_intensity >= 0.0
    }
}

/// The outcome of the hurdle rule for one site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hurdle {
    /// The instrumental intensity: at least `felt_floor` if felt, at most
    /// `unfelt_ceiling` if not, or NaN if an input is NaN or infinite inputs
    /// combine to NaN.
    pub instrumental: f64,
    /// The probability that the site is felt, with the offset applied. It is
    /// the value compared with `felt_probability`.
    pub probability: f64,
}

/// Returns the logistic function of `x`, `1 / (1 + e^-x)`.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-x))
}

/// Combines the raw scores of the felt classifier and the intensity regressor
/// for one site.
///
/// `offset` is added to the regressor's intensity, and `logit_per_intensity`
/// times `offset` to the classifier's logit. If the resulting probability is
/// at least `felt_probability`, the site is felt and its intensity is raised to
/// at least `felt_floor`; otherwise it is lowered to at most `unfelt_ceiling`.
/// If a raw score or `offset` is NaN, or infinite inputs combine to NaN (such
/// as inf - inf), the intensity is NaN rather than a bound.
#[must_use]
pub fn combine(
    classifier_raw: f64,
    regressor_raw: f64,
    offset: f64,
    params: &HurdleParams,
) -> Hurdle {
    let intensity = regressor_raw + offset;
    let probability = sigmoid(classifier_raw + params.logit_per_intensity * offset);
    let instrumental = if intensity.is_nan() || probability.is_nan() {
        f64::NAN
    } else if probability >= params.felt_probability {
        intensity.max(params.felt_floor)
    } else {
        intensity.min(params.unfelt_ceiling)
    };
    Hurdle {
        instrumental,
        probability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: HurdleParams = HurdleParams {
        felt_probability: 0.35,
        felt_floor: 0.5,
        unfelt_ceiling: 0.49,
        logit_per_intensity: 1.5,
    };

    #[test]
    fn felt_raises_to_the_floor() {
        assert_eq!(super::combine(10.0, 0.2, 0.0, &PARAMS).instrumental, 0.5);
    }

    #[test]
    fn felt_keeps_a_larger_value() {
        assert_eq!(super::combine(10.0, 3.2, 0.0, &PARAMS).instrumental, 3.2);
    }

    #[test]
    fn unfelt_caps_at_the_ceiling() {
        assert_eq!(super::combine(-10.0, 2.0, 0.0, &PARAMS).instrumental, 0.49);
    }

    #[test]
    fn unfelt_keeps_a_smaller_value() {
        assert_eq!(super::combine(-10.0, -0.3, 0.0, &PARAMS).instrumental, -0.3);
    }

    #[test]
    fn probability_equal_to_the_threshold_is_felt() {
        let params = HurdleParams {
            felt_probability: 0.5,
            ..PARAMS
        };
        let hurdle = super::combine(0.0, 0.2, 0.0, &params);
        assert_eq!(hurdle.probability, 0.5);
        assert_eq!(hurdle.instrumental, 0.5);
    }

    #[test]
    fn offset_shifts_both_parts() {
        let without = super::combine(-1.0, 1.0, 0.0, &PARAMS);
        assert_eq!(without.probability, super::sigmoid(-1.0));
        assert_eq!(without.instrumental, 0.49);

        // The logit rises by 1.5 x 0.5, so the probability passes 0.35.
        let with = super::combine(-1.0, 1.0, 0.5, &PARAMS);
        assert_eq!(with.probability, super::sigmoid(-0.25));
        assert_eq!(with.instrumental, 1.5);
    }

    #[test]
    fn offset_lowers_an_unfelt_value() {
        // The logit falls to -10.75, so the site stays unfelt.
        let hurdle = super::combine(-10.0, -0.25, -0.5, &PARAMS);
        assert_eq!(hurdle.instrumental, -0.75);
    }

    #[test]
    fn infinite_raw_scores_follow_the_rule() {
        let cases = [
            (f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0),
            (f64::INFINITY, f64::NEG_INFINITY, 0.5, 1.0),
            (f64::NEG_INFINITY, f64::INFINITY, 0.49, 0.0),
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0),
        ];
        for (classifier, regressor, instrumental, probability) in cases {
            assert_eq!(
                super::combine(classifier, regressor, 0.0, &PARAMS),
                Hurdle {
                    instrumental,
                    probability
                },
                "{classifier}, {regressor}"
            );
        }
    }

    #[test]
    fn nan_inputs_and_nan_results_give_nan() {
        for (classifier, regressor) in [(f64::NAN, 1.0), (10.0, f64::NAN), (-10.0, f64::NAN)] {
            let hurdle = super::combine(classifier, regressor, 0.0, &PARAMS);
            assert!(
                hurdle.instrumental.is_nan(),
                "{classifier}, {regressor}: {hurdle:?}"
            );
        }
        let nan_offset = super::combine(10.0, 1.0, f64::NAN, &PARAMS);
        assert!(nan_offset.instrumental.is_nan(), "{nan_offset:?}");
        assert!(nan_offset.probability.is_nan(), "{nan_offset:?}");

        // Each input is a number, but inf - inf in the intensity or in the
        // logit, or 0 x inf in the logit, is NaN.
        let flat = HurdleParams {
            logit_per_intensity: 0.0,
            ..PARAMS
        };
        let cases = [
            (10.0, f64::INFINITY, f64::NEG_INFINITY, PARAMS),
            (f64::NEG_INFINITY, 1.0, f64::MAX, PARAMS),
            (1.0, 1.0, f64::INFINITY, flat),
        ];
        for (classifier, regressor, offset, params) in cases {
            let hurdle = super::combine(classifier, regressor, offset, &params);
            assert!(
                hurdle.instrumental.is_nan(),
                "{classifier}, {regressor}, {offset}: {hurdle:?}"
            );
        }
    }

    #[test]
    fn sigmoid_of_zero_is_half() {
        assert_eq!(super::sigmoid(0.0), 0.5);
    }

    #[test]
    fn sigmoid_saturates_without_nan() {
        assert_eq!(super::sigmoid(1000.0), 1.0);
        assert_eq!(super::sigmoid(-1000.0), 0.0);
    }

    #[test]
    fn validates_params() {
        assert!(PARAMS.is_valid(), "{PARAMS:?}");
        let flat = HurdleParams {
            logit_per_intensity: 0.0,
            ..PARAMS
        };
        assert!(flat.is_valid(), "{flat:?}");

        let invalid = [
            HurdleParams {
                felt_probability: f64::NAN,
                ..PARAMS
            },
            HurdleParams {
                felt_probability: 0.0,
                ..PARAMS
            },
            HurdleParams {
                felt_probability: 1.0,
                ..PARAMS
            },
            HurdleParams {
                felt_floor: f64::INFINITY,
                ..PARAMS
            },
            HurdleParams {
                unfelt_ceiling: f64::NEG_INFINITY,
                ..PARAMS
            },
            HurdleParams {
                unfelt_ceiling: 0.5,
                ..PARAMS
            },
            HurdleParams {
                logit_per_intensity: -1.0,
                ..PARAMS
            },
            HurdleParams {
                logit_per_intensity: f64::INFINITY,
                ..PARAMS
            },
        ];
        for params in invalid {
            assert!(!params.is_valid(), "{params:?}");
        }
    }
}

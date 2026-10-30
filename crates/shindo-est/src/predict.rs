//! Intensity prediction for one hypocenter and many sites.

use crate::features::{self, FEATURE_COUNT};
use crate::forest::Forest;
use crate::hurdle::{self, HurdleParams};
use crate::input::{Hypocenter, Site};
use crate::model::ModelError;
use crate::slab::SlabGeometry;
use crate::validate::{self, InputError, Scope};

/// Settings of one call to [`Predictor::predict`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Options {
    /// Amount added to the regressor's instrumental intensity, which also
    /// raises the classifier's logit by `logit_per_intensity` times it.
    pub offset: f64,
}

/// Whether a site got an intensity.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[repr(i32)]
pub enum PointStatus {
    /// The site got an intensity.
    Ok = 0,
    /// The call is out of scope, as with an unknown depth or magnitude.
    OutOfScope = 1,
    /// The site has an invalid latitude, longitude, or ARV.
    InvalidInput = 2,
}

/// The prediction for one site.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PointResult {
    /// Whether the site got an intensity. If not, the values are NaN.
    pub status: PointStatus,
    /// The instrumental intensity after the hurdle rule, with the offset.
    pub instrumental: f64,
    /// The probability that the site is felt, with the offset applied.
    pub probability: f64,
    /// The regressor's instrumental intensity before the offset, for
    /// diagnosis.
    pub regression: f64,
}

impl PointResult {
    /// Returns a result whose values are NaN.
    fn without_values(status: PointStatus) -> Self {
        Self {
            status,
            instrumental: f64::NAN,
            probability: f64::NAN,
            regression: f64::NAN,
        }
    }
}

/// The felt classifier, the intensity regressor, and the data they need.
#[derive(Clone, Debug)]
pub struct Predictor {
    classifier: Forest,
    regressor: Forest,
    slabs: SlabGeometry,
    hurdle: HurdleParams,
}

impl Predictor {
    /// Combines a felt classifier and an intensity regressor with the slab
    /// geometry for their features and the hurdle settings.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::WrongFeatureCount`] when a forest does not take
    /// [`FEATURE_COUNT`] features, checking the classifier first, and
    /// [`ModelError::InvalidHurdleParams`] when `hurdle` is not valid.
    pub fn new(
        classifier: Forest,
        regressor: Forest,
        slabs: SlabGeometry,
        hurdle: HurdleParams,
    ) -> Result<Self, ModelError> {
        for forest in [&classifier, &regressor] {
            if forest.feature_count() != FEATURE_COUNT {
                return Err(ModelError::WrongFeatureCount {
                    feature_count: forest.feature_count(),
                    expected: FEATURE_COUNT,
                });
            }
        }
        if !hurdle.is_valid() {
            return Err(ModelError::InvalidHurdleParams);
        }
        Ok(Self {
            classifier,
            regressor,
            slabs,
            hurdle,
        })
    }

    /// Returns the hurdle settings.
    #[must_use]
    pub fn hurdle(&self) -> &HurdleParams {
        &self.hurdle
    }

    /// Predicts the intensity at each site, returning one result per site in
    /// the order of `sites`.
    ///
    /// A site with an invalid latitude, longitude, or ARV gets
    /// [`PointStatus::InvalidInput`], and the other sites are still predicted.
    /// When the call is out of scope, the other sites get
    /// [`PointStatus::OutOfScope`].
    ///
    /// # Errors
    ///
    /// Returns an [`InputError`] when the hypocenter or the offset is invalid,
    /// as [`validate::check_call`] describes.
    pub fn predict(
        &self,
        hypocenter: &Hypocenter,
        sites: &[Site],
        options: &Options,
    ) -> Result<Vec<PointResult>, InputError> {
        let scope = validate::check_call(hypocenter, options.offset)?;
        let mut results = vec![PointResult::without_values(PointStatus::InvalidInput); sites.len()];
        let mut rows = Vec::new();
        let mut computed = Vec::new();
        for (index, site) in sites.iter().enumerate() {
            if !validate::site_is_valid(site) {
                continue;
            }
            if scope == Scope::OutOfScope {
                results[index].status = PointStatus::OutOfScope;
            } else {
                rows.extend_from_slice(&features::compute(hypocenter, site, &self.slabs));
                computed.push(index);
            }
        }

        let mut classifier_raw = vec![0.0; computed.len()];
        let mut regressor_raw = vec![0.0; computed.len()];
        self.classifier.raw_batch(&rows, &mut classifier_raw);
        self.regressor.raw_batch(&rows, &mut regressor_raw);
        let scores = computed
            .iter()
            .zip(classifier_raw.iter().zip(&regressor_raw));
        for (&index, (&classifier, &regressor)) in scores {
            let combined = hurdle::combine(classifier, regressor, options.offset, &self.hurdle);
            results[index] = PointResult {
                status: PointStatus::Ok,
                instrumental: combined.instrumental,
                probability: combined.probability,
                regression: regressor,
            };
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FEATURE_NAMES;
    use crate::forest::Tree;
    use crate::slab::test_support;
    use crate::validate::HypocenterField;

    const PARAMS: HurdleParams = HurdleParams {
        felt_probability: 0.35,
        felt_floor: 0.5,
        unfelt_ceiling: 0.49,
        logit_per_intensity: 1.5,
    };

    const HYPOCENTER: Hypocenter = Hypocenter {
        latitude: 36.0,
        longitude: 140.0,
        depth: 10.0,
        magnitude: 6.0,
    };

    const SITE: Site = Site {
        latitude: 36.5,
        longitude: 140.5,
        arv: 1.0,
    };

    // Logit of the one-leaf classifier, far above the felt boundary.
    const CLASSIFIER_LEAF: f64 = 10.0;

    fn feature_index(name: &str) -> u32 {
        let index = FEATURE_NAMES
            .iter()
            .position(|&candidate| candidate == name)
            .expect("the name should be a feature");
        u32::try_from(index).expect("the index should fit in u32")
    }

    fn one_leaf(feature_count: usize, value: f64) -> Forest {
        let tree =
            Tree::new(&[], &[], &[], &[], &[], &[value]).expect("a one-leaf tree should be valid");
        Forest::new(feature_count, vec![tree]).expect("the forest should be valid")
    }

    // One split on feature `name`: at or below `threshold` gives 1, above gives 3.
    fn split_on(name: &str, threshold: f64) -> Forest {
        let tree = Tree::new(
            &[feature_index(name)],
            &[threshold],
            &[0],
            &[-1],
            &[-2],
            &[1.0, 3.0],
        )
        .expect("the tree should be valid");
        Forest::new(FEATURE_COUNT, vec![tree]).expect("the forest should be valid")
    }

    fn predictor_with(regressor: Forest) -> Predictor {
        Predictor::new(
            one_leaf(FEATURE_COUNT, CLASSIFIER_LEAF),
            regressor,
            test_support::geometry(),
            PARAMS,
        )
        .expect("the predictor should be valid")
    }

    // The regressor gives 3 from Mj 6 and 1 from Mj 4.
    fn mj_predictor() -> Predictor {
        predictor_with(split_on("mj", 5.0))
    }

    fn predict(hypocenter: &Hypocenter, sites: &[Site]) -> Vec<PointResult> {
        mj_predictor()
            .predict(hypocenter, sites, &Options::default())
            .expect("the call should be valid")
    }

    fn assert_without_values(result: &PointResult, status: PointStatus) {
        assert_eq!(result.status, status, "{result:?}");
        assert!(result.instrumental.is_nan(), "{result:?}");
        assert!(result.probability.is_nan(), "{result:?}");
        assert!(result.regression.is_nan(), "{result:?}");
    }

    #[test]
    fn predicts_each_site() {
        let results = predict(&HYPOCENTER, &[SITE, SITE]);
        assert_eq!(results.len(), 2, "{results:?}");
        for result in &results {
            assert_eq!(result.status, PointStatus::Ok, "{result:?}");
            assert_eq!(result.instrumental, 3.0, "{result:?}");
        }

        let small = Hypocenter {
            magnitude: 4.0,
            ..HYPOCENTER
        };
        let results = predict(&small, &[SITE]);
        assert_eq!(results[0].instrumental, 1.0, "{results:?}");
    }

    #[test]
    fn results_follow_the_site_order() {
        // The regressor gives 1 for ARV 1 (log 0) and 3 for ARV 100 (log 2).
        let predictor = predictor_with(split_on("log_arv", 1.0));
        let amplified = Site { arv: 100.0, ..SITE };
        let invalid = Site { arv: 0.0, ..SITE };
        let results = predictor
            .predict(
                &HYPOCENTER,
                &[SITE, invalid, amplified, SITE, amplified],
                &Options::default(),
            )
            .expect("the call should be valid");

        let statuses: Vec<PointStatus> = results.iter().map(|result| result.status).collect();
        assert_eq!(
            statuses,
            [
                PointStatus::Ok,
                PointStatus::InvalidInput,
                PointStatus::Ok,
                PointStatus::Ok,
                PointStatus::Ok,
            ]
        );
        let regressions: Vec<f64> = results.iter().map(|result| result.regression).collect();
        assert_eq!(regressions[0], 1.0, "{regressions:?}");
        assert!(regressions[1].is_nan(), "{regressions:?}");
        assert_eq!(regressions[2..], [3.0, 1.0, 3.0], "{regressions:?}");
    }

    #[test]
    fn reports_regression_before_offset() {
        let offset = 0.2;
        let results = mj_predictor()
            .predict(&HYPOCENTER, &[SITE], &Options { offset })
            .expect("the call should be valid");

        let result = results[0];
        assert_eq!(result.status, PointStatus::Ok, "{result:?}");
        assert_eq!(result.instrumental, 3.0 + offset, "{result:?}");
        assert_eq!(result.regression, 3.0, "{result:?}");
        assert_eq!(
            result.probability,
            hurdle::sigmoid(CLASSIFIER_LEAF + PARAMS.logit_per_intensity * offset),
            "{result:?}"
        );
    }

    #[test]
    fn unfelt_site_is_capped_at_the_ceiling() {
        // A logit of -10 is unfelt, so the regressor's 3 is lowered to the ceiling.
        let predictor = Predictor::new(
            one_leaf(FEATURE_COUNT, -CLASSIFIER_LEAF),
            split_on("mj", 5.0),
            test_support::geometry(),
            PARAMS,
        )
        .expect("the predictor should be valid");
        let results = predictor
            .predict(&HYPOCENTER, &[SITE], &Options::default())
            .expect("the call should be valid");

        assert_eq!(
            results[0].instrumental, PARAMS.unfelt_ceiling,
            "{results:?}"
        );
        assert_eq!(results[0].regression, 3.0, "{results:?}");
        assert_eq!(
            results[0].probability,
            hurdle::sigmoid(-CLASSIFIER_LEAF),
            "{results:?}"
        );
    }

    #[test]
    fn out_of_scope_marks_every_valid_site() {
        let invalid = Site { arv: 0.0, ..SITE };
        let unknown_depth = Hypocenter {
            depth: f64::NAN,
            ..HYPOCENTER
        };
        let zero_magnitude = Hypocenter {
            magnitude: 0.0,
            ..HYPOCENTER
        };
        for hypocenter in [unknown_depth, zero_magnitude] {
            let results = predict(&hypocenter, &[SITE, invalid, SITE]);
            assert_eq!(results.len(), 3, "{results:?}");
            assert_without_values(&results[0], PointStatus::OutOfScope);
            // An invalid site is the caller's mistake even when the call is
            // out of scope, as invalid calls are.
            assert_without_values(&results[1], PointStatus::InvalidInput);
            assert_without_values(&results[2], PointStatus::OutOfScope);
        }
    }

    #[test]
    fn invalid_site_does_not_stop_the_others() {
        let invalid = Site { arv: 0.0, ..SITE };
        let results = predict(&HYPOCENTER, &[invalid, SITE]);

        assert_eq!(results.len(), 2, "{results:?}");
        assert_without_values(&results[0], PointStatus::InvalidInput);
        assert_eq!(results[1].status, PointStatus::Ok, "{results:?}");
        assert_eq!(results[1].instrumental, 3.0, "{results:?}");
    }

    #[test]
    fn invalid_coordinates_of_a_site_are_invalid_input() {
        let sites = [
            Site {
                latitude: 91.0,
                ..SITE
            },
            Site {
                longitude: f64::NAN,
                ..SITE
            },
        ];
        for result in predict(&HYPOCENTER, &sites) {
            assert_without_values(&result, PointStatus::InvalidInput);
        }
    }

    #[test]
    fn invalid_call_is_an_error() {
        let predictor = mj_predictor();
        let wrong_latitude = Hypocenter {
            latitude: 91.0,
            ..HYPOCENTER
        };
        assert_eq!(
            predictor.predict(&wrong_latitude, &[SITE], &Options::default()),
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
        assert_eq!(
            predictor.predict(&HYPOCENTER, &[SITE], &Options { offset: f64::NAN }),
            Err(InputError::InvalidOffset)
        );
        // The call is checked even without sites.
        assert_eq!(
            predictor.predict(&wrong_latitude, &[], &Options::default()),
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
    }

    #[test]
    fn no_sites_give_no_results() {
        assert_eq!(predict(&HYPOCENTER, &[]), []);
    }

    #[test]
    fn invalid_site_alone_is_invalid_input() {
        let results = predict(&HYPOCENTER, &[Site { arv: 0.0, ..SITE }]);

        assert_eq!(results.len(), 1, "{results:?}");
        assert_without_values(&results[0], PointStatus::InvalidInput);
    }

    #[test]
    fn far_away_site_is_computed() {
        let far = Site {
            latitude: -35.0,
            longitude: -170.0,
            arv: 1.0,
        };
        let results = predict(&HYPOCENTER, &[far]);

        let result = results[0];
        assert_eq!(result.status, PointStatus::Ok, "{result:?}");
        assert!(result.instrumental.is_finite(), "{result:?}");
        assert!(result.probability.is_finite(), "{result:?}");
        assert!(result.regression.is_finite(), "{result:?}");
    }

    #[test]
    fn rejects_forests_with_the_wrong_feature_count() {
        let wrong = FEATURE_COUNT - 1;
        let error = Err(ModelError::WrongFeatureCount {
            feature_count: wrong,
            expected: FEATURE_COUNT,
        });
        let classifier = Predictor::new(
            one_leaf(wrong, CLASSIFIER_LEAF),
            split_on("mj", 5.0),
            test_support::geometry(),
            PARAMS,
        );
        assert_eq!(classifier.map(|_| ()), error);
        let regressor = Predictor::new(
            one_leaf(FEATURE_COUNT, CLASSIFIER_LEAF),
            one_leaf(wrong, 1.0),
            test_support::geometry(),
            PARAMS,
        );
        assert_eq!(regressor.map(|_| ()), error);
    }

    #[test]
    fn rejects_invalid_hurdle_params() {
        // A NaN boundary would send every site to the unfelt side.
        let params = HurdleParams {
            felt_probability: f64::NAN,
            ..PARAMS
        };
        let predictor = Predictor::new(
            one_leaf(FEATURE_COUNT, CLASSIFIER_LEAF),
            split_on("mj", 5.0),
            test_support::geometry(),
            params,
        );
        assert_eq!(predictor.map(|_| ()), Err(ModelError::InvalidHurdleParams));
    }

    #[test]
    fn keeps_the_hurdle_params() {
        assert_eq!(*mj_predictor().hurdle(), PARAMS);
    }

    #[test]
    fn default_offset_is_zero() {
        assert_eq!(Options::default(), Options { offset: 0.0 });
    }

    #[test]
    fn statuses_have_fixed_codes() {
        assert_eq!(PointStatus::Ok as i32, 0);
        assert_eq!(PointStatus::OutOfScope as i32, 1);
        assert_eq!(PointStatus::InvalidInput as i32, 2);
    }
}

//! Intensity prediction for one hypocenter and many sites.

use std::num::NonZeroUsize;
use std::panic;
use std::sync::OnceLock;
use std::thread::{self, ScopedJoinHandle};

use crate::features::{self, FEATURE_COUNT, FEATURE_NAMES};
use crate::forest::Forest;
use crate::hurdle::{self, HurdleParams};
use crate::input::{Hypocenter, Site};
use crate::model::{Model, ModelError, Objective};
use crate::slab::SlabGeometry;
use crate::validate::{self, InputError, Scope};

/// Settings of one call to [`Predictor::predict`] or [`Predictor::predict_into`].
///
/// Start from [`Options::default`] and change what differs, so that code
/// keeps compiling when settings are added.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Options {
    /// Amount added to the regressor's instrumental intensity, which also
    /// raises the classifier's logit by `logit_per_intensity` times it.
    pub offset: f64,
    /// Largest number of threads the call uses, counting the calling thread,
    /// or `None` for the number of logical CPUs the system reports, asked
    /// once per process and taken as 1 where it is unknown, as on wasm32.
    ///
    /// A call uses one thread for every 256 sites or part of them, up to this
    /// number, so `Some(1)` starts no thread. A chunk whose thread cannot be
    /// started is predicted on the calling thread. The results do not depend
    /// on the number of threads.
    pub max_threads: Option<NonZeroUsize>,
}

impl Options {
    /// Returns the options with `offset` instead.
    #[must_use]
    pub fn with_offset(self, offset: f64) -> Self {
        Self { offset, ..self }
    }

    /// Returns the options with at most `threads` threads instead.
    #[must_use]
    pub fn with_max_threads(self, threads: NonZeroUsize) -> Self {
        Self {
            max_threads: Some(threads),
            ..self
        }
    }
}

/// Sites for which a call uses one more thread, so that a small call does
/// not pay for starting threads.
const SITES_PER_THREAD: usize = 256;

fn thread_count(sites: usize, max_threads: usize) -> usize {
    max_threads.min(sites.div_ceil(SITES_PER_THREAD)).max(1)
}

/// Returns the number of logical CPUs, or 1 where it is unknown, as on
/// wasm32.
fn automatic_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| thread::available_parallelism().map_or(1, NonZeroUsize::get))
}

struct SpawnedChunk<'scope, 'out> {
    handle: ScopedJoinHandle<'scope, Vec<PointResult>>,
    out: &'out mut [PointResult],
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
///
/// [`PointResult::default`] is a result without values, marked
/// [`PointStatus::InvalidInput`], for filling a buffer that
/// [`Predictor::predict_into`] overwrites.
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

impl Default for PointResult {
    fn default() -> Self {
        Self::without_values(PointStatus::InvalidInput)
    }
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
    /// Checks the classifier, then the regressor, then `hurdle`, and returns
    /// [`ModelError::WrongObjective`] when the classifier is not a
    /// [`Objective::Binary`] model or the regressor not a
    /// [`Objective::Regression`] model, [`ModelError::WrongFeatures`] when a
    /// model's feature names are not [`FEATURE_NAMES`] in their order, and
    /// [`ModelError::InvalidHurdleParams`] when `hurdle` is not valid.
    pub fn new(
        classifier: Model,
        regressor: Model,
        slabs: SlabGeometry,
        hurdle: HurdleParams,
    ) -> Result<Self, ModelError> {
        check_model(&classifier, Objective::Binary)?;
        check_model(&regressor, Objective::Regression)?;
        if !hurdle.is_valid() {
            return Err(ModelError::InvalidHurdleParams);
        }
        Ok(Self {
            classifier: classifier.into_forest(),
            regressor: regressor.into_forest(),
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
        let mut results = vec![PointResult::default(); sites.len()];
        self.predict_into(hypocenter, sites, options, &mut results)?;
        Ok(results)
    }

    /// Writes into `out` what [`Predictor::predict`] returns, one result per
    /// site in the order of `sites`, so that a caller can reuse its buffer.
    ///
    /// The sites are split among up to [`Options::max_threads`] threads, and
    /// the results are the same, bit for bit, whatever the number of threads.
    ///
    /// # Errors
    ///
    /// Returns an [`InputError`] when the hypocenter or the offset is invalid,
    /// as [`validate::check_call`] describes, and leaves `out` as it was.
    ///
    /// # Panics
    ///
    /// Panics if `out` does not have the length of `sites`.
    pub fn predict_into(
        &self,
        hypocenter: &Hypocenter,
        sites: &[Site],
        options: &Options,
        out: &mut [PointResult],
    ) -> Result<(), InputError> {
        assert_eq!(
            out.len(),
            sites.len(),
            "out must have one result for each of the {} sites",
            sites.len()
        );
        let call_scope = validate::check_call(hypocenter, options.offset)?;
        if sites.is_empty() {
            return Ok(());
        }
        let max_threads = options
            .max_threads
            .map_or_else(automatic_threads, NonZeroUsize::get);
        let chunk_size = sites.len().div_ceil(thread_count(sites.len(), max_threads));
        let mut chunks = sites.chunks(chunk_size).zip(out.chunks_mut(chunk_size));
        let first = chunks.next();
        thread::scope(|threads| {
            let mut spawned: Vec<SpawnedChunk<'_, '_>> = Vec::new();
            for (chunk_sites, chunk_out) in chunks {
                let predict_on_thread = move || {
                    let mut results = vec![PointResult::default(); chunk_sites.len()];
                    self.predict_chunk(
                        hypocenter,
                        chunk_sites,
                        call_scope,
                        options.offset,
                        &mut results,
                    );
                    results
                };
                match thread::Builder::new().spawn_scoped(threads, predict_on_thread) {
                    Ok(handle) => spawned.push(SpawnedChunk {
                        handle,
                        out: chunk_out,
                    }),
                    Err(_) => self.predict_chunk(
                        hypocenter,
                        chunk_sites,
                        call_scope,
                        options.offset,
                        chunk_out,
                    ),
                }
            }
            if let Some((chunk_sites, chunk_out)) = first {
                self.predict_chunk(
                    hypocenter,
                    chunk_sites,
                    call_scope,
                    options.offset,
                    chunk_out,
                );
            }
            for chunk in spawned {
                match chunk.handle.join() {
                    Ok(results) => chunk.out.copy_from_slice(&results),
                    Err(payload) => panic::resume_unwind(payload),
                }
            }
        });
        Ok(())
    }

    /// Predicts `sites` into `out` for a call already checked to have
    /// `call_scope`.
    fn predict_chunk(
        &self,
        hypocenter: &Hypocenter,
        sites: &[Site],
        call_scope: Scope,
        offset: f64,
        out: &mut [PointResult],
    ) {
        out.fill(PointResult::default());
        let mut rows = Vec::new();
        let mut computed = Vec::new();
        for (index, site) in sites.iter().enumerate() {
            if !validate::site_is_valid(site) {
                continue;
            }
            if call_scope == Scope::OutOfScope {
                out[index].status = PointStatus::OutOfScope;
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
            let combined = hurdle::combine(classifier, regressor, offset, &self.hurdle);
            out[index] = PointResult {
                status: PointStatus::Ok,
                instrumental: combined.instrumental,
                probability: combined.probability,
                regression: regressor,
            };
        }
    }
}

/// Checks that a model has `objective` and takes the features predictions
/// compute.
fn check_model(model: &Model, objective: Objective) -> Result<(), ModelError> {
    if model.objective() != objective {
        return Err(ModelError::WrongObjective {
            expected: objective,
            found: model.objective(),
        });
    }
    let names = model.feature_names();
    let differing = names
        .iter()
        .zip(FEATURE_NAMES)
        .position(|(name, expected)| name != expected);
    if let Some(index) = differing {
        return Err(ModelError::WrongFeatures { index });
    }
    if names.len() != FEATURE_COUNT {
        return Err(ModelError::WrongFeatures {
            index: names.len().min(FEATURE_COUNT),
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::features::{FEATURE_COUNT, FEATURE_NAMES};
    use crate::forest::{Forest, Tree};
    use crate::input::Site;
    use crate::model::{Model, Objective};

    const INVALID_SITE_PERIOD: usize = 97;

    fn feature_index(name: &str) -> u32 {
        let index = FEATURE_NAMES
            .iter()
            .position(|&candidate| candidate == name)
            .expect("the name should be a feature");
        u32::try_from(index).expect("the index should fit in u32")
    }

    fn split(name: &str, threshold: f64, left: f64, right: f64) -> Tree {
        Tree::new(
            &[feature_index(name)],
            &[threshold],
            &[0],
            &[-1],
            &[-2],
            &[left, right],
        )
        .expect("the tree should be valid")
    }

    fn model(trees: Vec<Tree>, objective: Objective) -> Model {
        let forest = Forest::new(FEATURE_COUNT, trees).expect("the forest should be valid");
        let names: Vec<String> = FEATURE_NAMES.map(String::from).to_vec();
        Model::new(forest, objective, names).expect("the model should be valid")
    }

    /// Returns a classifier whose logit is 1 within 200 km of the epicenter
    /// and -3 beyond.
    pub fn classifier() -> Model {
        model(vec![split("epi", 200.0, 1.0, -3.0)], Objective::Binary)
    }

    /// Returns a regressor that depends on the site latitude and longitude and
    /// the epicentral distance, so that results written to the wrong site give
    /// other values.
    pub fn regressor() -> Model {
        let trees = vec![
            split("slat", 36.0, 1.0, 2.0),
            split("slon", 140.0, 0.1, 0.2),
            split("epi", 100.0, 0.01, 0.02),
        ];
        model(trees, Objective::Regression)
    }

    /// Returns `count` sites along a line from 33 N 137 E toward the
    /// northeast. Every 97th site, starting with the first, has an invalid ARV
    /// of 0.
    pub fn sites(count: usize) -> Vec<Site> {
        let mut sites = Vec::with_capacity(count);
        let mut latitude = 33.0;
        let mut longitude = 137.0;
        for index in 0..count {
            let arv = if is_invalid(index) { 0.0 } else { 1.0 };
            sites.push(Site {
                latitude,
                longitude,
                arv,
            });
            latitude += 0.01;
            longitude += 0.007;
        }
        sites
    }

    /// Returns whether the site at `index` of [`sites`] has an invalid ARV.
    pub fn is_invalid(index: usize) -> bool {
        index.is_multiple_of(INVALID_SITE_PERIOD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FEATURE_NAMES;
    use crate::forest::Tree;
    use crate::model::Objective;
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

    fn feature_names() -> Vec<String> {
        FEATURE_NAMES.map(String::from).to_vec()
    }

    fn model(forest: Forest, objective: Objective, feature_names: Vec<String>) -> Model {
        Model::new(forest, objective, feature_names).expect("the model should be valid")
    }

    fn one_leaf(feature_count: usize, value: f64) -> Forest {
        let tree =
            Tree::new(&[], &[], &[], &[], &[], &[value]).expect("a one-leaf tree should be valid");
        Forest::new(feature_count, vec![tree]).expect("the forest should be valid")
    }

    fn classifier(logit: f64) -> Model {
        model(
            one_leaf(FEATURE_COUNT, logit),
            Objective::Binary,
            feature_names(),
        )
    }

    // One split on feature `name`: at or below `threshold` gives 1, above gives 3.
    fn split_on(name: &str, threshold: f64) -> Model {
        let tree = Tree::new(
            &[feature_index(name)],
            &[threshold],
            &[0],
            &[-1],
            &[-2],
            &[1.0, 3.0],
        )
        .expect("the tree should be valid");
        let forest = Forest::new(FEATURE_COUNT, vec![tree]).expect("the forest should be valid");
        model(forest, Objective::Regression, feature_names())
    }

    fn predictor_with(regressor: Model) -> Predictor {
        Predictor::new(
            classifier(CLASSIFIER_LEAF),
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

    fn new_predictor(classifier: Model, regressor: Model) -> Result<Predictor, ModelError> {
        Predictor::new(classifier, regressor, test_support::geometry(), PARAMS)
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
            .predict(
                &HYPOCENTER,
                &[SITE],
                &Options::default().with_offset(offset),
            )
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
        let predictor = new_predictor(classifier(-CLASSIFIER_LEAF), split_on("mj", 5.0))
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
            predictor.predict(
                &HYPOCENTER,
                &[SITE],
                &Options::default().with_offset(f64::NAN)
            ),
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
    fn rejects_models_with_other_features() {
        let mut swapped = feature_names();
        swapped.swap(0, 1);
        let mut longer = feature_names();
        longer.push("extra".to_owned());
        let shorter = feature_names()[..FEATURE_COUNT - 1].to_vec();
        let mut padded = feature_names();
        padded[0].push('\u{a0}');
        let cases = [
            (swapped, FEATURE_COUNT, 0),
            (padded, FEATURE_COUNT, 0),
            (longer, FEATURE_COUNT + 1, FEATURE_COUNT),
            (shorter, FEATURE_COUNT - 1, FEATURE_COUNT - 1),
        ];
        for (names, feature_count, index) in cases {
            let error = Err(ModelError::WrongFeatures { index });
            let other = |objective| model(one_leaf(feature_count, 1.0), objective, names.clone());

            let as_classifier = new_predictor(other(Objective::Binary), split_on("mj", 5.0));
            assert_eq!(as_classifier.map(|_| ()), error, "{names:?}");
            let as_regressor = new_predictor(classifier(1.0), other(Objective::Regression));
            assert_eq!(as_regressor.map(|_| ()), error, "{names:?}");
        }
    }

    #[test]
    fn rejects_models_with_the_wrong_objective() {
        let classifier_as_regressor = new_predictor(classifier(1.0), classifier(1.0));
        let regressor_as_classifier = new_predictor(split_on("mj", 5.0), split_on("mj", 5.0));

        assert_eq!(
            classifier_as_regressor.map(|_| ()),
            Err(ModelError::WrongObjective {
                expected: Objective::Regression,
                found: Objective::Binary,
            })
        );
        assert_eq!(
            regressor_as_classifier.map(|_| ()),
            Err(ModelError::WrongObjective {
                expected: Objective::Binary,
                found: Objective::Regression,
            })
        );
    }

    #[test]
    fn rejects_invalid_hurdle_params() {
        // A NaN boundary would send every site to the unfelt side.
        let params = HurdleParams {
            felt_probability: f64::NAN,
            ..PARAMS
        };
        let predictor = Predictor::new(
            classifier(CLASSIFIER_LEAF),
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
        assert_eq!(
            Options::default(),
            Options {
                offset: 0.0,
                max_threads: None,
            }
        );
    }

    #[test]
    fn default_result_has_no_values() {
        assert_without_values(&PointResult::default(), PointStatus::InvalidInput);
    }

    #[test]
    fn with_offset_sets_the_offset() {
        assert_eq!(Options::default().with_offset(0.2).offset, 0.2);
    }

    // A finite result that no prediction below gives, to see what is written.
    const UNTOUCHED: PointResult = PointResult {
        status: PointStatus::OutOfScope,
        instrumental: -1.0,
        probability: -1.0,
        regression: -1.0,
    };

    #[test]
    fn predict_into_writes_what_predict_returns() {
        let predictor = mj_predictor();
        let sites = [SITE, Site { arv: 0.0, ..SITE }, SITE];
        let options = Options::default().with_offset(0.2);
        let expected = predictor
            .predict(&HYPOCENTER, &sites, &options)
            .expect("the call should be valid");

        let mut out = [UNTOUCHED; 3];
        predictor
            .predict_into(&HYPOCENTER, &sites, &options, &mut out)
            .expect("the call should be valid");

        assert_eq!(out[0], expected[0]);
        assert_without_values(&out[1], PointStatus::InvalidInput);
        assert_eq!(out[2], expected[2]);
    }

    #[test]
    fn predict_into_leaves_the_output_alone_on_an_error() {
        let wrong_latitude = Hypocenter {
            latitude: 91.0,
            ..HYPOCENTER
        };
        let mut out = [UNTOUCHED; 1];

        let result =
            mj_predictor().predict_into(&wrong_latitude, &[SITE], &Options::default(), &mut out);

        assert_eq!(
            result,
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
        assert_eq!(out, [UNTOUCHED]);
    }

    #[test]
    #[should_panic(expected = "one result for each of the 2 sites")]
    fn predict_into_rejects_an_output_of_another_length() {
        let mut out = [UNTOUCHED; 1];
        let _ =
            mj_predictor().predict_into(&HYPOCENTER, &[SITE, SITE], &Options::default(), &mut out);
    }

    fn threads(count: usize) -> NonZeroUsize {
        NonZeroUsize::new(count).expect("the thread count should be positive")
    }

    fn many_site_predictor() -> Predictor {
        Predictor::new(
            super::test_support::classifier(),
            super::test_support::regressor(),
            test_support::geometry(),
            PARAMS,
        )
        .expect("the predictor should be valid")
    }

    fn predict_with_threads(
        predictor: &Predictor,
        hypocenter: &Hypocenter,
        sites: &[Site],
        max_threads: usize,
    ) -> Vec<PointResult> {
        let options = Options::default().with_max_threads(threads(max_threads));
        predictor
            .predict(hypocenter, sites, &options)
            .expect("the call should be valid")
    }

    /// Asserts that two calls gave the same bits; `call` names the call that
    /// gave `actual`.
    fn assert_same_bits(actual: &[PointResult], expected: &[PointResult], call: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{call}: one result for each site"
        );
        let pairs = actual.iter().zip(expected).enumerate();
        for (index, (result, reference)) in pairs {
            let same = result.status == reference.status
                && result.instrumental.to_bits() == reference.instrumental.to_bits()
                && result.probability.to_bits() == reference.probability.to_bits()
                && result.regression.to_bits() == reference.regression.to_bits();
            assert!(
                same,
                "{call}, site {index}: {result:?}, expected {reference:?}"
            );
        }
    }

    #[test]
    fn thread_count_adds_a_thread_for_every_256_sites() {
        assert_eq!(super::thread_count(0, 8), 1);
        assert_eq!(super::thread_count(256, 8), 1);
        assert_eq!(super::thread_count(257, 8), 2);
        assert_eq!(super::thread_count(4400, 24), 18);
        assert_eq!(super::thread_count(4400, 4), 4);
        assert_eq!(super::thread_count(10, 1), 1);
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        let predictor = many_site_predictor();
        let sites = super::test_support::sites(1000);
        let sequential = predict_with_threads(&predictor, &HYPOCENTER, &sites, 1);

        let mut distinct: Vec<u64> = Vec::new();
        for result in &sequential {
            if result.status == PointStatus::Ok {
                distinct.push(result.instrumental.to_bits());
            }
        }
        distinct.sort_unstable();
        distinct.dedup();
        assert!(distinct.len() >= 4, "{} distinct values", distinct.len());

        for max_threads in [2, 3, 8] {
            let threaded = predict_with_threads(&predictor, &HYPOCENTER, &sites, max_threads);
            assert_same_bits(&threaded, &sequential, &format!("{max_threads} threads"));
        }
    }

    #[test]
    fn chunk_boundaries_keep_site_order() {
        let predictor = many_site_predictor();
        for count in [255, 256, 257, 513] {
            let sites = super::test_support::sites(count);
            let sequential = predict_with_threads(&predictor, &HYPOCENTER, &sites, 1);
            let threaded = predict_with_threads(&predictor, &HYPOCENTER, &sites, 4);
            assert_same_bits(
                &threaded,
                &sequential,
                &format!("{count} sites on 4 threads"),
            );
        }
    }

    #[test]
    fn out_of_scope_reaches_every_chunk() {
        let unknown_depth = Hypocenter {
            depth: f64::NAN,
            ..HYPOCENTER
        };
        let sites = super::test_support::sites(600);
        let results = predict_with_threads(&many_site_predictor(), &unknown_depth, &sites, 3);

        for (index, result) in results.iter().enumerate() {
            let expected = if super::test_support::is_invalid(index) {
                PointStatus::InvalidInput
            } else {
                PointStatus::OutOfScope
            };
            assert_without_values(result, expected);
        }
    }

    #[test]
    fn no_sites_with_threads() {
        let predictor = many_site_predictor();
        let options = Options::default().with_max_threads(threads(4));
        assert_eq!(
            predictor.predict_into(&HYPOCENTER, &[], &options, &mut []),
            Ok(())
        );

        let wrong_latitude = Hypocenter {
            latitude: 91.0,
            ..HYPOCENTER
        };
        assert_eq!(
            predictor.predict_into(&wrong_latitude, &[], &options, &mut []),
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
    }

    #[test]
    fn errors_leave_out_untouched_with_threads() {
        let wrong_latitude = Hypocenter {
            latitude: 91.0,
            ..HYPOCENTER
        };
        let sites = super::test_support::sites(600);
        let mut out = vec![UNTOUCHED; sites.len()];
        let options = Options::default().with_max_threads(threads(4));

        let result =
            many_site_predictor().predict_into(&wrong_latitude, &sites, &options, &mut out);

        assert_eq!(
            result,
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
        assert!(out.iter().all(|&result| result == UNTOUCHED), "{out:?}");
    }

    #[test]
    fn max_threads_defaults_to_automatic() {
        assert_eq!(Options::default().max_threads, None);

        let options = Options::default()
            .with_offset(0.2)
            .with_max_threads(threads(3));
        assert_eq!(options.max_threads, Some(threads(3)));
        assert_eq!(options.offset, 0.2);
    }

    #[test]
    fn predictor_can_be_shared_between_threads() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<Predictor>();
    }

    #[test]
    fn statuses_have_fixed_codes() {
        assert_eq!(PointStatus::Ok as i32, 0);
        assert_eq!(PointStatus::OutOfScope as i32, 1);
        assert_eq!(PointStatus::InvalidInput as i32, 2);
    }
}

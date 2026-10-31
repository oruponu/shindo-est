//! Measures prediction for one hypocenter and 4,400 sites with synthetic
//! forests as large as a full model: 3,300 trees of 63 leaves each.
//!
//! The trees are grown on the features of several hypocenters, as `LightGBM`
//! grows them on training data: every split divides the rows that reach it,
//! so that branches are closer to a trained model's than splits at random
//! thresholds would be. They are not grown by gain, so a trained model may
//! still be faster or slower to evaluate.
//!
//! Run with `cargo bench -p shindo-est --bench predict`.

use std::fs;
use std::hint;
use std::num::NonZeroUsize;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use shindo_est::features::{self, FEATURE_COUNT, FEATURE_NAMES};
use shindo_est::forest::{Forest, Tree};
use shindo_est::{
    HurdleParams, Hypocenter, Model, Objective, Options, PointResult, Predictor, Site,
    SlabGeometry, SlabGrid,
};

/// Number of trees in each forest: the largest model expected, since
/// training stops at 3,000 rounds and the shipped models are refit with 1.1
/// times the rounds early stopping chose.
const TREE_COUNT: usize = 3_300;
/// Number of leaves in each tree.
const LEAF_COUNT: usize = 63;

/// Number of site latitudes.
const SITE_ROWS: usize = 55;
/// Number of site longitudes.
const SITE_COLUMNS: usize = 80;
/// Latitude of the southernmost sites in degrees.
const FIRST_SITE_LATITUDE: f64 = 24.0;
/// Longitude of the westernmost sites in degrees.
const FIRST_SITE_LONGITUDE: f64 = 122.0;
/// Spacing of the sites in degrees.
const SITE_SPACING: f64 = 0.4;
/// ARVs given to the sites in turn.
const SITE_ARVS: [f64; 3] = [0.5, 1.0, 2.0];

/// The hypocenter whose prediction is measured.
const HYPOCENTER: Hypocenter = Hypocenter {
    latitude: 38.0,
    longitude: 142.5,
    depth: 40.0,
    magnitude: 7.0,
};

/// Hypocenters whose features the trees are grown on, so that splits on the
/// hypocenter features divide rows as they do in training.
const TRAINING_HYPOCENTERS: [Hypocenter; 4] = [
    HYPOCENTER,
    Hypocenter {
        latitude: 35.5,
        longitude: 139.5,
        depth: 60.0,
        magnitude: 5.5,
    },
    Hypocenter {
        latitude: 33.0,
        longitude: 132.0,
        depth: 350.0,
        magnitude: 6.0,
    },
    Hypocenter {
        latitude: 42.5,
        longitude: 145.0,
        depth: 100.0,
        magnitude: 8.0,
    },
];
/// Training rows drawn with replacement for each tree, as bagging does.
const BAGGED_ROWS: usize = 4_000;
/// Fewest rows each side of a split keeps, as `min_data_in_leaf` does.
const MIN_LEAF_ROWS: usize = 10;
/// Candidate splits tried for one split before giving up.
const MAX_SPLIT_ATTEMPTS: usize = 1_000;

/// The hurdle values of the previous model. They do not affect the speed.
const HURDLE: HurdleParams = HurdleParams {
    felt_probability: 0.35,
    felt_floor: 0.5,
    unfelt_ceiling: 0.49,
    logit_per_intensity: 1.5,
};

/// Index of `slab_top`, the first feature that can be NaN; `updip` and
/// `path_frac` follow.
const FIRST_SLAB_FEATURE: usize = 11;
/// Decision type of a split without missing values.
const MISSING_NONE: u8 = 0;
/// Decision type bit sending missing values left.
const DEFAULT_LEFT: u8 = 2;
/// Decision type of a split that treats NaN as missing.
const MISSING_NAN: u8 = 8;
/// `LightGBM`'s `kZeroThreshold`: the trees read values this close to zero as
/// zero.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;
/// Leaf values are drawn uniformly from minus to plus this value.
const LEAF_VALUE_RANGE: f64 = 0.05;

const SEED: u64 = 0;

const WARMUP_RUNS: usize = 3;
const MEASURED_RUNS: usize = 20;
const MILLISECONDS_PER_SECOND: f64 = 1000.0;

/// Increment of the `SplitMix64` state, the golden ratio as a 64-bit fraction.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;

/// The `SplitMix64` generator, so that every run builds the same forests.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("the bound should fit in u64");
        usize::try_from(self.next_u64() % bound)
            .expect("an index below the bound should fit in usize")
    }

    /// Returns a value in `[0, 1)`: 52 random bits become the mantissa of a
    /// number in `[1, 2)`, from which 1 is subtracted.
    fn unit(&mut self) -> f64 {
        f64::from_bits((self.next_u64() >> 12) | 1.0_f64.to_bits()) - 1.0
    }

    fn flag(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }
}

/// Where a leaf hangs from: a node and its side.
#[derive(Clone, Copy)]
struct Parent {
    node: usize,
    left: bool,
}

fn sites() -> Vec<Site> {
    let mut sites = Vec::with_capacity(SITE_ROWS * SITE_COLUMNS);
    let mut latitude = FIRST_SITE_LATITUDE;
    for _ in 0..SITE_ROWS {
        let mut longitude = FIRST_SITE_LONGITUDE;
        for _ in 0..SITE_COLUMNS {
            sites.push(Site {
                latitude,
                longitude,
                arv: SITE_ARVS[sites.len() % SITE_ARVS.len()],
            });
            longitude += SITE_SPACING;
        }
        latitude += SITE_SPACING;
    }
    sites
}

fn slab_geometry() -> SlabGeometry {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/slab2");
    let grid = |file: &str| {
        let path = directory.join(file);
        let text =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        SlabGrid::parse(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    };
    SlabGeometry::new(grid("kur.csv"), grid("izu.csv"))
}

/// A split of one leaf into a node with two leaves.
struct Split {
    leaf: usize,
    feature: usize,
    threshold: f64,
    missing: u8,
    default_left: bool,
    left_rows: Vec<usize>,
    right_rows: Vec<usize>,
}

/// Returns whether a node sends a value left, as `LightGBM` decides it for a
/// value from [`as_the_trees_see`].
fn goes_left(value: f64, threshold: f64, missing: u8, default_left: bool) -> bool {
    if value.is_nan() {
        if missing == MISSING_NAN {
            default_left
        } else {
            // A node that does not treat NaN as missing reads it as zero. No
            // feature that such nodes split on is NaN here, but the rule
            // keeps the rows where the evaluation sends them.
            0.0 <= threshold
        }
    } else {
        value <= threshold
    }
}

/// Returns a leaf picked with a probability proportional to its rows, so that
/// large leaves are split more often, as they usually gain the most.
fn weighted_leaf(random: &mut SplitMix64, leaf_rows: &[Vec<usize>]) -> usize {
    let total: usize = leaf_rows.iter().map(Vec::len).sum();
    let mut target = random.below(total);
    for (leaf, rows) in leaf_rows.iter().enumerate() {
        if target < rows.len() {
            return leaf;
        }
        target -= rows.len();
    }
    unreachable!("the target is below the total number of rows")
}

/// Returns a split that leaves at least `MIN_LEAF_ROWS` rows on each side,
/// with a random feature and a threshold at a random rank of its values.
fn find_split(
    random: &mut SplitMix64,
    rows: &[[f64; FEATURE_COUNT]],
    leaf_rows: &[Vec<usize>],
) -> Split {
    for _ in 0..MAX_SPLIT_ATTEMPTS {
        let leaf = weighted_leaf(random, leaf_rows);
        let feature = random.below(FEATURE_COUNT);
        let mut values: Vec<f64> = Vec::with_capacity(leaf_rows[leaf].len());
        for &row in &leaf_rows[leaf] {
            let value = rows[row][feature];
            if !value.is_nan() {
                values.push(value);
            }
        }
        if values.len() < 2 * MIN_LEAF_ROWS {
            continue;
        }
        // Ranks that leave `MIN_LEAF_ROWS` values at or below the threshold
        // and as many above it. Repeated values can move rows to the left, so
        // the sides are counted again after the split.
        let rank = MIN_LEAF_ROWS - 1 + random.below(values.len() - 2 * MIN_LEAF_ROWS + 1);
        let (_, threshold, _) = values.select_nth_unstable_by(rank, f64::total_cmp);
        let threshold = *threshold;
        let missing = if feature >= FIRST_SLAB_FEATURE {
            MISSING_NAN
        } else {
            MISSING_NONE
        };
        let default_left = random.flag();

        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for &row in &leaf_rows[leaf] {
            if goes_left(rows[row][feature], threshold, missing, default_left) {
                left_rows.push(row);
            } else {
                right_rows.push(row);
            }
        }
        if left_rows.len() >= MIN_LEAF_ROWS && right_rows.len() >= MIN_LEAF_ROWS {
            return Split {
                leaf,
                feature,
                threshold,
                missing,
                default_left,
                left_rows,
                right_rows,
            };
        }
    }
    panic!("no split left {MIN_LEAF_ROWS} rows on each side in {MAX_SPLIT_ATTEMPTS} tries");
}

/// Grows a tree leaf by leaf on rows bagged from `rows`, numbered as
/// `LightGBM` numbers them: split `k` turns a leaf into node `k`, whose left
/// child keeps the leaf's index and whose right child is the new leaf `k + 1`.
fn tree(random: &mut SplitMix64, rows: &[[f64; FEATURE_COUNT]]) -> Tree {
    let node_count = LEAF_COUNT - 1;
    let mut split_feature: Vec<u32> = Vec::with_capacity(node_count);
    let mut threshold: Vec<f64> = Vec::with_capacity(node_count);
    let mut decision_type: Vec<u8> = Vec::with_capacity(node_count);
    let mut left_child: Vec<i32> = Vec::with_capacity(node_count);
    let mut right_child: Vec<i32> = Vec::with_capacity(node_count);
    let mut leaf_parents: Vec<Option<Parent>> = vec![None];
    let mut bagged: Vec<usize> = Vec::with_capacity(BAGGED_ROWS);
    for _ in 0..BAGGED_ROWS {
        bagged.push(random.below(rows.len()));
    }
    let mut leaf_rows: Vec<Vec<usize>> = vec![bagged];
    let encode_leaf = |leaf: usize| !i32::try_from(leaf).expect("a leaf index should fit in i32");

    for node in 0..node_count {
        let split = find_split(random, rows, &leaf_rows);
        let new_leaf = leaf_parents.len();
        if let Some(parent) = leaf_parents[split.leaf] {
            let child = i32::try_from(node).expect("a node index should fit in i32");
            if parent.left {
                left_child[parent.node] = child;
            } else {
                right_child[parent.node] = child;
            }
        }

        let feature = u32::try_from(split.feature).expect("a feature index should fit in u32");
        split_feature.push(feature);
        threshold.push(split.threshold);
        let default_left = if split.default_left { DEFAULT_LEFT } else { 0 };
        decision_type.push(split.missing | default_left);
        left_child.push(encode_leaf(split.leaf));
        right_child.push(encode_leaf(new_leaf));

        leaf_parents[split.leaf] = Some(Parent { node, left: true });
        leaf_parents.push(Some(Parent { node, left: false }));
        leaf_rows[split.leaf] = split.left_rows;
        leaf_rows.push(split.right_rows);
    }

    let mut leaf_value: Vec<f64> = Vec::with_capacity(LEAF_COUNT);
    for _ in 0..LEAF_COUNT {
        leaf_value.push((random.unit() * 2.0 - 1.0) * LEAF_VALUE_RANGE);
    }
    Tree::new(
        &split_feature,
        &threshold,
        &decision_type,
        &left_child,
        &right_child,
        &leaf_value,
    )
    .expect("a grown tree should be valid")
}

fn forest(random: &mut SplitMix64, rows: &[[f64; FEATURE_COUNT]]) -> Forest {
    let mut trees = Vec::with_capacity(TREE_COUNT);
    for _ in 0..TREE_COUNT {
        trees.push(tree(random, rows));
    }
    Forest::new(FEATURE_COUNT, trees).expect("the forest should be valid")
}

/// Returns a feature as the trees see it: rounded to `f32`, and zero when
/// within [`ZERO_THRESHOLD`] of zero. NaN stays NaN.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the trees compare features rounded to f32"
)]
fn as_the_trees_see(value: f64) -> f64 {
    let rounded = f64::from(value as f32);
    if rounded.abs() <= ZERO_THRESHOLD {
        0.0
    } else {
        rounded
    }
}

/// Returns the features of every site for every training hypocenter, as the
/// trees see them.
fn training_rows(sites: &[Site], slabs: &SlabGeometry) -> Vec<[f64; FEATURE_COUNT]> {
    let mut rows = Vec::with_capacity(TRAINING_HYPOCENTERS.len() * sites.len());
    for hypocenter in &TRAINING_HYPOCENTERS {
        for site in sites {
            let row = features::compute(hypocenter, site, slabs);
            rows.push(row.map(as_the_trees_see));
        }
    }
    rows
}

fn model(forest: Forest, objective: Objective) -> Model {
    let names: Vec<String> = FEATURE_NAMES.iter().map(ToString::to_string).collect();
    Model::new(forest, objective, names).expect("the model should be valid")
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * MILLISECONDS_PER_SECOND
}

/// Runs `case` a few times, then prints the median and the minimum of the
/// measured runs.
fn measure(name: &str, mut case: impl FnMut()) {
    for _ in 0..WARMUP_RUNS {
        case();
    }
    let mut durations = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        let start = Instant::now();
        case();
        durations.push(start.elapsed());
    }
    durations.sort();
    // The runs are even in number, so the median is the mean of the middle two.
    let median = (durations[MEASURED_RUNS / 2 - 1] + durations[MEASURED_RUNS / 2]) / 2;
    println!(
        "{name}: median {:.2} ms, min {:.2} ms ({MEASURED_RUNS} runs)",
        milliseconds(median),
        milliseconds(durations[0]),
    );
}

fn main() {
    match thread::available_parallelism() {
        Ok(threads) => println!("available parallelism: {threads}"),
        Err(error) => println!("available parallelism: unknown ({error})"),
    }

    let sites = sites();
    let slabs = slab_geometry();
    let rows: Vec<[f64; FEATURE_COUNT]> = sites
        .iter()
        .map(|site| features::compute(&HYPOCENTER, site, &slabs))
        .collect();
    let training = training_rows(&sites, &slabs);
    let mut random = SplitMix64::new(SEED);
    let classifier = forest(&mut random, &training);
    let regressor = forest(&mut random, &training);
    println!(
        "forests: 2 x {TREE_COUNT} trees of {LEAF_COUNT} leaves, sites: {}",
        sites.len()
    );

    measure("features", || {
        for site in hint::black_box(&sites) {
            hint::black_box(features::compute(&HYPOCENTER, site, &slabs));
        }
    });

    let flat_rows = rows.as_flattened();
    let mut raw = vec![0.0; sites.len()];
    measure("classifier raw_batch", || {
        classifier.raw_batch(hint::black_box(flat_rows), &mut raw);
        hint::black_box(&raw);
    });

    let predictor = Predictor::new(
        model(classifier, Objective::Binary),
        model(regressor, Objective::Regression),
        slabs,
        HURDLE,
    )
    .expect("the predictor should be valid");
    let one_thread = Options::default().with_max_threads(NonZeroUsize::MIN);
    let mut results = vec![PointResult::default(); sites.len()];
    for (name, options) in [
        ("predict (1 thread)", one_thread),
        ("predict (auto)", Options::default()),
    ] {
        measure(name, || {
            predictor
                .predict_into(&HYPOCENTER, hint::black_box(&sites), &options, &mut results)
                .expect("the hypocenter should be valid");
            hint::black_box(&results);
        });
    }
}

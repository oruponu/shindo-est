//! Evaluation of `LightGBM` decision trees with numerical splits.

use std::fmt;

/// Bit of a decision type marking a categorical split.
const CATEGORICAL_MASK: u8 = 1;
/// Bit of a decision type sending missing values to the left child.
const DEFAULT_LEFT_MASK: u8 = 2;
/// Position of the missing type in a decision type.
const MISSING_TYPE_SHIFT: u8 = 2;
/// Bits of the missing type after shifting.
const MISSING_TYPE_MASK: u8 = 3;
/// Missing type of a node that treats zero as missing.
const MISSING_ZERO: u8 = 1;
/// Missing type of a node that treats NaN as missing.
const MISSING_NAN: u8 = 2;
/// `LightGBM`'s `kZeroThreshold`: values this close to zero count as zero.
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// An error in the structure of a model.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum ModelError {
    /// A tree has no leaves.
    NoLeaves,
    /// The node arrays of a tree are not one shorter than its leaves.
    MismatchedNodeCount,
    /// A child points to an earlier node, to a node or leaf that does not
    /// exist, or to itself.
    InvalidChild,
    /// A split uses a feature the model does not have.
    FeatureOutOfRange {
        /// Index of the feature.
        feature: u32,
        /// Number of features of the model.
        feature_count: usize,
    },
    /// The model has no trees.
    NoTrees,
    /// The model has no features.
    NoFeatures,
    /// A split is categorical; only numerical splits are supported.
    Categorical,
    /// A tree is linear; only constant leaves are supported.
    Linear,
    /// The model builds more than one tree per iteration, as for several
    /// classes; only single-output models are supported.
    MultiClass,
    /// A value in a model file cannot be read or is invalid.
    InvalidValue {
        /// Line number, starting at 1.
        line: usize,
    },
    /// A key appears twice in the same part of a model file.
    RepeatedKey {
        /// Line number of the second appearance, starting at 1.
        line: usize,
    },
    /// A model file lacks a required key.
    MissingKey {
        /// Name of the key.
        key: &'static str,
    },
    /// A model file ends before its trees do.
    Truncated,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLeaves => f.write_str("a tree has no leaves"),
            Self::MismatchedNodeCount => {
                f.write_str("a tree does not have one node fewer than its leaves")
            }
            Self::InvalidChild => f.write_str("a tree has a child index out of order or range"),
            Self::FeatureOutOfRange {
                feature,
                feature_count,
            } => write!(
                f,
                "a split uses feature {feature}, but the model has {feature_count} features"
            ),
            Self::NoTrees => f.write_str("the model has no trees"),
            Self::NoFeatures => f.write_str("the model has no features"),
            Self::Categorical => f.write_str("categorical splits are not supported"),
            Self::Linear => f.write_str("linear trees are not supported"),
            Self::MultiClass => f.write_str("models with several outputs are not supported"),
            Self::InvalidValue { line } => {
                write!(f, "line {line}: the value cannot be read or is invalid")
            }
            Self::RepeatedKey { line } => write!(f, "line {line}: the key appears again"),
            Self::MissingKey { key } => write!(f, "the model has no {key}"),
            Self::Truncated => f.write_str("the model ends before \"end of trees\""),
        }
    }
}

impl std::error::Error for ModelError {}

/// Where a split sends a row.
#[derive(Clone, Copy, Debug)]
enum Child {
    Node(usize),
    Leaf(usize),
}

/// An internal node of a tree.
#[derive(Clone, Debug)]
struct Node {
    feature: u32,
    threshold: f64,
    missing: u8,
    default_left: bool,
    left: Child,
    right: Child,
}

/// A decision tree whose splits are all numerical.
#[derive(Clone, Debug)]
pub struct Tree {
    nodes: Vec<Node>,
    leaf_values: Vec<f64>,
}

impl Tree {
    /// Builds a tree from `LightGBM`'s arrays. A tree with one leaf has empty
    /// node arrays. Children at or above zero are nodes, and a negative child
    /// `c` is leaf `!c`.
    ///
    /// # Errors
    ///
    /// Returns a [`ModelError`] when the tree has no leaves, the node arrays
    /// are not one shorter than `leaf_value`, a child does not point to a later
    /// node or an existing leaf, or a split is categorical.
    pub fn new(
        split_feature: &[u32],
        threshold: &[f64],
        decision_type: &[u8],
        left_child: &[i32],
        right_child: &[i32],
        leaf_value: &[f64],
    ) -> Result<Self, ModelError> {
        let node_count = leaf_value
            .len()
            .checked_sub(1)
            .ok_or(ModelError::NoLeaves)?;
        let lengths = [
            split_feature.len(),
            threshold.len(),
            decision_type.len(),
            left_child.len(),
            right_child.len(),
        ];
        if lengths.iter().any(|&length| length != node_count) {
            return Err(ModelError::MismatchedNodeCount);
        }

        let mut nodes = Vec::with_capacity(node_count);
        for index in 0..node_count {
            let decision = decision_type[index];
            if decision & CATEGORICAL_MASK != 0 {
                return Err(ModelError::Categorical);
            }
            let decode = |value: i32| child(value, index, node_count, leaf_value.len());
            nodes.push(Node {
                feature: split_feature[index],
                threshold: threshold[index],
                missing: (decision >> MISSING_TYPE_SHIFT) & MISSING_TYPE_MASK,
                default_left: decision & DEFAULT_LEFT_MASK != 0,
                left: decode(left_child[index])?,
                right: decode(right_child[index])?,
            });
        }
        Ok(Self {
            nodes,
            leaf_values: leaf_value.to_vec(),
        })
    }

    /// Returns the leaf that prepared features reach.
    fn leaf(&self, features: &[f64]) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            match next(node, features[node.feature as usize]) {
                Child::Node(child_index) => index = child_index,
                Child::Leaf(leaf) => return leaf,
            }
        }
    }
}

/// Decodes a child of node `index`. Nodes must come after their parent, which
/// also rules out cycles.
fn child(
    value: i32,
    index: usize,
    node_count: usize,
    leaf_count: usize,
) -> Result<Child, ModelError> {
    if let Ok(node) = usize::try_from(value)
        && node > index
        && node < node_count
    {
        return Ok(Child::Node(node));
    }
    if value < 0
        && let Ok(leaf) = usize::try_from(!value)
        && leaf < leaf_count
    {
        return Ok(Child::Leaf(leaf));
    }
    Err(ModelError::InvalidChild)
}

/// Returns the child a node sends a prepared value to, as in `LightGBM`'s
/// `Tree::NumericalDecision`.
fn next(node: &Node, value: f64) -> Child {
    let value = if value.is_nan() && node.missing != MISSING_NAN {
        0.0
    } else {
        value
    };
    let is_missing = (node.missing == MISSING_ZERO && value.abs() <= ZERO_THRESHOLD)
        || (node.missing == MISSING_NAN && value.is_nan());
    let goes_left = if is_missing {
        node.default_left
    } else {
        value <= node.threshold
    };
    if goes_left { node.left } else { node.right }
}

/// Rounds a feature to `f32` as in training, and reads values within
/// [`ZERO_THRESHOLD`] of zero as zero, as `LightGBM`'s prediction does. NaN
/// stays NaN.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the models were trained on f32 features, so the rounding is intended"
)]
fn prepare(value: f64) -> f64 {
    let rounded = f64::from(value as f32);
    if rounded.abs() <= ZERO_THRESHOLD {
        0.0
    } else {
        rounded
    }
}

/// Trees whose leaf values add up to a raw score.
#[derive(Clone, Debug)]
pub struct Forest {
    feature_count: usize,
    trees: Vec<Tree>,
}

impl Forest {
    /// Combines trees that take `feature_count` features.
    ///
    /// # Errors
    ///
    /// Returns a [`ModelError`] when there are no trees or no features, or a
    /// split uses a feature at or above `feature_count`.
    pub fn new(feature_count: usize, trees: Vec<Tree>) -> Result<Self, ModelError> {
        if trees.is_empty() {
            return Err(ModelError::NoTrees);
        }
        if feature_count == 0 {
            return Err(ModelError::NoFeatures);
        }
        for tree in &trees {
            for node in &tree.nodes {
                if node.feature as usize >= feature_count {
                    return Err(ModelError::FeatureOutOfRange {
                        feature: node.feature,
                        feature_count,
                    });
                }
            }
        }
        Ok(Self {
            feature_count,
            trees,
        })
    }

    /// Returns the number of features each row has.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Returns the number of trees.
    #[must_use]
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Returns the sum of the leaf values that one row reaches, adding the
    /// trees in order.
    ///
    /// # Panics
    ///
    /// Panics if `features` does not have [`Forest::feature_count`] values.
    #[must_use]
    pub fn raw(&self, features: &[f64]) -> f64 {
        let prepared = self.prepare_row(features);
        let mut sum = 0.0;
        for tree in &self.trees {
            sum += tree.leaf_values[tree.leaf(&prepared)];
        }
        sum
    }

    /// Writes into `out` the raw score of each row of `rows`, which holds
    /// `out.len()` rows of [`Forest::feature_count`] values one after another.
    /// Gives the same values as [`Forest::raw`] row by row, but visits each
    /// tree once for all rows.
    ///
    /// # Panics
    ///
    /// Panics if `rows` does not have `out.len()` rows.
    pub fn raw_batch(&self, rows: &[f64], out: &mut [f64]) {
        assert_eq!(
            Some(rows.len()),
            out.len().checked_mul(self.feature_count),
            "rows must hold {} features for each of the {} outputs",
            self.feature_count,
            out.len()
        );
        let prepared: Vec<f64> = rows.iter().copied().map(prepare).collect();
        out.fill(0.0);
        for tree in &self.trees {
            let rows_and_sums = prepared
                .chunks_exact(self.feature_count)
                .zip(out.iter_mut());
            for (row, sum) in rows_and_sums {
                *sum += tree.leaf_values[tree.leaf(row)];
            }
        }
    }

    /// Returns the leaf that one row reaches in each tree, for comparing with
    /// `LightGBM`.
    ///
    /// # Panics
    ///
    /// Panics if `features` does not have [`Forest::feature_count`] values.
    #[must_use]
    pub fn leaf_indices(&self, features: &[f64]) -> Vec<usize> {
        let prepared = self.prepare_row(features);
        self.trees.iter().map(|tree| tree.leaf(&prepared)).collect()
    }

    /// Returns a prepared copy of one row of features.
    fn prepare_row(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(
            features.len(),
            self.feature_count,
            "a row must have {} features",
            self.feature_count
        );
        features.iter().copied().map(prepare).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Node 0 sends feature 0 at or below 1.5 to leaf 0, and 0 (and missing
    // values, read as 0) to the right. Node 1 sends feature 1 at or below 0 to
    // leaf 1, and NaN to the right.
    fn two_trees() -> Forest {
        let first = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[4, 8],
            &[-1, -2],
            &[1, -3],
            &[10.0, 20.0, 30.0],
        )
        .expect("the first tree should be valid");
        let second =
            Tree::new(&[], &[], &[], &[], &[], &[0.5]).expect("a one-leaf tree should be valid");
        Forest::new(2, vec![first, second]).expect("the forest should be valid")
    }

    // One node on feature 0: at or below `threshold` gives 1, above gives 2.
    fn one_split(threshold: f64, decision_type: u8) -> Forest {
        let tree = Tree::new(
            &[0],
            &[threshold],
            &[decision_type],
            &[-1],
            &[-2],
            &[1.0, 2.0],
        )
        .expect("the tree should be valid");
        Forest::new(1, vec![tree]).expect("the forest should be valid")
    }

    #[test]
    fn counts_features_and_trees() {
        let forest = two_trees();

        assert_eq!(forest.feature_count(), 2);
        assert_eq!(forest.tree_count(), 2);
    }

    #[test]
    fn goes_left_when_at_or_below_threshold() {
        let forest = two_trees();

        assert_eq!(forest.raw(&[1.5, 5.0]), 10.5);
        assert_eq!(forest.raw(&[-1.0, 5.0]), 10.5);
        assert_eq!(forest.raw(&[2.0, 0.0]), 20.5);
        assert_eq!(forest.raw(&[2.0, 0.1]), 30.5);
    }

    #[test]
    fn zero_missing_sends_zero_and_nan_to_default_side() {
        let forest = two_trees();

        assert_eq!(forest.raw(&[0.0, -1.0]), 20.5);
        assert_eq!(forest.raw(&[f64::NAN, -1.0]), 20.5);
    }

    #[test]
    fn nan_missing_sends_nan_to_default_side() {
        assert_eq!(two_trees().raw(&[2.0, f64::NAN]), 30.5);
    }

    #[test]
    fn rounds_features_to_f32_first() {
        // 1.50000001 rounds to 1.5 in f32, so it goes left like 1.5.
        assert_eq!(two_trees().raw(&[1.500_000_01, 5.0]), 10.5);
    }

    #[test]
    fn threshold_neighbors_in_f32() {
        // 0.1 in f32 is 0.100000001..., above the f64 threshold 0.1.
        let tenth = one_split(0.1, 0);
        assert_eq!(tenth.raw(&[0.1]), 2.0);
        assert_eq!(tenth.raw(&[f64::from(0.1_f32)]), 2.0);
        assert_eq!(tenth.raw(&[f64::from(0.1_f32.next_down())]), 1.0);
        assert_eq!(tenth.raw(&[f64::from(0.1_f32.next_up())]), 2.0);

        let half = one_split(0.5, 0);
        assert_eq!(half.raw(&[0.5]), 1.0);
        assert_eq!(half.raw(&[f64::from(0.5_f32.next_up())]), 2.0);
    }

    #[test]
    fn missing_values_follow_the_default_side() {
        let zero_left = one_split(
            -1.0,
            (MISSING_ZERO << MISSING_TYPE_SHIFT) | DEFAULT_LEFT_MASK,
        );
        assert_eq!(zero_left.raw(&[0.0]), 1.0);
        assert_eq!(zero_left.raw(&[f64::NAN]), 1.0);
        assert_eq!(zero_left.raw(&[5.0]), 2.0);

        let nan_left = one_split(
            -1.0,
            (MISSING_NAN << MISSING_TYPE_SHIFT) | DEFAULT_LEFT_MASK,
        );
        assert_eq!(nan_left.raw(&[f64::NAN]), 1.0);
        // Zero is not missing here, and 0 is above -1.
        assert_eq!(nan_left.raw(&[0.0]), 2.0);
    }

    #[test]
    fn nan_reads_as_zero_without_nan_missing() {
        assert_eq!(one_split(0.5, 0).raw(&[f64::NAN]), 1.0);
        assert_eq!(one_split(-0.5, 0).raw(&[f64::NAN]), 2.0);
    }

    #[test]
    fn tiny_values_count_as_zero() {
        // 1e-36 is read as 0, so the zero-missing node 0 sends it right.
        assert_eq!(two_trees().raw(&[1e-36, -1.0]), 20.5);
        assert_eq!(two_trees().raw(&[-1e-36, -1.0]), 20.5);
        // Also at a node without a missing type: as 0 it is at or below 0.
        assert_eq!(one_split(0.0, 0).raw(&[1e-36]), 1.0);
    }

    #[test]
    fn zero_threshold_is_inclusive_whatever_the_missing_type() {
        let limit = 1e-35_f32;
        for decision_type in [0, MISSING_NAN << MISSING_TYPE_SHIFT] {
            let forest = one_split(0.0, decision_type);
            assert_eq!(
                forest.raw(&[f64::from(limit)]),
                1.0,
                "decision type {decision_type}"
            );
            assert_eq!(
                forest.raw(&[f64::from(-limit)]),
                1.0,
                "decision type {decision_type}"
            );
            assert_eq!(
                forest.raw(&[f64::from(limit.next_up())]),
                2.0,
                "decision type {decision_type}"
            );
        }
    }

    #[test]
    fn infinities_compare_like_numbers() {
        // 1e300 rounds to infinity in f32, which is above the largest f64.
        let below_max = one_split(f64::MAX, 0);
        assert_eq!(below_max.raw(&[1e300]), 2.0);
        assert_eq!(below_max.raw(&[-1e300]), 1.0);
        assert_eq!(below_max.raw(&[f64::INFINITY]), 2.0);
        assert_eq!(one_split(f64::INFINITY, 0).raw(&[f64::INFINITY]), 1.0);
    }

    #[test]
    fn batch_equals_one_by_one() {
        let forest = two_trees();
        let rows: Vec<[f64; 2]> = (0..300_u32)
            .map(|i| {
                let second = if i % 5 == 0 {
                    f64::NAN
                } else {
                    f64::from(i % 3) - 1.0
                };
                [f64::from(i % 7) * 0.5 - 1.0, second]
            })
            .collect();
        let mut batch = vec![0.0; rows.len()];

        forest.raw_batch(rows.as_flattened(), &mut batch);

        let rows_and_sums = rows.iter().zip(&batch).enumerate();
        for (index, (row, sum)) in rows_and_sums {
            assert_eq!(
                sum.to_bits(),
                forest.raw(row).to_bits(),
                "row {index}: {row:?}"
            );
        }
    }

    #[test]
    fn batch_adds_the_trees_in_order_into_a_cleared_output() {
        let leaf = |value: f64| {
            Tree::new(&[], &[], &[], &[], &[], &[value]).expect("a one-leaf tree should be valid")
        };
        // In order (1e16 - 1e16) + 1 is 1, but in reverse (1 - 1e16) + 1e16 is 0.
        let forest = Forest::new(1, vec![leaf(1e16), leaf(-1e16), leaf(1.0)])
            .expect("the forest should be valid");
        let mut out = [f64::NAN; 2];

        forest.raw_batch(&[0.0, 0.0], &mut out);

        assert_eq!(forest.raw(&[0.0]), 1.0);
        assert_eq!(out, [1.0, 1.0]);
        forest.raw_batch(&[], &mut []);
    }

    #[test]
    fn leaf_indices_name_the_reached_leaves() {
        assert_eq!(two_trees().leaf_indices(&[2.0, 0.0]), vec![1, 0]);
    }

    #[test]
    #[should_panic(expected = "features")]
    fn raw_rejects_a_wrong_feature_count() {
        let _ = two_trees().raw(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "features")]
    fn raw_batch_rejects_rows_not_matching_the_output() {
        two_trees().raw_batch(&[1.0, 2.0, 3.0], &mut [0.0; 2]);
    }

    #[test]
    #[should_panic(expected = "features")]
    fn raw_batch_rejects_lengths_whose_product_overflows() {
        // feature_count x 2 wraps to 0 in usize, matching the empty rows.
        let leaf =
            Tree::new(&[], &[], &[], &[], &[], &[0.5]).expect("a one-leaf tree should be valid");
        let forest =
            Forest::new(1 << (usize::BITS - 1), vec![leaf]).expect("the forest should be valid");

        forest.raw_batch(&[], &mut [0.0; 2]);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5],
            &[4, 8],
            &[-1, -2],
            &[1, -3],
            &[10.0, 20.0, 30.0],
        );

        assert_eq!(tree.unwrap_err(), ModelError::MismatchedNodeCount);
    }

    #[test]
    fn rejects_a_child_pointing_backwards() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[4, 8],
            &[-1, -2],
            &[0, -3],
            &[10.0, 20.0, 30.0],
        );

        assert_eq!(tree.unwrap_err(), ModelError::InvalidChild);
    }

    #[test]
    fn rejects_a_child_node_out_of_range() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[4, 8],
            &[-1, -2],
            &[2, -3],
            &[10.0, 20.0, 30.0],
        );

        assert_eq!(tree.unwrap_err(), ModelError::InvalidChild);
    }

    #[test]
    fn rejects_a_leaf_index_out_of_range() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[4, 8],
            &[-4, -2],
            &[1, -3],
            &[10.0, 20.0, 30.0],
        );

        assert_eq!(tree.unwrap_err(), ModelError::InvalidChild);
    }

    #[test]
    fn rejects_each_node_array_of_the_wrong_length() {
        for array in 0..5 {
            for length in [1, 3] {
                let mut features = vec![0_u32, 1];
                let mut thresholds = vec![1.5, 0.0];
                let mut decisions = vec![4_u8, 8];
                let mut lefts = vec![-1, -2];
                let mut rights = vec![1, -3];
                match array {
                    0 => features.resize(length, 0),
                    1 => thresholds.resize(length, 0.0),
                    2 => decisions.resize(length, 0),
                    3 => lefts.resize(length, -1),
                    _ => rights.resize(length, -1),
                }

                let tree = Tree::new(
                    &features,
                    &thresholds,
                    &decisions,
                    &lefts,
                    &rights,
                    &[10.0, 20.0, 30.0],
                );

                assert_eq!(
                    tree.unwrap_err(),
                    ModelError::MismatchedNodeCount,
                    "array {array} of length {length}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_children_on_either_side() {
        // (left children, right children) of the two-node tree, one bad child each.
        let cases = [
            ([-1, -2], [1, 0]),
            ([0, -2], [1, -3]),
            ([2, -2], [1, -3]),
            ([-1, -4], [1, -3]),
            ([-1, -2], [1, -4]),
            ([i32::MIN, -2], [1, -3]),
        ];

        for (lefts, rights) in cases {
            let tree = Tree::new(
                &[0, 1],
                &[1.5, 0.0],
                &[4, 8],
                &lefts,
                &rights,
                &[10.0, 20.0, 30.0],
            );

            assert_eq!(
                tree.unwrap_err(),
                ModelError::InvalidChild,
                "left {lefts:?}, right {rights:?}"
            );
        }
    }

    #[test]
    fn rejects_a_feature_out_of_range() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[4, 8],
            &[-1, -2],
            &[1, -3],
            &[10.0, 20.0, 30.0],
        )
        .expect("the tree should be valid");

        assert_eq!(
            Forest::new(1, vec![tree]).unwrap_err(),
            ModelError::FeatureOutOfRange {
                feature: 1,
                feature_count: 1
            }
        );
    }

    #[test]
    fn rejects_categorical_decision() {
        let tree = Tree::new(
            &[0, 1],
            &[1.5, 0.0],
            &[5, 8],
            &[-1, -2],
            &[1, -3],
            &[10.0, 20.0, 30.0],
        );

        assert_eq!(tree.unwrap_err(), ModelError::Categorical);
    }

    #[test]
    fn rejects_a_tree_without_leaves() {
        let tree = Tree::new(&[], &[], &[], &[], &[], &[]);

        assert_eq!(tree.unwrap_err(), ModelError::NoLeaves);
    }

    #[test]
    fn rejects_a_forest_without_trees_or_features() {
        let leaf =
            || Tree::new(&[], &[], &[], &[], &[], &[0.5]).expect("a one-leaf tree should be valid");

        assert_eq!(Forest::new(2, vec![]).unwrap_err(), ModelError::NoTrees);
        assert_eq!(
            Forest::new(0, vec![leaf()]).unwrap_err(),
            ModelError::NoFeatures
        );
    }
}

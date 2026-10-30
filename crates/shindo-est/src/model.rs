//! A model: decision trees together with what their raw score means and the
//! features they take.

use std::fmt;

use crate::forest::Forest;

/// How a model turns its raw score, the sum of its trees, into a prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Objective {
    /// A regression that predicts its raw score.
    Regression,
    /// A binary classifier that predicts the logistic function of its raw
    /// score as the probability of the positive class.
    Binary,
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regression => f.write_str("regression"),
            Self::Binary => f.write_str("binary classification"),
        }
    }
}

/// Decision trees with their objective and the names of their features.
#[derive(Clone, Debug)]
pub struct Model {
    forest: Forest,
    objective: Objective,
    feature_names: Vec<String>,
}

impl Model {
    /// Combines trees with their objective and one name per feature, in the
    /// order the trees take the features.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::FeatureNameCount`] when `feature_names` does not
    /// have one name for each feature of `forest`.
    pub fn new(
        forest: Forest,
        objective: Objective,
        feature_names: Vec<String>,
    ) -> Result<Self, ModelError> {
        if feature_names.len() != forest.feature_count() {
            return Err(ModelError::FeatureNameCount {
                names: feature_names.len(),
                feature_count: forest.feature_count(),
            });
        }
        Ok(Self {
            forest,
            objective,
            feature_names,
        })
    }

    /// Returns the trees.
    #[must_use]
    pub fn forest(&self) -> &Forest {
        &self.forest
    }

    /// Returns the trees, dropping the rest of the model.
    #[must_use]
    pub fn into_forest(self) -> Forest {
        self.forest
    }

    /// Returns how the model turns its raw score into a prediction.
    #[must_use]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Returns the names of the features, in the order the trees take them.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
}

/// An error in a model or in the settings that come with it.
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
    /// A split threshold is NaN.
    NanThreshold,
    /// A leaf value is NaN or infinite.
    NonFiniteLeafValue,
    /// A tree is linear; only constant leaves are supported.
    Linear,
    /// The model builds more than one tree per iteration, as for several
    /// classes; only single-output models are supported.
    MultiClass,
    /// The model predicts from the mean of its trees, as random forests do,
    /// while its raw score is their sum; only models that predict from the
    /// raw score are supported.
    AveragedOutput,
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
    /// A model's objective turns its raw score into a prediction in a way
    /// that is not supported.
    UnsupportedObjective {
        /// Line number, starting at 1.
        line: usize,
    },
    /// A model names a different number of features than its trees take.
    FeatureNameCount {
        /// Number of feature names.
        names: usize,
        /// Number of features the trees take.
        feature_count: usize,
    },
    /// A model has another objective than its part in a prediction needs.
    WrongObjective {
        /// Objective the part needs.
        expected: Objective,
        /// Objective of the model.
        found: Objective,
    },
    /// A model's features are not the ones predictions compute, in their
    /// order.
    WrongFeatures {
        /// Index of the first feature that differs.
        index: usize,
    },
    /// The hurdle settings fail [`HurdleParams::is_valid`].
    ///
    /// [`HurdleParams::is_valid`]: crate::hurdle::HurdleParams::is_valid
    InvalidHurdleParams,
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
            Self::NanThreshold => f.write_str("a split threshold is NaN"),
            Self::NonFiniteLeafValue => f.write_str("a leaf value is NaN or infinite"),
            Self::Linear => f.write_str("linear trees are not supported"),
            Self::MultiClass => f.write_str("models with several outputs are not supported"),
            Self::AveragedOutput => {
                f.write_str("models that average their trees are not supported")
            }
            Self::InvalidValue { line } => {
                write!(f, "line {line}: the value cannot be read or is invalid")
            }
            Self::RepeatedKey { line } => write!(f, "line {line}: the key appears again"),
            Self::MissingKey { key } => write!(f, "the model has no {key}"),
            Self::Truncated => f.write_str("the model ends before \"end of trees\""),
            Self::UnsupportedObjective { line } => {
                write!(f, "line {line}: the objective is not supported")
            }
            Self::FeatureNameCount {
                names,
                feature_count,
            } => write!(
                f,
                "the model names {names} features, but its trees take {feature_count}"
            ),
            Self::WrongObjective { expected, found } => write!(
                f,
                "the model is a {found} model, but a {expected} model is needed"
            ),
            Self::WrongFeatures { index } => write!(
                f,
                "feature {index} of the model is not the one predictions compute"
            ),
            Self::InvalidHurdleParams => f.write_str("the hurdle parameters are invalid"),
        }
    }
}

impl std::error::Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forest::Tree;

    fn one_leaf(feature_count: usize) -> Forest {
        let tree =
            Tree::new(&[], &[], &[], &[], &[], &[1.0]).expect("a one-leaf tree should be valid");
        Forest::new(feature_count, vec![tree]).expect("the forest should be valid")
    }

    fn names(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("f{index}")).collect()
    }

    #[test]
    fn keeps_the_forest_objective_and_names() {
        let model = Model::new(one_leaf(2), Objective::Binary, names(2))
            .expect("the model should be valid");

        assert_eq!(model.objective(), Objective::Binary);
        assert_eq!(model.feature_names(), ["f0", "f1"]);
        assert_eq!(model.forest().feature_count(), 2);
        assert_eq!(model.into_forest().raw(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn rejects_a_name_count_not_matching_the_features() {
        for count in [1, 3] {
            let error = Model::new(one_leaf(2), Objective::Regression, names(count)).unwrap_err();
            assert_eq!(
                error,
                ModelError::FeatureNameCount {
                    names: count,
                    feature_count: 2,
                }
            );
        }
    }

    #[test]
    fn objectives_name_themselves() {
        assert_eq!(Objective::Regression.to_string(), "regression");
        assert_eq!(Objective::Binary.to_string(), "binary classification");
    }
}

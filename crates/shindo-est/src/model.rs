//! Errors in a model or in the settings that come with it.

use std::fmt;

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
    /// A model does not take the number of features that predictions compute.
    WrongFeatureCount {
        /// Number of features of the model.
        feature_count: usize,
        /// Number of features that predictions compute.
        expected: usize,
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
            Self::WrongFeatureCount {
                feature_count,
                expected,
            } => write!(
                f,
                "the model has {feature_count} features, but predictions use {expected}"
            ),
            Self::InvalidHurdleParams => f.write_str("the hurdle parameters are invalid"),
        }
    }
}

impl std::error::Error for ModelError {}

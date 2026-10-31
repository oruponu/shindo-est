//! Model files: the models and the data they need, in one binary file.

use std::fmt;

use crate::model::ModelError;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the model file reader and writer come next")
)]
mod codec;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the model file reader and writer come next")
)]
mod forest_section;

/// An error in the bytes of a model file.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum FormatError {
    /// The bytes end before a value they promise.
    Truncated,
    /// Bytes follow where a part of the file should end.
    TrailingBytes,
    /// A count is larger than model files allow.
    LimitExceeded {
        /// What was counted.
        what: &'static str,
    },
    /// The code of a model's objective is not one model files have.
    InvalidObjective {
        /// The objective code read from the file.
        code: u8,
    },
    /// A model in the file is invalid.
    Model(ModelError),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("the model file ends too early"),
            Self::TrailingBytes => {
                f.write_str("the model file has extra bytes at the end of a part")
            }
            Self::LimitExceeded { what } => write!(f, "the model file has too many {what}"),
            Self::InvalidObjective { code } => {
                write!(f, "the model file has an unknown objective code {code}")
            }
            // The model error is the source, so it is not repeated here.
            Self::Model(_) => f.write_str("the model file has an invalid model"),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    #[test]
    fn model_errors_are_the_source() {
        let error = FormatError::Model(ModelError::NoLeaves);
        let source = error.source().expect("a model error should be the source");
        assert_eq!(source.to_string(), ModelError::NoLeaves.to_string());
        assert!(FormatError::Truncated.source().is_none(), "Truncated");
    }

    #[test]
    fn is_exported_at_the_crate_root() {
        let error: crate::FormatError = FormatError::Truncated;
        assert_eq!(error, FormatError::Truncated);
    }
}

//! Error messages: each error with the chain of its causes, and the file it
//! is about.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

/// Returns the message of `error` followed by those of its sources, joined by
/// ": ", as in "the model file has an invalid model: the hurdle parameters
/// are invalid".
pub(crate) fn describe(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// An error about one file, shown as "<path>: <error>".
#[derive(Debug)]
pub(crate) struct FileError {
    path: PathBuf,
    error: Box<dyn Error>,
}

impl FileError {
    pub(crate) fn new(path: &Path, error: Box<dyn Error>) -> Self {
        Self {
            path: path.to_path_buf(),
            error,
        }
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.error)
    }
}

impl Error for FileError {
    // The message already holds the wrapped error, so its causes come next.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.error.source()
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fmt;
    use std::path::Path;

    use shindo_est::{FormatError, ModelError};

    #[test]
    fn describe_follows_the_sources() {
        let error = FormatError::Model(ModelError::InvalidHurdleParams);

        assert_eq!(
            super::describe(&error),
            format!("{error}: {}", ModelError::InvalidHurdleParams)
        );
        assert_eq!(
            super::describe(&ModelError::InvalidHurdleParams),
            ModelError::InvalidHurdleParams.to_string()
        );
    }

    /// An error with a message and an optional cause, for chains of any depth.
    #[derive(Debug)]
    struct Layer {
        message: &'static str,
        cause: Option<Box<Layer>>,
    }

    impl fmt::Display for Layer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.message)
        }
    }

    impl Error for Layer {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            self.cause
                .as_deref()
                .map(|cause| cause as &(dyn Error + 'static))
        }
    }

    #[test]
    fn describe_follows_every_level() {
        let error = Layer {
            message: "a",
            cause: Some(Box::new(Layer {
                message: "b",
                cause: Some(Box::new(Layer {
                    message: "c",
                    cause: None,
                })),
            })),
        };

        assert_eq!(super::describe(&error), "a: b: c");
    }

    #[test]
    fn file_errors_name_the_file_before_the_error() {
        let error = super::FileError::new(
            Path::new("model.bin"),
            Box::new(FormatError::Model(ModelError::InvalidHurdleParams)),
        );

        assert_eq!(
            super::describe(&error),
            format!(
                "model.bin: {}: {}",
                FormatError::Model(ModelError::InvalidHurdleParams),
                ModelError::InvalidHurdleParams
            )
        );
    }
}

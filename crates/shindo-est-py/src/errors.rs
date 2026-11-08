//! The exceptions of the module, all subclasses of `ValueError`, and the
//! conversions of the core's errors to Python exceptions.

use std::error::Error;
use std::io;
use std::path::Path;

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use shindo_est::{FormatError, ReadError, SlabError};

pyo3::create_exception!(
    shindo_est,
    InputError,
    PyValueError,
    "Raised for an invalid hypocenter or offset."
);

pyo3::create_exception!(
    shindo_est,
    ModelFileError,
    PyValueError,
    "Raised when the bytes of a model file are not a valid model."
);

pyo3::create_exception!(
    shindo_est,
    SlabGridError,
    PyValueError,
    "Raised when a Slab2 CSV file is not a valid grid."
);

/// Returns the message of `error` followed by those of its sources, joined by
/// ": ", as the CLI shows errors.
fn describe(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// Returns the `InputError` for an invalid hypocenter or offset.
#[must_use]
pub(crate) fn from_input_error(error: shindo_est::InputError) -> PyErr {
    InputError::new_err(describe(&error))
}

/// Returns the `ModelFileError` for bytes that are not a valid model.
#[must_use]
pub(crate) fn from_format_error(error: &FormatError) -> PyErr {
    ModelFileError::new_err(describe(error))
}

/// Returns the exception for a failed read of the model file at `path`: an
/// `OSError` for a failed read, a `ModelFileError` for invalid bytes.
#[must_use]
pub(crate) fn from_read_error(path: &Path, error: ReadError) -> PyErr {
    match error {
        ReadError::Format(error) => {
            ModelFileError::new_err(format!("{}: {}", path.display(), describe(&error)))
        }
        ReadError::Io(error) => from_io_error(path, &error),
        // Any kind of error the core adds later is reported as a failed read.
        error => from_io_error(path, &io::Error::other(error)),
    }
}

/// Returns the `SlabGridError` for the invalid grid at `path`.
#[must_use]
pub(crate) fn from_slab_error(path: &Path, error: &SlabError) -> PyErr {
    SlabGridError::new_err(format!("{}: {}", path.display(), describe(error)))
}

/// Returns the subclass of `OSError` for the kind of `error`, such as
/// `FileNotFoundError`, with a message that starts with `path`.
#[must_use]
pub(crate) fn from_io_error(path: &Path, error: &io::Error) -> PyErr {
    let message = format!("{}: {}", path.display(), describe(error));
    PyErr::from(io::Error::new(error.kind(), message))
}

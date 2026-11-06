//! Checks of the arrays that a call takes together.

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Returns the length that the named arrays share.
///
/// Each element of `arrays` pairs the name of an array with its length.
///
/// # Errors
///
/// Returns a `ValueError` that names each array with its length when the
/// lengths differ, as in `the arrays must have the same length:
/// site_latitude 3, site_longitude 2, site_arv 3`.
pub(crate) fn common_length(arrays: &[(&str, usize)]) -> PyResult<usize> {
    let Some(&(_, length)) = arrays.first() else {
        return Ok(0);
    };
    if arrays.iter().all(|&(_, other)| other == length) {
        return Ok(length);
    }
    let lengths: Vec<String> = arrays
        .iter()
        .map(|(name, length)| format!("{name} {length}"))
        .collect();
    Err(PyValueError::new_err(format!(
        "the arrays must have the same length: {}",
        lengths.join(", ")
    )))
}

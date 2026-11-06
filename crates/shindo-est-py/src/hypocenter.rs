//! The hypocenter of a call to `Model.predict`.

use pyo3::types::PyFloat;
use pyo3::{PyResult, Python, pyclass, pymethods};

/// The hypocenter of an earthquake, an immutable value.
///
/// Args:
///     latitude: The latitude in degrees, from -90 to 90.
///     longitude: The longitude in degrees, from -180 to 180.
///     depth: The depth in km, at least 0, or NaN if unknown.
///     magnitude: The JMA magnitude (Mj), or NaN if unknown.
#[pyclass(frozen, module = "shindo_est")]
pub struct Hypocenter {
    /// The latitude in degrees.
    #[pyo3(get)]
    latitude: f64,
    /// The longitude in degrees.
    #[pyo3(get)]
    longitude: f64,
    /// The depth in km, or NaN if unknown.
    #[pyo3(get)]
    depth: f64,
    /// The JMA magnitude (Mj), or NaN if unknown.
    #[pyo3(get)]
    magnitude: f64,
}

#[pymethods]
impl Hypocenter {
    #[new]
    fn new(latitude: f64, longitude: f64, depth: f64, magnitude: f64) -> Self {
        Self {
            latitude,
            longitude,
            depth,
            magnitude,
        }
    }

    /// Returns the call that makes this hypocenter, with each value as
    /// Python's `repr` writes it.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        use pyo3::types::PyAnyMethods;

        Ok(format!(
            "Hypocenter(latitude={}, longitude={}, depth={}, magnitude={})",
            PyFloat::new(py, self.latitude).repr()?,
            PyFloat::new(py, self.longitude).repr()?,
            PyFloat::new(py, self.depth).repr()?,
            PyFloat::new(py, self.magnitude).repr()?
        ))
    }
}

impl Hypocenter {
    #[must_use]
    pub(crate) fn to_core(&self) -> shindo_est::Hypocenter {
        shindo_est::Hypocenter {
            latitude: self.latitude,
            longitude: self.longitude,
            depth: self.depth,
            magnitude: self.magnitude,
        }
    }
}

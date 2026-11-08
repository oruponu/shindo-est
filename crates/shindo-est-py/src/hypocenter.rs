//! The hypocenter of a call to `Model.predict`.

use std::hash::DefaultHasher;

use pyo3::types::PyFloat;
use pyo3::{PyResult, Python, pyclass, pymethods};

/// The hypocenter of an earthquake, an immutable value.
///
/// Hypocenters are equal when their values are equal as floats, so a NaN
/// value makes them unequal. Equal hypocenters hash alike, so they can be
/// dictionary keys, and a hypocenter can be pickled.
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

    fn __eq__(&self, other: &Self) -> bool {
        self.to_core() == other.to_core()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in [self.latitude, self.longitude, self.depth, self.magnitude] {
            // 0.0 and -0.0 are equal, so they must hash alike.
            let value = if value == 0.0 { 0.0 } else { value };
            value.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Returns the arguments of the constructor, so that `pickle` and
    /// `copy` can make this hypocenter again.
    fn __getnewargs__(&self) -> (f64, f64, f64, f64) {
        (self.latitude, self.longitude, self.depth, self.magnitude)
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

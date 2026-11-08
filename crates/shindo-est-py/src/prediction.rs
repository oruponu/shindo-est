//! The results of `Model.predict`.

use numpy::PyArray1;
use pyo3::types::{PyFloat, PyString};
use pyo3::{Py, PyResult, Python, pyclass, pymethods};
use shindo_est::PointResult;

use crate::intensity;

/// The prediction for each site of a call to `Model.predict`.
///
/// The attributes cannot be replaced. The arrays are new for each call and
/// belong to the caller, and an attribute returns the same array each time,
/// so changes to its elements stay in it. A prediction cannot be pickled;
/// to return one from another process, return its arrays.
#[pyclass(frozen, module = "shindo_est")]
pub struct Prediction {
    /// The ID of the model that made the prediction.
    #[pyo3(get)]
    model_id: String,
    /// The offset that the call added to the regressor's instrumental
    /// intensity.
    #[pyo3(get)]
    offset: f64,
    /// The status of each site, an int32 array of `STATUS_OK`,
    /// `STATUS_OUT_OF_SCOPE`, or `STATUS_INVALID_INPUT`.
    #[pyo3(get)]
    status: Py<PyArray1<i32>>,
    /// The JMA seismic intensity class of each site, an int32 array of 0 to
    /// 9, with -1 where the instrumental intensity is NaN.
    #[pyo3(get)]
    intensity_class: Py<PyArray1<i32>>,
    /// The instrumental intensity of each site after the hurdle rule, with
    /// the offset, a float64 array that is NaN for sites without a status of
    /// `STATUS_OK`.
    #[pyo3(get)]
    instrumental: Py<PyArray1<f64>>,
    /// The probability that each site is felt, with the offset, a float64
    /// array.
    #[pyo3(get)]
    probability: Py<PyArray1<f64>>,
    /// The regressor's instrumental intensity at each site before the
    /// offset, a float64 array for diagnosis.
    #[pyo3(get)]
    regression: Py<PyArray1<f64>>,
}

#[pymethods]
impl Prediction {
    /// Returns a summary with the model ID, the offset, and the number of
    /// sites.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        use numpy::PyUntypedArrayMethods;
        use pyo3::types::PyAnyMethods;

        Ok(format!(
            "Prediction(model_id={}, offset={}, sites={})",
            PyString::new(py, &self.model_id).repr()?,
            PyFloat::new(py, self.offset).repr()?,
            self.status.bind(py).len()
        ))
    }
}

impl Prediction {
    #[must_use]
    pub(crate) fn new(
        py: Python<'_>,
        model_id: &str,
        offset: f64,
        results: &[PointResult],
    ) -> Self {
        let mut status = Vec::with_capacity(results.len());
        let mut intensity_class = Vec::with_capacity(results.len());
        let mut instrumental = Vec::with_capacity(results.len());
        let mut probability = Vec::with_capacity(results.len());
        let mut regression = Vec::with_capacity(results.len());
        for result in results {
            status.push(result.status as i32);
            intensity_class.push(intensity::class_number(result.instrumental));
            instrumental.push(result.instrumental);
            probability.push(result.probability);
            regression.push(result.regression);
        }
        Self {
            model_id: model_id.to_owned(),
            offset,
            status: PyArray1::from_vec(py, status).unbind(),
            intensity_class: PyArray1::from_vec(py, intensity_class).unbind(),
            instrumental: PyArray1::from_vec(py, instrumental).unbind(),
            probability: PyArray1::from_vec(py, probability).unbind(),
            regression: PyArray1::from_vec(py, regression).unbind(),
        }
    }
}

//! The JMA seismic intensity classes and their labels.

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};
use pyo3::exceptions::PyOverflowError;
use pyo3::types::PyAny;
use pyo3::{Bound, PyResult, Python, pyfunction};
use shindo_est::IntensityClass;

/// The class of a NaN instrumental intensity, as the C ABI gives it.
const NO_INTENSITY_CLASS: i32 = -1;

/// The classes in the order of their numbers, 0 to 9.
const CLASSES: [IntensityClass; 10] = [
    IntensityClass::Zero,
    IntensityClass::One,
    IntensityClass::Two,
    IntensityClass::Three,
    IntensityClass::Four,
    IntensityClass::FiveLower,
    IntensityClass::FiveUpper,
    IntensityClass::SixLower,
    IntensityClass::SixUpper,
    IntensityClass::Seven,
];

/// Returns the JMA seismic intensity classes of instrumental intensities.
///
/// Each class includes its lower bound, and the values are compared as they
/// are, without rounding to one decimal place.
///
/// Args:
///     instrumental: A one-dimensional array of instrumental intensities.
///
/// Returns:
///     A new int32 array of the classes, 0 to 9, with -1 where the
///     intensity is NaN.
///
/// Raises:
///     `ValueError`: NumPy cannot convert an element to a float, as with a
///         string that is not a number.
///     `TypeError`: `instrumental` is not one-dimensional, or an element has
///         a type that NumPy cannot convert to a float.
#[pyfunction]
#[must_use]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 passes arguments by value"
)]
pub fn intensity_class<'py>(
    py: Python<'py>,
    instrumental: PyArrayLike1<'py, f64, AllowTypeChange>,
) -> Bound<'py, PyArray1<i32>> {
    let values = instrumental.as_array();
    let classes: Vec<i32> = values.iter().copied().map(class_number).collect();
    PyArray1::from_vec(py, classes)
}

/// Returns the class of an instrumental intensity, 0 to 9, or -1 for NaN.
#[must_use]
pub(crate) fn class_number(instrumental: f64) -> i32 {
    IntensityClass::from_instrumental(instrumental).map_or(NO_INTENSITY_CLASS, |class| class as i32)
}

/// Returns the label of a JMA seismic intensity class, such as "5-".
///
/// Args:
///     value: A class as an integer, such as an element of the array that
///         `intensity_class` returns.
///
/// Returns:
///     The label, or None if `value` is not a class from 0 to 9.
///
/// Raises:
///     `TypeError`: `value` is not an integer.
#[pyfunction]
pub fn intensity_label(value: &Bound<'_, PyAny>) -> PyResult<Option<&'static str>> {
    use pyo3::types::PyAnyMethods;

    match value.extract::<i64>() {
        Ok(number) => Ok(label(number)),
        // An integer that i64 cannot hold is not a class either.
        Err(error) if error.is_instance_of::<PyOverflowError>(value.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Returns the label of the class `value`, or `None` outside 0 to 9.
fn label(value: i64) -> Option<&'static str> {
    let index = usize::try_from(value).ok()?;
    CLASSES.get(index).map(|class| class.label())
}

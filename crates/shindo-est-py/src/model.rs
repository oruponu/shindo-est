//! A model file, opened for predictions.

use std::fs::File;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use numpy::{AllowTypeChange, PyArrayLike1};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyString};
use pyo3::{Bound, PyRef, PyResult, Python, pyclass, pymethods};
use shindo_est::{ModelBundle, Options, Site};

use crate::arrays;
use crate::errors;
use crate::hypocenter::Hypocenter;
use crate::prediction::Prediction;

/// A model file: the felt classifier, the intensity regressor, and the slab
/// geometry they take.
///
/// Open one with `Model.open` or `Model.from_bytes`. A model never changes,
/// and `predict` releases the GIL while it computes, so Python threads can
/// share one model, as with `concurrent.futures.ThreadPoolExecutor`. A model
/// cannot be pickled, so it cannot be passed to `multiprocessing`; open the
/// file in each process instead.
#[pyclass(frozen, module = "shindo_est")]
pub struct Model {
    bundle: ModelBundle,
}

#[pymethods]
impl Model {
    /// Reads a model file.
    ///
    /// Only the first 8 bytes are read from a file that is not a model file.
    ///
    /// Args:
    ///     path: The path of the file.
    ///
    /// Returns:
    ///     The model.
    ///
    /// Raises:
    ///     `OSError`: The file cannot be read; a subclass such as
    ///         `FileNotFoundError` tells why. The message starts with the
    ///         path.
    ///     `ModelFileError`: The file is not a valid model file. The message
    ///         starts with the path.
    #[staticmethod]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 passes arguments by value"
    )]
    fn open(path: PathBuf) -> PyResult<Self> {
        let file = File::open(&path).map_err(|error| errors::from_io_error(&path, &error))?;
        let bundle =
            ModelBundle::read(file).map_err(|error| errors::from_read_error(&path, error))?;
        Ok(Self { bundle })
    }

    /// Reads a model file from its bytes.
    ///
    /// Args:
    ///     data: The bytes of the file, as `bytes` or `bytearray`.
    ///
    /// Returns:
    ///     The model.
    ///
    /// Raises:
    ///     `ModelFileError`: The bytes are not a valid model file.
    #[staticmethod]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 passes arguments by value"
    )]
    fn from_bytes(data: PyBackedBytes) -> PyResult<Self> {
        let bundle =
            ModelBundle::from_bytes(&data).map_err(|error| errors::from_format_error(&error))?;
        Ok(Self { bundle })
    }

    /// The ID of the model, from its manifest.
    #[getter]
    fn model_id(&self) -> &str {
        self.bundle.manifest().model_id()
    }

    /// The manifest of the model in its canonical text form.
    #[getter]
    fn manifest(&self) -> String {
        self.bundle.manifest().to_string()
    }

    /// Returns a summary with the model ID.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        use pyo3::types::PyAnyMethods;

        let model_id = PyString::new(py, self.model_id()).repr()?;
        Ok(format!("Model(model_id={model_id})"))
    }

    /// Predicts the JMA instrumental seismic intensity at each site.
    ///
    /// The GIL is released while the sites are predicted, so other Python
    /// threads keep running and can use the same model.
    ///
    /// Args:
    ///     hypocenter: The hypocenter.
    ///     site_latitude: The latitude of each site in degrees, a
    ///         one-dimensional array.
    ///     site_longitude: The longitude of each site in degrees, a
    ///         one-dimensional array of the same length.
    ///     site_arv: The ARV of each site, a one-dimensional array of the
    ///         same length.
    ///     offset: The amount added to the regressor's instrumental
    ///         intensity, which also raises the felt probability.
    ///     max_threads: The largest number of threads the call uses, counting
    ///         the calling one, or None for the number of logical CPUs. An
    ///         integer too large for the platform sets no limit. A call uses
    ///         one thread for every 256 sites. The results do not depend on
    ///         it.
    ///
    /// Returns:
    ///     The `Prediction` for each site. A call out of scope, as with an
    ///     unknown depth or magnitude, returns `STATUS_OUT_OF_SCOPE` for its
    ///     valid sites instead of raising.
    ///
    /// Raises:
    ///     `InputError`: The hypocenter or the offset is invalid.
    ///     `ValueError`: The site arrays differ in length, or `max_threads`
    ///         is below 1.
    ///     `TypeError`: A site array is not one-dimensional, or `max_threads`
    ///         is not an integer.
    #[pyo3(signature = (
        hypocenter,
        site_latitude,
        site_longitude,
        site_arv,
        *,
        offset = 0.0,
        max_threads = None,
    ))]
    #[expect(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        reason = "PyO3 passes arguments by value, one for each Python parameter"
    )]
    #[expect(
        clippy::doc_markdown,
        reason = "the docstring is Python's, whose Args name parameters without backticks"
    )]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        hypocenter: PyRef<'py, Hypocenter>,
        site_latitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        site_longitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        site_arv: PyArrayLike1<'py, f64, AllowTypeChange>,
        offset: f64,
        max_threads: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Prediction> {
        let options = call_options(offset, max_threads)?;
        let latitudes = site_latitude.as_array();
        let longitudes = site_longitude.as_array();
        let arvs = site_arv.as_array();
        let length = arrays::common_length(&[
            ("site_latitude", latitudes.len()),
            ("site_longitude", longitudes.len()),
            ("site_arv", arvs.len()),
        ])?;
        let mut sites = Vec::with_capacity(length);
        let columns = latitudes.iter().zip(&longitudes).zip(&arvs);
        for ((&latitude, &longitude), &arv) in columns {
            sites.push(Site {
                latitude,
                longitude,
                arv,
            });
        }
        let hypocenter = hypocenter.to_core();
        let predictor = self.bundle.predictor();
        let results = py
            .detach(|| predictor.predict(&hypocenter, &sites, &options))
            .map_err(errors::from_input_error)?;
        Ok(Prediction::new(py, self.model_id(), offset, &results))
    }
}

/// Returns the options of a call with `offset` and at most `max_threads`
/// threads.
fn call_options(offset: f64, max_threads: Option<&Bound<'_, PyAny>>) -> PyResult<Options> {
    let options = Options::default().with_offset(offset);
    let Some(max_threads) = max_threads else {
        return Ok(options);
    };
    Ok(options.with_max_threads(thread_limit(max_threads)?))
}

/// Returns the number of threads that the Python integer `max_threads` allows,
/// with an integer beyond what `usize` holds setting no limit.
///
/// # Errors
///
/// Returns a `ValueError` when `max_threads` is below 1, and a `TypeError`
/// when it is not an integer.
fn thread_limit(max_threads: &Bound<'_, PyAny>) -> PyResult<NonZeroUsize> {
    use pyo3::types::PyAnyMethods;

    let below_one_error = || PyValueError::new_err("max_threads must be at least 1");
    match max_threads.extract::<usize>() {
        Ok(count) => NonZeroUsize::new(count).ok_or_else(below_one_error),
        // A negative integer, or one too large for usize, does not fit.
        Err(error) if error.is_instance_of::<PyOverflowError>(max_threads.py()) => {
            if max_threads.gt(0)? {
                Ok(NonZeroUsize::MAX)
            } else {
                Err(below_one_error())
            }
        }
        Err(error) => Err(error),
    }
}

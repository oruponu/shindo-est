//! Python bindings of shindo-est: the extension module `shindo_est`.
//!
//! The doc comments of the functions and classes that the module exports are
//! their Python docstrings, so they follow the Google Python style. The types
//! are in the hand-written stub `shindo_est.pyi`.

mod arrays;
mod errors;
mod hypocenter;
mod intensity;
mod model;
mod prediction;

use pyo3::pymodule;

/// Predicts JMA instrumental seismic intensities at sites for a given earthquake.
#[pymodule(name = "shindo_est")]
mod shindo_est {
    use ::shindo_est::PointStatus;
    use ::shindo_est::features;
    use pyo3::types::{PyModule, PyTuple};
    use pyo3::{Bound, PyResult};

    #[pymodule_export]
    use crate::errors::{InputError, ModelFileError, SlabGridError};
    #[pymodule_export]
    use crate::hypocenter::Hypocenter;
    #[pymodule_export]
    use crate::intensity::{intensity_class, intensity_label};
    #[pymodule_export]
    use crate::model::Model;
    #[pymodule_export]
    use crate::prediction::Prediction;

    /// The version of the features the models take.
    #[pymodule_export]
    const FEATURE_VERSION: u32 = features::FEATURE_VERSION;

    /// The status of a site that got an intensity.
    #[pymodule_export]
    const STATUS_OK: i32 = PointStatus::Ok as i32;

    /// The status of the sites of a call out of scope.
    #[pymodule_export]
    const STATUS_OUT_OF_SCOPE: i32 = PointStatus::OutOfScope as i32;

    /// The status of a site with an invalid latitude, longitude, or ARV.
    #[pymodule_export]
    const STATUS_INVALID_INPUT: i32 = PointStatus::InvalidInput as i32;

    /// Adds the names that cannot be constants.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        use pyo3::types::PyModuleMethods;

        module.add("__version__", env!("CARGO_PKG_VERSION"))?;
        let names = PyTuple::new(module.py(), features::FEATURE_NAMES)?;
        module.add("FEATURE_NAMES", names)?;
        Ok(())
    }
}

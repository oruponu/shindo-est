//! The exceptions of the module, all subclasses of `ValueError`.

use pyo3::exceptions::PyValueError;

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

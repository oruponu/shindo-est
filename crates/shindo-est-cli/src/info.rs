//! The `info` command: prints the manifest of a model file.

use std::error::Error;
use std::fs;
use std::path::Path;

use shindo_est::ModelBundle;

use crate::report::FileError;

/// Reads the model file at `model` and returns its manifest in canonical form.
pub(crate) fn run(model: &Path) -> Result<String, Box<dyn Error>> {
    Ok(load(model)?.manifest().to_string())
}

/// Reads the model file at `model`, naming the file in an error.
pub(crate) fn load(model: &Path) -> Result<ModelBundle, FileError> {
    let bytes = fs::read(model).map_err(|error| FileError::new(model, error.into()))?;
    ModelBundle::from_bytes(&bytes).map_err(|error| FileError::new(model, error.into()))
}

//! The `info` command: prints the manifest of a model file.

use std::error::Error;
use std::fs::File;
use std::path::Path;

use shindo_est::ModelBundle;

use crate::report::FileError;

/// Reads the model file at `model` and returns its manifest in canonical form.
pub(crate) fn run(model: &Path) -> Result<String, Box<dyn Error>> {
    Ok(load(model)?.manifest().to_string())
}

/// Reads the model file at `model`, naming the file in an error. Stops at the
/// first bytes of a file that does not start like a model file.
pub(crate) fn load(model: &Path) -> Result<ModelBundle, FileError> {
    let file = File::open(model).map_err(|error| FileError::new(model, error.into()))?;
    ModelBundle::read(file).map_err(|error| FileError::new(model, error.into()))
}

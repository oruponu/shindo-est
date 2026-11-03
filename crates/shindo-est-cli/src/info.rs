//! The `info` command: prints the manifest of a model file.

use std::error::Error;
use std::fs;
use std::path::Path;

use shindo_est::ModelBundle;

use crate::report::FileError;

/// Reads the model file at `model` and returns its manifest in canonical form.
pub(crate) fn run(model: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(model).map_err(|error| FileError::new(model, error.into()))?;
    let bundle =
        ModelBundle::from_bytes(&bytes).map_err(|error| FileError::new(model, error.into()))?;
    Ok(bundle.manifest().to_string())
}

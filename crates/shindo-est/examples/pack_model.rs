//! Packs a manifest, two `LightGBM` models in `LightGBM`'s text format, and
//! the Kuril and Izu-Bonin Slab2 grids into a model file.
//!
//! ```text
//! cargo run --release -p shindo-est --example pack_model -- \
//!     <manifest> <classifier> <regressor> <kuril> <izu_bonin> <out>
//! ```

use std::env;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use shindo_est::{Manifest, ModelBundle, SlabGeometry, SlabGrid, lightgbm_text};

/// The line printed when the arguments are not the six paths.
const USAGE: &str =
    "usage: pack_model <manifest> <classifier> <regressor> <kuril> <izu_bonin> <out>";

/// Exit code when the arguments are not the six paths.
const USAGE_EXIT_CODE: i32 = 2;

fn main() -> Result<(), Box<dyn Error>> {
    // From `args_os`, so that a path that is not Unicode still works.
    let paths: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    let [manifest, classifier, regressor, kuril, izu_bonin, out] = paths.as_slice() else {
        eprintln!("{USAGE}");
        process::exit(USAGE_EXIT_CODE);
    };
    let slabs = SlabGeometry::new(
        load(kuril, SlabGrid::parse)?,
        load(izu_bonin, SlabGrid::parse)?,
    );
    let bundle = ModelBundle::new(
        load(manifest, Manifest::parse)?,
        load(classifier, lightgbm_text::parse)?,
        load(regressor, lightgbm_text::parse)?,
        slabs,
    )?;
    let bytes = bundle.to_bytes();
    let shown = out.display();
    fs::write(out, &bytes).map_err(|error| format!("{shown}: {error}"))?;
    // Reads the file back, so that a file the library cannot read is
    // reported here rather than where it is used.
    let written = fs::read(out).map_err(|error| format!("{shown}: {error}"))?;
    ModelBundle::from_bytes(&written).map_err(|error| format!("{shown}: {error}"))?;
    println!(
        "wrote {shown}: {} bytes, model {}",
        bytes.len(),
        bundle.manifest().model_id()
    );
    Ok(())
}

/// Reads the text file at `path` and parses it with `parse`, naming the file
/// in an error.
fn load<T, E: fmt::Display>(
    path: &Path,
    parse: impl FnOnce(&str) -> Result<T, E>,
) -> Result<T, String> {
    let shown = path.display();
    let text = fs::read_to_string(path).map_err(|error| format!("{shown}: {error}"))?;
    parse(&text).map_err(|error| format!("{shown}: {error}"))
}

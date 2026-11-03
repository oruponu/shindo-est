//! The `pack` command: packs the text files of a model into a model file.

use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

use shindo_est::{Manifest, ModelBundle, SlabGeometry, SlabGrid, lightgbm_text};

use crate::args::PackPaths;
use crate::report::FileError;

/// Packs the files of `paths` into `paths.out`, and returns the line that
/// reports it.
///
/// The bytes are read back as a model file before they are written, so that
/// no file the library cannot read is left behind. They are written with
/// [`write_checked`], so that a failed write keeps a file that was there.
pub(crate) fn run(paths: &PackPaths) -> Result<String, Box<dyn Error>> {
    let slabs = SlabGeometry::new(
        load(&paths.kuril, SlabGrid::parse)?,
        load(&paths.izu_bonin, SlabGrid::parse)?,
    );
    let out = &paths.out;
    let bundle = ModelBundle::new(
        load(&paths.manifest, Manifest::parse)?,
        load(&paths.classifier, lightgbm_text::parse)?,
        load(&paths.regressor, lightgbm_text::parse)?,
        slabs,
    )
    .map_err(|error| FileError::new(out, error.into()))?;
    let bytes = bundle.to_bytes();
    ModelBundle::from_bytes(&bytes).map_err(|error| FileError::new(out, error.into()))?;
    write_checked(out, &bytes)?;
    Ok(format!(
        "wrote {}: {} bytes, model {}\n",
        out.display(),
        bytes.len(),
        bundle.manifest().model_id()
    ))
}

/// Writes `bytes` to `out` through a new file next to it, which replaces `out`
/// only after it has been read back and found equal to `bytes`. The new file
/// is removed if any step fails. Errors name `out`.
fn write_checked(out: &Path, bytes: &[u8]) -> Result<(), FileError> {
    let temporary = temporary_path(out)?;
    let written = write_and_compare(&temporary, bytes)
        .and_then(|()| fs::rename(&temporary, out))
        .map_err(|error| FileError::new(out, error.into()));
    if written.is_err() {
        // The new file may not exist, and a leftover one only clutters the
        // folder, so a failure to remove it is not reported.
        let _ = fs::remove_file(&temporary);
    }
    written
}

/// Returns the path of the new file that [`write_checked`] writes first: the
/// name of `out` with the process ID and ".tmp" after it, in the same folder,
/// so that moving it over `out` stays within one file system.
fn temporary_path(out: &Path) -> Result<PathBuf, FileError> {
    let Some(name) = out.file_name() else {
        return Err(FileError::new(out, "the path names no file".into()));
    };
    let mut temporary = OsString::from(name);
    temporary.push(format!(".{}.tmp", process::id()));
    Ok(out.with_file_name(temporary))
}

fn write_and_compare(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    if fs::read(path)? != bytes {
        return Err(io::Error::other(
            "the file read back differs from the bytes written",
        ));
    }
    Ok(())
}

/// Reads the text file at `path` and parses it with `parse`, naming the file
/// in an error.
fn load<T, E: Error + 'static>(
    path: &Path,
    parse: impl FnOnce(&str) -> Result<T, E>,
) -> Result<T, FileError> {
    let text = fs::read_to_string(path).map_err(|error| FileError::new(path, error.into()))?;
    parse(&text).map_err(|error| FileError::new(path, error.into()))
}

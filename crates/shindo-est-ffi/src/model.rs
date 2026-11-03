//! Model handles: opening a model file, the strings of a model, and closing
//! it.

use std::ffi::{CStr, CString, c_char};
use std::fs::File;
use std::io::ErrorKind;
use std::ptr;

use shindo_est::ModelBundle;
use shindo_est::bundle::MAGIC;

use crate::{
    SHINDO_EST_ERROR_INVALID_MODEL_FILE, SHINDO_EST_ERROR_INVALID_PATH, SHINDO_EST_ERROR_IO,
    SHINDO_EST_ERROR_NOT_A_MODEL_FILE, SHINDO_EST_ERROR_PANIC, SHINDO_EST_OK, codes, guard,
    pointer,
};

/// An open model. C callers see only pointers to it, which
/// `shindo_est_model_open` and `shindo_est_model_open_file` return and
/// `shindo_est_model_close` frees.
#[expect(non_camel_case_types, reason = "the name is the C name")]
pub struct shindo_est_model {
    bundle: ModelBundle,
    model_id: CString,
    manifest: CString,
}

impl shindo_est_model {
    /// Wraps a bundle with the strings the C ABI returns for it.
    fn new(bundle: ModelBundle) -> Result<Self, i32> {
        let manifest = bundle.manifest();
        // The manifest rules leave no NUL in either string, so this error
        // only guards against a broken rule.
        let model_id =
            CString::new(manifest.model_id()).map_err(|_| SHINDO_EST_ERROR_INVALID_MODEL_FILE)?;
        let manifest =
            CString::new(manifest.to_string()).map_err(|_| SHINDO_EST_ERROR_INVALID_MODEL_FILE)?;
        Ok(Self {
            bundle,
            model_id,
            manifest,
        })
    }

    /// Returns the model file the handle holds.
    #[must_use]
    #[expect(dead_code, reason = "shindo_est_predict will predict with it")]
    pub(crate) fn bundle(&self) -> &ModelBundle {
        &self.bundle
    }
}

/// Opens a model from the bytes of a model file.
fn open(bytes: &[u8]) -> Result<Box<shindo_est_model>, i32> {
    let bundle =
        ModelBundle::from_bytes(bytes).map_err(|error| codes::from_format_error(&error))?;
    shindo_est_model::new(bundle).map(Box::new)
}

/// Reads a model file, checking its magic before reading the rest so that a
/// large file of another kind is not read whole.
fn read_file(utf8_path: &CStr) -> Result<Vec<u8>, i32> {
    use std::io::Read;

    let path = utf8_path
        .to_str()
        .map_err(|_| SHINDO_EST_ERROR_INVALID_PATH)?;
    let mut file = File::open(path).map_err(|_| SHINDO_EST_ERROR_IO)?;
    let mut magic = [0; MAGIC.len()];
    match file.read_exact(&mut magic) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
            return Err(SHINDO_EST_ERROR_NOT_A_MODEL_FILE);
        }
        Err(_) => return Err(SHINDO_EST_ERROR_IO),
    }
    if magic != MAGIC {
        return Err(SHINDO_EST_ERROR_NOT_A_MODEL_FILE);
    }
    let mut bytes = magic.to_vec();
    file.read_to_end(&mut bytes)
        .map_err(|_| SHINDO_EST_ERROR_IO)?;
    Ok(bytes)
}

/// Checks `out`, writes NULL there, runs `open_model`, and writes the handle
/// it opens over the NULL. Returns the code of the first error, or
/// `SHINDO_EST_OK`.
///
/// # Safety
///
/// If `out` passes [`pointer::check`], it must be valid for writing one
/// pointer.
unsafe fn open_into(
    out: *mut *mut shindo_est_model,
    open_model: impl FnOnce() -> Result<Box<shindo_est_model>, i32>,
) -> i32 {
    if let Err(code) = pointer::check(out.cast_const(), 1) {
        return code;
    }
    // SAFETY: non-null and aligned: checked by `pointer::check` above. Valid
    // for writing: the caller's promise. Writing NULL first leaves NULL there
    // on every error, a panic included.
    unsafe { out.write(ptr::null_mut()) };
    match open_model() {
        Ok(model) => {
            // SAFETY: as for the write above.
            unsafe { out.write(Box::into_raw(model)) };
            SHINDO_EST_OK
        }
        Err(code) => code,
    }
}

/// Opens a model from the `length` bytes of a model file at `bytes`. The
/// library keeps no pointer to the bytes after the call, so the caller may
/// free them. On success, writes the handle to `*out` and returns
/// `SHINDO_EST_OK`. On any other code, writes NULL to `*out`, unless `out`
/// itself is NULL or misaligned. NULL `bytes` with `length` 0 reads as no
/// bytes.
///
/// # Safety
///
/// The library checks NULL pointers and alignment and returns an error for
/// them. Beyond that:
///
/// - If `bytes` is not NULL and `length` is not 0, `bytes` must point to
///   `length` bytes within one allocation (such as one array), valid for
///   reading, which nothing writes during the call.
/// - If `out` is not NULL and is aligned for a pointer, it must be valid for
///   writing one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shindo_est_model_open(
    bytes: *const u8,
    length: usize,
    out: *mut *mut shindo_est_model,
) -> i32 {
    guard::catch(SHINDO_EST_ERROR_PANIC, || {
        let open_bytes = || {
            // SAFETY: `length` bytes within one allocation, valid for reading
            // and not written during the call, if not NULL and `length` is not
            // 0: the caller's promise.
            let bytes = unsafe { pointer::slice(bytes, length) }?;
            open(bytes)
        };
        // SAFETY: `out` is valid for writing if it passes the check: the
        // caller's promise.
        unsafe { open_into(out, open_bytes) }
    })
}

/// Opens the model file at the NUL-terminated UTF-8 path `utf8_path`, after
/// checking that it starts like a model file. On success, writes the handle
/// to `*out` and returns `SHINDO_EST_OK`. On any other code, writes NULL to
/// `*out`, unless `out` itself is NULL or misaligned.
///
/// # Safety
///
/// The library checks NULL pointers and alignment and returns an error for
/// them. Beyond that:
///
/// - If `utf8_path` is not NULL, it must point to a NUL-terminated string
///   within one allocation (such as one array) and at most `PTRDIFF_MAX`
///   bytes with the NUL, which nothing writes during the call.
/// - If `out` is not NULL and is aligned for a pointer, it must be valid for
///   writing one pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shindo_est_model_open_file(
    utf8_path: *const c_char,
    out: *mut *mut shindo_est_model,
) -> i32 {
    guard::catch(SHINDO_EST_ERROR_PANIC, || {
        let open_path = || {
            // SAFETY: NUL-terminated within one allocation and `PTRDIFF_MAX`
            // (`isize::MAX`) bytes, valid for reading, and not written during
            // the call, if not NULL: the caller's promise.
            let path = unsafe { pointer::c_str(utf8_path) }?;
            open(&read_file(path)?)
        };
        // SAFETY: `out` is valid for writing if it passes the check: the
        // caller's promise.
        unsafe { open_into(out, open_path) }
    })
}

/// Closes a model and frees it. NULL does nothing.
///
/// # Safety
///
/// `model` must be NULL or a handle that `shindo_est_model_open` or
/// `shindo_est_model_open_file` returned and that is not closed yet. No other
/// thread may use the handle during or after the call, and the strings taken
/// from it become invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shindo_est_model_close(model: *mut shindo_est_model) {
    guard::catch((), || {
        if pointer::check(model.cast_const(), 1).is_err() {
            return;
        }
        // SAFETY: a handle from `Box::into_raw` in `open_into` that is not
        // closed yet and that nothing else uses: the caller's promise.
        drop(unsafe { Box::from_raw(model) });
    });
}

/// Returns the model ID of an open model, or NULL for NULL. The string
/// belongs to the handle and stays valid until the handle is closed.
///
/// # Safety
///
/// `model` must be NULL or an open handle.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn shindo_est_model_id(model: *const shindo_est_model) -> *const c_char {
    guard::catch(ptr::null(), || {
        // SAFETY: an open handle: the caller's promise.
        let model = unsafe { pointer::reference(model) };
        model.map_or(ptr::null(), |model| model.model_id.as_ptr())
    })
}

/// Returns the manifest of an open model, in its canonical text form, or NULL
/// for NULL. The string belongs to the handle and stays valid until the
/// handle is closed.
///
/// # Safety
///
/// `model` must be NULL or an open handle.
#[unsafe(no_mangle)]
#[must_use]
pub unsafe extern "C" fn shindo_est_model_manifest(
    model: *const shindo_est_model,
) -> *const c_char {
    guard::catch(ptr::null(), || {
        // SAFETY: an open handle: the caller's promise.
        let model = unsafe { pointer::reference(model) };
        model.map_or(ptr::null(), |model| model.manifest.as_ptr())
    })
}

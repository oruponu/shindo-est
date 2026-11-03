//! Calls the C ABI the way a C caller does, through raw pointers.

use std::env;
use std::ffi::{CStr, CString, c_char};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr::{self, NonNull};

use shindo_est::ModelBundle;
use shindo_est::bundle::MAGIC;
use shindo_est_ffi::{
    SHINDO_EST_ERROR_INVALID_MODEL_FILE, SHINDO_EST_ERROR_INVALID_PATH, SHINDO_EST_ERROR_IO,
    SHINDO_EST_ERROR_MISALIGNED_POINTER, SHINDO_EST_ERROR_NOT_A_MODEL_FILE,
    SHINDO_EST_ERROR_NULL_POINTER, SHINDO_EST_OK, shindo_est_model, shindo_est_model_close,
    shindo_est_model_id, shindo_est_model_manifest, shindo_est_model_open,
    shindo_est_model_open_file,
};

/// The synthetic model file of the core's tests.
const MODEL: &[u8] = include_bytes!("../../../tests/fixtures/bundle/model.bin");

/// Opens a model from `bytes` and returns the code and the written handle.
fn open(bytes: &[u8]) -> (i32, *mut shindo_est_model) {
    let mut model = ptr::null_mut();
    // SAFETY: `bytes` is valid for its length, and `model` for writing.
    let code = unsafe { shindo_est_model_open(bytes.as_ptr(), bytes.len(), &raw mut model) };
    (code, model)
}

/// Opens a model file at `path` and returns the code and the written handle.
fn open_file(path: &CStr) -> (i32, *mut shindo_est_model) {
    let mut model = ptr::null_mut();
    // SAFETY: `path` is NUL-terminated, and `model` is valid for writing.
    let code = unsafe { shindo_est_model_open_file(path.as_ptr(), &raw mut model) };
    (code, model)
}

fn close(model: *mut shindo_est_model) {
    // SAFETY: `model` is NULL or an open handle that nothing else uses.
    unsafe { shindo_est_model_close(model) };
}

/// Copies a string the library returned.
fn text(string: *const c_char) -> String {
    assert!(!string.is_null(), "the library returned NULL");
    // SAFETY: the library returns NUL-terminated strings that stay valid
    // while the model is open.
    let string = unsafe { CStr::from_ptr(string) };
    string.to_str().expect("the strings are UTF-8").to_owned()
}

fn model_id(model: *const shindo_est_model) -> *const c_char {
    // SAFETY: `model` is NULL or an open handle.
    unsafe { shindo_est_model_id(model) }
}

fn manifest(model: *const shindo_est_model) -> *const c_char {
    // SAFETY: `model` is NULL or an open handle.
    unsafe { shindo_est_model_manifest(model) }
}

/// Returns a pointer one byte past the start of `words`, which no pointer
/// type with an alignment above 1 accepts.
#[expect(
    clippy::cast_ptr_alignment,
    reason = "the test needs a misaligned pointer"
)]
fn misaligned(words: &mut [u64; 2]) -> *mut *mut shindo_est_model {
    words
        .as_mut_ptr()
        .cast::<u8>()
        .wrapping_add(1)
        .cast::<*mut shindo_est_model>()
}

#[test]
fn opens_the_fixture_from_bytes() {
    let bundle = ModelBundle::from_bytes(MODEL).expect("the fixture should open");

    let (code, model) = open(MODEL);

    assert_eq!(code, SHINDO_EST_OK);
    assert!(!model.is_null(), "{model:?}");
    assert_eq!(text(model_id(model)), "synthetic-fixture");
    assert_eq!(text(manifest(model)), bundle.manifest().to_string());
    close(model);
}

#[test]
fn open_needs_somewhere_to_write() {
    // SAFETY: `MODEL` is valid for its length; `out` is NULL, which the
    // library checks.
    let code = unsafe { shindo_est_model_open(MODEL.as_ptr(), MODEL.len(), ptr::null_mut()) };
    assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER);

    // SAFETY: the path is NUL-terminated; `out` is NULL, which the library
    // checks before it opens the path.
    let code = unsafe { shindo_est_model_open_file(c"model.bin".as_ptr(), ptr::null_mut()) };
    assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER);
}

#[test]
fn open_rejects_a_misaligned_out() {
    let mut words = [0_u64; 2];
    let out = misaligned(&mut words);
    // SAFETY: `MODEL` is valid for its length; the library checks the
    // alignment of `out` before it writes there.
    let code = unsafe { shindo_est_model_open(MODEL.as_ptr(), MODEL.len(), out) };
    assert_eq!(code, SHINDO_EST_ERROR_MISALIGNED_POINTER);
    assert_eq!(words, [0, 0]);

    let out = misaligned(&mut words);
    // SAFETY: the path is NUL-terminated; the library checks the alignment
    // of `out` before it opens the path or writes there.
    let code = unsafe { shindo_est_model_open_file(c"model.bin".as_ptr(), out) };
    assert_eq!(code, SHINDO_EST_ERROR_MISALIGNED_POINTER);
    assert_eq!(words, [0, 0]);
}

#[test]
fn null_bytes_are_empty_only_with_length_0() {
    for (length, expected) in [
        (0, SHINDO_EST_ERROR_NOT_A_MODEL_FILE),
        (5, SHINDO_EST_ERROR_NULL_POINTER),
    ] {
        let mut model: *mut shindo_est_model = NonNull::dangling().as_ptr();
        // SAFETY: the bytes are NULL, which the library checks, and `model`
        // is valid for writing.
        let code = unsafe { shindo_est_model_open(ptr::null(), length, &raw mut model) };
        assert_eq!(code, expected, "{length}");
        assert!(model.is_null(), "{length}: {model:?}");
    }
}

#[test]
fn failed_open_writes_null() {
    let mut model: *mut shindo_est_model = NonNull::dangling().as_ptr();
    let bytes = b"hello";
    // SAFETY: `bytes` is valid for its length, and `model` for writing.
    let code = unsafe { shindo_est_model_open(bytes.as_ptr(), bytes.len(), &raw mut model) };

    assert_eq!(code, SHINDO_EST_ERROR_NOT_A_MODEL_FILE);
    assert!(model.is_null(), "{model:?}");
}

#[test]
fn damaged_bytes_are_invalid_model_files() {
    let mut damaged = MODEL.to_vec();
    let last = damaged.len() - 1;
    damaged[last] ^= 0x01;

    for bytes in [damaged.as_slice(), MAGIC.as_slice()] {
        let (code, model) = open(bytes);
        assert_eq!(
            code,
            SHINDO_EST_ERROR_INVALID_MODEL_FILE,
            "{} bytes",
            bytes.len()
        );
        assert!(model.is_null(), "{} bytes: {model:?}", bytes.len());
    }
}

#[test]
fn close_accepts_null() {
    close(ptr::null_mut());
}

#[test]
fn strings_of_a_null_model_are_null() {
    let id = model_id(ptr::null());
    let contents = manifest(ptr::null());

    assert!(id.is_null(), "{id:?}");
    assert!(contents.is_null(), "{contents:?}");
}

#[test]
fn model_is_send_and_sync() {
    fn assert_send_and_sync<T: Send + Sync>() {}

    assert_send_and_sync::<shindo_est_model>();
}

/// A folder of its own for one test, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> Self {
        let path = env::temp_dir().join(format!("shindo-est-ffi-{}-{test}", process::id()));
        fs::create_dir_all(&path).expect("the temporary folder should be creatable");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// Writes `bytes` to the file `name` in the folder and returns its path.
    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).expect("the file should be writable");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A folder left behind is only clutter in the temporary folder.
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Returns `path` as a C string.
fn c_path(path: &Path) -> CString {
    let path = path.to_str().expect("the test paths are UTF-8");
    CString::new(path).expect("the test paths have no NUL")
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn opens_the_fixture_file() {
    let dir = TempDir::new("opens_the_fixture_file");
    let path = dir.file("model.bin", MODEL);

    let (code, model) = open_file(&c_path(&path));

    assert_eq!(code, SHINDO_EST_OK);
    assert_eq!(text(model_id(model)), "synthetic-fixture");
    close(model);
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn opens_a_path_with_japanese_characters() {
    let dir = TempDir::new("opens_a_path_with_japanese_characters");
    let path = dir.file("\u{30e2}\u{30c7}\u{30eb}.bin", MODEL);

    let (code, model) = open_file(&c_path(&path));

    assert_eq!(code, SHINDO_EST_OK, "{}", path.display());
    assert_eq!(text(model_id(model)), "synthetic-fixture");
    close(model);
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn rejects_a_path_that_is_not_utf8() {
    let (code, model) = open_file(c"\xff\xfe.bin");

    assert_eq!(code, SHINDO_EST_ERROR_INVALID_PATH);
    assert!(model.is_null(), "{model:?}");
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn reports_a_missing_file() {
    let dir = TempDir::new("reports_a_missing_file");

    let (code, model) = open_file(&c_path(&dir.path().join("missing.bin")));

    assert_eq!(code, SHINDO_EST_ERROR_IO);
    assert!(model.is_null(), "{model:?}");
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn reports_a_directory() {
    let dir = TempDir::new("reports_a_directory");

    let (code, model) = open_file(&c_path(dir.path()));

    assert_eq!(code, SHINDO_EST_ERROR_IO);
    assert!(model.is_null(), "{model:?}");
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn rejects_a_file_that_is_not_a_model() {
    let dir = TempDir::new("rejects_a_file_that_is_not_a_model");
    let zeros = vec![0; 1 << 20];
    let contents: [&[u8]; 3] = [b"", b"abc", &zeros];
    for (index, bytes) in contents.into_iter().enumerate() {
        let path = dir.file(&format!("{index}.bin"), bytes);

        let (code, model) = open_file(&c_path(&path));

        assert_eq!(
            code,
            SHINDO_EST_ERROR_NOT_A_MODEL_FILE,
            "{} bytes",
            bytes.len()
        );
        assert!(model.is_null(), "{} bytes: {model:?}", bytes.len());
    }
}

#[cfg(unix)]
#[test]
#[cfg_attr(miri, ignore = "Miri cannot open files")]
fn stops_at_the_magic_of_an_endless_file() {
    let (code, model) = open_file(c"/dev/zero");

    assert_eq!(code, SHINDO_EST_ERROR_NOT_A_MODEL_FILE);
    assert!(model.is_null(), "{model:?}");
}

//! Error codes, the statuses of results, and the messages of the codes.

use std::ffi::{CStr, c_char};

use shindo_est::validate::HypocenterField;
use shindo_est::{FormatError, InputError, PointStatus, ReadError};

use crate::guard;

/// Success.
pub const SHINDO_EST_OK: i32 = 0;
/// A pointer that must not be NULL is NULL.
pub const SHINDO_EST_ERROR_NULL_POINTER: i32 = 1;
/// A pointer is not aligned for the type it points to.
pub const SHINDO_EST_ERROR_MISALIGNED_POINTER: i32 = 2;
/// A length or a count is too large for the address space.
pub const SHINDO_EST_ERROR_TOO_LARGE: i32 = 3;
/// The sites and the results overlap in memory.
pub const SHINDO_EST_ERROR_OVERLAPPING_BUFFERS: i32 = 4;
/// `options->size` is not a multiple of 8 from 8 to 4096.
pub const SHINDO_EST_ERROR_INVALID_OPTIONS: i32 = 5;
/// An option this library does not know is set: `options->reserved`, or a
/// byte past the options this library knows, is not 0.
pub const SHINDO_EST_ERROR_UNSUPPORTED_OPTION: i32 = 6;
/// The latitude of the hypocenter is not within -90 to 90.
pub const SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE: i32 = 10;
/// The longitude of the hypocenter is not within -180 to 180.
pub const SHINDO_EST_ERROR_INVALID_HYPOCENTER_LONGITUDE: i32 = 11;
/// The depth of the hypocenter is negative or infinite.
pub const SHINDO_EST_ERROR_INVALID_HYPOCENTER_DEPTH: i32 = 12;
/// The magnitude of the hypocenter is positive infinity.
pub const SHINDO_EST_ERROR_INVALID_HYPOCENTER_MAGNITUDE: i32 = 13;
/// The offset is not finite.
pub const SHINDO_EST_ERROR_INVALID_OFFSET: i32 = 14;
/// An input is invalid in a way that has no code of its own.
pub const SHINDO_EST_ERROR_INVALID_INPUT: i32 = 15;
/// The path is not valid UTF-8.
pub const SHINDO_EST_ERROR_INVALID_PATH: i32 = 20;
/// The file cannot be opened or read.
pub const SHINDO_EST_ERROR_IO: i32 = 21;
/// The bytes are not a model file.
pub const SHINDO_EST_ERROR_NOT_A_MODEL_FILE: i32 = 30;
/// The model file has a layout version this library does not read.
pub const SHINDO_EST_ERROR_UNSUPPORTED_VERSION: i32 = 31;
/// The models take features of a version this library does not compute.
pub const SHINDO_EST_ERROR_UNSUPPORTED_FEATURE_VERSION: i32 = 32;
/// The model file is invalid in another way: a wrong checksum, missing or
/// extra bytes, or an invalid section, tree, grid, or manifest.
pub const SHINDO_EST_ERROR_INVALID_MODEL_FILE: i32 = 33;
/// The library panicked, which is a bug in the library.
pub const SHINDO_EST_ERROR_PANIC: i32 = 99;

/// The site got an intensity.
pub const SHINDO_EST_STATUS_OK: i32 = 0;
/// The call is out of scope, as with an unknown depth or magnitude, so the
/// site got no intensity.
pub const SHINDO_EST_STATUS_OUT_OF_SCOPE: i32 = 1;
/// The site has an invalid latitude, longitude, or ARV, so it got no
/// intensity.
pub const SHINDO_EST_STATUS_INVALID_INPUT: i32 = 2;

/// Returns the code of an error in the inputs of a prediction.
#[must_use]
pub(crate) fn from_input_error(error: InputError) -> i32 {
    match error {
        InputError::InvalidHypocenter(HypocenterField::Latitude) => {
            SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE
        }
        InputError::InvalidHypocenter(HypocenterField::Longitude) => {
            SHINDO_EST_ERROR_INVALID_HYPOCENTER_LONGITUDE
        }
        InputError::InvalidHypocenter(HypocenterField::Depth) => {
            SHINDO_EST_ERROR_INVALID_HYPOCENTER_DEPTH
        }
        InputError::InvalidHypocenter(HypocenterField::Magnitude) => {
            SHINDO_EST_ERROR_INVALID_HYPOCENTER_MAGNITUDE
        }
        InputError::InvalidOffset => SHINDO_EST_ERROR_INVALID_OFFSET,
        // The core may add errors; they share one code until they get theirs.
        _ => SHINDO_EST_ERROR_INVALID_INPUT,
    }
}

/// Returns the code of an error in a model file.
#[must_use]
pub(crate) fn from_format_error(error: &FormatError) -> i32 {
    match error {
        FormatError::NotAModelFile => SHINDO_EST_ERROR_NOT_A_MODEL_FILE,
        FormatError::UnsupportedVersion { .. } => SHINDO_EST_ERROR_UNSUPPORTED_VERSION,
        FormatError::FeatureVersion { .. } => SHINDO_EST_ERROR_UNSUPPORTED_FEATURE_VERSION,
        _ => SHINDO_EST_ERROR_INVALID_MODEL_FILE,
    }
}

/// Returns the code of an error from reading a model file.
#[must_use]
pub(crate) fn from_read_error(error: &ReadError) -> i32 {
    match error {
        ReadError::Format(error) => from_format_error(error),
        // Any kind of error the core adds later is also an I/O error.
        ReadError::Io(_) | _ => SHINDO_EST_ERROR_IO,
    }
}

/// Returns the status code of a result.
#[must_use]
pub(crate) fn from_status(status: PointStatus) -> i32 {
    match status {
        PointStatus::Ok => SHINDO_EST_STATUS_OK,
        PointStatus::OutOfScope => SHINDO_EST_STATUS_OUT_OF_SCOPE,
        // A status the core adds later also means the site got no intensity.
        _ => SHINDO_EST_STATUS_INVALID_INPUT,
    }
}

/// Returns the message of an error code.
#[must_use]
pub(crate) fn message(code: i32) -> &'static CStr {
    match code {
        SHINDO_EST_OK => c"success",
        SHINDO_EST_ERROR_NULL_POINTER => c"a pointer that must not be NULL is NULL",
        SHINDO_EST_ERROR_MISALIGNED_POINTER => {
            c"a pointer is not aligned for the type it points to"
        }
        SHINDO_EST_ERROR_TOO_LARGE => c"a length or a count is too large for the address space",
        SHINDO_EST_ERROR_OVERLAPPING_BUFFERS => c"the sites and the results overlap in memory",
        SHINDO_EST_ERROR_INVALID_OPTIONS => c"options->size is not a multiple of 8 from 8 to 4096",
        SHINDO_EST_ERROR_UNSUPPORTED_OPTION => c"an option this library does not know is set",
        SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE => {
            c"the hypocenter latitude must be within -90..=90"
        }
        SHINDO_EST_ERROR_INVALID_HYPOCENTER_LONGITUDE => {
            c"the hypocenter longitude must be within -180..=180"
        }
        SHINDO_EST_ERROR_INVALID_HYPOCENTER_DEPTH => {
            c"the hypocenter depth must be finite and at least 0, or NaN if unknown"
        }
        SHINDO_EST_ERROR_INVALID_HYPOCENTER_MAGNITUDE => {
            c"the hypocenter magnitude must not be positive infinity"
        }
        SHINDO_EST_ERROR_INVALID_OFFSET => c"the offset must be finite",
        SHINDO_EST_ERROR_INVALID_INPUT => c"an input is invalid",
        SHINDO_EST_ERROR_INVALID_PATH => c"the path is not valid UTF-8",
        SHINDO_EST_ERROR_IO => c"the file cannot be opened or read",
        SHINDO_EST_ERROR_NOT_A_MODEL_FILE => c"the bytes are not a model file",
        SHINDO_EST_ERROR_UNSUPPORTED_VERSION => {
            c"the model file has a layout version this library does not read"
        }
        SHINDO_EST_ERROR_UNSUPPORTED_FEATURE_VERSION => {
            c"the models take features of a version this library does not compute"
        }
        SHINDO_EST_ERROR_INVALID_MODEL_FILE => c"the model file is invalid",
        SHINDO_EST_ERROR_PANIC => c"the library panicked, which is a bug in the library",
        _ => c"unknown error code",
    }
}

/// Returns a static English message for an error code, or "unknown error code"
/// for a code the library does not know. Never returns NULL.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn shindo_est_error_message(code: i32) -> *const c_char {
    guard::catch(message(SHINDO_EST_ERROR_PANIC).as_ptr(), || {
        message(code).as_ptr()
    })
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    /// Every error code with the value the header gives it.
    const CODES: [(i32, i32); 20] = [
        (SHINDO_EST_OK, 0),
        (SHINDO_EST_ERROR_NULL_POINTER, 1),
        (SHINDO_EST_ERROR_MISALIGNED_POINTER, 2),
        (SHINDO_EST_ERROR_TOO_LARGE, 3),
        (SHINDO_EST_ERROR_OVERLAPPING_BUFFERS, 4),
        (SHINDO_EST_ERROR_INVALID_OPTIONS, 5),
        (SHINDO_EST_ERROR_UNSUPPORTED_OPTION, 6),
        (SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE, 10),
        (SHINDO_EST_ERROR_INVALID_HYPOCENTER_LONGITUDE, 11),
        (SHINDO_EST_ERROR_INVALID_HYPOCENTER_DEPTH, 12),
        (SHINDO_EST_ERROR_INVALID_HYPOCENTER_MAGNITUDE, 13),
        (SHINDO_EST_ERROR_INVALID_OFFSET, 14),
        (SHINDO_EST_ERROR_INVALID_INPUT, 15),
        (SHINDO_EST_ERROR_INVALID_PATH, 20),
        (SHINDO_EST_ERROR_IO, 21),
        (SHINDO_EST_ERROR_NOT_A_MODEL_FILE, 30),
        (SHINDO_EST_ERROR_UNSUPPORTED_VERSION, 31),
        (SHINDO_EST_ERROR_UNSUPPORTED_FEATURE_VERSION, 32),
        (SHINDO_EST_ERROR_INVALID_MODEL_FILE, 33),
        (SHINDO_EST_ERROR_PANIC, 99),
    ];

    #[test]
    fn error_codes_have_fixed_values() {
        for (code, value) in CODES {
            assert_eq!(code, value);
        }
    }

    #[test]
    fn statuses_have_fixed_values() {
        assert_eq!(SHINDO_EST_STATUS_OK, 0);
        assert_eq!(SHINDO_EST_STATUS_OUT_OF_SCOPE, 1);
        assert_eq!(SHINDO_EST_STATUS_INVALID_INPUT, 2);
    }

    #[test]
    fn every_code_has_its_own_ascii_message() {
        let mut messages = Vec::new();
        for (code, _) in CODES {
            let message = super::message(code).to_str().expect("messages are UTF-8");
            assert!(!message.is_empty(), "{code}");
            assert!(message.is_ascii(), "{code}: {message}");
            assert_ne!(message, "unknown error code", "{code}");
            assert!(!messages.contains(&message), "{code}: {message}");
            messages.push(message);
            assert_eq!(
                shindo_est_error_message(code),
                super::message(code).as_ptr(),
                "{code}"
            );
        }
        for code in [-1, 7, 100] {
            assert_eq!(super::message(code), c"unknown error code", "{code}");
        }
    }

    #[test]
    fn input_errors_map_to_codes() {
        let cases = [
            (
                InputError::InvalidHypocenter(HypocenterField::Latitude),
                SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE,
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Longitude),
                SHINDO_EST_ERROR_INVALID_HYPOCENTER_LONGITUDE,
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Depth),
                SHINDO_EST_ERROR_INVALID_HYPOCENTER_DEPTH,
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Magnitude),
                SHINDO_EST_ERROR_INVALID_HYPOCENTER_MAGNITUDE,
            ),
            (InputError::InvalidOffset, SHINDO_EST_ERROR_INVALID_OFFSET),
        ];
        for (error, code) in cases {
            assert_eq!(super::from_input_error(error), code, "{error:?}");
        }
    }

    #[test]
    fn format_errors_map_to_codes() {
        let cases = [
            (
                FormatError::NotAModelFile,
                SHINDO_EST_ERROR_NOT_A_MODEL_FILE,
            ),
            (
                FormatError::UnsupportedVersion { found: 2 },
                SHINDO_EST_ERROR_UNSUPPORTED_VERSION,
            ),
            (
                FormatError::FeatureVersion { found: 2 },
                SHINDO_EST_ERROR_UNSUPPORTED_FEATURE_VERSION,
            ),
            (
                FormatError::ChecksumMismatch,
                SHINDO_EST_ERROR_INVALID_MODEL_FILE,
            ),
            (FormatError::Truncated, SHINDO_EST_ERROR_INVALID_MODEL_FILE),
            (
                FormatError::InvalidGrid,
                SHINDO_EST_ERROR_INVALID_MODEL_FILE,
            ),
        ];
        for (error, code) in cases {
            assert_eq!(super::from_format_error(&error), code, "{error:?}");
        }
    }

    #[test]
    fn read_errors_map_to_codes() {
        let cases = [
            (
                ReadError::Io(io::Error::other("disk on fire")),
                SHINDO_EST_ERROR_IO,
            ),
            (
                ReadError::Format(FormatError::NotAModelFile),
                SHINDO_EST_ERROR_NOT_A_MODEL_FILE,
            ),
            (
                ReadError::Format(FormatError::ChecksumMismatch),
                SHINDO_EST_ERROR_INVALID_MODEL_FILE,
            ),
        ];
        for (error, code) in cases {
            assert_eq!(super::from_read_error(&error), code, "{error:?}");
        }
    }
}

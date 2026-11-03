//! Versions of the library and the JMA intensity classes.

use std::ffi::{CStr, c_char};
use std::ptr;

use shindo_est::IntensityClass;

use crate::guard;

/// Version of the ABI: the functions, the layouts of the structs, and the
/// meanings of the constants. It changes only when one of them changes in a
/// way that breaks callers; adding a function or a field of
/// `shindo_est_options` keeps it.
pub const SHINDO_EST_ABI_VERSION: u32 = 1;

/// Intensity class of a NaN instrumental intensity.
const NO_INTENSITY_CLASS: i32 = -1;

/// Version of the library.
static VERSION: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(version) => version,
        Err(_) => panic!("the package version contains a NUL"),
    };

/// Labels of the intensity classes, at the index of each class.
const LABELS: [&CStr; 10] = [
    c"0", c"1", c"2", c"3", c"4", c"5-", c"5+", c"6-", c"6+", c"7",
];

/// Returns the class of an instrumental intensity, 0 to 9, or -1 for NaN.
#[must_use]
pub(crate) fn intensity_class(instrumental: f64) -> i32 {
    IntensityClass::from_instrumental(instrumental).map_or(NO_INTENSITY_CLASS, |class| class as i32)
}

/// Returns the label of an intensity class, or `None` outside 0 to 9.
fn label(intensity_class: i32) -> Option<&'static CStr> {
    let index = usize::try_from(intensity_class).ok()?;
    LABELS.get(index).copied()
}

/// Returns `SHINDO_EST_ABI_VERSION` of the library, which can differ from the
/// one in the header the caller was built with.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn shindo_est_abi_version() -> u32 {
    guard::catch(SHINDO_EST_ABI_VERSION, || SHINDO_EST_ABI_VERSION)
}

/// Returns the version of the library, such as "0.1.0", as a static string.
/// A caller that relies on a function or an options field added in some
/// version checks this.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn shindo_est_version() -> *const c_char {
    guard::catch(VERSION.as_ptr(), || VERSION.as_ptr())
}

/// Returns the JMA intensity class of an instrumental intensity: 0 to 9 for
/// "0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", and "7", or -1 for NaN.
/// Each class includes its lower bound, and the value is not rounded first.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn shindo_est_intensity_class(instrumental: f64) -> i32 {
    guard::catch(NO_INTENSITY_CLASS, || intensity_class(instrumental))
}

/// Returns the static ASCII label of an intensity class 0 to 9, such as "5-",
/// or NULL for any other value.
#[unsafe(no_mangle)]
#[must_use]
pub extern "C" fn shindo_est_intensity_label(intensity_class: i32) -> *const c_char {
    guard::catch(ptr::null(), || {
        label(intensity_class).map_or(ptr::null(), CStr::as_ptr)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_1() {
        assert_eq!(shindo_est_abi_version(), 1);
        assert_eq!(shindo_est_abi_version(), SHINDO_EST_ABI_VERSION);
    }

    #[test]
    fn version_is_the_crate_version() {
        assert_eq!(VERSION.to_str(), Ok(env!("CARGO_PKG_VERSION")));
        assert_eq!(shindo_est_version(), VERSION.as_ptr());
    }

    #[test]
    fn intensity_class_matches_the_core() {
        let cases = [
            (f64::NAN, -1),
            (-1.0, 0),
            (0.49, 0),
            (0.5, 1),
            (4.5, 5),
            (4.99, 5),
            (5.0, 6),
            (6.5, 9),
            (7.2, 9),
        ];
        for (instrumental, class) in cases {
            assert_eq!(
                shindo_est_intensity_class(instrumental),
                class,
                "{instrumental}"
            );
        }
    }

    #[test]
    fn intensity_labels_cover_0_to_9() {
        let lower_bounds = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5];
        for instrumental in lower_bounds {
            let class =
                IntensityClass::from_instrumental(instrumental).expect("the bounds are not NaN");
            let expected = class.label();
            let text = super::label(class as i32).map(CStr::to_str);
            assert_eq!(text, Some(Ok(expected)), "{class:?}");
            assert_eq!(
                shindo_est_intensity_label(class as i32),
                super::label(class as i32).map_or(ptr::null(), CStr::as_ptr),
                "{class:?}"
            );
        }
        for class in [-1, 10] {
            assert_eq!(super::label(class), None, "{class}");
            assert!(shindo_est_intensity_label(class).is_null(), "{class}");
        }
    }
}

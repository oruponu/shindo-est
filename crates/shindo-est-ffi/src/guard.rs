//! Keeps panics from crossing the C ABI.
//!
//! A panic that reaches an `extern "C"` function aborts the process. Every
//! exported function runs its body through [`catch`], as defense in depth: the
//! library is written not to panic, and a panic that still happens is a bug,
//! reported to the caller as `SHINDO_EST_ERROR_PANIC` or a value of the same
//! meaning.

use std::mem;
use std::panic::{self, AssertUnwindSafe};

/// Runs `body` and returns its value, or `fallback` if it panics.
///
/// The bodies only read what they share with other calls, such as a model, so
/// a panic leaves nothing half-changed for a later call to see.
///
/// Dropping the panic payload runs code of the payload, which could panic
/// again, so the payload is dropped under the guard too. The payload of such a
/// second panic is leaked.
pub(crate) fn catch<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            if let Err(second) = panic::catch_unwind(AssertUnwindSafe(|| drop(payload))) {
                mem::forget(second);
            }
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use crate::SHINDO_EST_ERROR_PANIC;

    #[test]
    fn catch_returns_the_body_value() {
        assert_eq!(super::catch(SHINDO_EST_ERROR_PANIC, || 7), 7);
    }

    #[test]
    fn catch_turns_a_panic_into_the_fallback() {
        let code = super::catch(SHINDO_EST_ERROR_PANIC, || -> i32 {
            panic!("a panic inside the library");
        });

        assert_eq!(code, SHINDO_EST_ERROR_PANIC);
    }

    /// A panic payload that panics again when dropped.
    struct PanicsOnDrop;

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            panic!("a panic while dropping the payload");
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "the second panic's payload is leaked on purpose")]
    fn catch_survives_a_payload_that_panics_on_drop() {
        let code = super::catch(SHINDO_EST_ERROR_PANIC, || -> i32 {
            panic::panic_any(PanicsOnDrop);
        });

        assert_eq!(code, SHINDO_EST_ERROR_PANIC);
    }
}

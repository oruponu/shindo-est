//! Checks of the raw pointers that callers pass, and the conversions of them
//! to references.
//!
//! The library checks what it can see in an address: NULL, alignment, and a
//! length that does not fit in the address space. What it cannot see, that
//! the memory is valid and not changed meanwhile, is the caller's promise in
//! the `# Safety` section of each exported function.

use std::ffi::{CStr, c_char};
use std::mem::{self, MaybeUninit};
use std::slice;

use crate::{
    SHINDO_EST_ERROR_MISALIGNED_POINTER, SHINDO_EST_ERROR_NULL_POINTER, SHINDO_EST_ERROR_TOO_LARGE,
};

/// Checks that `pointer` can address `count` values of `T`, looking only at
/// the address: it creates no reference.
///
/// A count of 0 accepts any pointer. Otherwise the pointer must not be NULL,
/// must be aligned for `T`, and must leave room for the values within
/// `isize::MAX` bytes and without wrapping the address space.
pub(crate) fn check<T>(pointer: *const T, count: usize) -> Result<(), i32> {
    if count == 0 {
        return Ok(());
    }
    if pointer.is_null() {
        return Err(SHINDO_EST_ERROR_NULL_POINTER);
    }
    if !pointer.is_aligned() {
        return Err(SHINDO_EST_ERROR_MISALIGNED_POINTER);
    }
    let bytes = count
        .checked_mul(mem::size_of::<T>())
        .filter(|&bytes| isize::try_from(bytes).is_ok())
        .ok_or(SHINDO_EST_ERROR_TOO_LARGE)?;
    match pointer.addr().checked_add(bytes) {
        Some(_) => Ok(()),
        None => Err(SHINDO_EST_ERROR_TOO_LARGE),
    }
}

/// Returns the value at `pointer` after [`check`].
///
/// # Safety
///
/// If `pointer` passes [`check`], it must point to a value valid for reading
/// for `'call`, which nothing writes meanwhile.
pub(crate) unsafe fn reference<'call, T>(pointer: *const T) -> Result<&'call T, i32> {
    check(pointer, 1)?;
    // SAFETY: non-null and aligned: checked by `check` above. Valid for
    // reading and not written for 'call: the caller's promise.
    Ok(unsafe { &*pointer })
}

/// Returns the `count` values at `pointer` after [`check`]. A count of 0 gives
/// an empty slice whatever the pointer.
///
/// # Safety
///
/// If `count` is not 0 and `pointer` passes [`check`], it must point to
/// `count` values within one allocation, valid for reading for `'call`, which
/// nothing writes meanwhile.
pub(crate) unsafe fn slice<'call, T>(pointer: *const T, count: usize) -> Result<&'call [T], i32> {
    check(pointer, count)?;
    if count == 0 {
        return Ok(&[]);
    }
    // SAFETY: non-null, aligned, at most `isize::MAX` bytes, and not wrapping
    // the address space: checked by `check` above. Within one allocation,
    // valid for reading `count` values, and not written for 'call: the
    // caller's promise.
    Ok(unsafe { slice::from_raw_parts(pointer, count) })
}

/// Returns the room for `count` values at `pointer` after [`check`], for
/// writing: the values may be uninitialized, as in an array a C caller has not
/// filled, so the slice holds `MaybeUninit`. A count of 0 gives an empty slice
/// whatever the pointer.
///
/// # Safety
///
/// If `count` is not 0 and `pointer` passes [`check`], it must point to room
/// for `count` values within one allocation, valid for writing for `'call`,
/// which nothing else reads or writes meanwhile.
pub(crate) unsafe fn uninit_slice<'call, T>(
    pointer: *mut T,
    count: usize,
) -> Result<&'call mut [MaybeUninit<T>], i32> {
    check(pointer.cast_const(), count)?;
    if count == 0 {
        return Ok(&mut []);
    }
    // SAFETY: non-null, aligned, at most `isize::MAX` bytes, and not wrapping
    // the address space: checked by `check` above. Within one allocation,
    // valid for writing, and not used by anything else for 'call: the caller's
    // promise. `MaybeUninit` needs no initialized values.
    Ok(unsafe { slice::from_raw_parts_mut(pointer.cast::<MaybeUninit<T>>(), count) })
}

/// Returns whether the bytes of `a_count` values at `a` and of `b_count`
/// values at `b` overlap, comparing the addresses only. A count of 0 overlaps
/// nothing. The ranges are expected to pass [`check`]; a range that would
/// pass the end of the address space is cut at the end.
pub(crate) fn overlaps<A, B>(a: *const A, a_count: usize, b: *const B, b_count: usize) -> bool {
    if a_count == 0 || b_count == 0 {
        return false;
    }
    let a_start = a.addr();
    let b_start = b.addr();
    let a_end = a_start.saturating_add(a_count.saturating_mul(mem::size_of::<A>()));
    let b_end = b_start.saturating_add(b_count.saturating_mul(mem::size_of::<B>()));
    a_start < b_end && b_start < a_end
}

/// Returns the C string at `pointer`.
///
/// # Safety
///
/// If `pointer` is not NULL, it must point to a NUL-terminated string within
/// one allocation, at most `isize::MAX` bytes with the NUL, valid for reading
/// for `'call`, which nothing writes meanwhile.
pub(crate) unsafe fn c_str<'call>(pointer: *const c_char) -> Result<&'call CStr, i32> {
    if pointer.is_null() {
        return Err(SHINDO_EST_ERROR_NULL_POINTER);
    }
    // SAFETY: non-null: checked above. NUL-terminated within one allocation
    // and `isize::MAX` bytes, valid for reading, and not written for 'call:
    // the caller's promise.
    Ok(unsafe { CStr::from_ptr(pointer) })
}

#[cfg(test)]
mod tests {
    use std::ptr::{self, NonNull};

    use super::*;

    /// Returns a pointer one byte past the start of `words`.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "the test needs a misaligned pointer"
    )]
    fn misaligned(words: &[u64; 2]) -> *const f64 {
        words.as_ptr().cast::<u8>().wrapping_add(1).cast::<f64>()
    }

    #[test]
    fn empty_slices_ignore_the_pointer() {
        let pointers = [
            ptr::null(),
            NonNull::<f64>::dangling().as_ptr().cast_const(),
        ];
        for pointer in pointers {
            assert_eq!(super::check(pointer, 0), Ok(()), "{pointer:?}");
            // SAFETY: no value is read when the count is 0.
            let values = unsafe { super::slice(pointer, 0) };
            assert_eq!(values, Ok(&[][..]), "{pointer:?}");
        }
    }

    #[test]
    fn slice_rejects_null_misaligned_and_too_large() {
        let words = [0_u64; 2];
        let dangling = NonNull::<f64>::dangling().as_ptr().cast_const();
        let cases = [
            (ptr::null(), 1, SHINDO_EST_ERROR_NULL_POINTER),
            (misaligned(&words), 1, SHINDO_EST_ERROR_MISALIGNED_POINTER),
            // The byte count overflows usize.
            (dangling, usize::MAX / 8 + 1, SHINDO_EST_ERROR_TOO_LARGE),
            // The byte count fits in usize but not in isize, which slices need.
            (
                dangling,
                isize::MAX.unsigned_abs() / 8 + 1,
                SHINDO_EST_ERROR_TOO_LARGE,
            ),
        ];
        for (pointer, count, code) in cases {
            assert_eq!(
                super::check(pointer, count),
                Err(code),
                "{pointer:?}, {count}"
            );
            // SAFETY: the library rejects these before it reads anything.
            let values = unsafe { super::slice(pointer, count) };
            assert_eq!(values, Err(code), "{pointer:?}, {count}");
        }
    }

    #[test]
    fn check_rejects_ranges_that_wrap_the_address_space() {
        // The last 8 bytes of the address space, aligned for f64 everywhere.
        let pointer = ptr::without_provenance::<f64>(usize::MAX - 7);

        assert_eq!(super::check(pointer, 2), Err(SHINDO_EST_ERROR_TOO_LARGE));
    }

    #[test]
    fn c_str_rejects_null() {
        // SAFETY: the pointer is NULL, which the library checks.
        let string = unsafe { super::c_str(ptr::null()) };

        assert_eq!(string, Err(SHINDO_EST_ERROR_NULL_POINTER));
    }

    #[test]
    fn uninit_slice_checks_like_slice() {
        let mut words = [0_u64; 2];
        let misaligned = misaligned(&words).cast_mut();
        let cases = [
            (ptr::null_mut(), 1, Err(SHINDO_EST_ERROR_NULL_POINTER)),
            (misaligned, 1, Err(SHINDO_EST_ERROR_MISALIGNED_POINTER)),
            (ptr::null_mut(), 0, Ok(0)),
            (words.as_mut_ptr().cast::<f64>(), 2, Ok(2)),
        ];
        for (pointer, count, expected) in cases {
            // SAFETY: the library rejects the first two before it writes, the
            // third has no values, and the fourth is the two words.
            let values = unsafe { super::uninit_slice(pointer, count) };
            assert_eq!(
                values.map(|values| values.len()),
                expected,
                "{pointer:?}, {count}"
            );
        }
    }

    #[test]
    fn overlaps_compares_byte_ranges() {
        let words = [0_u64; 4];
        let base = words.as_ptr();
        let second = base.wrapping_add(1);
        let third = base.wrapping_add(2);
        // (a, a count, b, b count, overlap): words [0, 2) and [2, 4) touch
        // without overlapping, and bytes [0, 9) reach into word 1.
        let cases = [
            (base, 2, third, 2, false),
            (third, 2, base, 2, false),
            (base, 3, third, 2, true),
            (third, 2, base, 3, true),
            (base, 0, base, 2, false),
            (base, 2, base, 0, false),
            (second, 1, second, 1, true),
        ];
        for (a, a_count, b, b_count, expected) in cases {
            assert_eq!(
                super::overlaps(a, a_count, b, b_count),
                expected,
                "{a:?}, {a_count}, {b:?}, {b_count}"
            );
        }
        assert!(super::overlaps(base.cast::<u8>(), 9, second, 1), "{base:?}");
        assert!(
            !super::overlaps(base.cast::<u8>(), 8, second, 1),
            "{base:?}"
        );
    }
}

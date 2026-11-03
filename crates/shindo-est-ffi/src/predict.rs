//! Prediction through the C ABI.

use shindo_est::{Hypocenter, PointResult, Site, validate};

use crate::{
    SHINDO_EST_ERROR_OVERLAPPING_BUFFERS, SHINDO_EST_ERROR_PANIC, SHINDO_EST_OK, codes, guard,
    pointer, shindo_est_hypocenter, shindo_est_model, shindo_est_options, shindo_est_result,
    shindo_est_site, types,
};

/// Predicts the JMA intensity at `count` sites for one hypocenter, writing
/// one result per site to `out` in the order of `sites`. A NULL `options`
/// gives the defaults. With `count` 0, `sites` and `out` may be NULL.
///
/// Returns `SHINDO_EST_OK`, or the code of the first error found, checking in
/// this order: `model`, `hypocenter`, `sites`, `out`, an overlap of `sites`
/// and `out`, `options`, and then the values of the hypocenter and the
/// offset. On any error, `out` is left as it was. A site that is out of
/// scope or invalid is not an error: its result says so.
///
/// Any number of threads may call this with the same model at once.
///
/// # Safety
///
/// The library checks NULL pointers, alignment, and counts too large for the
/// address space, and returns an error for them. Beyond that:
///
/// - `model` must be NULL or an open handle, which no thread closes during
///   the call.
/// - If `hypocenter` is not NULL and is aligned, it must be valid for reading
///   one hypocenter.
/// - If `count` is not 0 and `sites` is not NULL and is aligned, `sites` must
///   point to `count` sites within one allocation (such as one array), valid
///   for reading, which nothing writes during the call.
/// - If `count` is not 0 and `out` is not NULL and is aligned, `out` must
///   point to `count` results within one allocation, valid for writing, which
///   nothing else reads or writes during the call.
/// - If `options` is not NULL and is aligned, it must point to at least 8
///   bytes, and to `options->size` bytes when that size is valid, within one
///   allocation, valid for reading, which nothing writes during the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn shindo_est_predict(
    model: *const shindo_est_model,
    hypocenter: *const shindo_est_hypocenter,
    sites: *const shindo_est_site,
    count: usize,
    options: *const shindo_est_options,
    out: *mut shindo_est_result,
) -> i32 {
    guard::catch(SHINDO_EST_ERROR_PANIC, || {
        // SAFETY: the caller's promise in the `# Safety` section above.
        let predicted = unsafe { predict(model, hypocenter, sites, count, options, out) };
        predicted.err().unwrap_or(SHINDO_EST_OK)
    })
}

/// Does what [`shindo_est_predict`] describes, returning the code of the
/// first error.
///
/// # Safety
///
/// As for [`shindo_est_predict`].
unsafe fn predict(
    model: *const shindo_est_model,
    hypocenter: *const shindo_est_hypocenter,
    sites: *const shindo_est_site,
    count: usize,
    options: *const shindo_est_options,
    out: *mut shindo_est_result,
) -> Result<(), i32> {
    // SAFETY: an open handle that no thread closes during the call: the
    // caller's promise.
    let model = unsafe { pointer::reference(model) }?;
    // SAFETY: valid for reading one hypocenter if it passes the check: the
    // caller's promise.
    let hypocenter = unsafe { pointer::reference(hypocenter) }?;
    let hypocenter = Hypocenter::from(*hypocenter);
    // No slice is made before the checks of both buffers and of their
    // overlap, since a slice and a mutable slice of the same bytes are
    // undefined behavior even if never used.
    pointer::check(sites, count)?;
    pointer::check(out.cast_const(), count)?;
    if pointer::overlaps(sites, count, out.cast_const(), count) {
        return Err(SHINDO_EST_ERROR_OVERLAPPING_BUFFERS);
    }
    // SAFETY: if not NULL and aligned, the first 8 bytes, and the first
    // `options->size` bytes if that size is valid, are readable and not written
    // during the call: the caller's promise.
    let options = unsafe { types::read_options(options) }?;
    // The core checks these again, but checking first keeps an invalid call
    // from copying the sites and allocating results for them.
    validate::check_call(&hypocenter, options.offset).map_err(codes::from_input_error)?;

    // SAFETY: `count` sites within one allocation, readable and not written
    // during the call: the caller's promise. The checks passed above.
    let sites = unsafe { pointer::slice(sites, count) }?;
    let sites: Vec<Site> = sites.iter().copied().map(Site::from).collect();
    let mut results = vec![PointResult::default(); count];
    model
        .bundle()
        .predictor()
        .predict_into(&hypocenter, &sites, &options, &mut results)
        .map_err(codes::from_input_error)?;

    // SAFETY: room for `count` results within one allocation, writable and
    // used by nothing else during the call: the caller's promise. The checks
    // passed above, and `out` does not overlap the sites.
    let out = unsafe { pointer::uninit_slice(out, count) }?;
    let slots = out.iter_mut().zip(&results);
    for (slot, result) in slots {
        slot.write(shindo_est_result::from(result));
    }
    Ok(())
}

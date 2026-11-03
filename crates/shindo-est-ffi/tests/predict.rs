//! Calls `shindo_est_predict` the way a C caller does, through raw pointers,
//! and compares its results with the core's.

use std::alloc::{self, Layout};
use std::mem::{self, MaybeUninit};
use std::num::NonZeroUsize;
use std::ptr::{self, NonNull};
use std::thread;

use shindo_est::{Hypocenter, IntensityClass, ModelBundle, Options, PointResult, Site};
use shindo_est_ffi::{
    SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE, SHINDO_EST_ERROR_INVALID_OFFSET,
    SHINDO_EST_ERROR_INVALID_OPTIONS, SHINDO_EST_ERROR_MISALIGNED_POINTER,
    SHINDO_EST_ERROR_NULL_POINTER, SHINDO_EST_ERROR_OVERLAPPING_BUFFERS,
    SHINDO_EST_ERROR_TOO_LARGE, SHINDO_EST_ERROR_UNSUPPORTED_OPTION, SHINDO_EST_OK,
    SHINDO_EST_STATUS_INVALID_INPUT, SHINDO_EST_STATUS_OUT_OF_SCOPE, shindo_est_hypocenter,
    shindo_est_model, shindo_est_model_close, shindo_est_model_open, shindo_est_options,
    shindo_est_predict, shindo_est_result, shindo_est_site,
};

/// The synthetic model file of the core's tests.
const MODEL: &[u8] = include_bytes!("../../../tests/fixtures/bundle/model.bin");

/// The hypocenter of the core's model file tests.
const HYPOCENTER: shindo_est_hypocenter = shindo_est_hypocenter {
    latitude: 35.02,
    longitude: 140.02,
    depth: 80.0,
    magnitude: 6.5,
};

/// Index in [`SITES`] of the site with an invalid ARV.
const INVALID_ARV_SITE: usize = 5;

/// The sites of the core's model file tests: over the Kuril slab above and
/// below the classifier's split, in a cell touching a node without a point,
/// over the Izu-Bonin slab, outside both grids, and with an invalid ARV.
const SITES: [shindo_est_site; 6] = [
    site(35.02, 140.02, 1.0),
    site(35.02, 140.17, 2.0),
    site(35.12, 140.12, 1.0),
    site(34.9, 140.1, 1.0),
    site(40.0, 135.0, 1.0),
    site(35.02, 140.02, 0.0),
];

/// A result no prediction writes, for checking that `out` stays unchanged.
const UNTOUCHED: shindo_est_result = shindo_est_result {
    status: 77,
    intensity_class: 77,
    instrumental: 77.0,
    probability: 77.0,
    regression: 77.0,
};

const fn site(latitude: f64, longitude: f64, arv: f64) -> shindo_est_site {
    shindo_est_site {
        latitude,
        longitude,
        arv,
    }
}

/// An open model handle, closed when dropped.
struct Model(NonNull<shindo_est_model>);

// SAFETY: the library lets any number of threads predict with one handle at
// once, and the handle is closed only when the last borrow has ended.
unsafe impl Sync for Model {}

impl Model {
    fn open() -> Self {
        let mut model = ptr::null_mut();
        // SAFETY: `MODEL` is valid for its length, and `model` for writing.
        let code = unsafe { shindo_est_model_open(MODEL.as_ptr(), MODEL.len(), &raw mut model) };
        assert_eq!(code, SHINDO_EST_OK);
        Self(NonNull::new(model).expect("a successful open writes the handle"))
    }

    fn as_ptr(&self) -> *const shindo_est_model {
        self.0.as_ptr().cast_const()
    }

    /// Predicts `sites` with `options`, and returns the code and the results,
    /// which start as [`UNTOUCHED`].
    fn predict(
        &self,
        hypocenter: &shindo_est_hypocenter,
        sites: &[shindo_est_site],
        options: *const shindo_est_options,
    ) -> (i32, Vec<shindo_est_result>) {
        let mut out = vec![UNTOUCHED; sites.len()];
        // SAFETY: the handle is open, `hypocenter` and `sites` are valid for
        // reading, `out` holds a result for each site, and `options` is NULL
        // or valid for reading its size.
        let code = unsafe {
            shindo_est_predict(
                self.as_ptr(),
                hypocenter,
                sites.as_ptr(),
                sites.len(),
                options,
                out.as_mut_ptr(),
            )
        };
        (code, out)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the handle is open and no borrow of it is left.
        unsafe { shindo_est_model_close(self.0.as_ptr()) };
    }
}

/// Returns what the core predicts for [`SITES`] with `options`.
fn core_results(hypocenter: &shindo_est_hypocenter, options: &Options) -> Vec<PointResult> {
    let bundle = ModelBundle::from_bytes(MODEL).expect("the fixture should open");
    let sites: Vec<Site> = SITES.into_iter().map(Site::from).collect();
    bundle
        .predictor()
        .predict(&Hypocenter::from(*hypocenter), &sites, options)
        .expect("the core should predict")
}

/// Asserts that each result is the core's, bit for bit, with its class.
fn assert_matches_core(results: &[shindo_est_result], core: &[PointResult]) {
    assert_eq!(results.len(), core.len());
    let pairs = results.iter().zip(core);
    for (index, (result, expected)) in pairs.enumerate() {
        let class = IntensityClass::from_instrumental(expected.instrumental)
            .map_or(-1, |class| class as i32);
        assert_eq!(result.status, expected.status as i32, "site {index}");
        assert_eq!(result.intensity_class, class, "site {index}");
        assert_eq!(
            result.instrumental.to_bits(),
            expected.instrumental.to_bits(),
            "site {index}"
        );
        assert_eq!(
            result.probability.to_bits(),
            expected.probability.to_bits(),
            "site {index}"
        );
        assert_eq!(
            result.regression.to_bits(),
            expected.regression.to_bits(),
            "site {index}"
        );
    }
}

/// The fields of each result, with the values as bits so that NaN compares
/// equal to itself.
fn bits(results: &[shindo_est_result]) -> Vec<[u64; 5]> {
    let mut fields = Vec::with_capacity(results.len());
    for result in results {
        fields.push([
            u64::from(result.status.cast_unsigned()),
            u64::from(result.intensity_class.cast_unsigned()),
            result.instrumental.to_bits(),
            result.probability.to_bits(),
            result.regression.to_bits(),
        ]);
    }
    fields
}

/// Asserts that `out` holds only [`UNTOUCHED`].
fn assert_untouched(out: &[shindo_est_result]) {
    for (index, result) in out.iter().enumerate() {
        assert_eq!(result, &UNTOUCHED, "result {index}");
    }
}

fn options_with(offset: f64, max_threads: u32) -> shindo_est_options {
    shindo_est_options {
        offset,
        max_threads,
        ..shindo_est_options::default()
    }
}

#[test]
fn predictions_match_the_core() {
    let model = Model::open();

    let (code, results) = model.predict(&HYPOCENTER, &SITES, ptr::null());

    assert_eq!(code, SHINDO_EST_OK);
    assert_matches_core(&results, &core_results(&HYPOCENTER, &Options::default()));
}

#[test]
fn offset_and_threads_match_the_core() {
    let model = Model::open();
    let expected = core_results(&HYPOCENTER, &Options::default().with_offset(0.2));

    for max_threads in [1, 0] {
        let options = options_with(0.2, max_threads);
        let (code, results) = model.predict(&HYPOCENTER, &SITES, &raw const options);

        assert_eq!(code, SHINDO_EST_OK, "{max_threads}");
        assert_matches_core(&results, &expected);
    }
}

#[test]
fn null_options_are_the_defaults() {
    let model = Model::open();
    let defaults = shindo_est_options::default();

    let (null_code, null_results) = model.predict(&HYPOCENTER, &SITES, ptr::null());
    let (code, results) = model.predict(&HYPOCENTER, &SITES, &raw const defaults);

    assert_eq!(null_code, SHINDO_EST_OK);
    assert_eq!(code, SHINDO_EST_OK);
    assert_matches_core(&results, &core_results(&HYPOCENTER, &Options::default()));
    assert_matches_core(
        &null_results,
        &core_results(&HYPOCENTER, &Options::default()),
    );
}

#[test]
fn out_of_scope_and_invalid_sites() {
    let model = Model::open();
    let unknown_depth = shindo_est_hypocenter {
        depth: f64::NAN,
        ..HYPOCENTER
    };

    let (code, results) = model.predict(&unknown_depth, &SITES, ptr::null());

    assert_eq!(code, SHINDO_EST_OK);
    for (index, result) in results.iter().enumerate() {
        let status = if index == INVALID_ARV_SITE {
            SHINDO_EST_STATUS_INVALID_INPUT
        } else {
            SHINDO_EST_STATUS_OUT_OF_SCOPE
        };
        assert_eq!(result.status, status, "site {index}");
        assert_eq!(result.intensity_class, -1, "site {index}");
        assert!(result.instrumental.is_nan(), "site {index}: {result:?}");
    }
}

#[test]
fn hypocenter_errors_leave_out_untouched() {
    let model = Model::open();
    let north_of_the_pole = shindo_est_hypocenter {
        latitude: 91.0,
        ..HYPOCENTER
    };
    let nan_offset = options_with(f64::NAN, 0);

    let (code, out) = model.predict(&north_of_the_pole, &SITES, ptr::null());
    assert_eq!(code, SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE);
    assert_untouched(&out);

    let (code, out) = model.predict(&HYPOCENTER, &SITES, &raw const nan_offset);
    assert_eq!(code, SHINDO_EST_ERROR_INVALID_OFFSET);
    assert_untouched(&out);
}

/// Calls `shindo_est_predict` with raw arguments and returns its code.
fn predict_raw(
    model: *const shindo_est_model,
    hypocenter: *const shindo_est_hypocenter,
    sites: *const shindo_est_site,
    count: usize,
    options: *const shindo_est_options,
    out: *mut shindo_est_result,
) -> i32 {
    // SAFETY: every caller passes pointers that the library either rejects
    // before use or that are valid for `count` values.
    unsafe { shindo_est_predict(model, hypocenter, sites, count, options, out) }
}

#[test]
fn out_may_be_uninitialized() {
    // The results start uninitialized, as in a C caller's
    // `shindo_est_result out[6];`. Miri reports any read of them before the
    // library writes them.
    let model = Model::open();
    let mut out: [MaybeUninit<shindo_est_result>; SITES.len()] =
        [MaybeUninit::uninit(); SITES.len()];

    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        SITES.as_ptr(),
        SITES.len(),
        ptr::null(),
        out.as_mut_ptr().cast(),
    );

    assert_eq!(code, SHINDO_EST_OK);
    let results = out.map(|result| {
        // SAFETY: a successful prediction writes every result.
        unsafe { result.assume_init() }
    });
    assert_matches_core(&results, &core_results(&HYPOCENTER, &Options::default()));
}

#[test]
fn zero_sites_may_be_null() {
    let model = Model::open();
    let north_of_the_pole = shindo_est_hypocenter {
        latitude: 91.0,
        ..HYPOCENTER
    };

    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        ptr::null(),
        0,
        ptr::null(),
        ptr::null_mut(),
    );
    assert_eq!(code, SHINDO_EST_OK);

    let code = predict_raw(
        model.as_ptr(),
        &raw const north_of_the_pole,
        ptr::null(),
        0,
        ptr::null(),
        ptr::null_mut(),
    );
    assert_eq!(code, SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE);
}

/// Returns a pointer one byte past the start of `words`, which no type with an
/// alignment above 1 accepts.
#[expect(
    clippy::cast_ptr_alignment,
    reason = "the test needs a misaligned pointer"
)]
fn misaligned(words: &mut [u64; 8]) -> *mut shindo_est_result {
    words
        .as_mut_ptr()
        .cast::<u8>()
        .wrapping_add(1)
        .cast::<shindo_est_result>()
}

#[test]
fn pointer_errors_come_in_order() {
    let model = Model::open();
    let model_ptr = model.as_ptr();
    let hypocenter: *const shindo_est_hypocenter = &HYPOCENTER;
    let first_site: *const shindo_est_site = &SITES[0];
    let mut out = [UNTOUCHED];
    let out_ptr = out.as_mut_ptr();

    let cases = [
        ("model", ptr::null(), hypocenter, first_site, out_ptr),
        ("hypocenter", model_ptr, ptr::null(), first_site, out_ptr),
        ("sites", model_ptr, hypocenter, ptr::null(), out_ptr),
        ("out", model_ptr, hypocenter, first_site, ptr::null_mut()),
    ];
    for (name, model, hypocenter, sites, out) in cases {
        let code = predict_raw(model, hypocenter, sites, 1, ptr::null(), out);
        assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER, "{name}");
    }
    assert_untouched(&out);

    let mut words = [0_u64; 8];
    let code = predict_raw(
        model_ptr,
        hypocenter,
        first_site,
        1,
        ptr::null(),
        misaligned(&mut words),
    );
    assert_eq!(code, SHINDO_EST_ERROR_MISALIGNED_POINTER);
    assert_eq!(words, [0; 8]);
}

#[test]
fn errors_are_checked_in_order() {
    let model = Model::open();
    let north_of_the_pole = shindo_est_hypocenter {
        latitude: 91.0,
        ..HYPOCENTER
    };
    let bad_size = shindo_est_options {
        size: 7,
        ..shindo_est_options::default()
    };
    let mut words = [0_u64; 8];

    // A NULL hypocenter comes before a misaligned `out`.
    let code = predict_raw(
        model.as_ptr(),
        ptr::null(),
        &SITES[0],
        1,
        ptr::null(),
        misaligned(&mut words),
    );
    assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER);

    // NULL sites come before a misaligned `out`.
    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        ptr::null(),
        1,
        ptr::null(),
        misaligned(&mut words),
    );
    assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER);
    assert_eq!(words, [0; 8]);

    // A misaligned `out` comes before invalid options.
    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        &SITES[0],
        1,
        &raw const bad_size,
        misaligned(&mut words),
    );
    assert_eq!(code, SHINDO_EST_ERROR_MISALIGNED_POINTER);
    assert_eq!(words, [0; 8]);

    // A NULL `out` comes before an invalid hypocenter.
    let code = predict_raw(
        model.as_ptr(),
        &raw const north_of_the_pole,
        &SITES[0],
        1,
        ptr::null(),
        ptr::null_mut(),
    );
    assert_eq!(code, SHINDO_EST_ERROR_NULL_POINTER);

    // Invalid options come before an invalid hypocenter.
    let mut out = [UNTOUCHED];
    let code = predict_raw(
        model.as_ptr(),
        &raw const north_of_the_pole,
        &SITES[0],
        1,
        &raw const bad_size,
        out.as_mut_ptr(),
    );
    assert_eq!(code, SHINDO_EST_ERROR_INVALID_OPTIONS);
    assert_untouched(&out);
}

#[test]
fn too_many_sites_are_rejected_before_reading() {
    let model = Model::open();

    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        NonNull::<shindo_est_site>::dangling().as_ptr(),
        usize::MAX / 16,
        ptr::null(),
        NonNull::<shindo_est_result>::dangling().as_ptr(),
    );

    assert_eq!(code, SHINDO_EST_ERROR_TOO_LARGE);
}

#[test]
fn overlapping_buffers_are_rejected() {
    let model = Model::open();
    // Room for 3 results (96 bytes) after 2 words, over 3 sites (72 bytes).
    let mut words = [0_u64; 16];
    let base = words.as_mut_ptr();
    let sites = base.cast::<shindo_est_site>().cast_const();
    let out = base.wrapping_add(2).cast::<shindo_est_result>();
    let bad_size = shindo_est_options {
        size: 7,
        ..shindo_est_options::default()
    };

    let code = predict_raw(model.as_ptr(), &HYPOCENTER, sites, 3, ptr::null(), out);
    assert_eq!(code, SHINDO_EST_ERROR_OVERLAPPING_BUFFERS);

    // The overlap comes before invalid options.
    let code = predict_raw(
        model.as_ptr(),
        &HYPOCENTER,
        sites,
        3,
        &raw const bad_size,
        out,
    );
    assert_eq!(code, SHINDO_EST_ERROR_OVERLAPPING_BUFFERS);
    assert_eq!(words, [0; 16]);
}

/// Options in a zeroed allocation of exactly `size` bytes, as a caller built
/// against another version of the header passes them.
struct RawOptions {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl RawOptions {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, mem::align_of::<shindo_est_options>())
            .expect("the layout should be valid");
        // SAFETY: not zero-sized: every caller passes at least the 8 bytes of the
        // size field.
        let pointer = unsafe { alloc::alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer).expect("the allocation should succeed");
        let options = Self { pointer, layout };
        options.write(mem::offset_of!(shindo_est_options, size), size as u64);
        options
    }

    /// Writes `value` at `offset` bytes into the allocation.
    fn write<T>(&self, offset: usize, value: T) {
        assert!(
            offset + mem::size_of::<T>() <= self.layout.size(),
            "{offset} is past the allocation of {} bytes",
            self.layout.size()
        );
        let field = self.pointer.as_ptr().wrapping_add(offset).cast::<T>();
        // SAFETY: within the allocation: checked above. Aligned: the
        // allocation is aligned for the options, and the offsets are those of
        // their fields, or of single bytes.
        unsafe { field.write(value) };
    }

    fn as_ptr(&self) -> *const shindo_est_options {
        self.pointer.as_ptr().cast_const().cast()
    }
}

impl Drop for RawOptions {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new` and not freed since.
        unsafe { alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[test]
fn options_smaller_than_the_struct_use_defaults_for_the_rest() {
    let model = Model::open();

    let older = RawOptions::new(mem::offset_of!(shindo_est_options, max_threads));
    older.write(mem::offset_of!(shindo_est_options, offset), 0.2_f64);
    let (code, results) = model.predict(&HYPOCENTER, &SITES, older.as_ptr());
    assert_eq!(code, SHINDO_EST_OK);
    assert_matches_core(
        &results,
        &core_results(&HYPOCENTER, &Options::default().with_offset(0.2)),
    );

    let oldest = RawOptions::new(mem::size_of::<u64>());
    let (code, results) = model.predict(&HYPOCENTER, &SITES, oldest.as_ptr());
    assert_eq!(code, SHINDO_EST_OK);
    assert_matches_core(&results, &core_results(&HYPOCENTER, &Options::default()));
}

/// Options of 64 bytes, as a caller built against a newer header passes them:
/// offset 0.2, one thread, and the rest 0.
fn newer_options() -> RawOptions {
    let newer = RawOptions::new(64);
    newer.write(mem::offset_of!(shindo_est_options, offset), 0.2_f64);
    newer.write(mem::offset_of!(shindo_est_options, max_threads), 1_u32);
    newer
}

#[test]
fn options_larger_than_the_struct_accept_a_zero_tail() {
    let model = Model::open();
    let expected = core_results(
        &HYPOCENTER,
        &Options::default()
            .with_offset(0.2)
            .with_max_threads(NonZeroUsize::MIN),
    );

    let (code, results) = model.predict(&HYPOCENTER, &SITES, newer_options().as_ptr());

    assert_eq!(code, SHINDO_EST_OK);
    assert_matches_core(&results, &expected);
}

#[test]
fn unknown_options_are_rejected() {
    let model = Model::open();

    for byte in [mem::size_of::<shindo_est_options>(), 63] {
        let newer = newer_options();
        newer.write(byte, 1_u8);

        let (code, out) = model.predict(&HYPOCENTER, &SITES, newer.as_ptr());

        assert_eq!(code, SHINDO_EST_ERROR_UNSUPPORTED_OPTION, "byte {byte}");
        assert_untouched(&out);
    }

    let reserved = shindo_est_options {
        reserved: 1,
        ..shindo_est_options::default()
    };
    let (code, out) = model.predict(&HYPOCENTER, &SITES, &raw const reserved);
    assert_eq!(code, SHINDO_EST_ERROR_UNSUPPORTED_OPTION);
    assert_untouched(&out);
}

#[test]
fn invalid_options_sizes_are_rejected() {
    let model = Model::open();

    // The library rejects these sizes before it reads past the size field.
    for size in [0, 7, 12, 4104] {
        let options = shindo_est_options {
            size,
            ..shindo_est_options::default()
        };

        let (code, out) = model.predict(&HYPOCENTER, &SITES, &raw const options);

        assert_eq!(code, SHINDO_EST_ERROR_INVALID_OPTIONS, "{size}");
        assert_untouched(&out);
    }
}

#[test]
fn one_model_serves_many_threads() {
    let model = Model::open();
    let (code, expected) = model.predict(&HYPOCENTER, &SITES, ptr::null());
    assert_eq!(code, SHINDO_EST_OK);
    let rounds = if cfg!(miri) { 2 } else { 50 };

    thread::scope(|threads| {
        for _ in 0..4 {
            threads.spawn(|| {
                for _ in 0..rounds {
                    let (code, results) = model.predict(&HYPOCENTER, &SITES, ptr::null());
                    assert_eq!(code, SHINDO_EST_OK);
                    assert_eq!(bits(&results), bits(&expected));
                }
            });
        }
    });
}

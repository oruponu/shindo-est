//! The structs of the C ABI, and their conversions to and from the core.
//!
//! The names are the C names in the header, so they are not in Rust's case.

use std::mem;

use shindo_est::{Hypocenter, PointResult, Site};

use crate::{codes, info};

/// An earthquake source.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct shindo_est_hypocenter {
    /// Latitude in degrees, -90 to 90.
    pub latitude: f64,
    /// Longitude in degrees, -180 to 180.
    pub longitude: f64,
    /// Depth in km, 0 or more. NaN means the depth is unknown.
    pub depth: f64,
    /// JMA magnitude (Mj). NaN means the magnitude is unknown.
    pub magnitude: f64,
}

impl From<shindo_est_hypocenter> for Hypocenter {
    fn from(hypocenter: shindo_est_hypocenter) -> Self {
        Self {
            latitude: hypocenter.latitude,
            longitude: hypocenter.longitude,
            depth: hypocenter.depth,
            magnitude: hypocenter.magnitude,
        }
    }
}

/// A place where the intensity is predicted.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct shindo_est_site {
    /// Latitude in degrees, -90 to 90.
    pub latitude: f64,
    /// Longitude in degrees, -180 to 180.
    pub longitude: f64,
    /// Amplification of peak ground velocity relative to engineering bedrock.
    /// Must be finite and positive.
    pub arv: f64,
}

impl From<shindo_est_site> for Site {
    fn from(site: shindo_est_site) -> Self {
        Self {
            latitude: site.latitude,
            longitude: site.longitude,
            arv: site.arv,
        }
    }
}

/// Options of a prediction. Every field but `size` is 0 by default, so a C
/// caller writes `shindo_est_options options = { .size = sizeof options };`
/// and then sets the fields it needs. A library too old to know a field rejects a
/// nonzero value in it with `SHINDO_EST_ERROR_UNSUPPORTED_OPTION` rather than
/// ignoring it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct shindo_est_options {
    /// `sizeof(shindo_est_options)` in the header the caller was built with: a
    /// multiple of 8 from 8 to 4096. The library reads only the fields that
    /// fit in this size and takes the defaults for the others.
    pub size: u64,
    /// Amount added to the regressor's instrumental intensity. The classifier's
    /// logit moves the same way, by the model's logit per intensity unit times
    /// this amount. Default 0.
    pub offset: f64,
    /// Largest number of threads a prediction uses, counting the calling
    /// thread, or 0 for the number of logical CPUs. Default 0. The results do
    /// not depend on it.
    pub max_threads: u32,
    /// Must be 0. It keeps the struct free of padding, and a later version may
    /// give it a meaning.
    pub reserved: u32,
}

/// The options with every field 0 but `size`, which are the defaults.
impl Default for shindo_est_options {
    fn default() -> Self {
        Self {
            size: mem::size_of::<Self>() as u64,
            offset: 0.0,
            max_threads: 0,
            reserved: 0,
        }
    }
}

/// The prediction for one site.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct shindo_est_result {
    /// One of the `SHINDO_EST_STATUS_` codes. The values are NaN unless it is
    /// `SHINDO_EST_STATUS_OK`.
    pub status: i32,
    /// JMA intensity class of `instrumental`, 0 to 9 as
    /// `shindo_est_intensity_class` returns it, or -1 when `instrumental` is
    /// NaN.
    pub intensity_class: i32,
    /// The instrumental intensity after the hurdle rule, with the offset.
    pub instrumental: f64,
    /// The probability that the site is felt, with the offset applied.
    pub probability: f64,
    /// The regressor's instrumental intensity before the offset, for
    /// diagnosis.
    pub regression: f64,
}

/// Converts the status to a `SHINDO_EST_STATUS_` code, copies the values, and
/// adds the intensity class.
impl From<&PointResult> for shindo_est_result {
    fn from(result: &PointResult) -> Self {
        Self {
            status: codes::from_status(result.status),
            intensity_class: info::intensity_class(result.instrumental),
            instrumental: result.instrumental,
            probability: result.probability,
            regression: result.regression,
        }
    }
}

#[cfg(test)]
mod tests {
    use shindo_est::PointStatus;

    use super::*;
    use crate::{SHINDO_EST_STATUS_INVALID_INPUT, SHINDO_EST_STATUS_OUT_OF_SCOPE};

    #[test]
    fn layouts_match_the_header() {
        assert_eq!(mem::size_of::<shindo_est_hypocenter>(), 32);
        assert_eq!(
            mem::align_of::<shindo_est_hypocenter>(),
            mem::align_of::<f64>()
        );
        assert_eq!(mem::offset_of!(shindo_est_hypocenter, latitude), 0);
        assert_eq!(mem::offset_of!(shindo_est_hypocenter, longitude), 8);
        assert_eq!(mem::offset_of!(shindo_est_hypocenter, depth), 16);
        assert_eq!(mem::offset_of!(shindo_est_hypocenter, magnitude), 24);

        assert_eq!(mem::size_of::<shindo_est_site>(), 24);
        assert_eq!(mem::align_of::<shindo_est_site>(), mem::align_of::<f64>());
        assert_eq!(mem::offset_of!(shindo_est_site, latitude), 0);
        assert_eq!(mem::offset_of!(shindo_est_site, longitude), 8);
        assert_eq!(mem::offset_of!(shindo_est_site, arv), 16);

        assert_eq!(mem::size_of::<shindo_est_result>(), 32);
        assert_eq!(mem::align_of::<shindo_est_result>(), mem::align_of::<f64>());
        assert_eq!(mem::offset_of!(shindo_est_result, status), 0);
        assert_eq!(mem::offset_of!(shindo_est_result, intensity_class), 4);
        assert_eq!(mem::offset_of!(shindo_est_result, instrumental), 8);
        assert_eq!(mem::offset_of!(shindo_est_result, probability), 16);
        assert_eq!(mem::offset_of!(shindo_est_result, regression), 24);

        // The options have no implicit padding on any target, so that a
        // caller's zero initialization covers every byte the library reads.
        assert_eq!(mem::size_of::<shindo_est_options>(), 24);
        assert_eq!(
            mem::align_of::<shindo_est_options>(),
            mem::align_of::<u64>()
        );
        assert_eq!(mem::offset_of!(shindo_est_options, size), 0);
        assert_eq!(mem::offset_of!(shindo_est_options, offset), 8);
        assert_eq!(mem::offset_of!(shindo_est_options, max_threads), 16);
        assert_eq!(mem::offset_of!(shindo_est_options, reserved), 20);
    }

    #[test]
    fn options_default_has_its_own_size() {
        let options = shindo_est_options::default();

        assert_eq!(options.size, 24);
        assert_eq!(options.offset.to_bits(), 0.0_f64.to_bits());
        assert_eq!(options.max_threads, 0);
        assert_eq!(options.reserved, 0);
    }

    #[test]
    fn conversions_keep_every_bit() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let hypocenter = Hypocenter::from(shindo_est_hypocenter {
            latitude: -0.0,
            longitude: 139.5,
            depth: nan,
            magnitude: 6.25,
        });
        let site = Site::from(shindo_est_site {
            latitude: 35.0,
            longitude: -0.0,
            arv: nan,
        });

        assert_eq!(hypocenter.latitude.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(hypocenter.longitude.to_bits(), 139.5_f64.to_bits());
        assert_eq!(hypocenter.depth.to_bits(), nan.to_bits());
        assert_eq!(hypocenter.magnitude.to_bits(), 6.25_f64.to_bits());
        assert_eq!(site.latitude.to_bits(), 35.0_f64.to_bits());
        assert_eq!(site.longitude.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(site.arv.to_bits(), nan.to_bits());
    }

    #[test]
    fn results_carry_the_intensity_class() {
        let mut computed = PointResult::default();
        computed.status = PointStatus::Ok;
        computed.instrumental = 4.5;
        computed.probability = 0.75;
        computed.regression = 4.25;
        let mut out_of_scope = PointResult::default();
        out_of_scope.status = PointStatus::OutOfScope;
        let invalid = PointResult::default();

        let result = shindo_est_result::from(&computed);
        assert_eq!(result.status, PointStatus::Ok as i32);
        assert_eq!(result.intensity_class, 5);
        assert_eq!(result.instrumental.to_bits(), 4.5_f64.to_bits());
        assert_eq!(result.probability.to_bits(), 0.75_f64.to_bits());
        assert_eq!(result.regression.to_bits(), 4.25_f64.to_bits());

        for (core, status) in [
            (out_of_scope, SHINDO_EST_STATUS_OUT_OF_SCOPE),
            (invalid, SHINDO_EST_STATUS_INVALID_INPUT),
        ] {
            let result = shindo_est_result::from(&core);
            assert_eq!(result.status, status, "{core:?}");
            assert_eq!(result.status, core.status as i32, "{core:?}");
            assert_eq!(result.intensity_class, -1, "{core:?}");
            assert!(result.instrumental.is_nan(), "{result:?}");
            assert!(result.probability.is_nan(), "{result:?}");
            assert!(result.regression.is_nan(), "{result:?}");
        }
    }
}

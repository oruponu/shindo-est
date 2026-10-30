//! Checks of the inputs to a prediction.
//!
//! An invalid input is the caller's mistake. A missing depth or magnitude, or
//! a magnitude at or below 0, is not a mistake: some earthquake reports, such
//! as the first Earthquake Early Warning reports, lack them, and such a call is
//! out of scope rather than invalid.

use std::fmt;

use crate::input::{Hypocenter, Site};

/// Largest absolute latitude in degrees.
const MAX_LATITUDE: f64 = 90.0;
/// Largest absolute longitude in degrees.
const MAX_LONGITUDE: f64 = 180.0;

/// A field of a [`Hypocenter`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HypocenterField {
    /// [`Hypocenter::latitude`].
    Latitude,
    /// [`Hypocenter::longitude`].
    Longitude,
    /// [`Hypocenter::depth`].
    Depth,
    /// [`Hypocenter::magnitude`].
    Magnitude,
}

/// An input that is invalid for every site of a call.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum InputError {
    /// A field of the hypocenter is invalid.
    InvalidHypocenter(HypocenterField),
    /// The offset is not finite.
    InvalidOffset,
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHypocenter(HypocenterField::Latitude) => write!(
                f,
                "the hypocenter latitude must be within -{MAX_LATITUDE}..={MAX_LATITUDE}"
            ),
            Self::InvalidHypocenter(HypocenterField::Longitude) => write!(
                f,
                "the hypocenter longitude must be within -{MAX_LONGITUDE}..={MAX_LONGITUDE}"
            ),
            Self::InvalidHypocenter(HypocenterField::Depth) => {
                f.write_str("the hypocenter depth must be finite and at least 0, or NaN if unknown")
            }
            Self::InvalidHypocenter(HypocenterField::Magnitude) => {
                f.write_str("the hypocenter magnitude must not be positive infinity")
            }
            Self::InvalidOffset => f.write_str("the offset must be finite"),
        }
    }
}

impl std::error::Error for InputError {}

/// Whether a valid call gives intensities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scope {
    /// The sites get intensities.
    InScope,
    /// The depth is unknown, or the magnitude is unknown or not positive, so
    /// no site gets an intensity.
    OutOfScope,
}

/// Checks the hypocenter and the offset of a call.
///
/// # Errors
///
/// Returns the first invalid input, checking the latitude, the longitude, the
/// depth, the magnitude, and the offset in this order: a latitude or longitude
/// that is not finite or out of range, a negative or infinite depth, a
/// magnitude of positive infinity, or an offset that is not finite. A call is
/// out of scope only when it has no invalid input.
pub fn check_call(hypocenter: &Hypocenter, offset: f64) -> Result<Scope, InputError> {
    if !latitude_is_valid(hypocenter.latitude) {
        return Err(InputError::InvalidHypocenter(HypocenterField::Latitude));
    }
    if !longitude_is_valid(hypocenter.longitude) {
        return Err(InputError::InvalidHypocenter(HypocenterField::Longitude));
    }
    let depth = hypocenter.depth;
    if depth < 0.0 || depth.is_infinite() {
        return Err(InputError::InvalidHypocenter(HypocenterField::Depth));
    }
    let magnitude = hypocenter.magnitude;
    if magnitude == f64::INFINITY {
        return Err(InputError::InvalidHypocenter(HypocenterField::Magnitude));
    }
    if !offset.is_finite() {
        return Err(InputError::InvalidOffset);
    }

    if depth.is_nan() || magnitude.is_nan() || magnitude <= 0.0 {
        Ok(Scope::OutOfScope)
    } else {
        Ok(Scope::InScope)
    }
}

/// Returns whether a site has a finite latitude within -90 to 90, a finite
/// longitude within -180 to 180, and a finite, positive ARV.
#[must_use]
pub fn site_is_valid(site: &Site) -> bool {
    latitude_is_valid(site.latitude)
        && longitude_is_valid(site.longitude)
        && site.arv.is_finite()
        && site.arv > 0.0
}

/// Returns whether a latitude lies within -90 to 90, which NaN does not.
fn latitude_is_valid(latitude: f64) -> bool {
    (-MAX_LATITUDE..=MAX_LATITUDE).contains(&latitude)
}

/// Returns whether a longitude lies within -180 to 180, which NaN does not.
fn longitude_is_valid(longitude: f64) -> bool {
    (-MAX_LONGITUDE..=MAX_LONGITUDE).contains(&longitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HYPOCENTER: Hypocenter = Hypocenter {
        latitude: 36.0,
        longitude: 140.0,
        depth: 10.0,
        magnitude: 6.0,
    };

    const SITE: Site = Site {
        latitude: 35.0,
        longitude: 139.0,
        arv: 1.5,
    };

    fn check(hypocenter: Hypocenter) -> Result<Scope, InputError> {
        super::check_call(&hypocenter, 0.0)
    }

    #[test]
    fn valid_call_is_in_scope() {
        assert_eq!(check(HYPOCENTER), Ok(Scope::InScope));
        let shallow_and_small = Hypocenter {
            depth: 0.0,
            magnitude: f64::MIN_POSITIVE,
            ..HYPOCENTER
        };
        assert_eq!(check(shallow_and_small), Ok(Scope::InScope));
        let negative_zero_depth = Hypocenter {
            depth: -0.0,
            ..HYPOCENTER
        };
        assert_eq!(check(negative_zero_depth), Ok(Scope::InScope));
    }

    #[test]
    fn nan_depth_is_out_of_scope() {
        let hypocenter = Hypocenter {
            depth: f64::NAN,
            ..HYPOCENTER
        };
        assert_eq!(check(hypocenter), Ok(Scope::OutOfScope));
    }

    #[test]
    fn nan_or_non_positive_magnitude_is_out_of_scope() {
        for magnitude in [f64::NAN, 0.0, -0.0, -1.0, f64::NEG_INFINITY] {
            let hypocenter = Hypocenter {
                magnitude,
                ..HYPOCENTER
            };
            assert_eq!(check(hypocenter), Ok(Scope::OutOfScope), "{magnitude}");
        }
    }

    #[test]
    fn negative_or_infinite_depth_is_an_error() {
        for depth in [-0.1, f64::INFINITY, f64::NEG_INFINITY] {
            let hypocenter = Hypocenter {
                depth,
                ..HYPOCENTER
            };
            assert_eq!(
                check(hypocenter),
                Err(InputError::InvalidHypocenter(HypocenterField::Depth)),
                "{depth}"
            );
        }
    }

    #[test]
    fn infinite_magnitude_is_an_error() {
        let hypocenter = Hypocenter {
            magnitude: f64::INFINITY,
            ..HYPOCENTER
        };
        assert_eq!(
            check(hypocenter),
            Err(InputError::InvalidHypocenter(HypocenterField::Magnitude))
        );
    }

    #[test]
    fn coordinate_bounds_are_inclusive() {
        for (latitude, longitude) in [(90.0, 180.0), (-90.0, -180.0)] {
            let hypocenter = Hypocenter {
                latitude,
                longitude,
                ..HYPOCENTER
            };
            assert_eq!(
                check(hypocenter),
                Ok(Scope::InScope),
                "{latitude}, {longitude}"
            );
        }

        for latitude in [90.0001, -90.0001, f64::NAN, f64::INFINITY] {
            let hypocenter = Hypocenter {
                latitude,
                ..HYPOCENTER
            };
            assert_eq!(
                check(hypocenter),
                Err(InputError::InvalidHypocenter(HypocenterField::Latitude)),
                "{latitude}"
            );
        }
        for longitude in [180.0001, -180.0001, f64::NAN, f64::NEG_INFINITY] {
            let hypocenter = Hypocenter {
                longitude,
                ..HYPOCENTER
            };
            assert_eq!(
                check(hypocenter),
                Err(InputError::InvalidHypocenter(HypocenterField::Longitude)),
                "{longitude}"
            );
        }
    }

    #[test]
    fn non_finite_offset_is_an_error() {
        for offset in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                super::check_call(&HYPOCENTER, offset),
                Err(InputError::InvalidOffset),
                "{offset}"
            );
        }
        assert_eq!(super::check_call(&HYPOCENTER, -0.5), Ok(Scope::InScope));
    }

    #[test]
    fn errors_win_over_out_of_scope() {
        let hypocenter = Hypocenter {
            latitude: 91.0,
            depth: f64::NAN,
            ..HYPOCENTER
        };
        assert_eq!(
            check(hypocenter),
            Err(InputError::InvalidHypocenter(HypocenterField::Latitude))
        );
        let unknown_depth = Hypocenter {
            depth: f64::NAN,
            ..HYPOCENTER
        };
        let infinite_magnitude = Hypocenter {
            magnitude: f64::INFINITY,
            ..unknown_depth
        };
        assert_eq!(
            check(infinite_magnitude),
            Err(InputError::InvalidHypocenter(HypocenterField::Magnitude))
        );
        assert_eq!(
            super::check_call(&unknown_depth, f64::NAN),
            Err(InputError::InvalidOffset)
        );
        let unknown_magnitude = Hypocenter {
            magnitude: f64::NAN,
            ..HYPOCENTER
        };
        assert_eq!(
            super::check_call(&unknown_magnitude, f64::NAN),
            Err(InputError::InvalidOffset)
        );
    }

    #[test]
    fn reports_the_first_error_in_field_order() {
        let all_wrong = Hypocenter {
            latitude: 91.0,
            longitude: 181.0,
            depth: -1.0,
            magnitude: f64::INFINITY,
        };
        let cases = [
            (all_wrong, HypocenterField::Latitude),
            (
                Hypocenter {
                    latitude: 0.0,
                    ..all_wrong
                },
                HypocenterField::Longitude,
            ),
            (
                Hypocenter {
                    latitude: 0.0,
                    longitude: 0.0,
                    ..all_wrong
                },
                HypocenterField::Depth,
            ),
            (
                Hypocenter {
                    latitude: 0.0,
                    longitude: 0.0,
                    depth: 0.0,
                    ..all_wrong
                },
                HypocenterField::Magnitude,
            ),
        ];
        for (hypocenter, field) in cases {
            assert_eq!(
                super::check_call(&hypocenter, f64::NAN),
                Err(InputError::InvalidHypocenter(field)),
                "{hypocenter:?}"
            );
        }
    }

    #[test]
    fn errors_name_the_input() {
        let cases = [
            (
                InputError::InvalidHypocenter(HypocenterField::Latitude),
                "latitude",
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Longitude),
                "longitude",
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Depth),
                "depth",
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Magnitude),
                "magnitude",
            ),
            (InputError::InvalidOffset, "offset"),
        ];
        for (error, name) in cases {
            let message = error.to_string();
            assert!(message.contains(name), "{name}: {message}");
        }
    }

    #[test]
    fn site_arv_must_be_positive() {
        assert!(super::site_is_valid(&SITE), "{SITE:?}");
        for arv in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY] {
            let site = Site { arv, ..SITE };
            assert!(!super::site_is_valid(&site), "{site:?}");
        }
    }

    #[test]
    fn site_coordinates_must_be_in_range() {
        for (latitude, longitude) in [(90.0, 180.0), (-90.0, -180.0)] {
            let site = Site {
                latitude,
                longitude,
                ..SITE
            };
            assert!(super::site_is_valid(&site), "{site:?}");
        }
        for (latitude, longitude) in [
            (-91.0, 139.0),
            (f64::NAN, 139.0),
            (35.0, 180.5),
            (35.0, f64::NAN),
        ] {
            let site = Site {
                latitude,
                longitude,
                ..SITE
            };
            assert!(!super::site_is_valid(&site), "{site:?}");
        }
    }
}

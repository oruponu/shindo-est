//! Distances between a hypocenter and a point on the surface.

use std::f64::consts::PI;

use crate::input::Hypocenter;

/// Mean radius of the Earth in km, used for the hypocentral distance.
pub const EARTH_RADIUS: f64 = 6371.008;

// GRS80 ellipsoid in km.
const MAJOR_AXIS: f64 = 6378.137;
const FLATTENING: f64 = 1.0 / 298.257_222_101;
const AXIS_RATIO: f64 = 1.0 - FLATTENING;
const MINOR_AXIS: f64 = MAJOR_AXIS * AXIS_RATIO;
const SECOND_ECCENTRICITY_SQUARED: f64 =
    (MAJOR_AXIS * MAJOR_AXIS - MINOR_AXIS * MINOR_AXIS) / (MINOR_AXIS * MINOR_AXIS);

// Vincenty's iteration stops when lambda changes by at most this many radians.
const VINCENTY_CONVERGENCE: f64 = 1e-12;
// Vincenty's formula gives NaN after this many iterations without converging.
const VINCENTY_MAX_ITERATIONS: u32 = 1000;
// Below this, sin^2(sigma) is treated as zero (the two points coincide or are antipodal).
const DEGENERATE_SIN_SIGMA_SQUARED: f64 = 1e-24;

/// Distances in km from a hypocenter to a point on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distances {
    /// Straight-line distance from the hypocenter.
    pub hypocentral: f64,
    /// Distance along the surface from the epicenter. NaN where Vincenty's formula does not converge.
    pub epicentral: f64,
}

/// Returns the distances from `hypocenter` to the point at `latitude` and `longitude` (degrees).
///
/// The hypocentral distance treats the Earth as a sphere of [`EARTH_RADIUS`] with
/// geocentric latitudes. The epicentral distance is the geodesic on GRS80 by
/// Vincenty's formula.
#[must_use]
pub fn distances(hypocenter: &Hypocenter, latitude: f64, longitude: f64) -> Distances {
    let lat1 = geocentric_latitude(hypocenter.latitude);
    let lat2 = geocentric_latitude(latitude);
    let lon1 = hypocenter.longitude.to_radians();
    let lon2 = longitude.to_radians();
    let dx = libm::cos(lat1) * libm::cos(lon1) - libm::cos(lat2) * libm::cos(lon2);
    let dy = libm::cos(lat1) * libm::sin(lon1) - libm::cos(lat2) * libm::sin(lon2);
    let dz = libm::sin(lat1) - libm::sin(lat2);
    let chord_squared = dx * dx + dy * dy + dz * dz;
    let cos_angle = 1.0 - chord_squared * 0.5;
    let radius = EARTH_RADIUS - hypocenter.depth;
    let hypocentral = (radius * radius + EARTH_RADIUS * EARTH_RADIUS
        - 2.0 * radius * EARTH_RADIUS * cos_angle)
        .sqrt();
    let epicentral = vincenty(
        hypocenter.latitude,
        hypocenter.longitude,
        latitude,
        longitude,
    );
    Distances {
        hypocentral,
        epicentral,
    }
}

/// Converts a geodetic latitude in degrees into a geocentric latitude in radians.
fn geocentric_latitude(latitude: f64) -> f64 {
    libm::atan(AXIS_RATIO * AXIS_RATIO * libm::tan(latitude.to_radians()))
}

/// Geodesic distance in km on GRS80 by Vincenty's inverse formula.
fn vincenty(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lon_diff = (lon2 - lon1).to_radians();
    let tan_u1 = AXIS_RATIO * libm::tan(lat1.to_radians());
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;
    let tan_u2 = AXIS_RATIO * libm::tan(lat2.to_radians());
    let cos_u2 = 1.0 / (1.0 + tan_u2 * tan_u2).sqrt();
    let sin_u2 = tan_u2 * cos_u2;
    let antipodal = lon_diff.abs() > PI / 2.0 || (lat2 - lat1).abs() > 90.0;

    let mut lambda = lon_diff;
    let mut sigma = if antipodal { PI } else { 0.0 };
    let mut sin_sigma = 0.0;
    let mut cos_sigma = if antipodal { -1.0 } else { 1.0 };
    let mut cos_2sigma_m = 1.0;
    let mut cos_alpha_squared = 1.0;
    let mut iterations = 0;
    loop {
        let (sin_lambda, cos_lambda) = (libm::sin(lambda), libm::cos(lambda));
        let a = cos_u2 * sin_lambda;
        let b = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda;
        let sin_sigma_squared = a * a + b * b;
        if sin_sigma_squared < DEGENERATE_SIN_SIGMA_SQUARED {
            break;
        }
        sin_sigma = sin_sigma_squared.sqrt();
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        sigma = libm::atan2(sin_sigma, cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        cos_alpha_squared = 1.0 - sin_alpha * sin_alpha;
        // On the equator cos^2(alpha) is zero and the term is defined as zero.
        cos_2sigma_m = if cos_alpha_squared == 0.0 {
            0.0
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_alpha_squared
        };
        let c = FLATTENING / 16.0
            * cos_alpha_squared
            * (4.0 + FLATTENING * (4.0 - 3.0 * cos_alpha_squared));
        let previous = lambda;
        lambda = lon_diff
            + (1.0 - c)
                * FLATTENING
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos_2sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));
        let overshoot = if antipodal {
            lambda.abs() - PI
        } else {
            lambda.abs()
        };
        if overshoot > PI {
            return f64::NAN;
        }
        if (lambda - previous).abs() <= VINCENTY_CONVERGENCE {
            break;
        }
        iterations += 1;
        if iterations >= VINCENTY_MAX_ITERATIONS {
            return f64::NAN;
        }
    }

    let u_squared = cos_alpha_squared * SECOND_ECCENTRICITY_SQUARED;
    let series_a = 1.0
        + u_squared / 16_384.0
            * (4096.0 + u_squared * (-768.0 + u_squared * (320.0 - 175.0 * u_squared)));
    let series_b =
        u_squared / 1024.0 * (256.0 + u_squared * (-128.0 + u_squared * (74.0 - 47.0 * u_squared)));
    let delta_sigma = series_b
        * sin_sigma
        * (cos_2sigma_m
            + series_b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                    - series_b / 6.0
                        * cos_2sigma_m
                        * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                        * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));
    MINOR_AXIS * series_a * (sigma - delta_sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Hypocenter;

    fn hypocenter(latitude: f64, longitude: f64, depth: f64) -> Hypocenter {
        Hypocenter {
            latitude,
            longitude,
            depth,
            magnitude: 5.0,
        }
    }

    #[test]
    fn at_epicenter_hypocentral_equals_depth() {
        let d = distances(&hypocenter(35.0, 135.0, 10.0), 35.0, 135.0);

        assert!((d.hypocentral - 10.0).abs() <= 1e-6, "{d:?}");
        assert!(d.epicentral.abs() <= 1e-9, "{d:?}");
    }

    #[test]
    fn one_degree_north_is_about_111_km() {
        let epicentral = distances(&hypocenter(35.0, 135.0, 0.0), 36.0, 135.0).epicentral;

        assert!((110.0..=112.0).contains(&epicentral), "{epicentral}");
    }

    #[test]
    fn epicentral_is_symmetric() {
        let ab = distances(&hypocenter(35.0, 135.0, 0.0), 38.0, 141.0).epicentral;
        let ba = distances(&hypocenter(38.0, 141.0, 0.0), 35.0, 135.0).epicentral;

        assert!((ab - ba).abs() <= 1e-9, "{ab} {ba}");
    }

    #[test]
    fn unknown_depth_gives_nan_hypocentral_only() {
        let d = distances(&hypocenter(35.0, 135.0, f64::NAN), 36.0, 135.0);

        assert!(d.hypocentral.is_nan(), "{d:?}");
        assert!((110.0..=112.0).contains(&d.epicentral), "{d:?}");
    }

    #[test]
    fn nan_coordinates_give_nan_without_hanging() {
        let d = distances(&hypocenter(f64::NAN, 135.0, 10.0), 36.0, 135.0);

        assert!(d.hypocentral.is_nan(), "{d:?}");
        assert!(d.epicentral.is_nan(), "{d:?}");
    }

    #[test]
    fn epicentral_matches_vincenty_example_on_grs80() {
        // Geoscience Australia's Vincenty example (Flinders Peak to Buninyong, 54972.271 m on GRS80)
        let flinders_peak = hypocenter(
            -(37.0 + 57.0 / 60.0 + 3.72030 / 3600.0),
            144.0 + 25.0 / 60.0 + 29.52440 / 3600.0,
            0.0,
        );

        let epicentral = distances(
            &flinders_peak,
            -(37.0 + 39.0 / 60.0 + 10.15610 / 3600.0),
            143.0 + 55.0 / 60.0 + 35.38390 / 3600.0,
        )
        .epicentral;

        assert!((epicentral - 54.972_271).abs() <= 1e-6, "{epicentral}");
    }
}

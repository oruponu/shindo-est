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

/// Vincenty's iteration stops when lambda changes by at most this many radians.
const VINCENTY_CONVERGENCE: f64 = 1e-12;
/// Vincenty's formula gives NaN after this many iterations without converging.
const VINCENTY_MAX_ITERATIONS: u32 = 1000;
/// Below this, sin^2(sigma) is treated as zero (the two points coincide or are antipodal).
const DEGENERATE_SIN_SIGMA_SQUARED: f64 = 1e-24;
/// Degrees in a full turn of longitude.
const FULL_TURN: f64 = 360.0;

/// Distances in km from a hypocenter to a point on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distances {
    /// Straight-line distance from the hypocenter.
    pub hypocentral: f64,
    /// Distance along the surface from the epicenter.
    pub epicentral: f64,
}

/// Returns the distances from `hypocenter` to the point at `latitude` and `longitude` (degrees).
///
/// The hypocentral distance treats the Earth as a sphere of [`EARTH_RADIUS`] with
/// geocentric latitudes. The epicentral distance is the geodesic on GRS80 by
/// Vincenty's formula, or, near the antipode where that formula does not
/// converge, the great-circle distance on the same sphere as the hypocentral
/// distance. That stand-in differs from the geodesic by up to about 22 km
/// (0.11%). Vincenty's formula stops converging no farther than about 0.6
/// degrees from the antipode, and the epicentral distance jumps by as much
/// there.
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
    let depth = hypocenter.depth;
    let radius = EARTH_RADIUS - depth;
    // Equal to r^2 + R^2 - 2rR cos(angle), but without the cancellation that
    // makes that form NaN or far off for depths near zero.
    let hypocentral = (depth * depth + radius * EARTH_RADIUS * chord_squared).sqrt();
    let geodesic = vincenty(
        hypocenter.latitude,
        hypocenter.longitude,
        latitude,
        longitude,
    );
    let epicentral = if geodesic.is_nan() {
        great_circle(chord_squared)
    } else {
        geodesic
    };
    Distances {
        hypocentral,
        epicentral,
    }
}

/// Converts a geodetic latitude in degrees into a geocentric latitude in radians.
fn geocentric_latitude(latitude: f64) -> f64 {
    libm::atan(AXIS_RATIO * AXIS_RATIO * libm::tan(latitude.to_radians()))
}

/// Returns `to - from` in degrees of longitude, wrapped into -180 (exclusive)
/// to 180 (inclusive), so that the same meridian written as 180 and -180
/// differs by 0.
#[must_use]
pub(crate) fn longitude_difference(from: f64, to: f64) -> f64 {
    let difference = (to - from) % FULL_TURN;
    if difference > FULL_TURN / 2.0 {
        difference - FULL_TURN
    } else if difference <= -FULL_TURN / 2.0 {
        difference + FULL_TURN
    } else {
        difference
    }
}

/// Returns the great-circle distance in km on a sphere of [`EARTH_RADIUS`]
/// from the squared distance between the unit vectors of two points.
fn great_circle(chord_squared: f64) -> f64 {
    let half_chord = chord_squared.sqrt() / 2.0;
    // Rounding can make half the chord slightly more than 1 for antipodes. Not
    // f64::min, which would turn a NaN chord into 1.
    let sine = if half_chord > 1.0 { 1.0 } else { half_chord };
    EARTH_RADIUS * 2.0 * libm::asin(sine)
}

/// Geodesic distance in km on GRS80 by Vincenty's inverse formula, or NaN
/// where it does not converge.
fn vincenty(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lon_diff = longitude_difference(lon1, lon2).to_radians();
    let tan_u1 = AXIS_RATIO * libm::tan(lat1.to_radians());
    let cos_u1 = 1.0 / (1.0 + tan_u1 * tan_u1).sqrt();
    let sin_u1 = tan_u1 * cos_u1;
    let tan_u2 = AXIS_RATIO * libm::tan(lat2.to_radians());
    let cos_u2 = 1.0 / (1.0 + tan_u2 * tan_u2).sqrt();
    let sin_u2 = tan_u2 * cos_u2;
    let antipodal = lon_diff.abs() > PI / 2.0 || (lat2 - lat1).abs() > 90.0;

    let mut lambda = lon_diff;
    let mut sigma;
    let mut sin_sigma;
    let mut cos_sigma;
    let mut cos_2sigma_m = 1.0;
    let mut cos_alpha_squared = 1.0;
    let mut iterations = 0;
    loop {
        let (sin_lambda, cos_lambda) = (libm::sin(lambda), libm::cos(lambda));
        let a = cos_u2 * sin_lambda;
        let b = cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda;
        let sin_sigma_squared = a * a + b * b;
        if sin_sigma_squared < DEGENERATE_SIN_SIGMA_SQUARED {
            // The points coincide or are antipodal, whatever their longitudes
            // say (as at a pole), and cos(sigma) tells which.
            sin_sigma = 0.0;
            cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            sigma = if cos_sigma > 0.0 { 0.0 } else { PI };
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
        let distances = super::distances(&hypocenter(35.0, 135.0, 10.0), 35.0, 135.0);

        assert!(
            (distances.hypocentral - 10.0).abs() <= 1e-6,
            "{distances:?}"
        );
        assert!(distances.epicentral.abs() <= 1e-9, "{distances:?}");
    }

    #[test]
    fn one_degree_north_is_about_111_km() {
        let epicentral = super::distances(&hypocenter(35.0, 135.0, 0.0), 36.0, 135.0).epicentral;

        assert!((110.0..=112.0).contains(&epicentral), "{epicentral}");
    }

    #[test]
    fn epicentral_is_symmetric() {
        let ab = super::distances(&hypocenter(35.0, 135.0, 0.0), 38.0, 141.0).epicentral;
        let ba = super::distances(&hypocenter(38.0, 141.0, 0.0), 35.0, 135.0).epicentral;

        assert!((ab - ba).abs() <= 1e-9, "{ab} {ba}");
    }

    #[test]
    fn unknown_depth_gives_nan_hypocentral_only() {
        let distances = super::distances(&hypocenter(35.0, 135.0, f64::NAN), 36.0, 135.0);

        assert!(distances.hypocentral.is_nan(), "{distances:?}");
        assert!(
            (110.0..=112.0).contains(&distances.epicentral),
            "{distances:?}"
        );
    }

    #[test]
    fn nan_coordinates_give_nan_without_hanging() {
        let distances = super::distances(&hypocenter(f64::NAN, 135.0, 10.0), 36.0, 135.0);

        assert!(distances.hypocentral.is_nan(), "{distances:?}");
        assert!(distances.epicentral.is_nan(), "{distances:?}");
    }

    #[test]
    fn same_point_across_the_date_line_or_at_a_pole_is_zero() {
        let cases = [
            (35.0, 180.0, 35.0, -180.0),
            (35.0, -180.0, 35.0, 180.0),
            (90.0, 0.0, 90.0, 120.0),
            (-90.0, 10.0, -90.0, -170.0),
        ];
        for (latitude, longitude, site_latitude, site_longitude) in cases {
            let distances = super::distances(
                &hypocenter(latitude, longitude, 10.0),
                site_latitude,
                site_longitude,
            );
            assert!(
                distances.epicentral.abs() <= 1e-9,
                "{latitude}, {longitude}: {distances:?}"
            );
            assert!(
                (distances.hypocentral - 10.0).abs() <= 1e-9,
                "{latitude}, {longitude}: {distances:?}"
            );
        }
    }

    #[test]
    fn crossing_the_date_line_matches_crossing_greenwich() {
        // The same longitude difference gives the same geodesic to the bit,
        // in either direction.
        let cases = [
            ((179.0, -179.0), (-1.0, 1.0)),
            ((-179.0, 179.0), (1.0, -1.0)),
        ];
        for ((from, to), (greenwich_from, greenwich_to)) in cases {
            let across = super::distances(&hypocenter(35.0, from, 10.0), 35.0, to);
            let greenwich =
                super::distances(&hypocenter(35.0, greenwich_from, 10.0), 35.0, greenwich_to);

            assert_eq!(
                across.epicentral.to_bits(),
                greenwich.epicentral.to_bits(),
                "{across:?} {greenwich:?}"
            );
            assert!(
                (across.hypocentral - greenwich.hypocentral).abs() <= 1e-9,
                "{across:?} {greenwich:?}"
            );
        }
    }

    #[test]
    fn antipodal_points_on_the_equator_are_half_a_meridian_apart() {
        // The shortest path runs over a pole: half the GRS80 meridian.
        let epicentral = super::distances(&hypocenter(0.0, 0.0, 0.0), 0.0, 180.0).epicentral;

        assert!((epicentral - 20_003.931_458).abs() <= 1e-6, "{epicentral}");
    }

    #[test]
    fn near_the_antipode_falls_back_to_a_great_circle() {
        // The expected distances are 6371.008 km times the angle between the
        // geocentric unit vectors, computed as atan2(|u x v|, u . v) in Python.
        let cases = [
            ((35.0, 140.0), (-35.41, -40.04), 19_969.479_819_476_015),
            ((0.0, 0.0), (0.0, 179.3965), 19_948.005_706_267_315),
        ];
        for ((latitude, longitude), (site_latitude, site_longitude), expected) in cases {
            let geodesic = super::vincenty(latitude, longitude, site_latitude, site_longitude);
            assert!(geodesic.is_nan(), "{geodesic}");

            let epicentral = super::distances(
                &hypocenter(latitude, longitude, 0.0),
                site_latitude,
                site_longitude,
            )
            .epicentral;
            assert!(
                (epicentral - expected).abs() <= 1e-9,
                "{epicentral} km, expected {expected} km"
            );
        }
    }

    #[test]
    fn tiny_depths_give_exact_hypocentral_distances() {
        // r^2 + R^2 - 2rR cos(angle) cancels badly for these depths.
        for depth in [1e-6, 1e-4, 0.0] {
            let distances = super::distances(&hypocenter(35.0, 135.0, depth), 35.0, 135.0);
            assert_eq!(distances.hypocentral, depth, "{distances:?}");
        }
    }

    #[test]
    fn epicentral_matches_vincenty_example_on_grs80() {
        // Geoscience Australia's Vincenty example: Flinders Peak to Buninyong,
        // 54972.271 m on GRS80.
        let flinders_peak = hypocenter(
            -(37.0 + 57.0 / 60.0 + 3.72030 / 3600.0),
            144.0 + 25.0 / 60.0 + 29.52440 / 3600.0,
            0.0,
        );

        let epicentral = super::distances(
            &flinders_peak,
            -(37.0 + 39.0 / 60.0 + 10.15610 / 3600.0),
            143.0 + 55.0 / 60.0 + 35.38390 / 3600.0,
        )
        .epicentral;

        assert!((epicentral - 54.972_271).abs() <= 1e-6, "{epicentral}");
    }
}

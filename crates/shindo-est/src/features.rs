//! Model input features computed from a hypocenter and a site.

use crate::geodesy;
use crate::input::{Hypocenter, Site};
use crate::slab::SlabGeometry;

/// Number of model features.
pub const FEATURE_COUNT: usize = 14;

/// Smallest hypocentral distance in km taken for `log_hypo`.
const MIN_HYPOCENTRAL_DISTANCE: f64 = 1.0;

/// Names of the model features, in the order the models were trained with.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "mj",
    "depth",
    "qlat",
    "qlon",
    "slat",
    "slon",
    "log_hypo",
    "epi",
    "az_sin",
    "az_cos",
    "log_arv",
    "slab_top",
    "updip",
    "path_frac",
];

/// Returns the features of a hypocenter and a site, in the order of
/// [`FEATURE_NAMES`].
///
/// Unknown values (NaN) pass through as NaN, except that the path fraction of
/// a hypocenter with an unknown depth is 0.
#[must_use]
pub fn compute(hypocenter: &Hypocenter, site: &Site, slabs: &SlabGeometry) -> [f64; FEATURE_COUNT] {
    let distances = geodesy::distances(hypocenter, site.latitude, site.longitude);
    let azimuth = site_azimuth(hypocenter, site);
    // Not f64::max, which would turn a NaN distance into 1.
    let floored_hypocentral = if distances.hypocentral < MIN_HYPOCENTRAL_DISTANCE {
        MIN_HYPOCENTRAL_DISTANCE
    } else {
        distances.hypocentral
    };
    [
        hypocenter.magnitude,
        hypocenter.depth,
        hypocenter.latitude,
        hypocenter.longitude,
        site.latitude,
        site.longitude,
        libm::log10(floored_hypocentral),
        distances.epicentral,
        libm::sin(azimuth),
        libm::cos(azimuth),
        libm::log10(site.arv),
        slabs.pacific_top_depth(site.latitude, site.longitude),
        slabs.updip_depth_difference(hypocenter, site.latitude, site.longitude),
        slabs.pacific_path_fraction(hypocenter, site.latitude, site.longitude),
    ]
}

/// Returns the azimuth in radians from the epicenter to the site, clockwise
/// from north, using geodetic latitudes on a sphere.
fn site_azimuth(hypocenter: &Hypocenter, site: &Site) -> f64 {
    let lat1 = hypocenter.latitude.to_radians();
    let lat2 = site.latitude.to_radians();
    let delta = geodesy::longitude_difference(hypocenter.longitude, site.longitude).to_radians();
    libm::atan2(
        libm::sin(delta) * libm::cos(lat2),
        libm::cos(lat1) * libm::sin(lat2) - libm::sin(lat1) * libm::cos(lat2) * libm::cos(delta),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slab::test_support;

    fn hypocenter(latitude: f64, longitude: f64, depth: f64) -> Hypocenter {
        Hypocenter {
            latitude,
            longitude,
            depth,
            magnitude: 6.0,
        }
    }

    fn site(latitude: f64, longitude: f64) -> Site {
        Site {
            latitude,
            longitude,
            arv: 1.0,
        }
    }

    fn features(hypocenter: &Hypocenter, site: &Site) -> [f64; FEATURE_COUNT] {
        super::compute(hypocenter, site, &test_support::geometry())
    }

    fn feature(values: &[f64; FEATURE_COUNT], name: &str) -> f64 {
        let index = FEATURE_NAMES
            .iter()
            .position(|&candidate| candidate == name)
            .expect("the name should be a feature");
        values[index]
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn names_follow_the_trained_order() {
        // Spelled out again so that swapping two names together with their
        // columns fails here, even though the other tests look columns up by name.
        assert_eq!(
            FEATURE_NAMES,
            [
                "mj",
                "depth",
                "qlat",
                "qlon",
                "slat",
                "slon",
                "log_hypo",
                "epi",
                "az_sin",
                "az_cos",
                "log_arv",
                "slab_top",
                "updip",
                "path_frac",
            ]
        );
    }

    #[test]
    fn same_point_has_zero_epicentral_and_north_azimuth() {
        let values = features(&hypocenter(35.0, 135.0, 10.0), &site(35.0, 135.0));

        assert_close(feature(&values, "mj"), 6.0, 0.0);
        assert_close(feature(&values, "depth"), 10.0, 0.0);
        assert_close(feature(&values, "log_hypo"), 1.0, 1e-9);
        assert_close(feature(&values, "epi"), 0.0, 1e-9);
        assert_close(feature(&values, "az_sin"), 0.0, 0.0);
        assert_close(feature(&values, "az_cos"), 1.0, 0.0);
        assert_close(feature(&values, "log_arv"), 0.0, 0.0);
    }

    #[test]
    fn coordinates_keep_hypocenter_and_site_apart() {
        let values = features(&hypocenter(35.0, 135.0, 10.0), &site(36.0, 137.0));

        assert_close(feature(&values, "qlat"), 35.0, 0.0);
        assert_close(feature(&values, "qlon"), 135.0, 0.0);
        assert_close(feature(&values, "slat"), 36.0, 0.0);
        assert_close(feature(&values, "slon"), 137.0, 0.0);
    }

    #[test]
    fn log_arv_is_base_10() {
        let site = Site {
            latitude: 35.0,
            longitude: 135.0,
            arv: 100.0,
        };
        let values = features(&hypocenter(35.0, 135.0, 10.0), &site);

        assert_close(feature(&values, "log_arv"), 2.0, 1e-12);
    }

    #[test]
    fn zero_depth_at_the_epicenter_has_zero_log_hypo() {
        let values = features(&hypocenter(35.0, 135.0, 0.0), &site(35.0, 135.0));

        assert_close(feature(&values, "log_hypo"), 0.0, 0.0);
    }

    #[test]
    fn distances_come_from_geodesy() {
        let shallow = hypocenter(35.0, 135.0, 0.0);
        let values = features(&shallow, &site(36.0, 137.0));
        let distances = geodesy::distances(&shallow, 36.0, 137.0);

        assert_close(
            feature(&values, "log_hypo"),
            libm::log10(distances.hypocentral),
            1e-12,
        );
        assert_close(feature(&values, "epi"), distances.epicentral, 1e-9);
        let epi = feature(&values, "epi");
        assert!(epi > 100.0, "{epi}");
    }

    #[test]
    fn distances_under_1_km_count_as_1_km() {
        let values = features(&hypocenter(35.0, 135.0, 0.5), &site(35.0, 135.0));

        assert_close(feature(&values, "log_hypo"), 0.0, 0.0);
    }

    #[test]
    fn unknown_depth_gives_nan_log_hypo() {
        // As in the trained data: the floor of 1 km does not hide a NaN distance.
        let values = features(&hypocenter(35.0, 135.0, f64::NAN), &site(36.0, 135.0));
        let log_hypo = feature(&values, "log_hypo");

        assert!(log_hypo.is_nan(), "{log_hypo}");
    }

    #[test]
    fn azimuth_points_north_and_east() {
        let north = features(&hypocenter(35.0, 135.0, 10.0), &site(36.0, 135.0));
        let east = features(&hypocenter(35.0, 135.0, 10.0), &site(35.0, 136.0));

        assert_close(feature(&north, "az_sin"), 0.0, 1e-12);
        assert_close(feature(&north, "az_cos"), 1.0, 1e-12);
        let east_sin = feature(&east, "az_sin");
        assert!(east_sin > 0.99, "{east_sin}");
    }

    #[test]
    fn azimuth_follows_the_great_circle() {
        // A flat atan2(longitude difference, latitude difference) would give
        // due east (1, 0) here.
        let values = features(&hypocenter(60.0, 0.0, 10.0), &site(60.0, 90.0));

        assert_close(feature(&values, "az_sin"), 0.755_928_946_018_454_4, 1e-12);
        assert_close(feature(&values, "az_cos"), 0.654_653_670_707_977_2, 1e-12);
    }

    #[test]
    fn same_point_across_the_date_line_points_north() {
        let values = features(&hypocenter(35.0, 180.0, 10.0), &site(35.0, -180.0));

        assert_close(feature(&values, "az_sin"), 0.0, 0.0);
        assert_close(feature(&values, "az_cos"), 1.0, 0.0);
        assert_close(feature(&values, "epi"), 0.0, 1e-9);
    }

    #[test]
    fn azimuth_crosses_the_date_line() {
        let east = features(&hypocenter(0.0, 179.0, 10.0), &site(0.0, -179.0));
        let west = features(&hypocenter(0.0, -179.0, 10.0), &site(0.0, 179.0));

        assert_close(feature(&east, "az_sin"), 1.0, 1e-12);
        assert_close(feature(&west, "az_sin"), -1.0, 1e-12);
    }

    #[test]
    fn slab_columns_use_the_geometry() {
        let values = features(&hypocenter(36.5, 140.0, 400.0), &site(37.0, 141.0));

        assert_close(feature(&values, "slab_top"), 300.0, 1e-9);
        assert_close(feature(&values, "updip"), 100.0, 1e-9);
    }

    #[test]
    fn slab_columns_take_the_top_under_the_site() {
        // The hypocenter is over the Kuril grid (300 km) and the site over the
        // Izu-Bonin grid (250 km).
        let values = features(&hypocenter(36.5, 140.0, 400.0), &site(34.9, 140.0));

        assert_close(feature(&values, "slab_top"), 250.0, 1e-9);
        assert_close(feature(&values, "updip"), 150.0, 1e-9);
    }

    #[test]
    fn path_fraction_uses_the_geometry() {
        // The vertical path up from 320 km is in the 300 to 350 km slab at 4 of
        // its 60 midpoints.
        let values = features(&hypocenter(36.5, 140.0, 320.0), &site(36.5, 140.0));

        assert_close(feature(&values, "path_frac"), 4.0 / 60.0, 1e-12);
    }

    #[test]
    fn far_away_site_does_not_panic() {
        let values = features(&hypocenter(36.5, 140.0, 400.0), &site(-35.0, -170.0));
        let slab_top = feature(&values, "slab_top");

        assert!(slab_top.is_nan(), "{slab_top}");
        assert_close(feature(&values, "path_frac"), 0.0, 0.0);
    }
}

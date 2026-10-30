//! Slab2 depth grids of the subducting Pacific plate and the slab features
//! derived from them.

use std::fmt;

use crate::input::Hypocenter;

/// Spacing of the Slab2 grids in degrees.
const SPACING: f64 = 0.05;
/// Earth radius in km used for distances on the grids and along paths.
const EARTH_RADIUS: f64 = 6371.0;
/// Latitude in degrees from which the Kuril grid is used instead of the
/// Izu-Bonin grid.
const BOUNDARY_LATITUDE: f64 = 35.0;
/// Number of equal segments of a path whose midpoints are tested.
const PATH_SAMPLES: u32 = 60;
/// Largest number of grid nodes accepted. The Kuril grid in `data/slab2` has about 155,000.
const MAX_NODES: f64 = 4_000_000.0;
/// Largest number of grid nodes accepted for each point, so that the memory a
/// grid takes stays proportional to its text. The grids in `data/slab2` have
/// 1.7 (Kuril) and 2.7 (Izu-Bonin) nodes for each point.
const MAX_NODES_PER_POINT: usize = 16;
/// Largest distance in nodes accepted between a point and its grid node.
const GRID_TOLERANCE: f64 = 1e-6;

/// The sample outside a grid or in a cell touching a node without a value.
const MISSING: SlabSample = SlabSample {
    top_depth: f64::NAN,
    thickness: f64::NAN,
    cos_dip: f64::NAN,
};

/// A Slab2 grid of the slab top depth and thickness, read from
/// `lon,lat,depth,thickness` rows (degrees and km, depth positive down).
#[derive(Clone, Debug)]
pub struct SlabGrid {
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    // Row-major values, one row per latitude. Nodes without a point are NaN.
    depth: Vec<f64>,
    thickness: Vec<f64>,
    cos_dip: Vec<f64>,
}

/// Values of a grid interpolated at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlabSample {
    /// Depth of the slab top in km.
    pub top_depth: f64,
    /// Thickness of the slab in km.
    pub thickness: f64,
    /// Cosine of the slab dip.
    pub cos_dip: f64,
}

/// An error from reading a grid.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum SlabError {
    /// The line does not have exactly four columns.
    WrongColumnCount {
        /// Line number, starting at 1.
        line: usize,
    },
    /// A column of the line is not a number.
    NotANumber {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The latitude or longitude is not finite or out of range.
    InvalidCoordinate {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The point is not on the 0.05 degree grid of the other points.
    NotOnGrid {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The point has the coordinates of an earlier point.
    RepeatedPoint {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The text has no points.
    NoPoints,
    /// The grid is smaller than 2 x 2 nodes.
    TooSmall,
    /// The grid has more than 4,000,000 nodes, far more than any Slab2 grid.
    TooLarge,
    /// The grid has more than 16 nodes for each point; the grids in
    /// `data/slab2` have at most 2.7.
    TooSparse,
}

impl fmt::Display for SlabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongColumnCount { line } => {
                write!(
                    f,
                    "line {line}: expected 4 columns (lon,lat,depth,thickness)"
                )
            }
            Self::NotANumber { line } => write!(f, "line {line}: a column is not a number"),
            Self::InvalidCoordinate { line } => write!(
                f,
                "line {line}: latitude must be within -90..=90 and longitude within -180..=360"
            ),
            Self::NotOnGrid { line } => {
                write!(f, "line {line}: the point is not on the 0.05 degree grid")
            }
            Self::RepeatedPoint { line } => {
                write!(f, "line {line}: the point repeats an earlier point")
            }
            Self::NoPoints => f.write_str("the grid has no points"),
            Self::TooSmall => f.write_str("the grid is smaller than 2 x 2 nodes"),
            Self::TooLarge => f.write_str("the grid has more than 4,000,000 nodes"),
            Self::TooSparse => write!(
                f,
                "the grid has more than {MAX_NODES_PER_POINT} nodes for each point"
            ),
        }
    }
}

impl std::error::Error for SlabError {}

impl std::str::FromStr for SlabGrid {
    type Err = SlabError;

    fn from_str(text: &str) -> Result<Self, SlabError> {
        Self::parse(text)
    }
}

struct Point {
    line: usize,
    latitude: f64,
    longitude: f64,
    depth: f64,
    thickness: f64,
}

impl SlabGrid {
    /// Reads a grid from `lon,lat,depth,thickness` rows.
    ///
    /// Empty lines are skipped and CRLF line endings are accepted. Depth and
    /// thickness may be NaN.
    ///
    /// # Errors
    ///
    /// Returns a [`SlabError`] when a line is malformed, a coordinate is invalid,
    /// a point is off the 0.05 degree grid or repeats an earlier point, or the
    /// grid is empty, smaller than 2 x 2, too large, or has far more nodes than
    /// points.
    pub fn parse(text: &str) -> Result<Self, SlabError> {
        let mut points = Vec::new();
        for (index, line) in text.lines().enumerate() {
            if !line.is_empty() {
                points.push(parse_point(line, index + 1)?);
            }
        }
        if points.is_empty() {
            return Err(SlabError::NoPoints);
        }

        let (min_latitude, max_latitude) = bounds(points.iter().map(|p| p.latitude));
        let (min_longitude, max_longitude) = bounds(points.iter().map(|p| p.longitude));
        let rows = ((max_latitude - min_latitude) / SPACING).round() + 1.0;
        let columns = ((max_longitude - min_longitude) / SPACING).round() + 1.0;
        if rows < 2.0 || columns < 2.0 {
            return Err(SlabError::TooSmall);
        }
        if rows * columns > MAX_NODES {
            return Err(SlabError::TooLarge);
        }
        let (rows, columns) = (to_index(rows), to_index(columns));
        if rows * columns > points.len().saturating_mul(MAX_NODES_PER_POINT) {
            return Err(SlabError::TooSparse);
        }

        let mut latitudes = axis(min_latitude, rows);
        let mut longitudes = axis(min_longitude, columns);
        let mut depth = vec![f64::NAN; rows * columns];
        let mut thickness = vec![f64::NAN; rows * columns];
        let mut filled = vec![false; rows * columns];
        for point in &points {
            let not_on_grid = SlabError::NotOnGrid { line: point.line };
            let i = grid_index(point.latitude - min_latitude).ok_or(not_on_grid)?;
            let j = grid_index(point.longitude - min_longitude).ok_or(not_on_grid)?;
            let node = i * columns + j;
            if filled[node] {
                return Err(SlabError::RepeatedPoint { line: point.line });
            }
            filled[node] = true;
            latitudes[i] = point.latitude;
            longitudes[j] = point.longitude;
            depth[node] = point.depth;
            thickness[node] = point.thickness;
        }
        Ok(Self::new(latitudes, longitudes, depth, thickness))
    }

    fn new(
        latitudes: Vec<f64>,
        longitudes: Vec<f64>,
        depth: Vec<f64>,
        thickness: Vec<f64>,
    ) -> Self {
        let cos_dip = cos_dip(&latitudes, longitudes.len(), &depth);
        Self {
            latitudes,
            longitudes,
            depth,
            thickness,
            cos_dip,
        }
    }

    /// Returns the values interpolated bilinearly at a point in degrees.
    ///
    /// Points outside the grid, NaN coordinates, and cells touching a node
    /// without a depth or thickness give NaN for all three values. The dip can
    /// also be NaN one node farther out, since it uses the neighboring depths.
    #[must_use]
    pub fn sample(&self, latitude: f64, longitude: f64) -> SlabSample {
        let inside = latitude >= self.latitudes[0]
            && latitude <= self.latitudes[self.latitudes.len() - 1]
            && longitude >= self.longitudes[0]
            && longitude <= self.longitudes[self.longitudes.len() - 1];
        if !inside {
            return MISSING;
        }
        let i = cell(&self.latitudes, latitude);
        let j = cell(&self.longitudes, longitude);
        let wy = (latitude - self.latitudes[i]) / (self.latitudes[i + 1] - self.latitudes[i]);
        let wx = (longitude - self.longitudes[j]) / (self.longitudes[j + 1] - self.longitudes[j]);
        let top_depth = self.interpolate(&self.depth, i, j, wy, wx);
        let thickness = self.interpolate(&self.thickness, i, j, wy, wx);
        if top_depth.is_nan() || thickness.is_nan() {
            return MISSING;
        }
        SlabSample {
            top_depth,
            thickness,
            cos_dip: self.interpolate(&self.cos_dip, i, j, wy, wx),
        }
    }

    fn interpolate(&self, values: &[f64], i: usize, j: usize, wy: f64, wx: f64) -> f64 {
        let columns = self.longitudes.len();
        let at = |i: usize, j: usize| values[i * columns + j];
        at(i, j) * (1.0 - wy) * (1.0 - wx)
            + at(i + 1, j) * wy * (1.0 - wx)
            + at(i, j + 1) * (1.0 - wy) * wx
            + at(i + 1, j + 1) * wy * wx
    }
}

/// The Pacific slab under Japan, from the Kuril and Izu-Bonin Slab2 grids.
#[derive(Clone, Debug)]
pub struct SlabGeometry {
    kuril: SlabGrid,
    izu_bonin: SlabGrid,
}

impl SlabGeometry {
    /// Combines the Kuril grid, used from 35 degrees north, with the Izu-Bonin
    /// grid, used south of it.
    #[must_use]
    pub fn new(kuril: SlabGrid, izu_bonin: SlabGrid) -> Self {
        Self { kuril, izu_bonin }
    }

    /// Returns the depth in km of the Pacific slab top under a point in
    /// degrees, or NaN where the grid has no value.
    #[must_use]
    pub fn pacific_top_depth(&self, latitude: f64, longitude: f64) -> f64 {
        self.pacific(latitude).sample(latitude, longitude).top_depth
    }

    /// Returns the hypocenter depth minus the Pacific slab top depth under a
    /// site, in km, or NaN when either depth is unknown.
    #[must_use]
    pub fn updip_depth_difference(
        &self,
        hypocenter: &Hypocenter,
        latitude: f64,
        longitude: f64,
    ) -> f64 {
        hypocenter.depth - self.pacific_top_depth(latitude, longitude)
    }

    /// Returns the fraction, from 0 to 1, of the straight path from the
    /// hypocenter to a site on the surface that lies inside the Pacific slab.
    ///
    /// The path is split into 60 equal segments, and a segment counts when its
    /// midpoint is between the slab top and bottom, measured perpendicular to
    /// the slab. Midpoints without slab values and an unknown hypocenter depth
    /// count as outside.
    #[must_use]
    pub fn pacific_path_fraction(
        &self,
        hypocenter: &Hypocenter,
        latitude: f64,
        longitude: f64,
    ) -> f64 {
        let start = cartesian(hypocenter.latitude, hypocenter.longitude, hypocenter.depth);
        let end = cartesian(latitude, longitude, 0.0);
        let mut inside = 0_u32;
        for k in 0..PATH_SAMPLES {
            let t = (f64::from(k) + 0.5) / f64::from(PATH_SAMPLES);
            let [x, y, z]: [f64; 3] =
                std::array::from_fn(|axis| start[axis] + t * (end[axis] - start[axis]));
            let radius = (x * x + y * y + z * z).sqrt();
            let point_latitude = libm::asin(z / radius).to_degrees();
            let point_longitude = libm::atan2(y, x).to_degrees();
            let sample = self
                .pacific(point_latitude)
                .sample(point_latitude, point_longitude);
            let perpendicular = (EARTH_RADIUS - radius - sample.top_depth) * sample.cos_dip;
            // Range checks are false for NaN, so points without values count as outside.
            if (0.0..=sample.thickness).contains(&perpendicular) {
                inside += 1;
            }
        }
        f64::from(inside) / f64::from(PATH_SAMPLES)
    }

    fn pacific(&self, latitude: f64) -> &SlabGrid {
        if latitude >= BOUNDARY_LATITUDE {
            &self.kuril
        } else {
            &self.izu_bonin
        }
    }
}

/// Returns the Earth-centered coordinates in km of a point at a depth in km
/// below the surface of a sphere with radius `EARTH_RADIUS`.
fn cartesian(latitude: f64, longitude: f64, depth: f64) -> [f64; 3] {
    let radius = EARTH_RADIUS - depth;
    let (latitude, longitude) = (latitude.to_radians(), longitude.to_radians());
    [
        radius * libm::cos(latitude) * libm::cos(longitude),
        radius * libm::cos(latitude) * libm::sin(longitude),
        radius * libm::sin(latitude),
    ]
}

fn parse_point(line: &str, line_number: usize) -> Result<Point, SlabError> {
    // Takes the columns one by one, so a long line is rejected without allocating.
    let mut columns = line.split(',');
    let (Some(longitude), Some(latitude), Some(depth), Some(thickness), None) = (
        columns.next(),
        columns.next(),
        columns.next(),
        columns.next(),
        columns.next(),
    ) else {
        return Err(SlabError::WrongColumnCount { line: line_number });
    };
    let number = |text: &str| -> Result<f64, SlabError> {
        text.parse()
            .map_err(|_| SlabError::NotANumber { line: line_number })
    };
    let point = Point {
        line: line_number,
        latitude: number(latitude)?,
        longitude: number(longitude)?,
        depth: number(depth)?,
        thickness: number(thickness)?,
    };
    // Range checks are false for NaN, so NaN coordinates are rejected too.
    if !((-90.0..=90.0).contains(&point.latitude) && (-180.0..=360.0).contains(&point.longitude)) {
        return Err(SlabError::InvalidCoordinate { line: line_number });
    }
    Ok(point)
}

fn bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| {
        (min.min(v), max.max(v))
    })
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass a rounded value between 0 and the checked grid size"
)]
fn to_index(value: f64) -> usize {
    value as usize
}

/// Returns the node index `offset` degrees from the first node, or `None` when
/// the offset is more than `GRID_TOLERANCE` nodes away from a node.
fn grid_index(offset: f64) -> Option<usize> {
    let nodes = offset / SPACING;
    let rounded = nodes.round();
    ((nodes - rounded).abs() <= GRID_TOLERANCE).then(|| to_index(rounded))
}

/// Returns `count` values from `min` at the grid spacing.
fn axis(min: f64, count: usize) -> Vec<f64> {
    std::iter::successors(Some(0.0), |k| Some(k + 1.0))
        .take(count)
        .map(|k: f64| min + k * SPACING)
        .collect()
}

/// Returns the last index whose axis value is at most `value`, kept within a cell.
fn cell(axis: &[f64], value: f64) -> usize {
    axis.partition_point(|&v| v <= value)
        .saturating_sub(1)
        .min(axis.len() - 2)
}

/// Returns the cosine of the slab dip at each node from the depth gradient:
/// central differences inside the grid and one-sided differences at its edges.
fn cos_dip(latitudes: &[f64], columns: usize, depth: &[f64]) -> Vec<f64> {
    let rows = latitudes.len();
    let dy = SPACING.to_radians() * EARTH_RADIUS;
    let mut result = Vec::with_capacity(depth.len());
    for (i, latitude) in latitudes.iter().enumerate() {
        let dx = dy * libm::cos(latitude.to_radians());
        for j in 0..columns {
            let gx = difference(|k| depth[i * columns + k], j, columns) / dx;
            let gy = difference(|k| depth[k * columns + j], i, rows) / dy;
            result.push(1.0 / (1.0 + gx * gx + gy * gy).sqrt());
        }
    }
    result
}

/// Returns the change of a value per node at `index` along a row or column.
fn difference(at: impl Fn(usize) -> f64, index: usize, count: usize) -> f64 {
    if index == 0 {
        at(1) - at(0)
    } else if index == count - 1 {
        at(count - 1) - at(count - 2)
    } else {
        (at(index + 1) - at(index - 1)) / 2.0
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{SlabGeometry, SlabGrid};

    /// Returns a geometry whose Kuril grid has its top at 300 km from 35
    /// degrees north and whose Izu-Bonin grid has its top at 250 km up to
    /// 34.95 degrees north, both 50 km thick, between 138 and 142 degrees east.
    pub fn geometry() -> SlabGeometry {
        SlabGeometry::new(
            flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0),
            flat(32.0, 34.95, 138.0, 142.0, 250.0, 50.0),
        )
    }

    /// Returns a grid at the Slab2 spacing with the same depth and thickness everywhere.
    pub fn flat(
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
        depth: f64,
        thickness: f64,
    ) -> SlabGrid {
        let rows = super::to_index(((max_latitude - min_latitude) / super::SPACING).round()) + 1;
        let columns =
            super::to_index(((max_longitude - min_longitude) / super::SPACING).round()) + 1;
        SlabGrid::new(
            super::axis(min_latitude, rows),
            super::axis(min_longitude, columns),
            vec![depth; rows * columns],
            vec![thickness; rows * columns],
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::test_support;
    use super::*;

    // Every corner differs, so swapped or ignored weights change the result.
    const FOUR_NODES: &str = "140,35,10,1\n140.05,35,20,2\n140,35.05,30,3\n140.05,35.05,80,4";

    // Deepens 10 km and then 30 km per node eastward on the equator, where
    // both node spacings are equal.
    const EASTWARD: &str = "140,0,100,10\n140.05,0,110,10\n140.1,0,140,10\n\
                            140,0.05,100,10\n140.05,0.05,110,10\n140.1,0.05,140,10";

    fn node_spacing() -> f64 {
        0.05_f64.to_radians() * 6371.0
    }

    fn cos_of_slope(slope: f64) -> f64 {
        1.0 / (1.0 + slope * slope).sqrt()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    fn assert_all_nan(sample: SlabSample) {
        assert!(
            sample.top_depth.is_nan() && sample.thickness.is_nan() && sample.cos_dip.is_nan(),
            "{sample:?}"
        );
    }

    #[test]
    fn flat_grid_samples_its_depth_and_thickness() {
        let sample =
            test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0).sample(36.123, 140.456);

        assert_close(sample.top_depth, 300.0, 1e-12);
        assert_close(sample.thickness, 50.0, 1e-12);
        assert_close(sample.cos_dip, 1.0, 1e-12);
    }

    #[test]
    fn outside_the_grid_is_nan() {
        let grid = test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0);

        assert_all_nan(grid.sample(34.99, 140.0));
        assert_all_nan(grid.sample(36.0, 142.01));
    }

    #[test]
    fn nan_coordinates_are_nan() {
        let grid = test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0);

        assert_all_nan(grid.sample(f64::NAN, 140.0));
        assert_all_nan(grid.sample(36.0, f64::NAN));
    }

    #[test]
    fn edges_are_inside() {
        let grid = test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0);

        assert_close(grid.sample(35.0, 138.0).top_depth, 300.0, 1e-12);
        assert_close(grid.sample(38.0, 142.0).top_depth, 300.0, 1e-12);
    }

    #[test]
    fn interpolates_bilinearly() {
        let sample = SlabGrid::parse(FOUR_NODES)
            .expect("four nodes should parse")
            .sample(35.01, 140.015);

        // Weights 0.2 north and 0.3 east of the south-west corner.
        assert_close(sample.top_depth, 19.4, 1e-9);
        assert_close(sample.thickness, 1.7, 1e-9);
    }

    #[test]
    fn picks_the_cell_containing_the_point() {
        // Depth 10 (i^2 + j^2) km at node (i, j) is not bilinear, so a wrong
        // cell gives a wrong value.
        let grid = SlabGrid::parse(
            "140,35,0,10\n140.05,35,10,10\n140.1,35,40,10\n\
             140,35.05,10,10\n140.05,35.05,20,10\n140.1,35.05,50,10\n\
             140,35.1,40,10\n140.05,35.1,50,10\n140.1,35.1,80,10",
        )
        .expect("nine nodes should parse");

        assert_close(grid.sample(35.075, 140.0625).top_depth, 42.5, 1e-9);
        assert_close(grid.sample(35.05, 140.025).top_depth, 15.0, 1e-9);
        assert_close(grid.sample(35.1, 140.1).top_depth, 80.0, 1e-9);
    }

    #[test]
    fn cos_dip_uses_central_differences_inside_and_one_sided_at_the_edges() {
        let grid = SlabGrid::parse(EASTWARD).expect("the eastward grid should parse");
        let spacing = node_spacing();

        assert_close(
            grid.sample(0.0, 140.0).cos_dip,
            cos_of_slope(10.0 / spacing),
            1e-12,
        );
        assert_close(
            grid.sample(0.0, 140.05).cos_dip,
            cos_of_slope(20.0 / spacing),
            1e-12,
        );
        assert_close(
            grid.sample(0.0, 140.1).cos_dip,
            cos_of_slope(30.0 / spacing),
            1e-12,
        );
    }

    #[test]
    fn cos_dip_is_interpolated_between_nodes() {
        let grid = SlabGrid::parse(EASTWARD).expect("the eastward grid should parse");
        let spacing = node_spacing();

        assert_close(
            grid.sample(0.0, 140.025).cos_dip,
            f64::midpoint(cos_of_slope(10.0 / spacing), cos_of_slope(20.0 / spacing)),
            1e-12,
        );
    }

    #[test]
    fn cos_dip_follows_a_north_south_gradient() {
        // Deepens 10 km per node northward.
        let grid =
            SlabGrid::parse("140,0,100,10\n140.05,0,100,10\n140,0.05,110,10\n140.05,0.05,110,10")
                .expect("the northward grid should parse");

        assert_close(
            grid.sample(0.0, 140.0).cos_dip,
            cos_of_slope(10.0 / node_spacing()),
            1e-12,
        );
    }

    #[test]
    fn cos_dip_narrows_the_east_west_spacing_by_latitude() {
        // Deepens 10 km per node eastward at 60 degrees north, where a node
        // east is half as far as a node north.
        let grid = SlabGrid::parse(
            "140,60,100,10\n140.05,60,110,10\n140,60.05,100,10\n140.05,60.05,110,10",
        )
        .expect("the grid at 60 degrees north should parse");

        assert_close(
            grid.sample(60.0, 140.0).cos_dip,
            cos_of_slope(10.0 / (node_spacing() * 60.0_f64.to_radians().cos())),
            1e-12,
        );
    }

    #[test]
    fn missing_nodes_make_nan_nearby() {
        let grid = SlabGrid::parse("140,35,100,10\n140.05,35,200,10\n140,35.05,100,10")
            .expect("three nodes should parse");

        assert_all_nan(grid.sample(35.02, 140.025));
    }

    #[test]
    fn a_node_missing_one_value_makes_its_cells_nan() {
        let no_thickness = SlabGrid::parse(&FOUR_NODES.replace("140,35,10,1", "140,35,10,NaN"))
            .expect("a NaN thickness should parse");
        let no_depth = SlabGrid::parse(&FOUR_NODES.replace("140,35,10,1", "140,35,NaN,1"))
            .expect("a NaN depth should parse");

        assert_all_nan(no_thickness.sample(35.025, 140.025));
        assert_all_nan(no_depth.sample(35.025, 140.025));
    }

    #[test]
    fn str_parse_reads_the_same_as_parse() {
        let grid: SlabGrid = FOUR_NODES.parse().expect("four nodes should parse");

        assert_eq!(
            grid.sample(35.02, 140.025),
            SlabGrid::parse(FOUR_NODES)
                .expect("four nodes should parse")
                .sample(35.02, 140.025)
        );
    }

    #[test]
    fn crlf_reads_the_same_as_lf() {
        let lf = SlabGrid::parse(FOUR_NODES).expect("LF should parse");
        let crlf = SlabGrid::parse(&FOUR_NODES.replace('\n', "\r\n")).expect("CRLF should parse");

        assert_eq!(lf.sample(35.02, 140.025), crlf.sample(35.02, 140.025));
    }

    #[test]
    fn rejects_a_line_without_four_columns() {
        assert_eq!(
            SlabGrid::parse("140,35,100").unwrap_err(),
            SlabError::WrongColumnCount { line: 1 }
        );
        assert_eq!(
            SlabGrid::parse("140,35,100,10,5").unwrap_err(),
            SlabError::WrongColumnCount { line: 1 }
        );
    }

    #[test]
    fn rejects_a_value_that_is_not_a_number() {
        assert_eq!(
            SlabGrid::parse("140,35,100,10\n140.05,35,deep,10").unwrap_err(),
            SlabError::NotANumber { line: 2 }
        );
    }

    #[test]
    fn rejects_an_empty_file() {
        assert_eq!(SlabGrid::parse("").unwrap_err(), SlabError::NoPoints);
    }

    #[test]
    fn rejects_a_grid_smaller_than_2x2() {
        assert_eq!(
            SlabGrid::parse("140,35,100,10\n140,35.05,100,10").unwrap_err(),
            SlabError::TooSmall
        );
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        assert_eq!(
            SlabGrid::parse("NaN,35,100,10").unwrap_err(),
            SlabError::InvalidCoordinate { line: 1 }
        );
        assert_eq!(
            SlabGrid::parse("140,35,100,10\n140,inf,100,10").unwrap_err(),
            SlabError::InvalidCoordinate { line: 2 }
        );
    }

    #[test]
    fn rejects_out_of_range_coordinates() {
        assert_eq!(
            SlabGrid::parse("140,91,100,10").unwrap_err(),
            SlabError::InvalidCoordinate { line: 1 }
        );
    }

    #[test]
    fn rejects_a_point_off_the_grid() {
        assert_eq!(
            SlabGrid::parse(&FOUR_NODES.replace("140.05,35.05", "140.06,35.05")).unwrap_err(),
            SlabError::NotOnGrid { line: 4 }
        );
    }

    #[test]
    fn rejects_a_repeated_point() {
        assert_eq!(
            SlabGrid::parse(&format!("{FOUR_NODES}\n140,35,10,1")).unwrap_err(),
            SlabError::RepeatedPoint { line: 5 }
        );
    }

    #[test]
    fn rejects_a_huge_grid_without_allocating() {
        assert_eq!(
            SlabGrid::parse("-180,-90,1,1\n360,90,1,1").unwrap_err(),
            SlabError::TooLarge
        );
    }

    #[test]
    fn rejects_a_grid_far_larger_than_its_points() {
        // Two corners 2 degrees apart span 41 x 41 nodes, but give 2 points.
        assert_eq!(
            SlabGrid::parse("140,35,10,1\n142,37,10,1").unwrap_err(),
            SlabError::TooSparse
        );
        // Two corners spanning 4 x 8 nodes give 16 nodes for each point, the
        // most accepted; 3 x 11 give one node more.
        SlabGrid::parse("140,35,10,1\n140.35,35.15,10,1").expect("32 nodes should parse");
        assert_eq!(
            SlabGrid::parse("140,35,10,1\n140.5,35.1,10,1").unwrap_err(),
            SlabError::TooSparse
        );
        // A grid of the same size with a point on every node is read.
        let mut lines = Vec::new();
        for i in 0..=40 {
            for j in 0..=40 {
                let latitude = 35.0 + f64::from(i) * super::SPACING;
                let longitude = 140.0 + f64::from(j) * super::SPACING;
                lines.push(format!("{longitude},{latitude},10,1"));
            }
        }
        let full = SlabGrid::parse(&lines.join("\n")).expect("a full grid should parse");
        assert_eq!(full.sample(36.0, 141.0).top_depth, 10.0);
    }

    #[test]
    fn reads_the_repository_slab2() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/slab2");
        let kuril = fs::read_to_string(directory.join("kur.csv")).expect("kur.csv should exist");
        let izu_bonin =
            fs::read_to_string(directory.join("izu.csv")).expect("izu.csv should exist");

        SlabGrid::parse(&izu_bonin).expect("izu.csv should parse");
        let depth = SlabGrid::parse(&kuril)
            .expect("kur.csv should parse")
            .sample(40.0, 144.0)
            .top_depth;
        assert!(depth.is_finite(), "{depth}");
    }

    fn hypocenter(latitude: f64, longitude: f64, depth: f64) -> Hypocenter {
        Hypocenter {
            latitude,
            longitude,
            depth,
            magnitude: 6.0,
        }
    }

    #[test]
    fn top_depth_uses_kuril_north_of_35_and_izu_south() {
        let geometry = test_support::geometry();

        assert_close(geometry.pacific_top_depth(35.0, 140.0), 300.0, 1e-9);
        assert_close(geometry.pacific_top_depth(34.9, 140.0), 250.0, 1e-9);
        let between = geometry.pacific_top_depth(34.97, 140.0);
        assert!(between.is_nan(), "{between}");
    }

    #[test]
    fn updip_is_hypocenter_depth_minus_top() {
        let geometry = test_support::geometry();
        let deep = hypocenter(36.5, 140.0, 400.0);
        let shallow = hypocenter(36.5, 140.0, 200.0);

        assert_close(
            geometry.updip_depth_difference(&deep, 37.0, 141.0),
            100.0,
            1e-9,
        );
        assert_close(
            geometry.updip_depth_difference(&shallow, 37.0, 141.0),
            -100.0,
            1e-9,
        );
        // The top is taken under the site, here on the Izu-Bonin grid.
        assert_close(
            geometry.updip_depth_difference(&deep, 34.9, 140.0),
            150.0,
            1e-9,
        );
        let outside = geometry.updip_depth_difference(&deep, 40.0, 141.0);
        assert!(outside.is_nan(), "{outside}");
    }

    #[test]
    fn vertical_path_through_flat_slab() {
        // The slab is 300 to 350 km deep. Going up from 320 km, the path is in
        // it for its first 1/16, which holds the midpoints of 4 of the 60
        // segments. From 330 km it is the first 1/11: 5 midpoints but 6 segment
        // starts. From 380 km it is from 3/38 to 8/38 of the path: 8 midpoints.
        let geometry = test_support::geometry();
        let from = |depth: f64| {
            geometry.pacific_path_fraction(&hypocenter(36.5, 140.0, depth), 36.5, 140.0)
        };

        assert_close(from(320.0), 4.0 / 60.0, 1e-12);
        assert_close(from(330.0), 5.0 / 60.0, 1e-12);
        assert_close(from(380.0), 8.0 / 60.0, 1e-12);
    }

    #[test]
    fn path_through_a_dipping_slab_uses_the_perpendicular_thickness() {
        // On the equator the Izu-Bonin slab deepens eastward by 4/3 km per km,
        // so cos_dip is 0.6 and the 30 km thick slab spans 50 km vertically.
        // Going up from 80 km below its top, the path is in it from 30/D to
        // 80/D of the path (D is the hypocenter depth): 7 midpoints, but only
        // 4 when the thickness is taken as vertical.
        let step = 4.0 / 3.0 * node_spacing();
        let izu_bonin = SlabGrid::parse(&format!(
            "140,0,300,30\n140.05,0,{middle_depth},30\n140.1,0,{east_depth},30\n\
             140,0.05,300,30\n140.05,0.05,{middle_depth},30\n140.1,0.05,{east_depth},30",
            middle_depth = 300.0 + step,
            east_depth = 300.0 + 2.0 * step,
        ))
        .expect("the dipping grid should parse");
        let geometry = SlabGeometry::new(
            test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0),
            izu_bonin,
        );
        let depth = 300.0 + step + 80.0;

        assert_close(
            geometry.pacific_path_fraction(&hypocenter(0.0, 140.05, depth), 0.0, 140.05),
            7.0 / 60.0,
            1e-12,
        );
    }

    #[test]
    fn path_follows_the_chord_below_the_surface() {
        // Between two surface points 4 degrees apart on the equator, the chord
        // sinks to about 3.9 km and is 1 to 2 km deep for 10 of the 60 midpoints.
        let geometry = SlabGeometry::new(
            test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0),
            test_support::flat(-1.0, 1.0, 137.0, 143.0, 1.0, 1.0),
        );

        let fraction = geometry.pacific_path_fraction(&hypocenter(0.0, 138.0, 0.0), 0.0, 142.0);

        assert_close(fraction, 10.0 / 60.0, 1e-12);
    }

    #[test]
    fn path_switches_grids_at_35_degrees() {
        // Both grids cover the whole path, but only the Kuril slab (10 to 40
        // km) is shallow enough. The first 36 midpoints are north of 35 degrees.
        let geometry = SlabGeometry::new(
            test_support::flat(34.0, 38.0, 138.0, 142.0, 10.0, 30.0),
            test_support::flat(33.0, 37.0, 138.0, 142.0, 100.0, 30.0),
        );

        let fraction = geometry.pacific_path_fraction(&hypocenter(35.6, 140.0, 30.0), 34.6, 140.0);

        assert_close(fraction, 36.0 / 60.0, 1e-12);
    }

    #[test]
    fn slab_top_and_bottom_are_inside() {
        // Going up from 80 km, the midpoints at 7/8 and 5/8 of the path are
        // exactly 10 and 30 km deep, the top and bottom of the slab.
        let geometry = SlabGeometry::new(
            test_support::flat(35.0, 38.0, 138.0, 142.0, 300.0, 50.0),
            test_support::flat(-1.0, 1.0, -1.0, 1.0, 10.0, 20.0),
        );

        let fraction = geometry.pacific_path_fraction(&hypocenter(0.0, 0.0, 80.0), 0.0, 0.0);

        assert_close(fraction, 16.0 / 60.0, 1e-12);
    }

    #[test]
    fn path_outside_the_grids_is_zero() {
        let geometry = SlabGeometry::new(
            test_support::flat(37.0, 38.0, 138.0, 142.0, 300.0, 50.0),
            test_support::flat(32.0, 34.95, 138.0, 142.0, 250.0, 50.0),
        );

        let fraction = geometry.pacific_path_fraction(&hypocenter(36.5, 140.0, 320.0), 36.5, 140.0);

        assert_close(fraction, 0.0, 0.0);
    }

    #[test]
    fn unknown_depth_gives_nan_updip_and_zero_path_fraction() {
        let geometry = test_support::geometry();
        let hypocenter = hypocenter(36.5, 140.0, f64::NAN);

        let updip = geometry.updip_depth_difference(&hypocenter, 37.0, 141.0);
        assert!(updip.is_nan(), "{updip}");
        assert_close(
            geometry.pacific_path_fraction(&hypocenter, 36.5, 140.0),
            0.0,
            0.0,
        );
    }
}

//! The part of a model file that holds one Slab2 grid: its axes and its
//! row-major depth and thickness. The dip cosines are not stored but computed
//! again when the grid is read.

use crate::bundle::FormatError;
use crate::bundle::codec::{self, Reader, Writer};
use crate::slab::{self, SlabGrid};

/// Writes the axes and nodes of a grid.
///
/// # Panics
///
/// Panics if the number of rows or columns does not fit in `u32`.
pub(crate) fn write(writer: &mut Writer, grid: &SlabGrid) {
    writer.u32(codec::to_u32(grid.latitudes().len()));
    writer.u32(codec::to_u32(grid.longitudes().len()));
    writer.f64s(grid.latitudes());
    writer.f64s(grid.longitudes());
    writer.f64s(grid.depth());
    writer.f64s(grid.thickness());
}

/// Reads what [`write`] wrote.
pub(crate) fn read(reader: &mut Reader<'_>) -> Result<SlabGrid, FormatError> {
    let rows = reader.u32()?;
    let columns = reader.u32()?;
    // Two u32 values multiply without overflow in u64.
    let nodes = u64::from(rows) * u64::from(columns);
    if nodes > u64::from(slab::MAX_NODES) {
        return Err(FormatError::LimitExceeded {
            what: "nodes in a slab grid",
        });
    }
    let rows = codec::to_usize(u64::from(rows))?;
    let columns = codec::to_usize(u64::from(columns))?;
    // Checked before the arrays are read, so that an empty axis does not let
    // the other run to any length the bytes allow.
    if rows < 2 || columns < 2 {
        return Err(FormatError::InvalidGrid);
    }
    let latitudes = reader.f64s(rows)?;
    let longitudes = reader.f64s(columns)?;
    let nodes = codec::to_usize(nodes)?;
    let depth = reader.f64s(nodes)?;
    let thickness = reader.f64s(nodes)?;
    SlabGrid::from_nodes(latitudes, longitudes, depth, thickness).ok_or(FormatError::InvalidGrid)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    // Rows and columns of the sample points across a grid, 1,000 in all.
    const SAMPLE_ROWS: u32 = 40;
    const SAMPLE_COLUMNS: u32 = 25;

    fn round_trip(grid: &SlabGrid) -> SlabGrid {
        let mut writer = Writer::new();
        super::write(&mut writer, grid);
        let bytes = writer.into_bytes();
        let mut reader = Reader::new(&bytes);
        let read = super::read(&mut reader).expect("the written grid should be valid");
        assert_eq!(reader.finish(), Ok(()));
        read
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn assert_same_bits(actual: &SlabGrid, expected: &SlabGrid) {
        assert_eq!(bits(actual.latitudes()), bits(expected.latitudes()));
        assert_eq!(bits(actual.longitudes()), bits(expected.longitudes()));
        assert_eq!(bits(actual.depth()), bits(expected.depth()));
        assert_eq!(bits(actual.thickness()), bits(expected.thickness()));
    }

    /// Returns a grid section with the axes given and every node 1 km deep
    /// and thick.
    fn section(latitudes: &[f64], longitudes: &[f64]) -> Vec<u8> {
        let nodes = latitudes.len() * longitudes.len();
        let mut writer = Writer::new();
        writer.u32(codec::to_u32(latitudes.len()));
        writer.u32(codec::to_u32(longitudes.len()));
        writer.f64s(latitudes);
        writer.f64s(longitudes);
        writer.f64s(&vec![1.0; nodes]);
        writer.f64s(&vec![1.0; nodes]);
        writer.into_bytes()
    }

    fn read_error(bytes: &[u8]) -> FormatError {
        super::read(&mut Reader::new(bytes)).expect_err("the grid should be rejected")
    }

    /// Returns a section that declares `rows` x `columns` and has no arrays.
    fn header_only(rows: u32, columns: u32) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.u32(rows);
        writer.u32(columns);
        writer.into_bytes()
    }

    #[test]
    fn repository_slab2_round_trips_bit_for_bit() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/slab2");
        for name in ["kur.csv", "izu.csv"] {
            let text = fs::read_to_string(directory.join(name))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let grid = SlabGrid::parse(&text).expect("the grid should parse");
            let read = round_trip(&grid);
            assert_same_bits(&read, &grid);

            let latitudes = grid.latitudes();
            let longitudes = grid.longitudes();
            let (south, north) = (latitudes[0], latitudes[latitudes.len() - 1]);
            let (west, east) = (longitudes[0], longitudes[longitudes.len() - 1]);
            for i in 0..SAMPLE_ROWS {
                let latitude = south + (north - south) * f64::from(i) / f64::from(SAMPLE_ROWS - 1);
                for j in 0..SAMPLE_COLUMNS {
                    let longitude =
                        west + (east - west) * f64::from(j) / f64::from(SAMPLE_COLUMNS - 1);
                    let expected = grid.sample(latitude, longitude);
                    let actual = read.sample(latitude, longitude);
                    let point = format!("{name} at {latitude}, {longitude}");
                    assert_eq!(
                        actual.top_depth.to_bits(),
                        expected.top_depth.to_bits(),
                        "{point}"
                    );
                    assert_eq!(
                        actual.thickness.to_bits(),
                        expected.thickness.to_bits(),
                        "{point}"
                    );
                    assert_eq!(
                        actual.cos_dip.to_bits(),
                        expected.cos_dip.to_bits(),
                        "{point}"
                    );
                }
            }
        }
    }

    #[test]
    fn reads_a_section_written_by_hand() {
        let grid = super::read(&mut Reader::new(&section(
            &[35.0, 35.05],
            &[140.0, 140.05, 140.1],
        )))
        .expect("the grid should be valid");

        assert_eq!(grid.latitudes(), [35.0, 35.05]);
        assert_eq!(grid.longitudes(), [140.0, 140.05, 140.1]);
        assert_eq!(grid.sample(35.02, 140.07).top_depth, 1.0);
    }

    #[test]
    fn keeps_nan_nodes() {
        // 2 x 2 nodes, the one at 35.05, 140.05 without a point.
        let grid = SlabGrid::parse("140,35,10,1\n140.05,35,20,2\n140,35.05,30,3")
            .expect("the grid should parse");
        let read = round_trip(&grid);

        let nan_nodes: Vec<bool> = read.depth().iter().map(|value| value.is_nan()).collect();
        assert_eq!(nan_nodes, [false, false, false, true]);
        let top_depth = read.sample(35.02, 140.02).top_depth;
        assert!(top_depth.is_nan(), "{top_depth}");
    }

    #[test]
    fn rejects_fewer_than_two_rows_or_columns() {
        assert_eq!(
            read_error(&section(&[35.0], &[140.0, 140.05])),
            FormatError::InvalidGrid
        );
        assert_eq!(
            read_error(&section(&[35.0, 35.05], &[140.0])),
            FormatError::InvalidGrid
        );
    }

    #[test]
    fn rejects_a_short_axis_before_reading_the_arrays() {
        assert_eq!(
            read_error(&header_only(0, u32::MAX)),
            FormatError::InvalidGrid
        );
        assert_eq!(
            read_error(&header_only(3_000_000, 1)),
            FormatError::InvalidGrid
        );
    }

    #[test]
    fn rejects_more_nodes_than_the_limit() {
        let limit = FormatError::LimitExceeded {
            what: "nodes in a slab grid",
        };
        assert_eq!(read_error(&header_only(2001, 2001)), limit);
        assert_eq!(read_error(&header_only(u32::MAX, u32::MAX)), limit);
    }

    #[test]
    fn rejects_arrays_beyond_the_bytes() {
        assert_eq!(read_error(&header_only(2000, 2000)), FormatError::Truncated);
    }

    #[test]
    fn rejects_an_axis_off_the_grid() {
        assert_eq!(
            read_error(&section(&[35.0, 35.07], &[140.0, 140.05])),
            FormatError::InvalidGrid
        );
    }

    #[test]
    fn rejects_a_decreasing_axis() {
        assert_eq!(
            read_error(&section(&[35.05, 35.0], &[140.0, 140.05])),
            FormatError::InvalidGrid
        );
    }

    #[test]
    fn rejects_non_finite_or_out_of_range_axes() {
        let longitudes = [140.0, 140.05];
        for latitudes in [[f64::NAN, 35.05], [90.0, 90.05]] {
            assert_eq!(
                read_error(&section(&latitudes, &longitudes)),
                FormatError::InvalidGrid,
                "{latitudes:?}"
            );
        }
        assert_eq!(
            read_error(&section(&[35.0, 35.05], &[360.0, 360.05])),
            FormatError::InvalidGrid,
            "longitudes beyond 360"
        );
    }
}

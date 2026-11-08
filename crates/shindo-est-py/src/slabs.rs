//! The slab geometry, for computing the features of training rows.

use std::fs;
use std::path::{Path, PathBuf};
use std::str;

use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLike1};
use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use shindo_est::features::{self, FEATURE_COUNT};
use shindo_est::{Hypocenter, Site, SlabGrid};

use crate::arrays;
use crate::errors::{self, SlabGridError};

/// The Pacific slab under Japan, from the Kuril and Izu-Bonin Slab2 grids.
///
/// Read one with `SlabGeometry.read`. A slab geometry never changes, and
/// `features` releases the GIL while it computes, so Python threads can share
/// one. It cannot be pickled, so it cannot be passed to `multiprocessing`;
/// read the grids in each process instead.
#[pyclass(frozen, module = "shindo_est")]
pub struct SlabGeometry {
    slabs: shindo_est::SlabGeometry,
}

#[pymethods]
impl SlabGeometry {
    /// Reads the Kuril and Izu-Bonin grids.
    ///
    /// Each grid is a CSV file of `lon,lat,depth,thickness` rows, in degrees
    /// and km with depth positive down, as in `data/slab2`. The Kuril grid is
    /// used from 35 degrees north, and the Izu-Bonin grid south of it.
    ///
    /// Args:
    ///     kuril: The path of the Kuril grid.
    ///     izu_bonin: The path of the Izu-Bonin grid.
    ///
    /// Returns:
    ///     The slab geometry.
    ///
    /// Raises:
    ///     `OSError`: A file cannot be read; a subclass such as
    ///         `FileNotFoundError` tells why. The message starts with the
    ///         path.
    ///     `SlabGridError`: A file is not a valid grid. The message starts
    ///         with the path.
    #[staticmethod]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 passes arguments by value"
    )]
    #[expect(
        clippy::doc_markdown,
        reason = "the docstring is Python's, whose Args name parameters without backticks"
    )]
    fn read(kuril: PathBuf, izu_bonin: PathBuf) -> PyResult<Self> {
        let slabs = shindo_est::SlabGeometry::new(read_grid(&kuril)?, read_grid(&izu_bonin)?);
        Ok(Self { slabs })
    }

    /// Computes the features of rows of a hypocenter and a site.
    ///
    /// Each row pairs the hypocenter and the site at the same index of the
    /// arrays, as the rows of a training set pair earthquakes and stations.
    /// The arguments are one-dimensional arrays of the same length, with one
    /// value for each row. The values are not validated: NaN inputs
    /// propagate to the features that depend on them, except that `path_frac`
    /// is 0 when any latitude, longitude, or depth is NaN. The rows are
    /// computed on one thread with the GIL released, so Python threads that
    /// each take a part of the rows compute in parallel. Convert the result
    /// to float32 before passing it to LightGBM, which rounds the features to
    /// float32 when it compares them in its trees.
    ///
    /// Args:
    ///     hypocenter_latitude: The latitude of the hypocenter of each row in
    ///         degrees.
    ///     hypocenter_longitude: The longitude of the hypocenter of each row
    ///         in degrees.
    ///     hypocenter_depth: The depth of the hypocenter of each row in km.
    ///     hypocenter_magnitude: The JMA magnitude (Mj) of the hypocenter of
    ///         each row.
    ///     site_latitude: The latitude of the site of each row in degrees.
    ///     site_longitude: The longitude of the site of each row in degrees.
    ///     site_arv: The ARV of the site of each row.
    ///
    /// Returns:
    ///     A new C-ordered float64 array with one row for each index and one
    ///     column for each name in `FEATURE_NAMES`, in that order.
    ///
    /// Raises:
    ///     `ValueError`: The arrays differ in length.
    ///     `TypeError`: An array is not one-dimensional.
    #[pyo3(signature = (
        *,
        hypocenter_latitude,
        hypocenter_longitude,
        hypocenter_depth,
        hypocenter_magnitude,
        site_latitude,
        site_longitude,
        site_arv,
    ))]
    #[expect(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        reason = "PyO3 passes arguments by value, one for each Python parameter"
    )]
    #[expect(
        clippy::doc_markdown,
        reason = "the docstring is Python's, whose Args name parameters without backticks"
    )]
    fn features<'py>(
        &self,
        py: Python<'py>,
        hypocenter_latitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        hypocenter_longitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        hypocenter_depth: PyArrayLike1<'py, f64, AllowTypeChange>,
        hypocenter_magnitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        site_latitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        site_longitude: PyArrayLike1<'py, f64, AllowTypeChange>,
        site_arv: PyArrayLike1<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let hypocenter_latitudes = hypocenter_latitude.as_array();
        let hypocenter_longitudes = hypocenter_longitude.as_array();
        let depths = hypocenter_depth.as_array();
        let magnitudes = hypocenter_magnitude.as_array();
        let site_latitudes = site_latitude.as_array();
        let site_longitudes = site_longitude.as_array();
        let arvs = site_arv.as_array();
        let length = arrays::common_length(&[
            ("hypocenter_latitude", hypocenter_latitudes.len()),
            ("hypocenter_longitude", hypocenter_longitudes.len()),
            ("hypocenter_depth", depths.len()),
            ("hypocenter_magnitude", magnitudes.len()),
            ("site_latitude", site_latitudes.len()),
            ("site_longitude", site_longitudes.len()),
            ("site_arv", arvs.len()),
        ])?;
        // Copied while the GIL is held, so that another Python thread that
        // changes an input array cannot change the values computed from it.
        let mut hypocenters = Vec::with_capacity(length);
        let mut sites = Vec::with_capacity(length);
        for row in 0..length {
            hypocenters.push(Hypocenter {
                latitude: hypocenter_latitudes[row],
                longitude: hypocenter_longitudes[row],
                depth: depths[row],
                magnitude: magnitudes[row],
            });
            sites.push(Site {
                latitude: site_latitudes[row],
                longitude: site_longitudes[row],
                arv: arvs[row],
            });
        }
        let values = py.detach(|| {
            let mut values = Vec::with_capacity(length * FEATURE_COUNT);
            let pairs = hypocenters.iter().zip(&sites);
            for (hypocenter, site) in pairs {
                values.extend_from_slice(&features::compute(hypocenter, site, &self.slabs));
            }
            values
        });
        let rows = Array2::from_shape_vec((length, FEATURE_COUNT), values)
            .expect("each row has FEATURE_COUNT values");
        Ok(rows.into_pyarray(py))
    }
}

/// Reads the grid in the file at `path`.
///
/// # Errors
///
/// Returns an `OSError` when the file cannot be read, and a `SlabGridError`
/// when it is not UTF-8 text or not a valid grid.
fn read_grid(path: &Path) -> PyResult<SlabGrid> {
    let bytes = fs::read(path).map_err(|error| errors::from_io_error(path, &error))?;
    let text = str::from_utf8(&bytes).map_err(|_| {
        SlabGridError::new_err(format!("{}: the grid is not UTF-8 text", path.display()))
    })?;
    SlabGrid::parse(text).map_err(|error| errors::from_slab_error(path, &error))
}

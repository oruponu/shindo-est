//! Seismic intensity estimation from a hypocenter and site parameters using `LightGBM` trees.
//!
//! A [`Predictor`] combines a felt classifier and an intensity regressor,
//! read from `LightGBM`'s text format, with the Slab2 grids of the Pacific
//! slab and the hurdle settings, and predicts the instrumental intensity at
//! many sites for one hypocenter.
//!
//! # Examples
//!
//! ```
//! use shindo_est::features::FEATURE_NAMES;
//! use shindo_est::{
//!     HurdleParams, Hypocenter, IntensityClass, Options, PointResult, PointStatus, Predictor,
//!     Site, SlabGeometry, SlabGrid, lightgbm_text,
//! };
//!
//! // A one-tree model in LightGBM's text format that always gives `leaf`.
//! fn model(objective: &str, leaf: f64) -> String {
//!     format!(
//!         "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=13\n\
//!          objective={objective}\nfeature_names={}\n\nTree=0\nnum_leaves=1\nnum_cat=0\n\
//!          is_linear=0\nsplit_feature=\nthreshold=\ndecision_type=\nleft_child=\n\
//!          right_child=\nleaf_value={leaf}\nshrinkage=1\n\n\nend of trees\n",
//!         FEATURE_NAMES.join(" ")
//!     )
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let classifier = lightgbm_text::parse(&model("binary sigmoid:1", 2.0))?;
//! let regressor = lightgbm_text::parse(&model("regression", 3.1))?;
//! // A slab top at 100 km under 35.00 to 35.05 N, 140.00 to 140.05 E.
//! let grid = "140,35,100,50\n140.05,35,100,50\n140,35.05,100,50\n140.05,35.05,100,50";
//! let slabs = SlabGeometry::new(SlabGrid::parse(grid)?, SlabGrid::parse(grid)?);
//! let hurdle = HurdleParams {
//!     felt_probability: 0.35,
//!     felt_floor: 0.5,
//!     unfelt_ceiling: 0.49,
//!     logit_per_intensity: 1.5,
//! };
//! let predictor = Predictor::new(classifier, regressor, slabs, hurdle)?;
//!
//! let hypocenter = Hypocenter {
//!     latitude: 36.0,
//!     longitude: 140.0,
//!     depth: 10.0,
//!     magnitude: 6.0,
//! };
//! let sites = [Site {
//!     latitude: 35.7,
//!     longitude: 139.8,
//!     arv: 1.2,
//! }];
//! let results = predictor.predict(&hypocenter, &sites, &Options::default())?;
//!
//! // The classifier's logit of 2 is felt, so the regressor's 3.1 stands.
//! assert_eq!(results[0].status, PointStatus::Ok);
//! assert_eq!(results[0].instrumental, 3.1);
//! assert_eq!(
//!     IntensityClass::from_instrumental(results[0].instrumental),
//!     Some(IntensityClass::Three)
//! );
//!
//! // A caller that predicts again and again can reuse one buffer.
//! let mut buffer = vec![PointResult::default(); sites.len()];
//! predictor.predict_into(&hypocenter, &sites, &Options::default(), &mut buffer)?;
//! assert_eq!(buffer, results);
//! # Ok(())
//! # }
//! ```

pub mod features;
pub mod forest;
mod geodesy;
pub mod hurdle;
pub mod input;
pub mod intensity;
pub mod lightgbm_text;
pub mod model;
pub mod predict;
pub mod slab;
pub mod validate;

pub use hurdle::HurdleParams;
pub use input::{Hypocenter, Site};
pub use intensity::IntensityClass;
pub use model::{Model, ModelError, Objective};
pub use predict::{Options, PointResult, PointStatus, Predictor};
pub use slab::{SlabError, SlabGeometry, SlabGrid};
pub use validate::InputError;

//! Model files: the models and the data they need, in one binary file.

use std::fmt;
use std::str;

use sha2::{Digest, Sha256};

use crate::bundle::codec::{Reader, Writer};
use crate::features::{FEATURE_NAMES, FEATURE_VERSION};
use crate::manifest::{Manifest, ManifestError};
use crate::model::{Model, ModelError, Objective};
use crate::predict::Predictor;
use crate::slab::{SlabGeometry, SlabGrid};

mod codec;
mod forest_section;
mod grid_section;

/// The first bytes of every model file. As in PNG, the first byte has its
/// high bit set, so that a 7-bit transfer changes it; CR LF and LF change
/// under line-ending conversion; and 0x1A stops a text display.
pub const MAGIC: [u8; 8] = *b"\x89SHE\r\n\x1a\n";

/// The version of the layout of model files that this library reads and
/// writes.
pub const FORMAT_VERSION: u32 = 1;

/// Most bytes the manifest may take in a model file.
const MAX_MANIFEST_BYTES: usize = 65_536;

/// Bytes of the SHA-256 checksum that ends a model file.
const CHECKSUM_BYTES: usize = 32;

/// Bytes of the magic and the format version.
const HEADER_BYTES: usize = 12;

/// Bytes of a section name.
const TAG_BYTES: usize = 4;

/// The names of the sections, in the order a model file has them.
mod tags {
    pub(super) const MANIFEST: &str = "MNFT";
    pub(super) const CLASSIFIER: &str = "CLSF";
    pub(super) const REGRESSOR: &str = "RGSR";
    pub(super) const KURIL: &str = "SLBK";
    pub(super) const IZU_BONIN: &str = "SLBI";
}

/// An error in the bytes of a model file, or in the parts given to make one.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum FormatError {
    /// The bytes do not start with [`MAGIC`].
    NotAModelFile,
    /// The file has a layout version other than [`FORMAT_VERSION`].
    UnsupportedVersion {
        /// The version read from the file.
        found: u32,
    },
    /// The checksum at the end does not match the bytes before it.
    ChecksumMismatch,
    /// Another section is where a section should be.
    WrongSection {
        /// The name of the section that should be there.
        expected: &'static str,
    },
    /// The manifest is not UTF-8.
    ManifestNotUtf8,
    /// The manifest is invalid.
    Manifest(ManifestError),
    /// The models take features of a version other than
    /// [`FEATURE_VERSION`].
    FeatureVersion {
        /// The version in the manifest.
        found: u32,
    },
    /// The feature names in the manifest are not [`FEATURE_NAMES`] in their
    /// order.
    FeatureNames,
    /// The bytes end before a value they promise.
    Truncated,
    /// Bytes follow where a part of the file should end.
    TrailingBytes,
    /// A count is larger than model files allow.
    LimitExceeded {
        /// What was counted.
        what: &'static str,
    },
    /// The code of a model's objective is not one model files have.
    InvalidObjective {
        /// The objective code read from the file.
        code: u8,
    },
    /// A model in the file is invalid.
    Model(ModelError),
    /// A slab grid in the file is invalid: too small, or with an axis off the
    /// 0.05 degree grid or out of range.
    InvalidGrid,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAModelFile => f.write_str("the bytes are not a model file"),
            Self::UnsupportedVersion { found } => write!(
                f,
                "the model file has layout version {found}, not {FORMAT_VERSION}"
            ),
            Self::ChecksumMismatch => f.write_str("the model file does not match its checksum"),
            Self::WrongSection { expected } => {
                write!(
                    f,
                    "the model file has another section where {expected} should be"
                )
            }
            Self::ManifestNotUtf8 => f.write_str("the manifest of the model file is not UTF-8"),
            // The manifest error is the source, so it is not repeated here.
            Self::Manifest(_) => f.write_str("the model file has an invalid manifest"),
            Self::FeatureVersion { found } => write!(
                f,
                "the models take features of version {found}, not {FEATURE_VERSION}"
            ),
            Self::FeatureNames => f.write_str("the manifest names other features"),
            Self::Truncated => f.write_str("the model file ends too early"),
            Self::TrailingBytes => {
                f.write_str("the model file has extra bytes at the end of a part")
            }
            Self::LimitExceeded { what } => write!(f, "the model file has too many {what}"),
            Self::InvalidObjective { code } => {
                write!(f, "the model file has an unknown objective code {code}")
            }
            // The model error is the source, so it is not repeated here.
            Self::Model(_) => f.write_str("the model file has an invalid model"),
            Self::InvalidGrid => f.write_str("the model file has an invalid slab grid"),
        }
    }
}

impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Manifest(error) => Some(error),
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

/// A model file: the manifest, the felt classifier, the intensity regressor,
/// and the two slab grids, checked so that they predict together.
///
/// # Examples
///
/// ```
/// use shindo_est::features::FEATURE_NAMES;
/// use shindo_est::{
///     Hypocenter, Manifest, ModelBundle, Options, PointStatus, Site, SlabGeometry, SlabGrid,
///     lightgbm_text,
/// };
///
/// // A one-tree model in LightGBM's text format that always gives `leaf`.
/// fn model(objective: &str, leaf: f64) -> String {
///     format!(
///         "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nmax_feature_idx=13\n\
///          objective={objective}\nfeature_names={}\n\nTree=0\nnum_leaves=1\nnum_cat=0\n\
///          is_linear=0\nsplit_feature=\nthreshold=\ndecision_type=\nleft_child=\n\
///          right_child=\nleaf_value={leaf}\nshrinkage=1\n\n\nend of trees\n",
///         FEATURE_NAMES.join(" ")
///     )
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let manifest = Manifest::parse(&format!(
///     "model_id=example\nfeature_version=1\nfeature_names={}\n\
///      hurdle_felt_probability=0.35\nhurdle_felt_floor=0.5\nhurdle_unfelt_ceiling=0.49\n\
///      hurdle_logit_per_intensity=1.5\narv_source=Example ARVs\n\
///      training_period=2004-01-01 to 2016-12-31\ntraining_conditions=None\n",
///     FEATURE_NAMES.join(" ")
/// ))?;
/// let classifier = lightgbm_text::parse(&model("binary sigmoid:1", 2.0))?;
/// let regressor = lightgbm_text::parse(&model("regression", 3.1))?;
/// let grid = "140,35,100,50\n140.05,35,100,50\n140,35.05,100,50\n140.05,35.05,100,50";
/// let slabs = SlabGeometry::new(SlabGrid::parse(grid)?, SlabGrid::parse(grid)?);
/// let bytes = ModelBundle::new(manifest, classifier, regressor, slabs)?.to_bytes();
///
/// let bundle = ModelBundle::from_bytes(&bytes)?;
/// let hypocenter = Hypocenter {
///     latitude: 36.0,
///     longitude: 140.0,
///     depth: 10.0,
///     magnitude: 6.0,
/// };
/// let sites = [Site {
///     latitude: 35.7,
///     longitude: 139.8,
///     arv: 1.2,
/// }];
/// let results = bundle
///     .predictor()
///     .predict(&hypocenter, &sites, &Options::default())?;
/// assert_eq!(results[0].status, PointStatus::Ok);
/// assert_eq!(results[0].instrumental, 3.1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct ModelBundle {
    manifest: Manifest,
    predictor: Predictor,
}

impl ModelBundle {
    /// Puts the models together under their manifest.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FeatureVersion`] when the manifest's feature
    /// version is not [`FEATURE_VERSION`], [`FormatError::FeatureNames`] when
    /// its feature names are not [`FEATURE_NAMES`] in their order,
    /// [`FormatError::LimitExceeded`] when the manifest or a tree is too large
    /// for a model file to read back, and [`FormatError::Model`] with the
    /// error of [`Predictor::new`].
    pub fn new(
        manifest: Manifest,
        classifier: Model,
        regressor: Model,
        slabs: SlabGeometry,
    ) -> Result<Self, FormatError> {
        check_manifest(&manifest)?;
        for model in [&classifier, &regressor] {
            let trees = model.forest().trees();
            if trees
                .iter()
                .any(|tree| tree.leaf_count() > forest_section::MAX_LEAVES)
            {
                return Err(FormatError::LimitExceeded {
                    what: "leaves in a tree",
                });
            }
        }
        // The grids need no check here: `SlabGrid::parse` holds them to the
        // limit that reading a grid section applies.
        let hurdle = manifest.hurdle();
        let predictor =
            Predictor::new(classifier, regressor, slabs, hurdle).map_err(FormatError::Model)?;
        Ok(Self {
            manifest,
            predictor,
        })
    }

    /// Reads a model file that [`ModelBundle::to_bytes`] wrote.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NotAModelFile`] when the bytes do not start
    /// with [`MAGIC`], [`FormatError::UnsupportedVersion`] when the layout
    /// version is not [`FORMAT_VERSION`], [`FormatError::ChecksumMismatch`]
    /// when the bytes do not match their checksum, another [`FormatError`]
    /// when a section is missing, out of order, or invalid, and the errors of
    /// [`ModelBundle::new`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if !bytes.starts_with(&MAGIC) {
            return Err(FormatError::NotAModelFile);
        }
        if bytes.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(FormatError::Truncated);
        }
        let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
        let mut reader = Reader::new(&body[MAGIC.len()..]);
        let version = reader.u32()?;
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion { found: version });
        }
        if Sha256::digest(body).as_slice() != checksum {
            return Err(FormatError::ChecksumMismatch);
        }

        let manifest = read_section(&mut reader, tags::MANIFEST, read_manifest)?;
        // Checked before the models are read, so that a file for other
        // features is rejected as such.
        check_manifest(&manifest)?;
        let names = manifest.feature_names();
        let classifier = read_section(&mut reader, tags::CLASSIFIER, |section| {
            forest_section::read(section, names.to_vec())
        })?;
        let regressor = read_section(&mut reader, tags::REGRESSOR, |section| {
            forest_section::read(section, names.to_vec())
        })?;
        let kuril = read_section(&mut reader, tags::KURIL, grid_section::read)?;
        let izu_bonin = read_section(&mut reader, tags::IZU_BONIN, grid_section::read)?;
        reader.finish()?;
        Self::new(
            manifest,
            classifier,
            regressor,
            SlabGeometry::new(kuril, izu_bonin),
        )
    }

    /// Writes the model file, the same bytes every time for the same bundle.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let predictor = &self.predictor;
        let mut writer = Writer::new();
        writer.bytes(&MAGIC);
        writer.u32(FORMAT_VERSION);
        write_section(&mut writer, tags::MANIFEST, |section| {
            section.bytes(self.manifest.to_string().as_bytes());
        });
        write_section(&mut writer, tags::CLASSIFIER, |section| {
            forest_section::write(section, Objective::Binary, predictor.classifier());
        });
        write_section(&mut writer, tags::REGRESSOR, |section| {
            forest_section::write(section, Objective::Regression, predictor.regressor());
        });
        write_grid_section(&mut writer, tags::KURIL, predictor.slabs().kuril());
        write_grid_section(&mut writer, tags::IZU_BONIN, predictor.slabs().izu_bonin());
        let mut bytes = writer.into_bytes();
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }

    /// Returns the manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Returns the predictor the models make.
    #[must_use]
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}

/// Checks what the manifest alone decides: that the models take the features
/// this library computes, and that the manifest fits in a model file.
fn check_manifest(manifest: &Manifest) -> Result<(), FormatError> {
    if manifest.feature_version() != FEATURE_VERSION {
        return Err(FormatError::FeatureVersion {
            found: manifest.feature_version(),
        });
    }
    let names = manifest.feature_names().iter().map(String::as_str);
    if !names.eq(FEATURE_NAMES) {
        return Err(FormatError::FeatureNames);
    }
    if manifest.to_string().len() > MAX_MANIFEST_BYTES {
        return Err(FormatError::LimitExceeded {
            what: "bytes in the manifest",
        });
    }
    Ok(())
}

/// Reads the section named `tag` with `read`, which must use all of it.
fn read_section<T>(
    reader: &mut Reader<'_>,
    tag: &'static str,
    read: impl FnOnce(&mut Reader<'_>) -> Result<T, FormatError>,
) -> Result<T, FormatError> {
    if reader.bytes(TAG_BYTES)? != tag.as_bytes() {
        return Err(FormatError::WrongSection { expected: tag });
    }
    let length = codec::to_usize(reader.u64()?)?;
    let mut section = Reader::new(reader.bytes(length)?);
    let value = read(&mut section)?;
    section.finish()?;
    Ok(value)
}

/// Reads the manifest from the rest of its section.
fn read_manifest(section: &mut Reader<'_>) -> Result<Manifest, FormatError> {
    if section.remaining() > MAX_MANIFEST_BYTES {
        return Err(FormatError::LimitExceeded {
            what: "bytes in the manifest",
        });
    }
    let bytes = section.bytes(section.remaining())?;
    let text = str::from_utf8(bytes).map_err(|_| FormatError::ManifestNotUtf8)?;
    Manifest::parse(text).map_err(FormatError::Manifest)
}

/// Writes a section named `tag` with what `write` writes.
///
/// # Panics
///
/// Panics if the section is longer than `u64` can count.
fn write_section(writer: &mut Writer, tag: &str, write: impl FnOnce(&mut Writer)) {
    let mut section = Writer::new();
    write(&mut section);
    let content = section.into_bytes();
    writer.bytes(tag.as_bytes());
    writer.u64(u64::try_from(content.len()).expect("a section length should fit in u64"));
    writer.bytes(&content);
}

/// Writes a section named `tag` that holds `grid`.
fn write_grid_section(writer: &mut Writer, tag: &str, grid: &SlabGrid) {
    write_section(writer, tag, |section| grid_section::write(section, grid));
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::ops::Range;

    use super::*;
    use crate::features::FEATURE_COUNT;
    use crate::forest::{Forest, Tree};
    use crate::input::Hypocenter;
    use crate::manifest::test_support::CANONICAL;
    use crate::predict::{Options, PointResult, test_support};
    use crate::slab;

    /// Bytes of a section length.
    const LENGTH_BYTES: usize = 8;

    /// Bytes of a section name and length, before its content.
    const SECTION_HEADER_BYTES: usize = TAG_BYTES + LENGTH_BYTES;

    const HYPOCENTER: Hypocenter = Hypocenter {
        latitude: 36.0,
        longitude: 140.0,
        depth: 10.0,
        magnitude: 6.0,
    };

    /// Returns grids of 3 x 3 nodes, so that a file stays small enough for
    /// the tests that change each of its bytes.
    fn slabs() -> SlabGeometry {
        SlabGeometry::new(
            slab::test_support::flat(35.0, 35.1, 140.0, 140.1, 300.0, 50.0),
            slab::test_support::flat(34.0, 34.1, 140.0, 140.1, 250.0, 50.0),
        )
    }

    fn manifest(text: &str) -> Manifest {
        Manifest::parse(text).expect("the manifest should be valid")
    }

    fn new_bundle(text: &str, regressor: Model) -> Result<ModelBundle, FormatError> {
        ModelBundle::new(
            manifest(text),
            test_support::classifier(),
            regressor,
            slabs(),
        )
    }

    fn bundle() -> ModelBundle {
        new_bundle(CANONICAL, test_support::regressor()).expect("the bundle should be valid")
    }

    /// Returns a model of one tree with `leaves` leaves: a chain of splits on
    /// mj, each sending rows below its threshold to a leaf of its own.
    fn chain(leaves: usize, objective: Objective) -> Model {
        let node_count = leaves - 1;
        let splits = i32::try_from(node_count).expect("the splits should fit in i32");
        let left_child: Vec<i32> = (0..splits).map(|node| !node).collect();
        let right_child: Vec<i32> = (1..=splits)
            .map(|next| if next < splits { next } else { !next })
            .collect();
        let threshold: Vec<f64> = (0..splits).map(f64::from).collect();
        let tree = Tree::new(
            &vec![0; node_count],
            &threshold,
            &vec![0; node_count],
            &left_child,
            &right_child,
            &vec![0.0; leaves],
        )
        .expect("the tree should be valid");
        let forest = Forest::new(FEATURE_COUNT, vec![tree]).expect("the forest should be valid");
        let names: Vec<String> = FEATURE_NAMES.map(String::from).to_vec();
        Model::new(forest, objective, names).expect("the model should be valid")
    }

    fn read_error(bytes: &[u8]) -> FormatError {
        ModelBundle::from_bytes(bytes).expect_err("the file should be rejected")
    }

    /// Returns the bytes before the checksum.
    fn body(bytes: &[u8]) -> Vec<u8> {
        bytes[..bytes.len() - CHECKSUM_BYTES].to_vec()
    }

    /// Appends the checksum of `body`, as a forger would.
    fn seal(body: &[u8]) -> Vec<u8> {
        let mut bytes = body.to_vec();
        bytes.extend_from_slice(&Sha256::digest(body));
        bytes
    }

    /// Returns where the content of the section named `tag` is.
    fn section_range(bytes: &[u8], tag: [u8; TAG_BYTES]) -> Range<usize> {
        let mut start = HEADER_BYTES;
        loop {
            let length_bytes: [u8; LENGTH_BYTES] = bytes
                [start + TAG_BYTES..start + SECTION_HEADER_BYTES]
                .try_into()
                .expect("a section length should have 8 bytes");
            let length = usize::try_from(u64::from_le_bytes(length_bytes))
                .expect("the section length should fit in usize");
            let content = start + SECTION_HEADER_BYTES..start + SECTION_HEADER_BYTES + length;
            if bytes[start..start + TAG_BYTES] == tag {
                return content;
            }
            start = content.end;
        }
    }

    /// Returns `bytes` with the content of the section named `tag` replaced
    /// by `content`, sealed again.
    fn with_section(bytes: &[u8], tag: [u8; TAG_BYTES], content: &[u8]) -> Vec<u8> {
        let range = section_range(bytes, tag);
        let length = u64::try_from(content.len()).expect("the length should fit in u64");
        let mut body = body(bytes);
        body.splice(
            range.start - LENGTH_BYTES..range.end,
            length
                .to_le_bytes()
                .into_iter()
                .chain(content.iter().copied()),
        );
        seal(&body)
    }

    fn assert_same_results(actual: &[PointResult], expected: &[PointResult]) {
        let bits = |result: &PointResult| {
            [result.instrumental, result.probability, result.regression].map(f64::to_bits)
        };
        assert_eq!(actual.len(), expected.len());
        let pairs = actual.iter().zip(expected);
        for (index, (actual, expected)) in pairs.enumerate() {
            assert_eq!(actual.status, expected.status, "site {index}");
            assert_eq!(bits(actual), bits(expected), "site {index}");
        }
    }

    #[test]
    fn round_trips_a_bundle() {
        let bundle = bundle();
        let bytes = bundle.to_bytes();
        let read = ModelBundle::from_bytes(&bytes).expect("the written file should be valid");

        assert_eq!(read.to_bytes(), bytes);
        assert_eq!(read.manifest(), bundle.manifest());
        // The two grids have different depths, so a swap shows.
        let (slabs, read_slabs) = (bundle.predictor().slabs(), read.predictor().slabs());
        assert_eq!(read_slabs.kuril().depth(), slabs.kuril().depth(), "Kuril");
        assert_eq!(
            read_slabs.izu_bonin().depth(),
            slabs.izu_bonin().depth(),
            "Izu-Bonin"
        );
        let sites = test_support::sites(300);
        let predict = |bundle: &ModelBundle| {
            bundle
                .predictor()
                .predict(&HYPOCENTER, &sites, &Options::default())
                .expect("the call should be valid")
        };
        assert_same_results(&predict(&read), &predict(&bundle));
    }

    #[test]
    fn starts_with_the_magic_and_the_version() {
        let bytes = bundle().to_bytes();
        assert_eq!(bytes[0..8], MAGIC);
        assert_eq!(bytes[8..12], [1, 0, 0, 0]);
        assert_eq!(&bytes[12..16], b"MNFT");
    }

    #[test]
    fn rejects_every_truncation() {
        let bytes = bundle().to_bytes();
        for length in 0..bytes.len() {
            assert!(
                ModelBundle::from_bytes(&bytes[..length]).is_err(),
                "{length} bytes"
            );
        }
    }

    #[test]
    fn rejects_every_flipped_byte() {
        let bytes = bundle().to_bytes();
        for index in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[index] ^= 0x01;
            assert!(ModelBundle::from_bytes(&flipped).is_err(), "byte {index}");
        }
    }

    #[test]
    fn rejects_line_ending_conversion() {
        let bytes = bundle().to_bytes();
        // The CR LF in the magic turned into LF.
        let converted = [&bytes[..4], b"\n", &bytes[6..]].concat();
        assert_eq!(read_error(&converted), FormatError::NotAModelFile);
    }

    #[test]
    fn rejects_another_format_version() {
        let mut body = body(&bundle().to_bytes());
        body[MAGIC.len()..HEADER_BYTES].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            read_error(&seal(&body)),
            FormatError::UnsupportedVersion { found: 2 }
        );
    }

    #[test]
    fn rejects_a_wrong_checksum() {
        let mut bytes = bundle().to_bytes();
        let manifest = section_range(&bytes, *b"MNFT");
        bytes[manifest.start] = b'n';
        assert_eq!(read_error(&bytes), FormatError::ChecksumMismatch);
    }

    #[test]
    fn rejects_sections_out_of_order() {
        let bytes = bundle().to_bytes();
        let classifier = section_range(&bytes, *b"CLSF").start - SECTION_HEADER_BYTES;
        let regressor = section_range(&bytes, *b"RGSR").start - SECTION_HEADER_BYTES;
        let mut body = body(&bytes);
        body[classifier..classifier + TAG_BYTES].copy_from_slice(b"RGSR");
        body[regressor..regressor + TAG_BYTES].copy_from_slice(b"CLSF");
        assert_eq!(
            read_error(&seal(&body)),
            FormatError::WrongSection { expected: "CLSF" }
        );
    }

    #[test]
    fn rejects_a_section_length_past_the_end() {
        let bytes = bundle().to_bytes();
        let length = section_range(&bytes, *b"MNFT").start - LENGTH_BYTES;
        let mut body = body(&bytes);
        body[length..length + LENGTH_BYTES].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(read_error(&seal(&body)), FormatError::Truncated);
    }

    #[test]
    fn rejects_bytes_after_the_last_section() {
        let mut body = body(&bundle().to_bytes());
        body.push(0);
        assert_eq!(read_error(&seal(&body)), FormatError::TrailingBytes);
    }

    #[test]
    fn rejects_an_oversized_manifest() {
        let bytes = bundle().to_bytes();
        let oversized = with_section(&bytes, *b"MNFT", &vec![b'a'; MAX_MANIFEST_BYTES + 1]);
        assert_eq!(
            read_error(&oversized),
            FormatError::LimitExceeded {
                what: "bytes in the manifest"
            }
        );
    }

    #[test]
    fn rejects_a_manifest_that_is_not_utf8() {
        let bytes = bundle().to_bytes();
        let manifest = section_range(&bytes, *b"MNFT");
        let mut body = body(&bytes);
        body[manifest.start] = 0xFF;
        assert_eq!(read_error(&seal(&body)), FormatError::ManifestNotUtf8);
    }

    #[test]
    fn rejects_another_feature_version() {
        let text = CANONICAL.replace("feature_version=1", "feature_version=2");
        assert_eq!(
            new_bundle(&text, test_support::regressor())
                .expect_err("the version should be rejected"),
            FormatError::FeatureVersion { found: 2 }
        );
    }

    #[test]
    fn rejects_other_feature_names() {
        let text = CANONICAL.replace("feature_names=mj ", "feature_names=mw ");
        assert_eq!(
            new_bundle(&text, test_support::regressor()).expect_err("the names should be rejected"),
            FormatError::FeatureNames
        );
    }

    #[test]
    fn rejects_invalid_hurdle_values() {
        let text = CANONICAL.replace(
            "hurdle_felt_probability=0.35",
            "hurdle_felt_probability=1.5",
        );
        assert_eq!(
            new_bundle(&text, test_support::regressor())
                .expect_err("the hurdle should be rejected"),
            FormatError::Model(ModelError::InvalidHurdleParams)
        );
    }

    #[test]
    fn new_rejects_a_manifest_too_large_to_read_back() {
        let text = format!("{CANONICAL}source={}\n", "a".repeat(MAX_MANIFEST_BYTES));
        assert_eq!(
            new_bundle(&text, test_support::regressor())
                .expect_err("the manifest should be rejected"),
            FormatError::LimitExceeded {
                what: "bytes in the manifest"
            }
        );
    }

    #[test]
    fn new_rejects_a_tree_too_large_to_read_back() {
        let limit = FormatError::LimitExceeded {
            what: "leaves in a tree",
        };
        let leaves = forest_section::MAX_LEAVES + 1;
        let classifier = ModelBundle::new(
            manifest(CANONICAL),
            chain(leaves, Objective::Binary),
            test_support::regressor(),
            slabs(),
        );
        assert_eq!(
            classifier.expect_err("the classifier should be rejected"),
            limit
        );
        let regressor = new_bundle(CANONICAL, chain(leaves, Objective::Regression));
        assert_eq!(
            regressor.expect_err("the regressor should be rejected"),
            limit
        );
    }

    #[test]
    fn reads_back_a_tree_at_the_limit() {
        let bundle = new_bundle(
            CANONICAL,
            chain(forest_section::MAX_LEAVES, Objective::Regression),
        )
        .expect("the bundle should be valid");
        ModelBundle::from_bytes(&bundle.to_bytes()).expect("the file should be read back");
    }

    #[test]
    fn reads_back_a_manifest_at_the_limit() {
        let padding = MAX_MANIFEST_BYTES - CANONICAL.len() - "source=\n".len();
        let text = format!("{CANONICAL}source={}\n", "a".repeat(padding));
        let bundle =
            new_bundle(&text, test_support::regressor()).expect("the bundle should be valid");
        assert_eq!(bundle.manifest().to_string().len(), MAX_MANIFEST_BYTES);
        ModelBundle::from_bytes(&bundle.to_bytes()).expect("the file should be read back");
    }

    #[test]
    fn rejects_bytes_left_in_a_section() {
        let bytes = bundle().to_bytes();
        let classifier = [&bytes[section_range(&bytes, *b"CLSF")], &[0]].concat();
        assert_eq!(
            read_error(&with_section(&bytes, *b"CLSF", &classifier)),
            FormatError::TrailingBytes
        );
    }

    #[test]
    fn rejects_a_section_cut_short() {
        let bytes = bundle().to_bytes();
        let kuril = section_range(&bytes, *b"SLBK");
        let cut = &bytes[kuril.start..kuril.end - 1];
        assert_eq!(
            read_error(&with_section(&bytes, *b"SLBK", cut)),
            FormatError::Truncated
        );
    }

    #[test]
    fn rejects_swapped_models() {
        let bytes = bundle().to_bytes();
        let classifier = bytes[section_range(&bytes, *b"CLSF")].to_vec();
        let regressor = bytes[section_range(&bytes, *b"RGSR")].to_vec();
        let swapped = with_section(&bytes, *b"CLSF", &regressor);
        let swapped = with_section(&swapped, *b"RGSR", &classifier);
        assert_eq!(
            read_error(&swapped),
            FormatError::Model(ModelError::WrongObjective {
                expected: Objective::Binary,
                found: Objective::Regression,
            })
        );
    }

    #[test]
    fn checks_the_manifest_before_reading_the_models() {
        // A file for features of another version, whose classifier this
        // library cannot read either.
        let bytes = bundle().to_bytes();
        let text = CANONICAL.replace("feature_version=1", "feature_version=2");
        let other_version = with_section(&bytes, *b"MNFT", text.as_bytes());
        let unreadable = with_section(&other_version, *b"CLSF", &[0xFF]);
        assert_eq!(
            read_error(&unreadable),
            FormatError::FeatureVersion { found: 2 }
        );
    }

    #[test]
    fn bundle_is_send_and_sync() {
        fn assert_send_and_sync<T: Send + Sync>() {}
        assert_send_and_sync::<ModelBundle>();
    }

    #[test]
    fn manifest_errors_are_the_source() {
        let error = FormatError::Manifest(ManifestError::MissingKey { key: "model_id" });
        let source = error
            .source()
            .expect("a manifest error should be the source");
        assert_eq!(
            source.to_string(),
            ManifestError::MissingKey { key: "model_id" }.to_string()
        );
    }

    #[test]
    fn model_errors_are_the_source() {
        let error = FormatError::Model(ModelError::NoLeaves);
        let source = error.source().expect("a model error should be the source");
        assert_eq!(source.to_string(), ModelError::NoLeaves.to_string());
        assert!(FormatError::Truncated.source().is_none(), "Truncated");
    }

    #[test]
    fn is_exported_at_the_crate_root() {
        let error: crate::FormatError = FormatError::Truncated;
        assert_eq!(error, FormatError::Truncated);
    }
}

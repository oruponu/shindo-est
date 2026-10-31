//! The manifest of a model file: what the models are and how to use them, as
//! `key=value` lines.

use std::fmt;
use std::str::FromStr;

use crate::hurdle::HurdleParams;

/// The keys of a manifest.
mod keys {
    pub(super) const MODEL_ID: &str = "model_id";
    pub(super) const FEATURE_VERSION: &str = "feature_version";
    pub(super) const FEATURE_NAMES: &str = "feature_names";
    pub(super) const FELT_PROBABILITY: &str = "hurdle_felt_probability";
    pub(super) const FELT_FLOOR: &str = "hurdle_felt_floor";
    pub(super) const UNFELT_CEILING: &str = "hurdle_unfelt_ceiling";
    pub(super) const LOGIT_PER_INTENSITY: &str = "hurdle_logit_per_intensity";
    pub(super) const ARV_SOURCE: &str = "arv_source";
    pub(super) const TRAINING_PERIOD: &str = "training_period";
    pub(super) const TRAINING_CONDITIONS: &str = "training_conditions";
    /// The one key that may appear any number of times, in order.
    pub(super) const SOURCE: &str = "source";
}

/// The byte order mark that editors on Windows may put first.
const BYTE_ORDER_MARK: char = '\u{feff}';

/// What a model file says about its models: their identity, the features and
/// hurdle settings they need, and where their data came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    model_id: String,
    feature_version: u32,
    feature_names: Vec<String>,
    hurdle: HurdleParams,
    arv_source: String,
    training_period: String,
    training_conditions: String,
    sources: Vec<String>,
}

/// An error in the text of a manifest.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
#[non_exhaustive]
pub enum ManifestError {
    /// The line is not `key=value`.
    InvalidLine {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The key is not one a manifest has.
    UnknownKey {
        /// Line number, starting at 1.
        line: usize,
    },
    /// The key appears again, and only `source` may.
    RepeatedKey {
        /// Line number of the second appearance, starting at 1.
        line: usize,
    },
    /// The manifest lacks a required key.
    MissingKey {
        /// Name of the key.
        key: &'static str,
    },
    /// The value is empty, has a control character, or cannot be read.
    InvalidValue {
        /// Line number, starting at 1.
        line: usize,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLine { line } => write!(f, "line {line}: expected key=value"),
            Self::UnknownKey { line } => write!(f, "line {line}: the key is not known"),
            Self::RepeatedKey { line } => write!(f, "line {line}: the key appears again"),
            Self::MissingKey { key } => write!(f, "the manifest has no {key}"),
            Self::InvalidValue { line } => {
                write!(f, "line {line}: the value is empty or cannot be read")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

/// A value and the line it is on.
#[derive(Clone, Copy)]
struct Entry<'a> {
    line: usize,
    value: &'a str,
}

impl<'a> Entry<'a> {
    /// Returns the value, which must be neither empty nor hold a control
    /// character.
    fn text(self) -> Result<&'a str, ManifestError> {
        if self.value.is_empty() || self.value.chars().any(char::is_control) {
            return Err(ManifestError::InvalidValue { line: self.line });
        }
        Ok(self.value)
    }

    fn number<T: FromStr>(self) -> Result<T, ManifestError> {
        self.text()?
            .parse()
            .map_err(|_| ManifestError::InvalidValue { line: self.line })
    }

    /// Returns the names separated by single spaces.
    fn names(self) -> Result<Vec<String>, ManifestError> {
        let text = self.text()?;
        let mut names = Vec::new();
        for name in text.split(' ') {
            if name.is_empty() {
                return Err(ManifestError::InvalidValue { line: self.line });
            }
            names.push(name.to_owned());
        }
        Ok(names)
    }
}

/// The value of each key other than `source`, once it has been seen.
#[derive(Default)]
struct Entries<'a> {
    model_id: Option<Entry<'a>>,
    feature_version: Option<Entry<'a>>,
    feature_names: Option<Entry<'a>>,
    felt_probability: Option<Entry<'a>>,
    felt_floor: Option<Entry<'a>>,
    unfelt_ceiling: Option<Entry<'a>>,
    logit_per_intensity: Option<Entry<'a>>,
    arv_source: Option<Entry<'a>>,
    training_period: Option<Entry<'a>>,
    training_conditions: Option<Entry<'a>>,
}

impl<'a> Entries<'a> {
    /// Returns where the value of `key` goes, or `None` for an unknown key.
    fn slot(&mut self, key: &str) -> Option<&mut Option<Entry<'a>>> {
        match key {
            keys::MODEL_ID => Some(&mut self.model_id),
            keys::FEATURE_VERSION => Some(&mut self.feature_version),
            keys::FEATURE_NAMES => Some(&mut self.feature_names),
            keys::FELT_PROBABILITY => Some(&mut self.felt_probability),
            keys::FELT_FLOOR => Some(&mut self.felt_floor),
            keys::UNFELT_CEILING => Some(&mut self.unfelt_ceiling),
            keys::LOGIT_PER_INTENSITY => Some(&mut self.logit_per_intensity),
            keys::ARV_SOURCE => Some(&mut self.arv_source),
            keys::TRAINING_PERIOD => Some(&mut self.training_period),
            keys::TRAINING_CONDITIONS => Some(&mut self.training_conditions),
            _ => None,
        }
    }
}

/// Returns the entry of a key that must appear.
fn required<'a>(entry: Option<Entry<'a>>, key: &'static str) -> Result<Entry<'a>, ManifestError> {
    entry.ok_or(ManifestError::MissingKey { key })
}

impl Manifest {
    /// Reads a manifest from `key=value` lines.
    ///
    /// Each key other than `source` appears exactly once, and `source` any
    /// number of times. Empty lines are skipped, a leading byte order mark is
    /// ignored, and CRLF line endings are accepted. Whether the hurdle
    /// settings are valid is checked where the models are put together.
    ///
    /// # Errors
    ///
    /// Returns a [`ManifestError`] when a line is not `key=value`, a key is
    /// unknown, repeated, or missing, or a value is empty, has a control
    /// character, or cannot be read.
    pub fn parse(text: &str) -> Result<Self, ManifestError> {
        let text = text.strip_prefix(BYTE_ORDER_MARK).unwrap_or(text);
        let mut entries = Entries::default();
        let mut sources = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line_number = index + 1;
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(ManifestError::InvalidLine { line: line_number });
            };
            let entry = Entry {
                line: line_number,
                value,
            };
            if key == keys::SOURCE {
                sources.push(entry.text()?.to_owned());
                continue;
            }
            let Some(slot) = entries.slot(key) else {
                return Err(ManifestError::UnknownKey { line: line_number });
            };
            if slot.is_some() {
                return Err(ManifestError::RepeatedKey { line: line_number });
            }
            *slot = Some(entry);
        }

        Ok(Self {
            model_id: required(entries.model_id, keys::MODEL_ID)?
                .text()?
                .to_owned(),
            feature_version: required(entries.feature_version, keys::FEATURE_VERSION)?.number()?,
            feature_names: required(entries.feature_names, keys::FEATURE_NAMES)?.names()?,
            hurdle: HurdleParams {
                felt_probability: required(entries.felt_probability, keys::FELT_PROBABILITY)?
                    .number()?,
                felt_floor: required(entries.felt_floor, keys::FELT_FLOOR)?.number()?,
                unfelt_ceiling: required(entries.unfelt_ceiling, keys::UNFELT_CEILING)?.number()?,
                logit_per_intensity: required(
                    entries.logit_per_intensity,
                    keys::LOGIT_PER_INTENSITY,
                )?
                .number()?,
            },
            arv_source: required(entries.arv_source, keys::ARV_SOURCE)?
                .text()?
                .to_owned(),
            training_period: required(entries.training_period, keys::TRAINING_PERIOD)?
                .text()?
                .to_owned(),
            training_conditions: required(entries.training_conditions, keys::TRAINING_CONDITIONS)?
                .text()?
                .to_owned(),
            sources,
        })
    }

    /// Returns the identifier of the models.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the version of the feature computation the models were trained
    /// with, to compare with [`FEATURE_VERSION`](crate::features::FEATURE_VERSION).
    #[must_use]
    pub fn feature_version(&self) -> u32 {
        self.feature_version
    }

    /// Returns the names of the features, in the order the models take them.
    #[must_use]
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns the hurdle settings chosen for the models.
    #[must_use]
    pub fn hurdle(&self) -> HurdleParams {
        self.hurdle
    }

    /// Returns where the ARVs the models were trained with came from. ARVs
    /// from another source shift the predictions.
    #[must_use]
    pub fn arv_source(&self) -> &str {
        &self.arv_source
    }

    /// Returns the period of the earthquakes the models were trained on.
    #[must_use]
    pub fn training_period(&self) -> &str {
        &self.training_period
    }

    /// Returns the conditions the models were trained under.
    #[must_use]
    pub fn training_conditions(&self) -> &str {
        &self.training_conditions
    }

    /// Returns the sources of the data in the file and of the training data,
    /// with their SHA-256 hashes, in order.
    #[must_use]
    pub fn sources(&self) -> &[String] {
        &self.sources
    }
}

impl FromStr for Manifest {
    type Err = ManifestError;

    fn from_str(text: &str) -> Result<Self, ManifestError> {
        Self::parse(text)
    }
}

/// Writes the manifest in its canonical form: one line per key in a fixed
/// order, `source` lines last, each line ending in LF, and numbers written so
/// that they read back to the same value.
impl fmt::Display for Manifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hurdle = &self.hurdle;
        writeln!(f, "{}={}", keys::MODEL_ID, self.model_id)?;
        writeln!(f, "{}={}", keys::FEATURE_VERSION, self.feature_version)?;
        writeln!(
            f,
            "{}={}",
            keys::FEATURE_NAMES,
            self.feature_names.join(" ")
        )?;
        writeln!(
            f,
            "{}={:?}",
            keys::FELT_PROBABILITY,
            hurdle.felt_probability
        )?;
        writeln!(f, "{}={:?}", keys::FELT_FLOOR, hurdle.felt_floor)?;
        writeln!(f, "{}={:?}", keys::UNFELT_CEILING, hurdle.unfelt_ceiling)?;
        writeln!(
            f,
            "{}={:?}",
            keys::LOGIT_PER_INTENSITY,
            hurdle.logit_per_intensity
        )?;
        writeln!(f, "{}={}", keys::ARV_SOURCE, self.arv_source)?;
        writeln!(f, "{}={}", keys::TRAINING_PERIOD, self.training_period)?;
        writeln!(
            f,
            "{}={}",
            keys::TRAINING_CONDITIONS,
            self.training_conditions
        )?;
        for source in &self.sources {
            writeln!(f, "{}={source}", keys::SOURCE)?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// A manifest with every key in the canonical form, as
    /// [`Manifest`](super::Manifest) displays it.
    pub const CANONICAL: &str = "\
model_id=synthetic-2026-10
feature_version=1
feature_names=mj depth qlat qlon slat slon log_hypo epi az_sin az_cos log_arv slab_top updip path_frac
hurdle_felt_probability=0.35
hurdle_felt_floor=0.5
hurdle_unfelt_ceiling=0.49
hurdle_logit_per_intensity=1.5
arv_source=Synthetic ARVs for tests
training_period=2004-01-01 to 2016-12-31
training_conditions=Synthetic rows for tests
source=Trees written by hand for tests
source=Slab grids written by hand for tests
";
}

#[cfg(test)]
mod tests {
    use super::test_support::CANONICAL;
    use super::*;
    use crate::features::{FEATURE_NAMES, FEATURE_VERSION};

    fn canonical() -> Manifest {
        Manifest::parse(CANONICAL).expect("the canonical manifest should be valid")
    }

    /// Returns `CANONICAL` without the lines that start with `prefix`.
    fn without(prefix: &str) -> String {
        let mut text = String::new();
        for line in CANONICAL.lines() {
            if !line.starts_with(prefix) {
                text.push_str(line);
                text.push('\n');
            }
        }
        text
    }

    /// Returns `CANONICAL` with `old` replaced by `new` once.
    fn replaced(old: &str, new: &str) -> String {
        assert!(CANONICAL.contains(old), "{old} is not in the manifest");
        CANONICAL.replacen(old, new, 1)
    }

    #[test]
    fn reads_every_key() {
        let manifest = canonical();

        assert_eq!(manifest.model_id(), "synthetic-2026-10");
        assert_eq!(manifest.feature_version(), 1);
        assert_eq!(manifest.feature_names(), FEATURE_NAMES);
        assert_eq!(
            manifest.hurdle(),
            HurdleParams {
                felt_probability: 0.35,
                felt_floor: 0.5,
                unfelt_ceiling: 0.49,
                logit_per_intensity: 1.5,
            }
        );
        assert_eq!(manifest.arv_source(), "Synthetic ARVs for tests");
        assert_eq!(manifest.training_period(), "2004-01-01 to 2016-12-31");
        assert_eq!(manifest.training_conditions(), "Synthetic rows for tests");
        assert_eq!(
            manifest.sources(),
            [
                "Trees written by hand for tests",
                "Slab grids written by hand for tests",
            ]
        );
    }

    #[test]
    fn writes_the_canonical_text_back() {
        assert_eq!(canonical().to_string(), CANONICAL);
    }

    #[test]
    fn writes_keys_in_the_canonical_order() {
        let mut keys = Vec::new();
        let mut sources = Vec::new();
        for line in CANONICAL.lines() {
            if line.starts_with("source=") {
                sources.push(line);
            } else {
                keys.push(line);
            }
        }
        let mut lines = keys;
        lines.reverse();
        lines.insert(0, sources[0]);
        lines.insert(lines.len() / 2, sources[1]);
        let text = lines.join("\n");

        let manifest = Manifest::parse(&text).expect("the manifest should be valid");
        assert_eq!(manifest.to_string(), CANONICAL);
    }

    #[test]
    fn keeps_values_as_written() {
        // Spaces at both ends, an equals sign, and a letter outside ASCII.
        let value = " a=b \u{e9} ";
        let text = replaced(
            "training_conditions=Synthetic rows for tests",
            &format!("training_conditions={value}"),
        );
        let manifest = Manifest::parse(&text).expect("the manifest should be valid");

        assert_eq!(manifest.training_conditions(), value);
        assert_eq!(manifest.to_string(), text);
    }

    #[test]
    fn round_trips_numbers_exactly() {
        let text = replaced("hurdle_felt_floor=0.5", "hurdle_felt_floor=0.1").replacen(
            "hurdle_logit_per_intensity=1.5",
            "hurdle_logit_per_intensity=0.30000000000000004",
            1,
        );
        let manifest = Manifest::parse(&text).expect("the manifest should be valid");
        let again =
            Manifest::parse(&manifest.to_string()).expect("the written manifest should be valid");

        let first = manifest.hurdle();
        let second = again.hurdle();
        assert_eq!(first.felt_floor.to_bits(), 0.1_f64.to_bits());
        assert_eq!(
            first.logit_per_intensity.to_bits(),
            (0.1_f64 + 0.2).to_bits()
        );
        assert_eq!(second.felt_floor.to_bits(), first.felt_floor.to_bits());
        assert_eq!(
            second.logit_per_intensity.to_bits(),
            first.logit_per_intensity.to_bits()
        );
    }

    #[test]
    fn writes_numbers_in_the_round_trip_form() {
        // The line in CANONICAL, the line read, and the line written. A whole
        // number read without its decimal point is written with it.
        let cases = [
            (
                "hurdle_felt_probability=0.35",
                "hurdle_felt_probability=2",
                "hurdle_felt_probability=2.0",
            ),
            (
                "hurdle_felt_floor=0.5",
                "hurdle_felt_floor=-0.0",
                "hurdle_felt_floor=-0.0",
            ),
            (
                "hurdle_unfelt_ceiling=0.49",
                "hurdle_unfelt_ceiling=5e-324",
                "hurdle_unfelt_ceiling=5e-324",
            ),
            (
                "hurdle_logit_per_intensity=1.5",
                "hurdle_logit_per_intensity=1.7976931348623157e308",
                "hurdle_logit_per_intensity=1.7976931348623157e308",
            ),
        ];
        let mut text = CANONICAL.to_owned();
        for (old, read, _) in cases {
            assert!(text.contains(old), "{old} is not in the manifest");
            text = text.replacen(old, read, 1);
        }

        let manifest = Manifest::parse(&text).expect("the manifest should be valid");
        let written = manifest.to_string();
        for (_, _, line) in cases {
            assert!(
                written.contains(&format!("{line}\n")),
                "{line} in {written}"
            );
        }
        let hurdle = Manifest::parse(&written)
            .expect("the written manifest should be valid")
            .hurdle();
        assert_eq!(hurdle.felt_probability.to_bits(), 2.0_f64.to_bits());
        assert_eq!(hurdle.felt_floor.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(hurdle.unfelt_ceiling.to_bits(), 1, "the smallest subnormal");
        assert_eq!(hurdle.logit_per_intensity.to_bits(), f64::MAX.to_bits());
    }

    #[test]
    fn crlf_and_a_bom_read_the_same() {
        let windows = format!("\u{feff}{}", CANONICAL.replace('\n', "\r\n"));
        assert_eq!(Manifest::parse(&windows), Ok(canonical()));
    }

    #[test]
    fn skips_empty_lines() {
        let spaced = format!("\n{}\n\n", CANONICAL.replace('\n', "\n\n"));
        assert_eq!(Manifest::parse(&spaced), Ok(canonical()));
    }

    #[test]
    fn sources_are_optional() {
        let manifest = Manifest::parse(&without("source=")).expect("the manifest should be valid");
        assert!(manifest.sources().is_empty(), "{:?}", manifest.sources());
    }

    #[test]
    fn rejects_a_missing_key() {
        assert_eq!(
            Manifest::parse(&without("model_id=")),
            Err(ManifestError::MissingKey { key: "model_id" })
        );
    }

    #[test]
    fn rejects_a_repeated_key() {
        // The second arv_source line follows the first, on line 9.
        let text = replaced(
            "arv_source=Synthetic ARVs for tests\n",
            "arv_source=Synthetic ARVs for tests\narv_source=Other ARVs\n",
        );
        assert_eq!(
            Manifest::parse(&text),
            Err(ManifestError::RepeatedKey { line: 9 })
        );
    }

    #[test]
    fn rejects_an_unknown_key() {
        let text = format!("{CANONICAL}comment=x\n");
        assert_eq!(
            Manifest::parse(&text),
            Err(ManifestError::UnknownKey { line: 13 })
        );
    }

    #[test]
    fn rejects_a_line_without_equals() {
        let text = replaced("model_id=synthetic-2026-10", "model_id");
        assert_eq!(
            Manifest::parse(&text),
            Err(ManifestError::InvalidLine { line: 1 })
        );
    }

    #[test]
    fn rejects_empty_or_control_values() {
        for value in ["model_id=", "model_id=a\tb"] {
            let text = replaced("model_id=synthetic-2026-10", value);
            assert_eq!(
                Manifest::parse(&text),
                Err(ManifestError::InvalidValue { line: 1 }),
                "{value:?}"
            );
        }
    }

    #[test]
    fn rejects_unreadable_numbers() {
        let cases = [
            ("feature_version=1", "feature_version=-1", 2),
            ("feature_version=1", "feature_version=1.5", 2),
            ("hurdle_felt_floor=0.5", "hurdle_felt_floor=x", 5),
        ];
        for (old, new, line) in cases {
            assert_eq!(
                Manifest::parse(&replaced(old, new)),
                Err(ManifestError::InvalidValue { line }),
                "{new}"
            );
        }
    }

    #[test]
    fn rejects_empty_feature_names() {
        let text = replaced("feature_names=mj depth", "feature_names=mj  depth");
        assert_eq!(
            Manifest::parse(&text),
            Err(ManifestError::InvalidValue { line: 3 })
        );
    }

    #[test]
    fn feature_version_is_1() {
        assert_eq!(FEATURE_VERSION, 1);
    }
}

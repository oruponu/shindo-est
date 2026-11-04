//! The command line, parsed by hand, as it has only a few commands and
//! options.

use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// How to call the command, printed for `--help` and after a usage error.
pub(crate) const USAGE: &str = "\
usage: shindo-est info <model>
       shindo-est pack <manifest> <classifier> <regressor> <kuril> <izu_bonin> <out>
       shindo-est predict --model <model> [--max-threads <n>]
       shindo-est --help | --version";

#[derive(Debug, PartialEq)]
pub(crate) enum Command {
    /// Packs the text files of a model into a model file.
    Pack(PackPaths),
    /// Prints the manifest of a model file.
    Info {
        model: PathBuf,
    },
    /// Answers prediction requests on standard input with a model file.
    Predict {
        model: PathBuf,
        /// The largest number of threads for each request, or `None` for the
        /// number of logical CPUs.
        max_threads: Option<NonZeroUsize>,
    },
    Help,
    Version,
}

#[derive(Debug, PartialEq)]
pub(crate) struct PackPaths {
    /// The manifest, in its text form.
    pub manifest: PathBuf,
    /// The felt classifier, in `LightGBM`'s text format.
    pub classifier: PathBuf,
    /// The intensity regressor, in `LightGBM`'s text format.
    pub regressor: PathBuf,
    /// The Kuril grid of the Pacific slab, as `lon,lat,depth,thickness` lines.
    pub kuril: PathBuf,
    /// The Izu-Bonin grid of the Pacific slab, in the same form.
    pub izu_bonin: PathBuf,
    /// The model file to write.
    pub out: PathBuf,
}

#[derive(Debug, PartialEq)]
pub(crate) struct UsageError(String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for UsageError {}

/// Parses the arguments after the program name. Paths stay as given, so that
/// paths that are not Unicode work too.
///
/// `-h`, `--help`, `-V`, and `--version` count only alone, as the first
/// argument.
pub(crate) fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Command, UsageError> {
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Err(UsageError("no command given".to_owned()));
    };
    let rest: Vec<OsString> = arguments.collect();
    match command.to_str() {
        Some("-h" | "--help") if rest.is_empty() => Ok(Command::Help),
        Some("-V" | "--version") if rest.is_empty() => Ok(Command::Version),
        Some("info") => {
            let [model] = take_paths(rest, "info")?;
            Ok(Command::Info { model })
        }
        Some("pack") => {
            let [manifest, classifier, regressor, kuril, izu_bonin, out] =
                take_paths(rest, "pack")?;
            Ok(Command::Pack(PackPaths {
                manifest,
                classifier,
                regressor,
                kuril,
                izu_bonin,
                out,
            }))
        }
        Some("predict") => parse_predict(rest),
        _ => Err(UsageError(format!(
            "unknown command {:?}",
            command.to_string_lossy()
        ))),
    }
}

/// Parses the options of `predict`, which come in any order: `--model` once,
/// and `--max-threads` at most once.
fn parse_predict(arguments: Vec<OsString>) -> Result<Command, UsageError> {
    let mut model = None;
    let mut max_threads = None;
    let mut arguments = arguments.into_iter();
    while let Some(option) = arguments.next() {
        let name = option.to_string_lossy();
        let Some(value) = arguments.next() else {
            return Err(UsageError(format!("{name} needs a value")));
        };
        match option.to_str() {
            Some("--model") if model.is_none() => model = Some(PathBuf::from(value)),
            Some("--max-threads") if max_threads.is_none() => {
                max_threads = Some(parse_max_threads(&value)?);
            }
            Some("--model" | "--max-threads") => {
                return Err(UsageError(format!("{name} is given twice")));
            }
            _ => return Err(UsageError(format!("unknown option {name:?} for predict"))),
        }
    }
    let Some(model) = model else {
        return Err(UsageError("predict needs --model <model>".to_owned()));
    };
    Ok(Command::Predict { model, max_threads })
}

fn parse_max_threads(value: &OsStr) -> Result<NonZeroUsize, UsageError> {
    let threads = value.to_str().and_then(|text| text.parse().ok());
    threads.ok_or_else(|| {
        UsageError(format!(
            "--max-threads takes an integer of at least 1, not {:?}",
            value.to_string_lossy()
        ))
    })
}

/// Returns the `N` paths that `command` takes, or an error naming how many it
/// got.
fn take_paths<const N: usize>(
    arguments: Vec<OsString>,
    command: &str,
) -> Result<[PathBuf; N], UsageError> {
    let count = arguments.len();
    let paths: Vec<PathBuf> = arguments.into_iter().map(PathBuf::from).collect();
    paths
        .try_into()
        .map_err(|_| UsageError(format!("{command} takes {N} paths, not {count}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(words: &[&str]) -> Vec<OsString> {
        words.iter().map(OsString::from).collect()
    }

    fn parse_words(words: &[&str]) -> Result<Command, UsageError> {
        super::parse(arguments(words))
    }

    #[test]
    fn parses_pack() {
        let command = parse_words(&["pack", "m", "c", "r", "k", "i", "o"]);

        assert_eq!(
            command,
            Ok(Command::Pack(PackPaths {
                manifest: PathBuf::from("m"),
                classifier: PathBuf::from("c"),
                regressor: PathBuf::from("r"),
                kuril: PathBuf::from("k"),
                izu_bonin: PathBuf::from("i"),
                out: PathBuf::from("o"),
            }))
        );
        for words in [
            &["pack", "m", "c", "r", "k", "i"][..],
            &["pack", "m", "c", "r", "k", "i", "o", "x"],
        ] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn parses_info() {
        assert_eq!(
            parse_words(&["info", "a"]),
            Ok(Command::Info {
                model: PathBuf::from("a")
            })
        );
        for words in [&["info"][..], &["info", "a", "b"]] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn parses_help_and_version() {
        for flag in ["-h", "--help"] {
            assert_eq!(parse_words(&[flag]), Ok(Command::Help), "{flag}");
        }
        for flag in ["-V", "--version"] {
            assert_eq!(parse_words(&[flag]), Ok(Command::Version), "{flag}");
        }
        // The flags count only alone, as the first argument.
        for words in [&["pack", "--help"][..], &["--help", "pack"]] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn parses_predict() {
        assert_eq!(
            parse_words(&["predict", "--model", "a"]),
            Ok(Command::Predict {
                model: PathBuf::from("a"),
                max_threads: None,
            })
        );
        assert_eq!(
            parse_words(&["predict", "--max-threads", "3", "--model", "a"]),
            Ok(Command::Predict {
                model: PathBuf::from("a"),
                max_threads: NonZeroUsize::new(3),
            })
        );
    }

    #[test]
    fn rejects_bad_predict_arguments() {
        for words in [
            &["predict"][..],
            &["predict", "--max-threads", "3"],
            &["predict", "--model", "a", "--model", "b"],
            &["predict", "--model"],
            &["predict", "--model", "a", "--max-threads"],
            &["predict", "--model", "a", "--max-threads", "0"],
            &["predict", "--model", "a", "--max-threads", "-1"],
            &["predict", "--model", "a", "--max-threads", "x"],
            &[
                "predict",
                "--model",
                "a",
                "--max-threads",
                "2",
                "--max-threads",
                "3",
            ],
            &["predict", "--model", "a", "--threads", "3"],
            &["predict", "--model", "a", "b"],
        ] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }

    #[test]
    fn rejects_unknown_commands() {
        for words in [&[][..], &["run"], &["--model"]] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }
}

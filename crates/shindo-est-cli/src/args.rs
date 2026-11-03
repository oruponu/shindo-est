//! The command line, parsed by hand to keep the binary free of dependencies.

use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

/// How to call the command, printed for `--help` and after a usage error.
pub(crate) const USAGE: &str = "\
usage: shindo-est info <model>
       shindo-est pack <manifest> <classifier> <regressor> <kuril> <izu_bonin> <out>
       shindo-est --help | --version";

#[derive(Debug, PartialEq)]
pub(crate) enum Command {
    /// Packs the text files of a model into a model file.
    Pack(PackPaths),
    /// Prints the manifest of a model file.
    Info {
        model: PathBuf,
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
        _ => Err(UsageError(format!(
            "unknown command {:?}",
            command.to_string_lossy()
        ))),
    }
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
    fn rejects_unknown_commands() {
        for words in [&[][..], &["run"], &["--model"]] {
            assert!(parse_words(words).is_err(), "{words:?}");
        }
    }
}

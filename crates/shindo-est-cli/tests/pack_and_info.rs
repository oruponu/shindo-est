//! Runs the `shindo-est` binary on the model file fixture.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Output};

use shindo_est::{FormatError, ModelError, lightgbm_text};

/// The text files of the model file fixture.
const INPUTS: [&str; 5] = [
    "manifest.txt",
    "classifier.txt",
    "regressor.txt",
    "kuril.csv",
    "izu_bonin.csv",
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bundle")
        .join(name)
}

fn run(arguments: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_shindo-est"))
        .args(arguments)
        .output()
        .expect("the binary should run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("the output is UTF-8")
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("the errors are UTF-8")
}

/// A folder of its own for one test, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> Self {
        let path = env::temp_dir().join(format!("shindo-est-cli-{}-{test}", process::id()));
        // A run that was stopped may have left a folder of the same name, as
        // process IDs are reused, so the test starts from an empty one.
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).expect("the temporary folder should be creatable");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Returns the names of the files in the folder, sorted.
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.0)
            .expect("the temporary folder is readable")
            .map(|entry| {
                let entry = entry.expect("the temporary folder is readable");
                entry.file_name().to_string_lossy().into_owned()
            })
            .collect();
        names.sort();
        names
    }

    /// Copies the fixture's input `name` into the folder, with `old` replaced
    /// by `new`, and returns the copy's path.
    fn copy_with(&self, name: &str, old: &str, new: &str) -> PathBuf {
        let text = fs::read_to_string(fixture(name)).expect("the fixture is readable");
        assert!(text.contains(old), "{name} has no {old:?}");
        let path = self.path(name);
        fs::write(&path, text.replace(old, new)).expect("the copy should be writable");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // A folder left behind is only clutter in the temporary folder.
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Runs `pack` on the fixture's inputs, with `replaced` in place of the input
/// of the same file name, into `out`.
fn pack(replaced: Option<&Path>, out: &Path) -> Output {
    let mut paths: Vec<PathBuf> = Vec::new();
    for name in INPUTS {
        let path = match replaced {
            Some(path) if path.file_name() == Some(name.as_ref()) => path.to_path_buf(),
            _ => fixture(name),
        };
        paths.push(path);
    }
    let mut arguments: Vec<&Path> = vec![Path::new("pack")];
    arguments.extend(paths.iter().map(PathBuf::as_path));
    arguments.push(out);
    run(&arguments)
}

/// Asserts that the file at `path` holds `expected`, naming the first byte
/// that differs, as printing whole model files would bury it.
fn assert_same_bytes(path: &Path, expected: &[u8]) {
    let written = fs::read(path).expect("pack writes the model file");
    let first_difference = written
        .iter()
        .zip(expected)
        .position(|(written_byte, expected_byte)| written_byte != expected_byte);
    assert!(
        written == expected,
        "{} has {} bytes, expected {}; the first differing byte is at {:?}",
        path.display(),
        written.len(),
        expected.len(),
        first_difference
    );
}

#[test]
fn pack_rebuilds_the_fixture() {
    let dir = TempDir::new("pack_rebuilds_the_fixture");
    let out = dir.path("model.bin");

    let output = pack(None, &out);

    assert!(output.status.success(), "{}", stderr(&output));
    let expected = fs::read(fixture("model.bin")).expect("the fixture is readable");
    assert_same_bytes(&out, &expected);
    assert_eq!(
        stdout(&output),
        format!(
            "wrote {}: {} bytes, model synthetic-fixture\n",
            out.display(),
            expected.len()
        )
    );
    assert_eq!(dir.names(), ["model.bin"]);
}

#[test]
fn pack_replaces_an_existing_file() {
    let dir = TempDir::new("pack_replaces_an_existing_file");
    let out = dir.path("model.bin");
    fs::write(&out, "old").expect("the file should be writable");

    let output = pack(None, &out);

    assert!(output.status.success(), "{}", stderr(&output));
    let expected = fs::read(fixture("model.bin")).expect("the fixture is readable");
    assert_same_bytes(&out, &expected);
    assert_eq!(dir.names(), ["model.bin"]);
}

#[test]
fn pack_keeps_an_existing_file_when_it_fails() {
    let dir = TempDir::new("pack_keeps_an_existing_file_when_it_fails");
    let manifest = dir.copy_with(
        "manifest.txt",
        "hurdle_felt_probability=0.35",
        "hurdle_felt_probability=1.5",
    );
    let out = dir.path("model.bin");
    fs::write(&out, "old").expect("the file should be writable");

    let output = pack(Some(&manifest), &out);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert_eq!(fs::read(&out).expect("the file is readable"), b"old");
    assert_eq!(dir.names(), ["manifest.txt", "model.bin"]);
}

#[test]
fn pack_removes_its_new_file_when_it_cannot_replace_the_output() {
    let dir = TempDir::new("pack_removes_its_new_file_when_it_cannot_replace_the_output");
    // A file cannot be moved over a folder.
    let out = dir.path("model.bin");
    fs::create_dir(&out).expect("the folder should be creatable");

    let output = pack(None, &out);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(errors.contains(&out.display().to_string()), "{errors}");
    assert!(out.is_dir(), "pack removed the folder {}", out.display());
    assert_eq!(dir.names(), ["model.bin"]);
}

#[test]
fn pack_reports_the_cause_and_writes_nothing() {
    let dir = TempDir::new("pack_reports_the_cause_and_writes_nothing");
    let manifest = dir.copy_with(
        "manifest.txt",
        "hurdle_felt_probability=0.35",
        "hurdle_felt_probability=1.5",
    );
    let out = dir.path("model.bin");

    let output = pack(Some(&manifest), &out);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(!out.exists(), "pack wrote {}", out.display());
    let errors = stderr(&output);
    assert!(errors.starts_with("shindo-est: "), "{errors}");
    let format_error = FormatError::Model(ModelError::InvalidHurdleParams).to_string();
    let model_error = ModelError::InvalidHurdleParams.to_string();
    assert!(errors.contains(&format_error), "{errors}");
    assert!(errors.contains(&model_error), "{errors}");
}

#[test]
fn pack_names_the_file_it_cannot_parse() {
    let dir = TempDir::new("pack_names_the_file_it_cannot_parse");
    let classifier = dir.path("classifier.txt");
    fs::write(&classifier, "tree\n").expect("the copy should be writable");
    let error = lightgbm_text::parse("tree\n").expect_err("the text is not a model");

    let output = pack(Some(&classifier), &dir.path("model.bin"));

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(
        errors.contains(&classifier.display().to_string()),
        "{errors}"
    );
    assert!(errors.contains(&error.to_string()), "{errors}");
}

#[test]
fn info_prints_the_manifest() {
    let output = run(&[Path::new("info"), &fixture("model.bin")]);

    assert!(output.status.success(), "{}", stderr(&output));
    let manifest = fs::read_to_string(fixture("manifest.txt")).expect("the fixture is readable");
    assert_eq!(stdout(&output), manifest);
}

#[test]
fn info_reports_a_file_that_is_not_a_model() {
    let manifest = fixture("manifest.txt");

    let output = run(&[Path::new("info"), &manifest]);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(errors.contains(&manifest.display().to_string()), "{errors}");
    assert!(errors.contains("not a model file"), "{errors}");
}

#[test]
fn info_names_a_file_it_cannot_read() {
    let dir = TempDir::new("info_names_a_file_it_cannot_read");
    let missing = dir.path("missing.bin");

    let output = run(&[Path::new("info"), &missing]);

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(errors.contains(&missing.display().to_string()), "{errors}");
}

#[test]
fn prints_the_usage_on_request() {
    let output = run(&[Path::new("--help")]);

    assert!(output.status.success(), "{}", stderr(&output));
    let usage = stdout(&output);
    assert!(usage.starts_with("usage:"), "{usage}");
    assert_eq!(stderr(&output), "");
}

#[test]
fn usage_errors_exit_with_2() {
    let output = run(&[]);

    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(errors.contains("usage:"), "{errors}");
}

#[test]
fn prints_the_version() {
    let output = run(&[Path::new("--version")]);

    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        stdout(&output),
        format!("shindo-est {}\n", env!("CARGO_PKG_VERSION"))
    );
}

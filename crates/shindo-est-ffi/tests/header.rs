//! Checks that the committed C header is what cbindgen makes from the crate.
//!
//! To regenerate the header, run
//! `SHINDO_EST_FFI_WRITE_HEADER=1 cargo test -p shindo-est-ffi --test header`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Functions the header must declare. cbindgen skips an item it cannot
/// translate without failing, so the names are listed here.
const FUNCTIONS: [&str; 5] = [
    "shindo_est_abi_version",
    "shindo_est_version",
    "shindo_est_error_message",
    "shindo_est_intensity_class",
    "shindo_est_intensity_label",
];

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn header_path() -> PathBuf {
    crate_dir().join("include").join("shindo_est.h")
}

/// Returns the header that cbindgen makes from the crate now.
fn generate() -> String {
    let config = cbindgen::Config::from_file(crate_dir().join("cbindgen.toml"))
        .expect("cbindgen.toml should be readable");
    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir())
        .with_config(config)
        .generate()
        .expect("cbindgen should translate the crate");
    let mut bytes = Vec::new();
    bindings.write(&mut bytes);
    String::from_utf8(bytes).expect("the header should be UTF-8")
}

fn committed() -> String {
    fs::read_to_string(header_path()).expect("include/shindo_est.h should be readable")
}

#[test]
#[cfg_attr(miri, ignore)]
fn header_matches_cbindgen() {
    let generated = generate();
    if env::var_os("SHINDO_EST_FFI_WRITE_HEADER").is_some_and(|value| value == "1") {
        fs::write(header_path(), &generated).expect("include/shindo_est.h should be writable");
    }

    assert!(
        committed() == generated,
        "include/shindo_est.h is out of date; regenerate it with \
         SHINDO_EST_FFI_WRITE_HEADER=1 cargo test -p shindo-est-ffi --test header"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn header_is_ascii() {
    let header = committed();
    let line = header.lines().position(|line| !line.is_ascii());

    assert_eq!(line, None, "line index of the first non-ASCII line");
}

#[test]
#[cfg_attr(miri, ignore)]
fn header_declares_every_function() {
    let header = committed();
    for name in FUNCTIONS {
        assert!(header.contains(&format!("{name}(")), "{name}");
    }
}

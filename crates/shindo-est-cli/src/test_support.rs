//! The model file fixture, for the tests of the modules.

use shindo_est::{ModelBundle, Site};

const MODEL: &[u8] = include_bytes!("../../../tests/fixtures/bundle/model.bin");

/// The model ID in the fixture's manifest.
pub(crate) const MODEL_ID: &str = "synthetic-fixture";

/// The sites of the fixture's tests in `crates/shindo-est/tests`: five valid
/// sites and one with an invalid ARV.
pub(crate) const SITES: [Site; 6] = [
    Site {
        latitude: 35.02,
        longitude: 140.02,
        arv: 1.0,
    },
    Site {
        latitude: 35.02,
        longitude: 140.17,
        arv: 2.0,
    },
    Site {
        latitude: 35.12,
        longitude: 140.12,
        arv: 1.0,
    },
    Site {
        latitude: 34.9,
        longitude: 140.1,
        arv: 1.0,
    },
    Site {
        latitude: 40.0,
        longitude: 135.0,
        arv: 1.0,
    },
    Site {
        latitude: 35.02,
        longitude: 140.02,
        arv: 0.0,
    },
];

pub(crate) fn bundle() -> ModelBundle {
    ModelBundle::from_bytes(MODEL).expect("the fixture is a model file")
}

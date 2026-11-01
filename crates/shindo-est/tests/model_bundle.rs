//! Checks the model file in `tests/fixtures/bundle` against the text sources it
//! was packed from, so that a change to the layout of model files shows.

use std::fs;
use std::path::{Path, PathBuf};

use shindo_est::{
    HurdleParams, Hypocenter, Manifest, Model, ModelBundle, Options, PointResult, PointStatus,
    Predictor, Site, SlabGeometry, SlabGrid, lightgbm_text,
};

/// A hypocenter under the Kuril grid, closer than 100 km to every site but
/// the one outside the grids, with a magnitude above the regressor's split.
const HYPOCENTER: Hypocenter = Hypocenter {
    latitude: 35.02,
    longitude: 140.02,
    depth: 80.0,
    magnitude: 6.5,
};

/// [`HYPOCENTER`] with a magnitude below the regressor's split.
const SMALLER: Hypocenter = Hypocenter {
    magnitude: 5.5,
    ..HYPOCENTER
};

/// Index in [`SITES`] of the site with an invalid ARV.
const INVALID_ARV_SITE: usize = 5;

const SITES: [Site; 6] = [
    // Over the Kuril slab at 54 km, shallower than the classifier's split at 60 km.
    Site {
        latitude: 35.02,
        longitude: 140.02,
        arv: 1.0,
    },
    // Over the Kuril slab at 84 km, deeper than the split, with an ARV above 1.
    Site {
        latitude: 35.02,
        longitude: 140.17,
        arv: 2.0,
    },
    // In a cell that touches the Kuril node without a point.
    Site {
        latitude: 35.12,
        longitude: 140.12,
        arv: 1.0,
    },
    // Over the Izu-Bonin slab at 30 km.
    Site {
        latitude: 34.9,
        longitude: 140.1,
        arv: 1.0,
    },
    // Outside both grids, farther than 100 km from the hypocenter.
    Site {
        latitude: 40.0,
        longitude: 135.0,
        arv: 1.0,
    },
    // An invalid ARV.
    Site {
        latitude: 35.02,
        longitude: 140.02,
        arv: 0.0,
    },
];

/// The raw scores of the fixture's two models at one site.
struct RawScores {
    classifier: f64,
    regressor: f64,
}

/// The raw scores at each valid site of [`SITES`] for [`HYPOCENTER`], written
/// as the sum of the leaves each site reaches, tree by tree. Between them, the
/// sites take both branches of every split but the one on `mj`, whose other
/// branch [`SMALLER`] takes.
const RAW_SCORES: [RawScores; 5] = [
    // log_hypo under 2, slab_top 54; mj above 6, log_arv 0, path_frac 19/60.
    RawScores {
        classifier: 2.0 + 0.3,
        regressor: 4.0 - 0.3 + 0.2,
    },
    // log_hypo under 2, slab_top 84; log_arv above 0, path_frac 12/60.
    RawScores {
        classifier: 2.0 - 0.2,
        regressor: 4.0 + 0.3 + 0.1,
    },
    // log_hypo under 2, slab_top NaN (left); log_arv 0, path_frac 16/60.
    RawScores {
        classifier: 2.0 + 0.3,
        regressor: 4.0 - 0.3 + 0.2,
    },
    // log_hypo under 2, slab_top 30; log_arv 0, path_frac 13/60.
    RawScores {
        classifier: 2.0 + 0.3,
        regressor: 4.0 - 0.3 + 0.1,
    },
    // log_hypo above 2, slab_top NaN (left); log_arv 0, path_frac 0.
    RawScores {
        classifier: -1.0 + 0.3,
        regressor: 4.0 - 0.3 + 0.1,
    },
];

/// How much lower the regressor's leaf for [`SMALLER`] is than for
/// [`HYPOCENTER`].
const SMALLER_REGRESSOR_DROP: f64 = 2.0;

/// Largest difference accepted between a score worked out here and the
/// library's, which may round differently.
const TOLERANCE: f64 = 1e-12;

/// The parts of a model file, read from the fixture's text sources.
struct Sources {
    manifest: Manifest,
    classifier: Model,
    regressor: Model,
    slabs: SlabGeometry,
}

fn fixture_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bundle")
        .join(file)
}

fn read(file: &str) -> String {
    let path = fixture_path(file);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn committed_file() -> Vec<u8> {
    let path = fixture_path("model.bin");
    fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn sources() -> Sources {
    let grid = |file| SlabGrid::parse(&read(file)).expect("the grid should be valid");
    Sources {
        manifest: Manifest::parse(&read("manifest.txt")).expect("the manifest should be valid"),
        classifier: lightgbm_text::parse(&read("classifier.txt"))
            .expect("the classifier should be valid"),
        regressor: lightgbm_text::parse(&read("regressor.txt"))
            .expect("the regressor should be valid"),
        slabs: SlabGeometry::new(grid("kuril.csv"), grid("izu_bonin.csv")),
    }
}

/// Returns the predictor made from the sources directly, without a model file.
fn source_predictor() -> Predictor {
    let sources = sources();
    let hurdle = sources.manifest.hurdle();
    Predictor::new(sources.classifier, sources.regressor, sources.slabs, hurdle)
        .expect("the sources should make a predictor")
}

fn predict(predictor: &Predictor, hypocenter: &Hypocenter) -> Vec<PointResult> {
    predictor
        .predict(hypocenter, &SITES, &Options::default())
        .expect("the call should be valid")
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "{context}: {actual} against {expected}"
    );
}

#[test]
fn packing_the_fixture_sources_gives_the_committed_file() {
    let sources = sources();
    let bundle = ModelBundle::new(
        sources.manifest,
        sources.classifier,
        sources.regressor,
        sources.slabs,
    )
    .expect("the sources should make a model file");
    assert!(
        bundle.to_bytes() == committed_file(),
        "the packed bytes differ from model.bin; see tests/fixtures/bundle/README.md"
    );
}

#[test]
fn fixture_sites_take_every_branch() {
    let predictor = source_predictor();
    let results = predict(&predictor, &HYPOCENTER);
    let smaller = predict(&predictor, &SMALLER);
    for (index, scores) in RAW_SCORES.iter().enumerate() {
        let result = &results[index];
        let site = format!("site {index}");
        assert_eq!(result.status, PointStatus::Ok, "{site}");
        assert_close(result.probability, sigmoid(scores.classifier), &site);
        assert_close(result.regression, scores.regressor, &site);
        assert_close(
            smaller[index].regression,
            scores.regressor - SMALLER_REGRESSOR_DROP,
            &format!("{site}, smaller"),
        );
    }
    assert_eq!(results[INVALID_ARV_SITE].status, PointStatus::InvalidInput);
}

#[test]
fn committed_file_predicts_like_the_sources() {
    let bundle = ModelBundle::from_bytes(&committed_file()).expect("model.bin should be valid");
    let predictor = source_predictor();
    let bits = |result: &PointResult| {
        [result.instrumental, result.probability, result.regression].map(f64::to_bits)
    };
    for hypocenter in [HYPOCENTER, SMALLER] {
        let expected = predict(&predictor, &hypocenter);
        let actual = predict(bundle.predictor(), &hypocenter);
        assert_eq!(actual.len(), expected.len());
        let magnitude = hypocenter.magnitude;
        let pairs = actual.iter().zip(&expected);
        for (index, (actual, expected)) in pairs.enumerate() {
            assert_eq!(
                actual.status, expected.status,
                "Mj {magnitude}, site {index}"
            );
            assert_eq!(bits(actual), bits(expected), "Mj {magnitude}, site {index}");
        }
    }
}

#[test]
fn committed_file_carries_its_manifest() {
    let bundle = ModelBundle::from_bytes(&committed_file()).expect("model.bin should be valid");
    let manifest = bundle.manifest();
    assert_eq!(manifest.model_id(), "synthetic-fixture");
    assert_eq!(manifest.feature_version(), 1);
    assert_eq!(
        manifest.hurdle(),
        HurdleParams {
            felt_probability: 0.35,
            felt_floor: 0.5,
            unfelt_ceiling: 0.49,
            logit_per_intensity: 1.5,
        }
    );
}

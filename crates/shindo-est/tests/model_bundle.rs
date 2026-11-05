//! Checks the model file in `tests/fixtures/bundle` against the text sources it
//! was packed from, so that a change to the layout of model files shows, and
//! checks the library's features and predictions against `expected.csv`.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use shindo_est::features::{self, FEATURE_NAMES};
use shindo_est::{
    HurdleParams, Hypocenter, IntensityClass, Manifest, Model, ModelBundle, Options, PointResult,
    PointStatus, Predictor, Site, SlabGeometry, SlabGrid, lightgbm_text,
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

/// A hypocenter under the Izu-Bonin grid that differs from [`HYPOCENTER`] in
/// every field, so that a binding that gives one row the hypocenter of
/// another fails on `expected.csv`.
const ELSEWHERE: Hypocenter = Hypocenter {
    latitude: 34.9,
    longitude: 140.1,
    depth: 50.0,
    magnitude: 6.8,
};

/// [`HYPOCENTER`] with an unknown depth, which puts the call out of scope.
const UNKNOWN_DEPTH: Hypocenter = Hypocenter {
    depth: f64::NAN,
    ..HYPOCENTER
};

/// One call written to `expected.csv`.
struct ExpectedCall {
    hypocenter: Hypocenter,
    offset: f64,
}

/// The calls written to `expected.csv`, in order.
const EXPECTED_CALLS: [ExpectedCall; 5] = [
    ExpectedCall {
        hypocenter: HYPOCENTER,
        offset: 0.0,
    },
    ExpectedCall {
        hypocenter: SMALLER,
        offset: 0.0,
    },
    ExpectedCall {
        hypocenter: HYPOCENTER,
        offset: 0.2,
    },
    ExpectedCall {
        hypocenter: UNKNOWN_DEPTH,
        offset: 0.0,
    },
    ExpectedCall {
        hypocenter: ELSEWHERE,
        offset: 0.0,
    },
];

/// The `intensity_class` of a row without an instrumental intensity, as the
/// C ABI gives it.
const NO_INTENSITY_CLASS: i32 = -1;

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

/// Returns the text of `expected.csv`: a header, then a row for each site of
/// [`SITES`] in each call of [`EXPECTED_CALLS`], holding the inputs, the
/// features, and the prediction from `model.bin`. Numbers are written as
/// `{:?}` writes them, which reads back as the same bits.
fn expected_values() -> String {
    let slabs = sources().slabs;
    let bundle = ModelBundle::from_bytes(&committed_file()).expect("model.bin should be valid");
    let mut columns: Vec<String> = [
        "hypocenter_latitude",
        "hypocenter_longitude",
        "hypocenter_depth",
        "hypocenter_magnitude",
        "offset",
        "site_latitude",
        "site_longitude",
        "site_arv",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    columns.extend(FEATURE_NAMES.map(|name| format!("feature_{name}")));
    columns.extend(
        [
            "status",
            "intensity_class",
            "instrumental",
            "probability",
            "regression",
        ]
        .map(String::from),
    );
    let mut text = columns.join(",");
    text.push('\n');
    let number = |value: &f64| format!("{value:?}");
    for call in &EXPECTED_CALLS {
        let hypocenter = &call.hypocenter;
        let options = Options::default().with_offset(call.offset);
        let results = bundle
            .predictor()
            .predict(hypocenter, &SITES, &options)
            .expect("the call should be valid");
        assert_eq!(
            results.len(),
            SITES.len(),
            "the zip below would drop results past the last site; {hypocenter:?}"
        );
        let pairs = SITES.iter().zip(&results);
        for (site, result) in pairs {
            let inputs = [
                hypocenter.latitude,
                hypocenter.longitude,
                hypocenter.depth,
                hypocenter.magnitude,
                call.offset,
                site.latitude,
                site.longitude,
                site.arv,
            ];
            let mut fields: Vec<String> = inputs.iter().map(number).collect();
            let feature_values = features::compute(hypocenter, site, &slabs);
            fields.extend(feature_values.iter().map(number));
            let class = IntensityClass::from_instrumental(result.instrumental)
                .map_or(NO_INTENSITY_CLASS, |class| class as i32);
            fields.push((result.status as i32).to_string());
            fields.push(class.to_string());
            let values = [result.instrumental, result.probability, result.regression];
            fields.extend(values.iter().map(number));
            text.push_str(&fields.join(","));
            text.push('\n');
        }
    }
    text
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

#[test]
fn committed_expected_values_match() {
    let computed = expected_values();
    if env::var_os("SHINDO_EST_WRITE_EXPECTED").is_some_and(|value| value == "1") {
        let path = fixture_path("expected.csv");
        fs::write(&path, &computed).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }
    let committed = read("expected.csv");
    let mut committed_lines = committed.split('\n');
    let mut computed_lines = computed.split('\n');
    for line_number in 1.. {
        match (committed_lines.next(), computed_lines.next()) {
            (None, None) => break,
            (committed_line, computed_line) if committed_line == computed_line => {}
            (committed_line, computed_line) => panic!(
                "line {line_number} differs: expected.csv has {committed_line:?}, the library \
                 gives {computed_line:?}; see tests/fixtures/bundle/README.md"
            ),
        }
    }
}

#[test]
fn expected_values_cover_every_status() {
    let text = expected_values();
    let lines: Vec<Vec<&str>> = text.lines().map(|line| line.split(',').collect()).collect();
    let (header, rows) = lines.split_first().expect("there is a header");
    assert_eq!(header.len(), 27);
    assert_eq!(rows.len(), 30);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), 27, "row {index}");
    }
    let column = |name| {
        header
            .iter()
            .position(|column| *column == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    };
    let status = column("status");
    for code in ["0", "1", "2"] {
        assert!(
            rows.iter().any(|row| row[status] == code),
            "no row has status {code}"
        );
    }
    let class = column("intensity_class");
    let no_class = NO_INTENSITY_CLASS.to_string();
    assert!(
        rows.iter().any(|row| row[class] == no_class),
        "no row is without an intensity class"
    );
    let hypocenters: BTreeSet<&[&str]> = rows.iter().map(|row| &row[..4]).collect();
    assert!(hypocenters.len() >= 4, "{hypocenters:?}");
    let positions: BTreeSet<&[&str]> = rows.iter().map(|row| &row[..2]).collect();
    assert!(positions.len() >= 2, "{positions:?}");
}

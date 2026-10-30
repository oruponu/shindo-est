//! Checks tree evaluation against `LightGBM`'s own predictions for the models in
//! `tests/fixtures/lightgbm`.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use shindo_est::forest::Forest;
use shindo_est::lightgbm_text;

/// Header lines before the first row of each CSV file.
const HEADER_LINES: usize = 1;

/// `LightGBM`'s prediction for one input row.
struct Expected {
    raw: f64,
    leaves: Vec<usize>,
}

struct Row<'a> {
    line: usize,
    fields: Vec<&'a str>,
}

/// A model with its input rows and `LightGBM`'s predictions for them.
struct Fixture {
    forest: Forest,
    inputs: Vec<Vec<f64>>,
    expected: Vec<Expected>,
}

fn fixture_directory(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/lightgbm")
        .join(name)
}

fn read(directory: &Path, file: &str) -> String {
    let path = directory.join(file);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// Returns the line number, counted from 1, of the CSV row at `index`.
fn csv_line(index: usize) -> usize {
    index + HEADER_LINES + 1
}

fn rows(text: &str) -> Vec<Row<'_>> {
    let mut rows = Vec::new();
    let numbered_lines = text.lines().skip(HEADER_LINES).enumerate();
    for (index, content) in numbered_lines {
        rows.push(Row {
            line: csv_line(index),
            fields: content.split(',').collect(),
        });
    }
    rows
}

/// Parses the field at 0-based `column`, panicking with the 1-based position.
fn number<T: FromStr>(file: &str, row: &Row<'_>, column: usize) -> T {
    let field = row.fields[column];
    field.parse().unwrap_or_else(|_| {
        panic!(
            "{file} line {}, column {}: unreadable number {field:?}",
            row.line,
            column + 1
        )
    })
}

fn load(name: &str) -> Fixture {
    let directory = fixture_directory(name);
    let forest = lightgbm_text::parse(&read(&directory, "model.txt"))
        .unwrap_or_else(|error| panic!("{name}/model.txt: {error}"))
        .into_forest();

    let inputs_file = format!("{name}/inputs.csv");
    let inputs_text = read(&directory, "inputs.csv");
    let mut inputs = Vec::new();
    for row in rows(&inputs_text) {
        let mut input: Vec<f64> = Vec::new();
        for column in 0..row.fields.len() {
            input.push(number(&inputs_file, &row, column));
        }
        inputs.push(input);
    }

    let expected_file = format!("{name}/expected.csv");
    let expected_text = read(&directory, "expected.csv");
    let mut expected = Vec::new();
    for row in rows(&expected_text) {
        let raw: f64 = number(&expected_file, &row, 0);
        let mut leaves: Vec<usize> = Vec::new();
        for column in 1..row.fields.len() {
            leaves.push(number(&expected_file, &row, column));
        }
        expected.push(Expected { raw, leaves });
    }
    Fixture {
        forest,
        inputs,
        expected,
    }
}

/// Checks every row's leaves and raw score, then the batch scores.
fn assert_matches_lightgbm(name: &str) {
    let fixture = load(name);
    let forest = &fixture.forest;
    assert!(
        !fixture.inputs.is_empty(),
        "{name}: inputs.csv has no rows, so nothing would be compared"
    );
    assert_eq!(
        fixture.inputs.len(),
        fixture.expected.len(),
        "{name}: input and expected row counts"
    );

    let mut scores = Vec::new();
    let numbered_rows = fixture.inputs.iter().zip(&fixture.expected).enumerate();
    for (index, (input, expected)) in numbered_rows {
        let line = csv_line(index);
        assert_eq!(
            input.len(),
            forest.feature_count(),
            "{name}: features on inputs.csv line {line}"
        );
        assert_eq!(
            expected.leaves.len(),
            forest.tree_count(),
            "{name}: leaves on expected.csv line {line}"
        );
        let leaves = forest.leaf_indices(input);
        assert_eq!(
            leaves.len(),
            expected.leaves.len(),
            "{name}: leaves from leaf_indices for line {line}, {input:?}"
        );
        let mismatch = leaves
            .iter()
            .zip(&expected.leaves)
            .position(|(leaf, lightgbm_leaf)| leaf != lightgbm_leaf);
        if let Some(tree) = mismatch {
            panic!(
                "{name}: line {line}, tree {tree}: leaf {} instead of {} for {input:?}",
                leaves[tree], expected.leaves[tree]
            );
        }
        let raw = forest.raw(input);
        // The same leaf values added in the same order give the same bits. A
        // tolerance would hide leaf values read with fewer digits or trees
        // added in another order.
        assert!(
            raw.to_bits() == expected.raw.to_bits(),
            "{name}: line {line}: raw {raw} instead of {} for {input:?}",
            expected.raw
        );
        scores.push(raw);
    }

    let flat = fixture.inputs.concat();
    let mut batch = vec![0.0; fixture.inputs.len()];
    forest.raw_batch(&flat, &mut batch);
    let numbered_scores = batch.iter().zip(&scores).enumerate();
    for (index, (batch_score, score)) in numbered_scores {
        let line = csv_line(index);
        assert_eq!(
            batch_score.to_bits(),
            score.to_bits(),
            "{name}: line {line}: raw_batch {batch_score} and raw {score} for {:?}",
            fixture.inputs[index]
        );
    }
}

#[test]
fn nan_missing_model_matches_lightgbm() {
    assert_matches_lightgbm("nan");
}

#[test]
fn zero_missing_model_matches_lightgbm() {
    assert_matches_lightgbm("zero");
}

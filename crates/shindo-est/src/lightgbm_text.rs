//! Reading `LightGBM` models saved in its text format.

use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use crate::forest::{Forest, Tree};
use crate::model::{Model, ModelError, Objective};

/// Prefix of the line that starts a tree.
const TREE_START: &str = "Tree=";
/// Line that follows the last tree. Feature importances and parameters come
/// after it and are not read.
const TREES_END: &str = "end of trees";
/// Header key of a model that predicts from the mean of its trees rather than
/// their sum, as random forests do. `LightGBM` averages whenever the key is
/// present, with or without a value.
const AVERAGE_OUTPUT: &str = "average_output";

/// Keys read before the trees, each of which may appear once.
const HEADER_KEYS: [&str; 5] = [
    "max_feature_idx",
    "num_tree_per_iteration",
    "num_class",
    "objective",
    "feature_names",
];

/// Objectives, as `LightGBM` writes them, whose prediction is the raw score.
/// `LightGBM` adds "sqrt" to a regression trained on square-rooted labels,
/// which predicts the square of its raw score with the sign kept, so that is
/// not among them.
const REGRESSION_OBJECTIVES: [&str; 6] = [
    "regression",
    "regression_l1",
    "huber",
    "fair",
    "quantile",
    "mape",
];

/// Objectives, as `LightGBM` writes them, whose probability is the logistic
/// function of the raw score.
const BINARY_OBJECTIVES: [&str; 2] = ["binary sigmoid:1", "cross_entropy"];

/// The value of a `key=value` line and where it is.
#[derive(Clone, Copy, Debug)]
struct Entry<'a> {
    line: usize,
    value: &'a str,
}

/// The `key=value` lines of one tree, by key.
type TreeLines<'a> = HashMap<&'a str, Entry<'a>>;

/// Reads a model with numerical splits from `LightGBM`'s text format.
///
/// # Errors
///
/// Returns a [`ModelError`] when a value cannot be read, a required key is
/// missing or repeated, the text ends before "end of trees", the model has no
/// trees, several outputs, averaged trees, an objective that [`Objective`]
/// does not cover, feature names that do not match its features, categorical
/// splits, or linear trees, or its trees are malformed.
pub fn parse(text: &str) -> Result<Model, ModelError> {
    let mut feature_count = None;
    let mut objective = None;
    let mut feature_names = None;
    let mut trees = Vec::new();
    let mut tree: Option<TreeLines<'_>> = None;
    let mut ended = false;
    let mut header_keys: HashSet<&str> = HashSet::new();
    let numbered_lines = text.lines().enumerate();
    for (index, line) in numbered_lines {
        let line_number = index + 1;
        let line = line.trim_end_matches('\r');
        if line.starts_with(TREE_START) || line == TREES_END {
            if let Some(lines) = tree.take() {
                trees.push(build_tree(&lines)?);
            }
            if line == TREES_END {
                ended = true;
                break;
            }
            tree = Some(HashMap::new());
            continue;
        }
        let split = line.split_once('=');
        if tree.is_none() && split.map_or(line, |(key, _)| key) == AVERAGE_OUTPUT {
            return Err(ModelError::AveragedOutput);
        }
        let Some((key, value)) = split else {
            continue;
        };
        if let Some(lines) = tree.as_mut() {
            let entry = Entry {
                line: line_number,
                value,
            };
            if lines.insert(key, entry).is_some() {
                return Err(ModelError::RepeatedKey { line: line_number });
            }
            continue;
        }
        if HEADER_KEYS.contains(&key) && !header_keys.insert(key) {
            return Err(ModelError::RepeatedKey { line: line_number });
        }
        match key {
            "max_feature_idx" => {
                let max_index: usize = number(value, line_number)?;
                let count = max_index
                    .checked_add(1)
                    .ok_or(ModelError::InvalidValue { line: line_number })?;
                feature_count = Some(count);
            }
            "num_tree_per_iteration" | "num_class" => {
                let count: usize = number(value, line_number)?;
                if count != 1 {
                    return Err(ModelError::MultiClass);
                }
            }
            "objective" => objective = Some(read_objective(value, line_number)?),
            "feature_names" => {
                // LightGBM splits the names at spaces only and skips empty
                // ones; other white space stays in a name.
                let names = value.split(' ').filter(|name| !name.is_empty());
                feature_names = Some(names.map(String::from).collect());
            }
            _ => {}
        }
    }
    if !ended {
        return Err(ModelError::Truncated);
    }
    let feature_count = feature_count.ok_or(ModelError::MissingKey {
        key: "max_feature_idx",
    })?;
    let objective = objective.ok_or(ModelError::MissingKey { key: "objective" })?;
    let feature_names = feature_names.ok_or(ModelError::MissingKey {
        key: "feature_names",
    })?;
    Model::new(Forest::new(feature_count, trees)?, objective, feature_names)
}

/// Reads the value of an `objective` line.
fn read_objective(value: &str, line: usize) -> Result<Objective, ModelError> {
    let words: Vec<&str> = value.split_whitespace().collect();
    let normalized = words.join(" ");
    if REGRESSION_OBJECTIVES.contains(&normalized.as_str()) {
        Ok(Objective::Regression)
    } else if BINARY_OBJECTIVES.contains(&normalized.as_str()) {
        Ok(Objective::Binary)
    } else {
        Err(ModelError::UnsupportedObjective { line })
    }
}

/// Builds a tree from its lines.
fn build_tree(lines: &TreeLines<'_>) -> Result<Tree, ModelError> {
    let categorical_count: usize = single(lines, "num_cat")?;
    if categorical_count != 0 {
        return Err(ModelError::Categorical);
    }
    let linear: u8 = single(lines, "is_linear")?;
    if linear != 0 {
        return Err(ModelError::Linear);
    }
    let leaf_count: usize = single(lines, "num_leaves")?;
    let leaf_value: Vec<f64> = numbers(lines, "leaf_value")?;
    // Tree::new counts the leaves in leaf_value, so the declared count is
    // checked here.
    if leaf_value.len() != leaf_count {
        return Err(ModelError::InvalidValue {
            line: required(lines, "num_leaves")?.line,
        });
    }
    let split_feature: Vec<u32> = numbers(lines, "split_feature")?;
    let threshold: Vec<f64> = numbers(lines, "threshold")?;
    let decision_type: Vec<u8> = numbers(lines, "decision_type")?;
    let left_child: Vec<i32> = numbers(lines, "left_child")?;
    let right_child: Vec<i32> = numbers(lines, "right_child")?;
    Tree::new(
        &split_feature,
        &threshold,
        &decision_type,
        &left_child,
        &right_child,
        &leaf_value,
    )
}

/// Returns the entry of a key that must be present.
fn required<'a>(lines: &TreeLines<'a>, key: &'static str) -> Result<Entry<'a>, ModelError> {
    lines
        .get(key)
        .copied()
        .ok_or(ModelError::MissingKey { key })
}

/// Reads the one number of a key.
fn single<T: FromStr>(lines: &TreeLines<'_>, key: &'static str) -> Result<T, ModelError> {
    let entry = required(lines, key)?;
    number(entry.value, entry.line)
}

/// Reads the whitespace-separated numbers of a key. A blank value has none.
fn numbers<T: FromStr>(lines: &TreeLines<'_>, key: &'static str) -> Result<Vec<T>, ModelError> {
    let entry = required(lines, key)?;
    entry
        .value
        .split_whitespace()
        .map(|item| number(item, entry.line))
        .collect()
}

/// Reads one number, such as `1.5`, `inf`, or `-nan`, ignoring spaces around it.
fn number<T: FromStr>(text: &str, line: usize) -> Result<T, ModelError> {
    text.trim()
        .parse()
        .map_err(|_| ModelError::InvalidValue { line })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Node 0 sends feature 0 at or below 1.5 to leaf 0, and 0 (and missing
    // values, read as 0) to the right. Node 1 sends feature 1 at or below 0 to
    // leaf 1, and NaN to the right.
    const TWO_TREES: &str = "tree
version=v4
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=regression
feature_names=depth arv

Tree=0
num_leaves=3
num_cat=0
is_linear=0
split_feature=0 1
threshold=1.5 0
decision_type=4 8
left_child=-1 -2
right_child=1 -3
leaf_value=10 20 30
shrinkage=1

Tree=1
num_leaves=1
num_cat=0
is_linear=0
split_feature=
threshold=
decision_type=
left_child=
right_child=
leaf_value=0.5
shrinkage=1


end of trees

Tree=999
num_leaves=x

feature_importances:
Column_0=1

parameters:
[max_feature_idx: 99]
max_feature_idx=99
end of parameters
";

    fn parse_model_with(old: &str, new: &str) -> Result<Model, ModelError> {
        assert!(TWO_TREES.contains(old), "{old:?}");
        super::parse(&TWO_TREES.replacen(old, new, 1))
    }

    fn parse_with(old: &str, new: &str) -> Result<Forest, ModelError> {
        parse_model_with(old, new).map(Model::into_forest)
    }

    fn parse_forest(text: &str) -> Forest {
        super::parse(text)
            .unwrap_or_else(|error| panic!("{error}"))
            .into_forest()
    }

    #[test]
    fn reads_feature_and_tree_counts() {
        let forest = parse_forest(TWO_TREES);

        assert_eq!(forest.feature_count(), 2);
        assert_eq!(forest.tree_count(), 2);
    }

    #[test]
    fn evaluates_like_the_hand_built_trees() {
        let forest = parse_forest(TWO_TREES);

        assert_eq!(forest.raw(&[1.5, 5.0]), 10.5);
        assert_eq!(forest.raw(&[-1.0, 5.0]), 10.5);
        assert_eq!(forest.raw(&[2.0, 0.0]), 20.5);
        assert_eq!(forest.raw(&[2.0, 0.1]), 30.5);
        assert_eq!(forest.raw(&[0.0, -1.0]), 20.5);
        assert_eq!(forest.raw(&[2.0, f64::NAN]), 30.5);
    }

    #[test]
    fn crlf_reads_the_same_as_lf() {
        let crlf = TWO_TREES.replace('\n', "\r\n");
        let whole = parse_forest(&crlf);
        // A text may also stop right after the "\r" of "end of trees".
        let end = crlf.find("end of trees").expect("the model ends its trees");
        let cut = parse_forest(&crlf[..end + "end of trees\r".len()]);

        assert_eq!(whole.raw(&[2.0, 0.0]), 20.5);
        assert_eq!(whole.raw(&[2.0, 0.1]), 30.5);
        assert_eq!(cut.raw(&[2.0, 0.1]), 30.5);
    }

    #[test]
    fn reads_infinite_thresholds_and_exponents() {
        let infinite = parse_with("threshold=1.5 0", "threshold=inf 0")
            .expect("an infinite threshold should parse");
        assert_eq!(infinite.raw(&[1e300, 5.0]), 10.5);

        let negative = parse_with("threshold=1.5 0", "threshold=-inf 0")
            .expect("a negative infinite threshold should parse");
        // -1 is above -inf, though at or below the original 1.5.
        assert_eq!(negative.raw(&[-1.0, 5.0]), 30.5);
        let exponent =
            parse_with("leaf_value=0.5", "leaf_value=5e-1").expect("an exponent should parse");
        assert_eq!(exponent.raw(&[1.5, 5.0]), 10.5);
    }

    #[test]
    fn reads_values_with_extra_spaces() {
        let cases = [
            ("leaf_value=10 20 30", "leaf_value=10  20 30"),
            ("leaf_value=10 20 30", "leaf_value=10\t20\t30"),
            ("leaf_value=10 20 30", "leaf_value= 10 20 30 "),
            ("num_leaves=3", "num_leaves= 3 "),
            ("split_feature=\n", "split_feature= \n"),
        ];

        for (old, new) in cases {
            let forest = parse_with(old, new).unwrap_or_else(|error| panic!("{new:?}: {error}"));
            assert_eq!(forest.raw(&[2.0, 0.1]), 30.5, "{new:?}");
        }
    }

    #[test]
    fn ignores_what_follows_the_trees() {
        // After "end of trees" come a broken tree and max_feature_idx=99.
        let forest = parse_forest(TWO_TREES);

        assert_eq!(forest.feature_count(), 2);
        assert_eq!(forest.tree_count(), 2);
    }

    #[test]
    fn rejects_non_finite_leaf_values() {
        for value in ["nan", "-nan", "inf", "-inf"] {
            assert_eq!(
                parse_with("leaf_value=0.5", &format!("leaf_value={value}")).unwrap_err(),
                ModelError::NonFiniteLeafValue,
                "{value}"
            );
        }
    }

    #[test]
    fn rejects_a_nan_threshold() {
        assert_eq!(
            parse_with("threshold=1.5 0", "threshold=nan 0").unwrap_err(),
            ModelError::NanThreshold
        );
    }

    #[test]
    fn rejects_categorical_splits() {
        let error = parse_with(
            "Tree=0\nnum_leaves=3\nnum_cat=0",
            "Tree=0\nnum_leaves=3\nnum_cat=1",
        )
        .unwrap_err();

        assert_eq!(error, ModelError::Categorical);
    }

    #[test]
    fn rejects_linear_trees() {
        let error = parse_with("is_linear=0", "is_linear=1").unwrap_err();

        assert_eq!(error, ModelError::Linear);
    }

    #[test]
    fn rejects_unreadable_counts_and_flags() {
        let cases = [
            ("num_class=1", "num_class=x", 3),
            ("num_tree_per_iteration=1", "num_tree_per_iteration=", 4),
            ("num_cat=0", "num_cat=x", 11),
            ("is_linear=0", "is_linear=x", 12),
        ];

        for (old, new, line) in cases {
            assert_eq!(
                parse_with(old, new).unwrap_err(),
                ModelError::InvalidValue { line },
                "{new:?}"
            );
        }
    }

    #[test]
    fn rejects_repeated_keys() {
        let in_tree = parse_with(
            "leaf_value=10 20 30",
            "leaf_value=10 20 30\nleaf_value=1 2 3",
        )
        .unwrap_err();
        let in_header =
            parse_with("max_feature_idx=1", "max_feature_idx=1\nmax_feature_idx=5").unwrap_err();

        assert_eq!(in_tree, ModelError::RepeatedKey { line: 19 });
        assert_eq!(in_header, ModelError::RepeatedKey { line: 6 });
    }

    #[test]
    fn reads_the_objective_and_the_feature_names() {
        let model = super::parse(TWO_TREES).expect("the model should parse");

        assert_eq!(model.objective(), Objective::Regression);
        assert_eq!(model.feature_names(), ["depth", "arv"]);
    }

    #[test]
    fn splits_feature_names_only_at_spaces() {
        // LightGBM separates the names with spaces and keeps other white space,
        // such as a tab or a no-break space, inside a name.
        let model = parse_model_with(
            "feature_names=depth arv",
            "feature_names= de\tpth  a\u{a0}rv ",
        )
        .expect("the model should parse");

        assert_eq!(model.feature_names(), ["de\tpth", "a\u{a0}rv"]);
    }

    #[test]
    fn reads_supported_objectives() {
        let cases = [
            ("regression_l1", Objective::Regression),
            ("huber", Objective::Regression),
            ("fair", Objective::Regression),
            ("quantile", Objective::Regression),
            ("mape", Objective::Regression),
            ("binary sigmoid:1", Objective::Binary),
            ("cross_entropy", Objective::Binary),
        ];
        for (value, objective) in cases {
            let model = parse_model_with("objective=regression", &format!("objective={value}"))
                .unwrap_or_else(|error| panic!("{value:?}: {error}"));
            assert_eq!(model.objective(), objective, "{value:?}");
        }
    }

    #[test]
    fn rejects_objectives_that_transform_the_raw_score() {
        // None is a regression whose prediction is the raw score or a binary
        // objective whose probability is the logistic function of it.
        let values = [
            "regression sqrt",
            "regression_l1 sqrt",
            "poisson",
            "gamma",
            "tweedie",
            "binary sigmoid:2",
            "binary",
            "cross_entropy_lambda",
            "multiclass num_class:3",
            "lambdarank",
            "",
        ];
        for value in values {
            let error =
                parse_with("objective=regression", &format!("objective={value}")).unwrap_err();
            assert_eq!(
                error,
                ModelError::UnsupportedObjective { line: 6 },
                "{value:?}"
            );
        }
    }

    #[test]
    fn rejects_a_model_without_an_objective() {
        // LightGBM writes no objective for a model trained with a custom one.
        let error = parse_with("objective=regression\n", "").unwrap_err();

        assert_eq!(error, ModelError::MissingKey { key: "objective" });
    }

    #[test]
    fn rejects_missing_or_miscounted_feature_names() {
        let missing = parse_with("feature_names=depth arv\n", "").unwrap_err();
        let one = parse_with("feature_names=depth arv", "feature_names=depth").unwrap_err();
        let three =
            parse_with("feature_names=depth arv", "feature_names=depth arv mj").unwrap_err();

        assert_eq!(
            missing,
            ModelError::MissingKey {
                key: "feature_names"
            }
        );
        assert_eq!(
            one,
            ModelError::FeatureNameCount {
                names: 1,
                feature_count: 2,
            }
        );
        assert_eq!(
            three,
            ModelError::FeatureNameCount {
                names: 3,
                feature_count: 2,
            }
        );
    }

    #[test]
    fn rejects_a_repeated_objective_or_feature_names() {
        let objective = parse_with(
            "objective=regression",
            "objective=regression\nobjective=regression",
        )
        .unwrap_err();
        let names = parse_with(
            "feature_names=depth arv",
            "feature_names=depth arv\nfeature_names=depth arv",
        )
        .unwrap_err();

        assert_eq!(objective, ModelError::RepeatedKey { line: 7 });
        assert_eq!(names, ModelError::RepeatedKey { line: 8 });
    }

    #[test]
    fn rejects_averaged_output() {
        // LightGBM averages the trees whenever the key is in the header,
        // whatever its value.
        let lines = ["average_output", "average_output=1", "average_output=0"];
        for line in lines {
            let error = parse_with(
                "objective=regression\n",
                &format!("objective=regression\n{line}\n"),
            )
            .unwrap_err();
            assert_eq!(error, ModelError::AveragedOutput, "{line:?}");
        }

        let crlf = TWO_TREES
            .replacen(
                "objective=regression\n",
                "objective=regression\naverage_output\n",
                1,
            )
            .replace('\n', "\r\n");
        assert_eq!(super::parse(&crlf).unwrap_err(), ModelError::AveragedOutput);
    }

    #[test]
    fn reads_average_output_only_in_the_header() {
        let in_tree = parse_with(
            "shrinkage=1\n\n\nend",
            "shrinkage=1\naverage_output\n\n\nend",
        );
        let after_trees = parse_with("end of trees\n", "end of trees\naverage_output\n");

        assert!(in_tree.is_ok(), "{in_tree:?}");
        assert!(after_trees.is_ok(), "{after_trees:?}");
    }

    #[test]
    fn rejects_multiclass() {
        let per_iteration =
            parse_with("num_tree_per_iteration=1", "num_tree_per_iteration=3").unwrap_err();
        let classes = parse_with("num_class=1", "num_class=3").unwrap_err();

        assert_eq!(per_iteration, ModelError::MultiClass);
        assert_eq!(classes, ModelError::MultiClass);
    }

    #[test]
    fn rejects_missing_max_feature_idx() {
        let error = parse_with("max_feature_idx=1\n", "").unwrap_err();

        assert_eq!(
            error,
            ModelError::MissingKey {
                key: "max_feature_idx"
            }
        );
    }

    #[test]
    fn rejects_each_missing_tree_key() {
        let keys = [
            "num_leaves",
            "num_cat",
            "is_linear",
            "split_feature",
            "threshold",
            "decision_type",
            "left_child",
            "right_child",
            "leaf_value",
        ];

        for tree in ["Tree=0", "Tree=1"] {
            let start = TWO_TREES.find(tree).expect("the model has the tree");
            for key in keys {
                let line_start = start
                    + TWO_TREES[start..]
                        .find(&format!("\n{key}="))
                        .expect("the tree has the key")
                    + 1;
                let line_end =
                    line_start + TWO_TREES[line_start..].find('\n').expect("the line ends") + 1;
                let text = format!("{}{}", &TWO_TREES[..line_start], &TWO_TREES[line_end..]);

                assert_eq!(
                    super::parse(&text).unwrap_err(),
                    ModelError::MissingKey { key },
                    "{tree} without {key}"
                );
            }
        }
    }

    #[test]
    fn rejects_a_file_without_trees() {
        let start = TWO_TREES.find("Tree=0").expect("the model has trees");
        let end = TWO_TREES
            .find("end of trees")
            .expect("the model ends its trees");
        let text = format!("{}{}", &TWO_TREES[..start], &TWO_TREES[end..]);

        assert_eq!(super::parse(&text).unwrap_err(), ModelError::NoTrees);
    }

    #[test]
    fn rejects_a_file_cut_before_the_end_of_trees() {
        let end = TWO_TREES
            .find("end of trees")
            .expect("the model ends its trees");

        assert_eq!(
            super::parse(&TWO_TREES[..end]).unwrap_err(),
            ModelError::Truncated
        );
    }

    #[test]
    fn rejects_a_broken_number() {
        let error = parse_with("leaf_value=10 20 30", "leaf_value=10 x 30").unwrap_err();

        assert_eq!(error, ModelError::InvalidValue { line: 18 });
    }

    #[test]
    fn rejects_zero_leaves() {
        let text = TWO_TREES
            .replacen("num_leaves=1\n", "num_leaves=0\n", 1)
            .replacen("leaf_value=0.5\n", "leaf_value=\n", 1);

        assert_eq!(super::parse(&text).unwrap_err(), ModelError::NoLeaves);
    }

    #[test]
    fn rejects_an_overflowing_max_feature_idx() {
        let largest =
            parse_with("max_feature_idx=1", "max_feature_idx=18446744073709551615").unwrap_err();
        let negative = parse_with("max_feature_idx=1", "max_feature_idx=-1").unwrap_err();

        assert_eq!(largest, ModelError::InvalidValue { line: 5 });
        assert_eq!(negative, ModelError::InvalidValue { line: 5 });
    }

    #[test]
    fn rejects_num_leaves_not_matching_leaf_values() {
        let error = parse_with("num_leaves=1\n", "num_leaves=3\n").unwrap_err();

        assert_eq!(error, ModelError::InvalidValue { line: 22 });
    }
}

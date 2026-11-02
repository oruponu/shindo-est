//! The part of a model file that holds one model's trees.

use crate::bundle::FormatError;
use crate::bundle::codec::{self, Reader, Writer};
use crate::forest::{Forest, Tree};
use crate::model::{Model, ModelError, Objective};

/// Most leaves a tree may have, `LightGBM`'s upper bound on `num_leaves`.
pub(crate) const MAX_LEAVES: usize = 131_072;

/// Most trees a forest may have, about 20 times the trees expected in a
/// model. It bounds the memory that forged one-leaf trees take beyond the
/// size of the file.
pub(crate) const MAX_TREES: usize = 65_536;

/// Code of [`Objective::Regression`].
const REGRESSION: u8 = 0;
/// Code of [`Objective::Binary`].
const BINARY: u8 = 1;

/// Fewest bytes a tree takes: its leaf count and one leaf value.
const MIN_TREE_BYTES: usize = 12;

/// Writes the objective and the trees of a model. A forest with more than
/// [`MAX_TREES`] trees, or a tree with more than [`MAX_LEAVES`] leaves, is
/// written but cannot be read back, so callers check the limits first.
///
/// # Panics
///
/// Panics if the number of features, trees, or leaves of a tree does not fit
/// in `u32`.
pub(crate) fn write(writer: &mut Writer, objective: Objective, forest: &Forest) {
    writer.u8(match objective {
        Objective::Regression => REGRESSION,
        Objective::Binary => BINARY,
    });
    writer.u32(codec::to_u32(forest.feature_count()));
    writer.u32(codec::to_u32(forest.tree_count()));
    for tree in forest.trees() {
        let arrays = tree.arrays();
        writer.u32(codec::to_u32(arrays.leaf_value.len()));
        writer.u32s(&arrays.split_feature);
        writer.f64s(&arrays.threshold);
        writer.u8s(&arrays.decision_type);
        writer.i32s(&arrays.left_child);
        writer.i32s(&arrays.right_child);
        writer.f64s(&arrays.leaf_value);
    }
}

/// Reads what [`write()`] wrote and names the features `feature_names`.
pub(crate) fn read(
    reader: &mut Reader<'_>,
    feature_names: Vec<String>,
) -> Result<Model, FormatError> {
    let objective = match reader.u8()? {
        REGRESSION => Objective::Regression,
        BINARY => Objective::Binary,
        code => return Err(FormatError::InvalidObjective { code }),
    };
    let feature_count = codec::to_usize(u64::from(reader.u32()?))?;
    let tree_count = codec::to_usize(u64::from(reader.u32()?))?;
    if tree_count > MAX_TREES {
        return Err(FormatError::LimitExceeded {
            what: "trees in a forest",
        });
    }
    if tree_count > reader.remaining() / MIN_TREE_BYTES {
        return Err(FormatError::Truncated);
    }
    // Not reserved up front: a tree takes more memory than the `MIN_TREE_BYTES`
    // it takes at least in the file, so a large count could ask for more than
    // the file is worth before any tree is checked.
    let mut trees = Vec::new();
    for _ in 0..tree_count {
        trees.push(read_tree(reader)?);
    }
    let forest = Forest::new(feature_count, trees).map_err(FormatError::Model)?;
    Model::new(forest, objective, feature_names).map_err(FormatError::Model)
}

fn read_tree(reader: &mut Reader<'_>) -> Result<Tree, FormatError> {
    let leaf_count = codec::to_usize(u64::from(reader.u32()?))?;
    let Some(node_count) = leaf_count.checked_sub(1) else {
        return Err(FormatError::Model(ModelError::NoLeaves));
    };
    if leaf_count > MAX_LEAVES {
        return Err(FormatError::LimitExceeded {
            what: "leaves in a tree",
        });
    }
    let split_feature = reader.u32s(node_count)?;
    let threshold = reader.f64s(node_count)?;
    let decision_type = reader.u8s(node_count)?;
    let left_child = reader.i32s(node_count)?;
    let right_child = reader.i32s(node_count)?;
    let leaf_value = reader.f64s(leaf_count)?;
    Tree::new(
        &split_feature,
        &threshold,
        &decision_type,
        &left_child,
        &right_child,
        &leaf_value,
    )
    .map_err(FormatError::Model)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::lightgbm_text;

    fn write_model(model: &Model) -> Vec<u8> {
        let mut writer = Writer::new();
        super::write(&mut writer, model.objective(), model.forest());
        writer.into_bytes()
    }

    fn read_bytes(bytes: &[u8], feature_names: Vec<String>) -> Result<Model, FormatError> {
        let mut reader = Reader::new(bytes);
        let model = super::read(&mut reader, feature_names)?;
        reader.finish()?;
        Ok(model)
    }

    fn names(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("f{index}")).collect()
    }

    /// Returns a forest section of one tree on `feature_count` features, the
    /// tree written by `write_tree`.
    fn one_tree(feature_count: u32, write_tree: impl FnOnce(&mut Writer)) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.u8(0);
        writer.u32(feature_count);
        writer.u32(1);
        write_tree(&mut writer);
        writer.into_bytes()
    }

    /// Writes a tree with one split on `feature` and the children given.
    fn one_split(writer: &mut Writer, feature: u32, left: i32, right: i32) {
        writer.u32(2);
        writer.u32s(&[feature]);
        writer.f64s(&[0.5]);
        writer.u8s(&[0]);
        writer.i32s(&[left]);
        writer.i32s(&[right]);
        writer.f64s(&[1.0, 2.0]);
    }

    #[test]
    fn lightgbm_fixtures_round_trip() {
        for name in ["nan", "zero"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/lightgbm")
                .join(name)
                .join("model.txt");
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let model = lightgbm_text::parse(&text).expect("the fixture should be valid");

            let bytes = write_model(&model);
            let read = read_bytes(&bytes, model.feature_names().to_vec())
                .expect("the written model should be valid");

            assert_eq!(read, model, "{name}");
            assert_eq!(write_model(&read), bytes, "{name}");
        }
    }

    #[test]
    fn reads_a_forest_written_by_hand() {
        let mut bytes = one_tree(1, |writer| one_split(writer, 0, -1, -2));
        let regression = read_bytes(&bytes, names(1)).expect("the forest should be valid");
        assert_eq!(regression.objective(), Objective::Regression);
        assert_eq!(regression.forest().raw(&[0.0]), 1.0);
        assert_eq!(regression.forest().raw(&[1.0]), 2.0);

        bytes[0] = 1;
        let binary = read_bytes(&bytes, names(1)).expect("the forest should be valid");
        assert_eq!(binary.objective(), Objective::Binary);
    }

    #[test]
    fn one_leaf_tree_round_trips() {
        let tree =
            Tree::new(&[], &[], &[], &[], &[], &[-0.25]).expect("a one-leaf tree should be valid");
        let forest = Forest::new(3, vec![tree]).expect("the forest should be valid");
        let model =
            Model::new(forest, Objective::Regression, names(3)).expect("the model should be valid");

        let bytes = write_model(&model);
        assert_eq!(read_bytes(&bytes, names(3)), Ok(model));
    }

    #[test]
    fn negative_zero_round_trips_bit_for_bit() {
        let tree = Tree::new(&[0], &[-0.0], &[0], &[-1], &[-2], &[-0.0, 1.0])
            .expect("the tree should be valid");
        let forest = Forest::new(1, vec![tree]).expect("the forest should be valid");
        let model =
            Model::new(forest, Objective::Regression, names(1)).expect("the model should be valid");

        let read = read_bytes(&write_model(&model), names(1)).expect("the model should be read");
        let arrays = read.forest().trees()[0].arrays();
        assert_eq!(arrays.threshold[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(arrays.leaf_value[0].to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn rejects_an_unknown_objective() {
        let mut bytes = one_tree(1, |writer| one_split(writer, 0, -1, -2));
        bytes[0] = 2;
        assert_eq!(
            read_bytes(&bytes, names(1)),
            Err(FormatError::InvalidObjective { code: 2 })
        );
    }

    /// Returns the start of a forest section that declares `tree_count` trees
    /// on one feature, followed by no trees.
    fn tree_count_only(tree_count: u32) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.u8(0);
        writer.u32(1);
        writer.u32(tree_count);
        writer.into_bytes()
    }

    #[test]
    fn rejects_a_tree_count_beyond_the_bytes() {
        assert_eq!(
            read_bytes(&tree_count_only(2), names(1)),
            Err(FormatError::Truncated)
        );
    }

    #[test]
    fn rejects_too_many_trees() {
        let limit = Err(FormatError::LimitExceeded {
            what: "trees in a forest",
        });
        assert_eq!(read_bytes(&tree_count_only(65_537), names(1)), limit);
        assert_eq!(read_bytes(&tree_count_only(u32::MAX), names(1)), limit);
        // At the limit, the trees are looked for and turn out to be missing.
        assert_eq!(
            read_bytes(&tree_count_only(65_536), names(1)),
            Err(FormatError::Truncated)
        );
    }

    #[test]
    fn rejects_too_many_trees_that_the_bytes_hold() {
        let tree =
            Tree::new(&[], &[], &[], &[], &[], &[0.0]).expect("a one-leaf tree should be valid");
        let forest = Forest::new(1, vec![tree; MAX_TREES + 1]).expect("the forest should be valid");
        let model =
            Model::new(forest, Objective::Regression, names(1)).expect("the model should be valid");

        // Compared with `err()`, so that a failure does not print every tree.
        assert_eq!(
            read_bytes(&write_model(&model), names(1)).err(),
            Some(FormatError::LimitExceeded {
                what: "trees in a forest"
            })
        );
    }

    #[test]
    fn rejects_too_many_leaves() {
        // Enough bytes after the leaf count for the tree count to pass.
        let tree_with = |leaf_count| {
            one_tree(1, |writer| {
                writer.u32(leaf_count);
                writer.u64(0);
            })
        };

        assert_eq!(
            read_bytes(&tree_with(131_073), names(1)),
            Err(FormatError::LimitExceeded {
                what: "leaves in a tree"
            })
        );
        // At the limit, the arrays are read and turn out to be cut short.
        assert_eq!(
            read_bytes(&tree_with(131_072), names(1)),
            Err(FormatError::Truncated)
        );
    }

    #[test]
    fn rejects_a_truncated_forest() {
        let bytes = one_tree(1, |writer| one_split(writer, 0, -1, -2));
        assert!(read_bytes(&bytes, names(1)).is_ok(), "the whole forest");
        for length in 0..bytes.len() {
            assert_eq!(
                read_bytes(&bytes[..length], names(1)),
                Err(FormatError::Truncated),
                "{length} bytes"
            );
        }
    }

    #[test]
    fn invalid_trees_are_model_errors() {
        let to_itself = one_tree(1, |writer| one_split(writer, 0, 0, -1));
        assert_eq!(
            read_bytes(&to_itself, names(1)),
            Err(FormatError::Model(ModelError::InvalidChild))
        );

        let second_feature = one_tree(1, |writer| one_split(writer, 1, -1, -2));
        assert_eq!(
            read_bytes(&second_feature, names(1)),
            Err(FormatError::Model(ModelError::FeatureOutOfRange {
                feature: 1,
                feature_count: 1,
            }))
        );

        let no_leaves = one_tree(1, |writer| {
            writer.u32(0);
            writer.u64(0);
        });
        assert_eq!(
            read_bytes(&no_leaves, names(1)),
            Err(FormatError::Model(ModelError::NoLeaves))
        );
    }
}

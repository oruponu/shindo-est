# Model file format

A model file holds everything shindo-est needs to predict intensities: a
manifest, the felt classifier and the intensity regressor (two LightGBM
forests), and the Kuril and Izu-Bonin Slab2 grids of the Pacific slab. This
document specifies layout version 1 for implementations in any language.

All integers are little-endian and unsigned unless noted. `f64` is an IEEE 754
binary64 value in little-endian byte order, and `i32` is a little-endian
two's complement integer. Every `f64` is stored bit for bit, including the
sign of zero and the payload of NaN.

## Layout

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | Magic: the bytes `89 53 48 45 0D 0A 1A 0A` |
| 8 | 4 | Layout version (`u32`), 1 |
| 12 | | Five sections, in the order below |
| end - 32 | 32 | SHA-256 of every byte before it |

As in PNG, the magic catches common damage: the first byte has its high bit
set, so a 7-bit transfer changes it; the CR LF and the LF detect line-ending
conversion in either direction; and the 0x1A byte (Control-Z) stops the
display of the file under MS-DOS.

Each section is:

| Size | Content |
|---|---|
| 4 | Name, four ASCII letters |
| 8 | Length of the content in bytes (`u64`) |
| length | Content |

| Order | Name | Content |
|---|---|---|
| 1 | `MNFT` | The manifest, as UTF-8 text |
| 2 | `CLSF` | The felt classifier, as a forest section |
| 3 | `RGSR` | The intensity regressor, as a forest section |
| 4 | `SLBK` | The Kuril grid, as a grid section |
| 5 | `SLBI` | The Izu-Bonin grid, as a grid section |

Writers must not write anything beyond what this document describes. With
the manifest in its canonical form and every `f64` stored bit for bit, the
same models then always give the same bytes.

## Manifest

The manifest is `key=value` lines. Each key other than `source` appears
exactly once; `source` may appear any number of times, and its lines keep
their order. A value must not be empty or contain a control character.

| Key | Value |
|---|---|
| `model_id` | Identifier of the models |
| `feature_version` | Version of the feature computation (`u32`), 1 |
| `feature_names` | The feature names, separated by single spaces |
| `hurdle_felt_probability` | Lowest felt probability at which a site counts as felt, strictly between 0 and 1 |
| `hurdle_felt_floor` | Lowest instrumental intensity of a felt site |
| `hurdle_unfelt_ceiling` | Highest instrumental intensity of an unfelt site, below the felt floor |
| `hurdle_logit_per_intensity` | Amount the classifier's logit rises for each unit of intensity offset, at least 0 |
| `arv_source` | Where the site amplification factors (ARV) used in training came from |
| `training_period` | The period of the training data |
| `training_conditions` | How the training rows were selected |
| `source` | A data source or credit; optional and repeatable |

The four hurdle values are decimal numbers and must be finite.

Readers skip empty lines, ignore a leading byte order mark, accept CR LF line
endings, accept the keys in any order, and accept any decimal notation for
numbers. Writers use the canonical form:

- the keys in the order of the table above, `source` lines last;
- every line ending in LF;
- `feature_version` as a decimal integer;
- each hurdle value written with the fewest digits that read back to the same
  value: in plain notation with at least one digit after the point (`0.35`,
  `1.0`, `0.0001`) when it is 0 or its magnitude is at least 1e-4 and below
  1e16, and otherwise with a lowercase `e` and no `+` (`9.9e-5`, `2.5e16`).

`feature_names` must be these 14 names in this order:

```text
mj depth qlat qlon slat slon log_hypo epi az_sin az_cos log_arv slab_top updip path_frac
```

`feature_version` 1 names the way the current library computes those
features. Any change that alters a feature value increments it. A reader
rejects a model file whose `feature_version` it does not support, because its
models were trained on features computed differently.

## Forest section

| Size | Content |
|---|---|
| 1 | Objective: 0 for regression, 1 for binary classification |
| 4 | Number of features (`u32`) |
| 4 | Number of trees (`u32`) |
| | The trees, one after another |

Each tree with `L` leaves is:

| Count | Type | Content |
|---|---|---|
| 1 | `u32` | `L`, the number of leaves |
| `L - 1` | `u32` | Feature index of each split |
| `L - 1` | `f64` | Threshold of each split |
| `L - 1` | `u8` | Decision type of each split |
| `L - 1` | `i32` | Left child of each split |
| `L - 1` | `i32` | Right child of each split |
| `L` | `f64` | Value of each leaf |

The arrays are LightGBM's arrays for the tree, as in its model text. A tree
with one leaf has empty split arrays. A nonnegative child is the index of a
split, which must come after its parent; a negative child `c` is leaf
`-c - 1`. In a decision type, bit 0 marks a categorical split, which is not
supported; bit 1 sends missing values to the left; bits 2 and 3 are the
missing type: 0 for none, 1 for zero, 2 for NaN.

The classifier must have objective 1 and the regressor objective 0. The
forests take the features in the order of `feature_names`, so the number of
features is 14.

## Grid section

| Count | Type | Content |
|---|---|---|
| 1 | `u32` | `R`, the number of rows (latitudes) |
| 1 | `u32` | `C`, the number of columns (longitudes) |
| `R` | `f64` | Latitude of each row in degrees, south to north |
| `C` | `f64` | Longitude of each column in degrees, west to east |
| `R x C` | `f64` | Depth of the slab top in km, positive down |
| `R x C` | `f64` | Thickness of the slab in km |

The depth and thickness are row-major: the node of row `i` and column `j` is
at index `i x C + j`. A node without a value is NaN; any NaN is accepted, and
its bits are kept like those of any other `f64`. The dip of the slab is not
stored; readers compute it from the depths.

Each axis increases from its first value along the 0.05 degree grid: value
`k` is the first value plus `k` times 0.05, within one millionth of the
spacing. Latitudes lie within -90 to 90 and longitudes within -180 to 360,
and both are finite.

## Checks and limits

A reader checks, in this order:

1. The file starts with the magic. A file shorter than 8 bytes does not.
2. The file has at least 44 bytes (the magic, the version, and the checksum).
3. The layout version is 1.
4. The SHA-256 at the end matches the bytes before it.
5. The sections have the names and order above, each length fits in the bytes
   left, and the content of each section is parsed exactly: no bytes are left
   over, and none past its end are read.
6. The manifest has at most 65,536 bytes and is valid UTF-8. If its first
   `feature_version` line specifies a version other than 1, the reader
   rejects the file and reports that version, even if the rest of the
   manifest violates the rules above. Otherwise, the manifest follows the
   rules above, with `feature_version` 1 and the 14 feature names.
7. Each forest: at most 65,536 trees, 1 to 131,072 leaves per tree, as many
   trees within the section as its tree count declares, finite leaf values,
   no NaN thresholds, no categorical splits, children as described, and
   feature indices below the number of features.
8. Each grid: at least 2 rows and 2 columns, at most 4,000,000 nodes, and axes
   as described.
9. No bytes follow the last section before the checksum.
10. The classifier and regressor have their objectives, and the hurdle values
    satisfy the bounds in the manifest table.

A writer must not write a file that a reader would reject: in particular, no
manifest of more than 65,536 bytes in canonical form, no forest with more
than 65,536 trees, no tree with more than 131,072 leaves, and no grid with
more than 4,000,000 nodes.

## Versions

The layout version (offset 8) changes when the layout of the file changes. A
reader rejects any layout version it does not know.

`feature_version` in the manifest changes when the way features are computed
changes, without changing the layout. Such a change may add manifest keys. A
reader rejects a model file whose `feature_version` it does not support and
reports the unsupported version, even if the manifest has unrecognized keys.
Any other change to the set of manifest keys requires a new layout version.

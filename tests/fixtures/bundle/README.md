# Model file fixture

A small model file and the text files used to pack it, for detecting
unintended changes to the model file layout. The text files are synthetic
and were written by hand for this repository; no real earthquake, station,
or Slab2 data is used.

Files:

- `manifest.txt`: the manifest in its canonical form.
- `classifier.txt` and `regressor.txt`: LightGBM models in text format, with
  single-split trees on `log_hypo` and `slab_top` (the classifier) and on
  `mj`, `log_arv`, and `path_frac` (the regressor). The test's hypocenters
  and sites exercise both branches of every split. Since `path_frac` depends
  on the slab thickness and dip, reading a grid incorrectly changes the
  predictions.
- `kuril.csv`: a 5 x 5 grid of nodes spanning 35.00 to 35.20 N and 140.00 to
  140.20 E, with depth increasing eastward by 10 km per column, starting at
  50 km, and a slab thickness of 40 km. No point is given for the node at
  35.10 N 140.10 E, so that node has no value.
- `izu_bonin.csv`: a 4 x 4 grid of nodes spanning 34.80 to 34.95 N and 140.00
  to 140.15 E, with a constant depth of 30 km and a slab thickness of 40 km.
- `model.bin`: the model file packed from the five files above.

## Regenerating

From the repository root:

```sh
cargo run --release -p shindo-est --example pack_model -- \
    tests/fixtures/bundle/manifest.txt \
    tests/fixtures/bundle/classifier.txt \
    tests/fixtures/bundle/regressor.txt \
    tests/fixtures/bundle/kuril.csv \
    tests/fixtures/bundle/izu_bonin.csv \
    tests/fixtures/bundle/model.bin
```

The same text files always give the same bytes.
`crates/shindo-est/tests/model_bundle.rs` fails when the packed bytes differ
from `model.bin`. If the change is intended, regenerate `model.bin` and
review the test expectations. A change that breaks compatibility with
existing readers also requires a new layout version (see
`docs/model-format.md`).

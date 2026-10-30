# LightGBM reference fixtures

Small LightGBM models and LightGBM's own predictions for them, used to check
that shindo-est evaluates trees the same way. Everything here comes from
synthetic random data; no earthquake or station data is used.

Each directory holds one model:

- `nan/`: missing values are NaN (LightGBM's default). Its splits use the
  missing types NaN and None.
- `zero/`: trained with `zero_as_missing=True`. Its splits use the missing
  type Zero.

After training, the thresholds of every other split in each tree (splits 0, 2,
4, and so on) were replaced with the nearest float32 values, and the model was
loaded back into LightGBM and saved. Inputs can then equal those thresholds
exactly. The other thresholds keep their values, most of which cannot be
represented exactly as float32.

Files in each directory:

- `model.txt`: the model saved by `Booster.save_model`. 40 trees, 18 features.
- `inputs.csv`: columns `f0` to `f17`. 300 random rows, then 10 rows for each
  of 100 split nodes picked at random. Each of the 10 rows sets the node's
  feature to one of these values and reaches the node: the threshold itself,
  the threshold rounded to float32, the float32 values just below and above
  it, `0`, `nan`, plus or minus LightGBM's zero threshold (`1e-35` as a
  float32), and the float32 values just outside the zero threshold. The other
  features come from the first random row that still reaches the node after
  the node's feature is set to that value.
- `expected.csv`: `raw` is `Booster.predict(inputs, raw_score=True)` and
  `leaf_0` to `leaf_39` are `Booster.predict(inputs, pred_leaf=True)`, with the
  inputs converted to float32 first.

## Regenerating

The files were generated on Linux (Ubuntu 26.04 under WSL, Python 3.14) with
the versions in `training/requirements.txt` (LightGBM 4.7.0, NumPy 2.5.4).
Running the script twice there gives the same bytes. On another OS, CPU, or
LightGBM build, the trained trees themselves may differ. The three files of a
model still agree with each other, because the inputs and predictions are made
from the saved model.

```bash
python3 -m venv training/.venv
training/.venv/bin/python -m pip install -r training/requirements.txt
training/.venv/bin/python training/fixtures/make_lightgbm_fixtures.py tests/fixtures/lightgbm
```

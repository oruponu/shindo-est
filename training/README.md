# Training

The `training` package builds training data from public sources and compares
three ARV cases: the ARV derived from the Cabinet Office's seismic intensity
increments (`cabinet_office`), the ARV provided by J-SHIS (`jshis`), and no
ARV (`none`, as a baseline). It saves the LightGBM models it trains for the
comparison, but does not produce a shindo-est model file yet.

For each split, all three cases use the same rows: earthquakes from 2004 to
2019 at stations with an ARV from both sources. About 20% of the stations
are held out from training, allowing separate evaluation on training and
held-out stations. The dataset also includes earthquakes from 2020 to 2024,
which are excluded from the comparison and reserved for a final evaluation.

## Requirements

- Linux or macOS. The times below were measured on a Mac mini (Apple M4, 10
  cores, 16 GB).
- Python 3.15.
- Rust, to build the `shindo_est` extension.
- On macOS, Homebrew's `libomp`, which LightGBM loads:

  ```sh
  brew install libomp
  ```

Run every command from the repository root. Create the virtual environment
and install the pinned dependencies and the extension:

```sh
python3.15 -m venv training/.venv
training/.venv/bin/python -m pip install -r training/requirements.txt
training/.venv/bin/python -m pip install ./crates/shindo-est-py
```

The extension is built from the repository when it is installed. Install it
again after changing the core or `crates/shindo-est-py`. Before training,
`compare_arv run` checks the features that the installed extension computes
against `tests/fixtures/bundle/expected.csv`, and raises
`StaleExtensionError` if they differ.

To run the tests:

```sh
training/.venv/bin/python -m unittest discover -s training/tests -t .
```

## Sources and terms

The source files and the data, models, and results generated from them go
under `training/data/`, which git ignores.

- Do not commit or redistribute the source files. The terms of both ARV
  sources forbid distributing copies of the data, including copies converted
  to another format.
- Do not distribute `arv.csv` or `dataset.npz`. They hold values taken from
  the sources for each station.
- The models are trained on the dataset, and those using ARV may contain
  split thresholds derived from ARV values. The predictions are probabilities
  and intensities computed by the models, and `results.md` and `results.json`
  hold aggregate measures. Both ARV sources allow distributing processed
  products under their respective terms, including attribution requirements.
  `dataset-summary.json` holds counts taken only from the JMA files, some of
  them for each station. Check that these files comply with the terms before
  publishing them.

| Source | Files | Place |
|---|---|---|
| JMA seismic intensity bulletin and station list | `i2004.zip` to `i2024.zip`, `code_p.zip` | `training/data/jma/` |
| Cabinet Office, nationwide seismic intensity increments on a 250 m mesh | `to250mr3japan.zip` | `training/data/ground/` |
| J-SHIS (NIED), surface ground model, Vs 400 m/s to the surface | `Z-V4-JAPAN-AMP-VS400_M250.zip` | `training/data/ground/` |

- **JMA**: `training.sources` fetches the files listed on
  https://www.data.jma.go.jp/eqev/data/bulletin/shindo_e.html . They may be
  used under the Public Data License (Version 1.0), citing the source as
  "Source: Japan Meteorological Agency website (URL of the page)" and,
  when the content is edited, stating so
  (https://www.jma.go.jp/jma/en/copyright.html).
- **Cabinet Office**: download `to250mr3japan.zip` from the dataset
  https://www.geospatial.jp/ckan/dataset/1312 on G-Spatial Information
  Center, which requires logging in. The data come from the Cabinet Office's
  study group on the model of earthquakes directly beneath the Tokyo
  metropolitan area. Its terms of use (published on the same page) require
  citing that study group, allow distributing products edited from the data,
  and forbid distributing copies of the data.
- **J-SHIS**: download
  https://www.j-shis.bosai.go.jp/map/JSHIS2/data/Z/V4/JAPAN/AMP/Z-V4-JAPAN-AMP-VS400_M250.zip .
  The terms (https://www.j-shis.bosai.go.jp/agreement) allow distributing
  edited products and forbid distributing copies of the data. Work that
  quotes the surface ground data cites the references in the rules for the
  surface ground data files on the J-SHIS download page, such as
  Wakamatsu, K. and Matsuoka, M. (2013): Nationwide 7.5-Arc-Second Japan
  Engineering Geomorphologic Classification Map and Vs30 Zoning, Journal of
  Disaster Research, Vol. 8, No. 5, pp. 904-911.

Keep the downloaded files unchanged; the ZIP files are read without
extracting them.

## Checking the sources

[`SHA256SUMS`](SHA256SUMS) records the SHA-256 of every source file.
`training.sources fetch`, `training.arv`, and `training.dataset` check the
source files they read against it and stop when one differs, so that a
replaced source does not silently change the data. To check files by hand:

```sh
training/.venv/bin/python -m training.sources verify training/data/ground/to250mr3japan.zip training/data/ground/Z-V4-JAPAN-AMP-VS400_M250.zip
```

When a check fails:

- If the file was downloaded by hand, download it again; the copy may be
  broken.
- A JMA file whose download differs is left as `<name>.part`. If JMA has
  replaced the file, the data built from it may change. To accept the new
  file, delete the old file, the `.part` file, and the line in `SHA256SUMS`,
  then run `training.sources fetch --record <name>`, which downloads the file
  and records it.
- To accept a new ground data file, delete its line in `SHA256SUMS` and run
  `training.sources record <path>`.

Commit the changed `SHA256SUMS`, since the results depend on it.

## Steps

1. Fetch the JMA files into `training/data/jma/` (about 30 s, 39 MB). Files
   already there are checked, not downloaded again.

   ```sh
   training/.venv/bin/python -m training.sources fetch
   ```

2. Look up each station's ARV in both ground data files and write
   `training/data/arv.csv` (about 3 s). Stations outside the first-order
   meshes, or in a mesh without data, get no ARV from that source.

   ```sh
   training/.venv/bin/python -m training.arv
   ```

3. Build the training data, `training/data/dataset.npz`, and its summary,
   `training/data/dataset-summary.json` (about 4 s, 0.8 GB of memory).

   ```sh
   training/.venv/bin/python -m training.dataset --jma training/data/jma --out training/data/dataset.npz
   ```

4. Train each ARV case on each split (about 90 min for the 21 trainings,
   2.4 GB of memory, 2.6 GB on disk). The `year` split trains on the
   earthquakes from 2004 to 2016 and tests on those from 2017 to 2019. Each
   `region:<name>` split groups the earthquakes from 2004 to 2019 into six
   regions by epicenter, and tests on one region after training on the
   others, so no earthquake appears in both the training and test sets.

   ```sh
   training/.venv/bin/python -m training.compare_arv run --dataset training/data/dataset.npz --arv training/data/arv.csv --out training/data/compare-arv
   ```

   `--case` and `--split` train only some of them:

   ```sh
   training/.venv/bin/python -m training.compare_arv run --dataset training/data/dataset.npz --arv training/data/arv.csv --out training/data/compare-arv --case cabinet_office --split year
   ```

   A stopped run picks up where it left off: a case and split is skipped
   when its files are there and were made from the same inputs and commit.
   If the working tree has changes, including untracked files, nothing is
   skipped: every selected case and split is trained again. Commit or remove
   the changes before running.

5. Evaluate the trained models and write `results.md` and `results.json` in
   the same directory (about 5 s). Pass the same `--dataset` and `--arv`; the
   report stops if the predictions were made from other files.

   ```sh
   training/.venv/bin/python -m training.compare_arv report --dataset training/data/dataset.npz --arv training/data/arv.csv --out training/data/compare-arv
   ```

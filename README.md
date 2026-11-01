# shindo-est

shindo-est predicts the JMA instrumental seismic intensity at given sites from a hypocenter (latitude, longitude, depth, JMA magnitude Mj) and site parameters (latitude, longitude, ARV). Instead of a ground-motion attenuation equation, it uses LightGBM models trained on past intensity observations. It is written in Rust.

> **Status:** This is an experimental project under active development. The API, the model format, and the estimated intensities may change at any time.

> **Notice:** shindo-est is for research and visualization only. Its output is not official information, it is not an earthquake prediction, and it must not be used for forecasting or warning services that require a license under the Japanese Meteorological Service Act.

## Model files

A model file is a binary file that contains everything needed to make predictions: the two LightGBM forests (a felt classifier and an intensity regressor), the Slab2 grids of the Pacific slab, and a manifest that describes the models. The format is described in [docs/model-format.md](docs/model-format.md).

To pack a model file from a manifest, two models in LightGBM's text format, and the Kuril and Izu-Bonin grids as `lon,lat,depth,thickness` CSV files, run:

```sh
cargo run --release -p shindo-est --example pack_model -- manifest.txt classifier.txt regressor.txt kuril.csv izu_bonin.csv model.bin
```

No trained model is published yet. [tests/fixtures/bundle](tests/fixtures/bundle) has a small synthetic one.

## License

The code is licensed under the MIT License. Data sources and their terms are listed in [NOTICE](NOTICE).

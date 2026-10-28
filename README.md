# shindo-est

shindo-est predicts the JMA instrumental seismic intensity at given sites from a hypocenter (latitude, longitude, depth, JMA magnitude Mj) and site parameters (latitude, longitude, ARV). Instead of a ground-motion attenuation equation, it uses LightGBM models trained on past intensity observations. It is written in Rust.

> **Status:** This is an experimental project under active development. The API, the model format, and the estimated intensities may change at any time.

> **Notice:** shindo-est is for research and visualization only. Its output is not official information, it is not an earthquake prediction, and it must not be used for forecasting or warning services that require a license under the Japanese Meteorological Service Act.

## License

The code is licensed under the MIT License. Data sources and their terms are listed in [NOTICE](NOTICE).

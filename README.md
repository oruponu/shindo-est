# shindo-est

shindo-est predicts the JMA instrumental seismic intensity at given sites from a hypocenter (latitude, longitude, depth, JMA magnitude Mj) and site parameters (latitude, longitude, ARV). Instead of a ground-motion attenuation equation, it uses LightGBM models trained on past intensity observations. It is written in Rust.

> **Status:** This is an experimental project under active development. The API, the model format, and the estimated intensities may change at any time.

> **Notice:** shindo-est is for research and visualization only. Its output is not official information, it is not an earthquake prediction, and it must not be used for forecasting or warning services that require a license under the Japanese Meteorological Service Act.

## Model files

A model file is a binary file that contains everything needed to make predictions: the two LightGBM forests (a felt classifier and an intensity regressor), the Slab2 grids of the Pacific slab, and a manifest that describes the models. The format is described in [docs/model-format.md](docs/model-format.md).

To pack a model file from a manifest, two models in LightGBM's text format, and the Kuril and Izu-Bonin grids as `lon,lat,depth,thickness` CSV files, run:

```sh
cargo run --release -p shindo-est-cli -- pack manifest.txt classifier.txt regressor.txt kuril.csv izu_bonin.csv model.bin
```

No trained model is published yet. [tests/fixtures/bundle](tests/fixtures/bundle) has a small synthetic one.

## C ABI

The `shindo-est-ffi` crate builds a shared library that exposes a C API:

```sh
cargo build --release -p shindo-est-ffi
```

The library is named `shindo_est_ffi.dll` on Windows, `libshindo_est_ffi.so` on Linux, and `libshindo_est_ffi.dylib` on macOS, and is placed in `target/release/`. On Windows, the import library is `shindo_est_ffi.dll.lib` with the MSVC toolchain and `libshindo_est_ffi.dll.a` with the GNU toolchain. The header [crates/shindo-est-ffi/include/shindo_est.h](crates/shindo-est-ffi/include/shindo_est.h) documents each function and its error codes, and the order in which `shindo_est_predict` checks its arguments.

```c
#include <stdint.h>
#include <stdio.h>

#include "shindo_est.h"

int main(void) {
  shindo_est_model *model = NULL;
  int32_t code = shindo_est_model_open_file("model.bin", &model);
  if (code != SHINDO_EST_OK) {
    fprintf(stderr, "%s\n", shindo_est_error_message(code));
    return 1;
  }

  shindo_est_hypocenter hypocenter = {35.02, 140.02, 80.0, 6.5};
  shindo_est_site sites[2] = {{35.02, 140.02, 1.0}, {40.0, 135.0, 1.0}};
  shindo_est_options options = {.size = sizeof options, .offset = 0.2};
  shindo_est_result results[2];
  code = shindo_est_predict(model, &hypocenter, sites, 2, &options, results);
  if (code == SHINDO_EST_OK) {
    for (int index = 0; index < 2; ++index) {
      const char *label =
          shindo_est_intensity_label(results[index].intensity_class);
      printf("%.2f %s\n", results[index].instrumental,
             label != NULL ? label : "-");
    }
  } else {
    fprintf(stderr, "%s\n", shindo_est_error_message(code));
  }

  shindo_est_model_close(model);
  return code == SHINDO_EST_OK ? 0 : 1;
}
```

- Functions that report errors return an `int32_t` status code, with `SHINDO_EST_OK` (0) indicating success. `shindo_est_error_message` returns a description of a code. If `shindo_est_predict` returns an error, it leaves the results unchanged.
- The caller allocates the results, one `shindo_est_result` per site. A result's `status` indicates whether an intensity was predicted for the site. If not, its floating-point fields are NaN and its `intensity_class` is -1.
- Multiple threads may call `shindo_est_predict` concurrently with the same model handle. Close the handle only once, after all calls have completed. The strings returned by `shindo_est_model_id` and `shindo_est_model_manifest` become invalid when the handle is closed.
- Initialize `shindo_est_options` with `{ .size = sizeof options }` and set any other fields as needed. A field left at zero takes its default, and passing NULL for the options uses the defaults for all of them. The `size` field allows callers and libraries built against different header versions to remain compatible. If a caller sets an option that the library does not recognize, the library returns `SHINDO_EST_ERROR_UNSUPPORTED_OPTION` instead of silently ignoring it.

## License

The code is licensed under the MIT License. Data sources and their terms are listed in [NOTICE](NOTICE).

/* SPDX-License-Identifier: MIT */

/*
 * Calls the shared library from C, as a C caller builds against the header.
 *
 * Usage: smoke <model file>
 *
 * Prints "smoke: ok" and exits 0 when every check passes. Otherwise prints
 * each failed check to standard error and exits 1. Exits 2 on a wrong number
 * of arguments.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "shindo_est.h"

/* The layouts the Rust side promises, on every target. */
_Static_assert(sizeof(shindo_est_hypocenter) == 32, "hypocenter size");
_Static_assert(sizeof(shindo_est_site) == 24, "site size");
_Static_assert(sizeof(shindo_est_result) == 32, "result size");
_Static_assert(offsetof(shindo_est_result, intensity_class) == 4,
               "result intensity_class offset");
_Static_assert(offsetof(shindo_est_result, instrumental) == 8,
               "result instrumental offset");
_Static_assert(offsetof(shindo_est_result, probability) == 16,
               "result probability offset");
_Static_assert(offsetof(shindo_est_result, regression) == 24,
               "result regression offset");
_Static_assert(sizeof(shindo_est_options) == 24, "options size");
_Static_assert(offsetof(shindo_est_options, size) == 0, "options size offset");
_Static_assert(offsetof(shindo_est_options, reserved) == 20,
               "options reserved offset");

/* Number of failed checks, which check() and check_site() count and main()
 * reports. */
static int failures = 0;

/* Records a failed check when `passed` is false. */
static void check(bool passed, const char *what) {
  if (!passed) {
    fprintf(stderr, "smoke: failed: %s\n", what);
    ++failures;
  }
}

/* Records a failed check about the site at index `site` when `passed` is
 * false. */
static void check_site(bool passed, int site, const char *what) {
  if (!passed) {
    fprintf(stderr, "smoke: failed: site %d: %s\n", site, what);
    ++failures;
  }
}

/* The hypocenter and sites of the core's model file tests: a site over the
 * Kuril slab, one outside both grids, and one with an invalid ARV. */
static const shindo_est_hypocenter HYPOCENTER = {35.02, 140.02, 80.0, 6.5};
static const shindo_est_site SITES[3] = {
    {35.02, 140.02, 1.0},
    {40.0, 135.0, 1.0},
    {35.02, 140.02, 0.0},
};
#define SITE_COUNT (sizeof SITES / sizeof SITES[0])
/* Index in SITES of the site with an invalid ARV. The sites before it are
 * valid. */
static const int INVALID_SITE = 2;

/* Checks the versions and the error messages, which need no model. */
static void check_versions(void) {
  check(shindo_est_abi_version() == SHINDO_EST_ABI_VERSION,
        "the library's ABI version is the header's");
  check(shindo_est_version() != NULL, "the library has a version");
  const char *message = shindo_est_error_message(SHINDO_EST_ERROR_NULL_POINTER);
  check(message != NULL && message[0] != '\0', "error codes have messages");
}

/* Predicts SITES for `hypocenter` into `out`, and records a failed check named
 * `what` unless the call returns SHINDO_EST_OK. Returns whether it did, as
 * `out` is left unset on an error. */
static bool predict(const shindo_est_model *model,
                    const shindo_est_hypocenter *hypocenter,
                    const shindo_est_options *options,
                    shindo_est_result out[SITE_COUNT], const char *what) {
  int32_t code =
      shindo_est_predict(model, hypocenter, SITES, SITE_COUNT, options, out);
  check(code == SHINDO_EST_OK, what);
  return code == SHINDO_EST_OK;
}

/* Checks the result of the site at index `site`, which gets an intensity. */
static void check_computed(const shindo_est_result *result, int site) {
  check_site(result->status == SHINDO_EST_STATUS_OK, site,
             "the site gets an intensity");
  check_site(result->intensity_class ==
                 shindo_est_intensity_class(result->instrumental),
             site, "the class is that of the instrumental intensity");
  check_site(isfinite(result->instrumental), site, "the intensity is finite");
  check_site(result->probability >= 0.0 && result->probability <= 1.0, site,
             "the felt probability is within 0 to 1");
  check_site(isfinite(result->regression), site,
             "the regressor's intensity is finite");
}

/* Checks the result of the site at index `site`, which gets no intensity for
 * the reason `status`. */
static void check_without_values(const shindo_est_result *result,
                                 int32_t status, int site) {
  check_site(result->status == status, site, "the status says why");
  check_site(result->intensity_class == -1, site, "the site has no class");
  check_site(isnan(result->instrumental) && isnan(result->probability) &&
                 isnan(result->regression),
             site, "the values are NaN");
}

/* Checks the results of each site, and that options of the current and of an
 * older header are read as the header describes. */
static void check_predictions(const shindo_est_model *model) {
  shindo_est_result defaults[SITE_COUNT];
  if (!predict(model, &HYPOCENTER, NULL, defaults,
               "predict with NULL options returns OK")) {
    return;
  }
  for (int site = 0; site < INVALID_SITE; ++site) {
    check_computed(&defaults[site], site);
  }
  check_without_values(&defaults[INVALID_SITE], SHINDO_EST_STATUS_INVALID_INPUT,
                       INVALID_SITE);

  /* The library gives the same bits for the same inputs, NaN included. */
  shindo_est_options zeroed = {.size = sizeof zeroed};
  shindo_est_result explicit_defaults[SITE_COUNT];
  if (predict(model, &HYPOCENTER, &zeroed, explicit_defaults,
              "predict with zeroed options returns OK")) {
    check(memcmp(defaults, explicit_defaults, sizeof defaults) == 0,
          "zeroed options give the results of NULL options");
  }

  shindo_est_options older = {.size = offsetof(shindo_est_options, max_threads),
                              .offset = 0.2};
  shindo_est_result shifted[SITE_COUNT];
  if (predict(model, &HYPOCENTER, &older, shifted,
              "options of an older header are accepted")) {
    check(shifted[0].instrumental > defaults[0].instrumental,
          "the offset of options of an older header is read");
  }
}

/* Checks that an unknown depth is out of scope, not an error, and that an
 * invalid latitude is an error that leaves the results unchanged. */
static void check_errors(const shindo_est_model *model) {
  shindo_est_hypocenter unknown_depth = HYPOCENTER;
  unknown_depth.depth = NAN;
  shindo_est_result results[SITE_COUNT];
  if (predict(model, &unknown_depth, NULL, results,
              "an unknown depth is not an error")) {
    for (int site = 0; site < INVALID_SITE; ++site) {
      check_without_values(&results[site], SHINDO_EST_STATUS_OUT_OF_SCOPE,
                           site);
    }
    /* An invalid site stays invalid, whatever the hypocenter. */
    check_without_values(&results[INVALID_SITE],
                         SHINDO_EST_STATUS_INVALID_INPUT, INVALID_SITE);
  }

  /* A result no prediction writes. */
  const shindo_est_result untouched = {77, 77, 77.0, 77.0, 77.0};
  shindo_est_result kept[SITE_COUNT];
  for (size_t index = 0; index < SITE_COUNT; ++index) {
    kept[index] = untouched;
  }
  shindo_est_hypocenter north_of_the_pole = HYPOCENTER;
  north_of_the_pole.latitude = 91.0;
  int32_t code = shindo_est_predict(model, &north_of_the_pole, SITES,
                                    SITE_COUNT, NULL, kept);
  check(code == SHINDO_EST_ERROR_INVALID_HYPOCENTER_LATITUDE,
        "a latitude of 91 is an error");
  for (size_t index = 0; index < SITE_COUNT; ++index) {
    check(memcmp(&kept[index], &untouched, sizeof untouched) == 0,
          "an error leaves the results unchanged");
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: smoke <model file>\n");
    return 2;
  }

  check_versions();

  shindo_est_model *model = NULL;
  int32_t code = shindo_est_model_open_file(argv[1], &model);
  check(code == SHINDO_EST_OK, "the model file opens");
  if (code != SHINDO_EST_OK) {
    fprintf(stderr, "smoke: %s\n", shindo_est_error_message(code));
    return 1;
  }
  check(strcmp(shindo_est_model_id(model), "synthetic-fixture") == 0,
        "the model ID is the fixture's");

  check_predictions(model);
  check_errors(model);

  shindo_est_model_close(model);
  shindo_est_model_close(NULL);

  if (failures != 0) {
    return 1;
  }
  printf("smoke: ok\n");
  return 0;
}

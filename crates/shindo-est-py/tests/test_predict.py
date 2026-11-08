# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks Model.predict against expected.csv bit for bit.

Also checks the arrays it accepts and rejects, the errors and statuses of
invalid and out-of-scope calls, and threads.
"""

import concurrent.futures
import math
import pickle
import unittest

import numpy as np

import shindo_est
from tests import expected
from tests import gil

_ARRAYS = ("status", "intensity_class") + expected.FLOAT_ARRAYS

# Sites over the fixture grids whose coordinates float32 holds exactly.
_LATITUDES = (35.0, 35.125, 34.875)
_LONGITUDES = (140.0, 140.125, 140.0625)
_ARVS = (1.0, 2.0, 0.5)


def _strided(values):
    """Returns the values as a view whose elements are two apart in memory."""
    doubled = np.repeat(np.array(values, dtype=np.float64), 2)
    return doubled[::2]


def _read_only(values):
    """Returns the values as an array that cannot be written to."""
    array = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array


def _grid():
    """Returns 600 sites on a grid over the fixture's slab grids.

    Returns:
        The latitudes, longitudes, and ARVs of the sites, as arrays.
    """
    latitudes, longitudes = np.meshgrid(
        np.linspace(35.0, 35.2, 20), np.linspace(140.0, 140.2, 30)
    )
    latitudes = latitudes.ravel()
    return latitudes, longitudes.ravel(), np.ones_like(latitudes)


def _many_sites():
    """Returns `_grid()` repeated to 6,000 sites, to keep a call busy."""
    return tuple(np.tile(values, 10) for values in _grid())


class PredictTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = shindo_est.Model.open(expected.FIXTURES / "model.bin")
        cls.calls = expected.calls()
        cls.hypocenter = expected.hypocenter(cls.calls[0])

    def predict_first_call(self, **options):
        call = self.calls[0]
        return self.model.predict(
            self.hypocenter,
            call.site_latitude,
            call.site_longitude,
            call.site_arv,
            **options,
        )

    def assert_same_results(self, actual, wanted, context):
        for name in _ARRAYS:
            expected.assert_same_bits(
                self,
                getattr(actual, name),
                getattr(wanted, name),
                f"{context}, {name}",
            )

    def test_matches_the_core_bit_for_bit(self):
        self.assertEqual(len(self.calls), 5)
        for number, call in enumerate(self.calls):
            with self.subTest(call=number):
                prediction = self.model.predict(
                    expected.hypocenter(call),
                    call.site_latitude,
                    call.site_longitude,
                    call.site_arv,
                    offset=call.offset,
                )
                self.assertEqual(prediction.status.tolist(), call.status)
                self.assertEqual(
                    prediction.intensity_class.tolist(), call.intensity_class
                )
                for name in expected.FLOAT_ARRAYS:
                    expected.assert_same_bits(
                        self,
                        getattr(prediction, name),
                        getattr(call, name),
                        f"call {number}, {name}",
                    )

    def test_carries_the_model_id_and_offset(self):
        prediction = self.predict_first_call(offset=0.2)
        self.assertEqual(prediction.offset, 0.2)
        self.assertEqual(prediction.model_id, "synthetic-fixture")

    def test_repr_counts_the_sites(self):
        self.assertEqual(
            repr(self.predict_first_call()),
            "Prediction(model_id='synthetic-fixture', offset=0.0, sites=6)",
        )

    def test_prediction_is_not_picklable(self):
        prediction = self.predict_first_call()
        with self.assertRaises(TypeError):
            pickle.dumps(prediction)

    def test_attributes_cannot_be_replaced(self):
        prediction = self.predict_first_call()
        with self.assertRaises(AttributeError):
            prediction.instrumental = np.zeros(6)
        with self.assertRaises(AttributeError):
            prediction.offset = 1.0

    def test_results_belong_to_the_caller(self):
        prediction = self.predict_first_call()
        original = prediction.instrumental[0]
        prediction.instrumental[0] = 99.0
        self.assertEqual(prediction.instrumental[0], 99.0)
        again = self.predict_first_call()
        self.assertEqual(again.instrumental[0], original)

    def test_classes_follow_the_instrumental_intensity(self):
        prediction = self.predict_first_call()
        self.assertEqual(
            prediction.intensity_class.tolist(),
            shindo_est.intensity_class(prediction.instrumental).tolist(),
        )

    def test_result_types(self):
        prediction = self.predict_first_call()
        dtypes = {
            "status": np.int32,
            "intensity_class": np.int32,
            "instrumental": np.float64,
            "probability": np.float64,
            "regression": np.float64,
        }
        for name, dtype in dtypes.items():
            with self.subTest(name=name):
                array = getattr(prediction, name)
                self.assertEqual(array.dtype, dtype)
                self.assertEqual(array.shape, (6,))

    def test_accepts_lists_float32_read_only_and_strided_arrays(self):
        forms = {
            "list": list,
            "float32": lambda values: np.array(values, dtype=np.float32),
            "read-only": _read_only,
            "strided": _strided,
        }
        wanted = self.model.predict(
            self.hypocenter,
            np.array(_LATITUDES),
            np.array(_LONGITUDES),
            np.array(_ARVS),
        )
        for name, form in forms.items():
            with self.subTest(form=name):
                prediction = self.model.predict(
                    self.hypocenter,
                    form(_LATITUDES),
                    form(_LONGITUDES),
                    form(_ARVS),
                )
                self.assert_same_results(prediction, wanted, name)

    def test_accepts_integer_arrays(self):
        # Two sites at a whole-degree point over the fixture grids.
        integers = [
            np.array(values, dtype=np.int64)
            for values in ([35, 35], [140, 140], [1, 2])
        ]
        floats = [values.astype(np.float64) for values in integers]
        prediction = self.model.predict(self.hypocenter, *integers)
        self.assertEqual(prediction.status.tolist(), [shindo_est.STATUS_OK] * 2)
        self.assert_same_results(
            prediction, self.model.predict(self.hypocenter, *floats), "int64"
        )

    def test_no_sites_give_empty_results(self):
        prediction = self.model.predict(self.hypocenter, [], [], [])
        for name in _ARRAYS:
            with self.subTest(name=name):
                self.assertEqual(getattr(prediction, name).shape, (0,))
        self.assertEqual(prediction.status.dtype, np.int32)
        self.assertEqual(prediction.instrumental.dtype, np.float64)

    def test_rejects_arrays_of_different_lengths(self):
        with self.assertRaises(ValueError) as context:
            self.model.predict(
                self.hypocenter, [35.0] * 3, [140.0] * 2, [1.0] * 3
            )
        self.assertIn("site_longitude 2", str(context.exception))

    def test_rejects_arrays_that_are_not_one_dimensional(self):
        for latitudes in [np.full((2, 2), 35.0), 35.0]:
            with self.subTest(latitudes=latitudes):
                with self.assertRaises(TypeError):
                    self.model.predict(
                        self.hypocenter, latitudes, [140.0] * 4, [1.0] * 4
                    )

    def test_rejects_elements_that_are_not_numbers(self):
        # NumPy raises ValueError for a string that is not a number, and
        # TypeError for an object of a type it cannot convert.
        for value, error in [("abc", ValueError), (object(), TypeError)]:
            with self.subTest(value=value):
                with self.assertRaises(error):
                    self.model.predict(self.hypocenter, [value], [140.0], [1.0])

    def test_rejects_arguments_of_other_types(self):
        call = self.calls[0]
        with self.subTest(argument="hypocenter"):
            with self.assertRaises(TypeError):
                self.model.predict(
                    (35.02, 140.02, 80.0, 6.5),
                    call.site_latitude,
                    call.site_longitude,
                    call.site_arv,
                )
        for name, value in [("offset", "0.2"), ("max_threads", 1.5)]:
            with self.subTest(argument=name):
                with self.assertRaises(TypeError):
                    self.predict_first_call(**{name: value})

    def test_invalid_hypocenters_raise_input_error(self):
        cases = [
            (
                (91.0, 140.0, 10.0, 6.0),
                0.0,
                "the hypocenter latitude must be within -90..=90",
            ),
            (
                (35.0, -181.0, 10.0, 6.0),
                0.0,
                "the hypocenter longitude must be within -180..=180",
            ),
            (
                (35.0, 140.0, -1.0, 6.0),
                0.0,
                "the hypocenter depth must be finite and at least 0, or NaN"
                " if unknown",
            ),
            (
                (35.0, 140.0, 10.0, math.inf),
                0.0,
                "the hypocenter magnitude must not be positive infinity",
            ),
            ((35.0, 140.0, 10.0, 6.0), math.nan, "the offset must be finite"),
        ]
        for values, offset, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(shindo_est.InputError) as context:
                    self.model.predict(
                        shindo_est.Hypocenter(*values),
                        _LATITUDES,
                        _LONGITUDES,
                        _ARVS,
                        offset=offset,
                    )
                self.assertEqual(str(context.exception), message)

    def test_out_of_scope_calls_return_statuses(self):
        call = self.calls[0]
        invalid_arv = [arv <= 0.0 for arv in call.site_arv]
        self.assertTrue(any(invalid_arv), call.site_arv)
        hypocenters = [
            shindo_est.Hypocenter(35.02, 140.02, math.nan, 6.5),
            shindo_est.Hypocenter(35.02, 140.02, 80.0, 0.0),
        ]
        for hypocenter in hypocenters:
            with self.subTest(hypocenter=hypocenter):
                prediction = self.model.predict(
                    hypocenter,
                    call.site_latitude,
                    call.site_longitude,
                    call.site_arv,
                )
                statuses = [
                    shindo_est.STATUS_INVALID_INPUT
                    if invalid
                    else shindo_est.STATUS_OUT_OF_SCOPE
                    for invalid in invalid_arv
                ]
                self.assertEqual(prediction.status.tolist(), statuses)
                self.assertEqual(
                    prediction.intensity_class.tolist(), [-1] * len(statuses)
                )
                for name in expected.FLOAT_ARRAYS:
                    values = getattr(prediction, name)
                    self.assertTrue(np.isnan(values).all(), f"{name}: {values}")

    def test_max_threads_do_not_change_the_results(self):
        sites = _grid()
        wanted = self.model.predict(self.hypocenter, *sites)
        for threads in [1, 3]:
            with self.subTest(threads=threads):
                prediction = self.model.predict(
                    self.hypocenter, *sites, max_threads=threads
                )
                self.assert_same_results(
                    prediction, wanted, f"max_threads={threads}"
                )

    def test_huge_max_threads_set_no_limit(self):
        wanted = self.predict_first_call()
        for threads in [2**63, 2**64, 2**100]:
            with self.subTest(threads=threads):
                prediction = self.predict_first_call(max_threads=threads)
                self.assert_same_results(
                    prediction, wanted, f"max_threads={threads}"
                )

    def test_rejects_max_threads_below_1(self):
        for threads in [0, -1, -(2**100)]:
            with self.subTest(threads=threads):
                with self.assertRaises(ValueError) as context:
                    self.predict_first_call(max_threads=threads)
                self.assertEqual(
                    str(context.exception), "max_threads must be at least 1"
                )

    def test_offset_and_max_threads_are_keyword_only(self):
        call = self.calls[0]
        with self.assertRaises(TypeError):
            self.model.predict(
                self.hypocenter,
                call.site_latitude,
                call.site_longitude,
                call.site_arv,
                0.2,
            )

    @unittest.skipUnless(gil.is_enabled(), "the GIL is disabled")
    def test_releases_the_gil(self):
        sites = _many_sites()
        # One thread makes the call take about the same time on any machine.
        self.assertTrue(
            gil.is_released_during(
                lambda: self.model.predict(
                    self.hypocenter, *sites, max_threads=1
                )
            )
        )

    def test_threads_can_share_a_model(self):
        sites = _grid()
        wanted = self.model.predict(self.hypocenter, *sites)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.model.predict, self.hypocenter, *sites)
                for _ in range(8)
            ]
            predictions = [future.result() for future in futures]
        for number, prediction in enumerate(predictions):
            with self.subTest(prediction=number):
                self.assert_same_results(prediction, wanted, f"thread {number}")


if __name__ == "__main__":
    unittest.main()

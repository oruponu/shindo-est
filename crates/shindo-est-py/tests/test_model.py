# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks opening model files by path and bytes, their errors, and Hypocenter.

Uses the fixture model in tests/fixtures/bundle and temporary copies of it.
"""

import hashlib
import math
import pathlib
import pickle
import shutil
import tempfile
import unittest

import shindo_est
from tests import expected

_MODEL = expected.FIXTURES / "model.bin"


class ModelTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.dir = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_opens_a_path_given_as_str_or_pathlib(self):
        for path in [str(_MODEL), _MODEL]:
            with self.subTest(path=type(path).__name__):
                model = shindo_est.Model.open(path)
                self.assertEqual(model.model_id, "synthetic-fixture")

    def test_reads_bytes_and_bytearray(self):
        data = _MODEL.read_bytes()
        call = expected.calls()[0]
        for form in [bytes, bytearray]:
            with self.subTest(form=form.__name__):
                model = shindo_est.Model.from_bytes(form(data))
                self.assertEqual(model.model_id, "synthetic-fixture")
                prediction = model.predict(
                    expected.hypocenter(call),
                    call.site_latitude,
                    call.site_longitude,
                    call.site_arv,
                    offset=call.offset,
                )
                for name in expected.FLOAT_ARRAYS:
                    expected.assert_same_bits(
                        self,
                        getattr(prediction, name),
                        getattr(call, name),
                        f"{form.__name__}, {name}",
                    )

    def test_from_bytes_rejects_other_types(self):
        with self.assertRaises(TypeError):
            shindo_est.Model.from_bytes("text")

    def test_carries_the_manifest(self):
        model = shindo_est.Model.open(_MODEL)
        manifest = (expected.FIXTURES / "manifest.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(model.manifest, manifest)

    def test_repr_names_the_model(self):
        model = shindo_est.Model.open(_MODEL)
        self.assertEqual(repr(model), "Model(model_id='synthetic-fixture')")

    def test_opens_a_path_with_japanese_characters(self):
        path = self.dir / "\u30e2\u30c7\u30eb.bin"
        shutil.copyfile(_MODEL, path)
        model = shindo_est.Model.open(path)
        self.assertEqual(model.model_id, "synthetic-fixture")

    def test_missing_file_raises_file_not_found_error(self):
        path = self.dir / "missing.bin"
        with self.assertRaises(FileNotFoundError) as context:
            shindo_est.Model.open(path)
        self.assertIn(str(path), str(context.exception))

    def test_directory_raises_os_error(self):
        with self.assertRaises(OSError) as context:
            shindo_est.Model.open(self.dir)
        self.assertNotIsInstance(context.exception, shindo_est.ModelFileError)
        self.assertIn(str(self.dir), str(context.exception))

    def test_other_files_raise_model_file_error(self):
        paths = [
            self.write("empty.bin", b""),
            expected.FIXTURES / "manifest.txt",
            self.write("zeros.bin", bytes(1 << 20)),
        ]
        for path in paths:
            with self.subTest(path=path.name):
                with self.assertRaises(shindo_est.ModelFileError) as context:
                    shindo_est.Model.open(path)
                message = str(context.exception)
                self.assertIn(str(path), message)
                self.assertIn("not a model file", message)

    def test_damaged_bytes_raise_model_file_error(self):
        data = bytearray(_MODEL.read_bytes())
        data[-1] ^= 0x01
        with self.assertRaises(shindo_est.ModelFileError) as context:
            shindo_est.Model.from_bytes(bytes(data))
        self.assertIn("checksum", str(context.exception))
        with self.assertRaises(shindo_est.ModelFileError):
            shindo_est.Model.from_bytes(b"hello")

    def test_errors_name_their_causes(self):
        # A felt probability above 1 in a file whose checksum still matches,
        # so that the error comes from the model, with its cause.
        data = bytearray(_MODEL.read_bytes())
        old = b"hurdle_felt_probability=0.35"
        start = data.index(old)
        data[start : start + len(old)] = b"hurdle_felt_probability=1.50"
        data[-32:] = hashlib.sha256(bytes(data[:-32])).digest()
        path = self.write("bad-hurdle.bin", bytes(data))
        message = (
            "the model file has an invalid model: the hurdle parameters are"
            " invalid"
        )
        with self.assertRaises(shindo_est.ModelFileError) as context:
            shindo_est.Model.from_bytes(bytes(data))
        self.assertEqual(str(context.exception), message)
        with self.assertRaises(shindo_est.ModelFileError) as context:
            shindo_est.Model.open(path)
        self.assertEqual(str(context.exception), f"{path}: {message}")

    def test_cannot_be_constructed_directly(self):
        with self.assertRaises(TypeError):
            shindo_est.Model()

    def test_is_not_picklable(self):
        model = shindo_est.Model.open(_MODEL)
        with self.assertRaises(TypeError):
            pickle.dumps(model)


class HypocenterTest(unittest.TestCase):

    def test_keeps_its_values(self):
        by_position = shindo_est.Hypocenter(35.02, 140.02, 80.0, 6.5)
        by_keyword = shindo_est.Hypocenter(
            latitude=35.02, longitude=140.02, depth=80.0, magnitude=6.5
        )
        for hypocenter in [by_position, by_keyword]:
            with self.subTest(hypocenter=hypocenter):
                self.assertEqual(hypocenter.latitude, 35.02)
                self.assertEqual(hypocenter.longitude, 140.02)
                self.assertEqual(hypocenter.depth, 80.0)
                self.assertEqual(hypocenter.magnitude, 6.5)

    def test_repr(self):
        hypocenter = shindo_est.Hypocenter(35.02, 140.02, 80.0, 6.5)
        self.assertEqual(
            repr(hypocenter),
            "Hypocenter(latitude=35.02, longitude=140.02, depth=80.0,"
            " magnitude=6.5)",
        )
        unknown_depth = shindo_est.Hypocenter(35.02, 140.02, math.nan, 6.5)
        self.assertIn("depth=nan", repr(unknown_depth))

    def test_is_immutable(self):
        hypocenter = shindo_est.Hypocenter(35.02, 140.02, 80.0, 6.5)
        with self.assertRaises(AttributeError):
            hypocenter.depth = 1.0


if __name__ == "__main__":
    unittest.main()

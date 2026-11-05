# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks the version, constants, errors, docstrings, and stub of the module.

The stub tests read `shindo_est.pyi` in the crate folder, so they also catch a
name added to the module without its stub.
"""

import ast
import importlib.metadata
import inspect
import pathlib
import unittest

import shindo_est

_STUB = pathlib.Path(__file__).resolve().parents[1] / "shindo_est.pyi"


def _is_public(name):
    return not name.startswith("_")


def _stub_tree():
    return ast.parse(_STUB.read_text(encoding="utf-8"))


def _defined_names(body):
    """Returns the names that a module or class body of the stub defines.

    Functions, classes, and annotated names count, but names that imports
    bring in do not.
    """
    names = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign):
            names.add(node.target.id)
    return names


def _module_names():
    """Returns the public names of the module and `__version__`.

    The extension itself, a submodule in maturin's package, is left out.
    """
    names = set()
    for name, value in vars(shindo_est).items():
        if inspect.ismodule(value):
            continue
        if _is_public(name) or name == "__version__":
            names.add(name)
    return names


class ModuleTest(unittest.TestCase):

    def test_version_matches_the_package_metadata(self):
        self.assertEqual(
            shindo_est.__version__, importlib.metadata.version("shindo-est")
        )

    def test_feature_names_follow_the_trained_order(self):
        self.assertEqual(
            shindo_est.FEATURE_NAMES,
            (
                "mj",
                "depth",
                "qlat",
                "qlon",
                "slat",
                "slon",
                "log_hypo",
                "epi",
                "az_sin",
                "az_cos",
                "log_arv",
                "slab_top",
                "updip",
                "path_frac",
            ),
        )

    def test_feature_version_is_1(self):
        self.assertEqual(shindo_est.FEATURE_VERSION, 1)

    def test_statuses_match_the_c_abi(self):
        self.assertEqual(shindo_est.STATUS_OK, 0)
        self.assertEqual(shindo_est.STATUS_OUT_OF_SCOPE, 1)
        self.assertEqual(shindo_est.STATUS_INVALID_INPUT, 2)

    def test_errors_are_value_errors(self):
        errors = [
            shindo_est.InputError,
            shindo_est.ModelFileError,
            shindo_est.SlabGridError,
        ]
        for error in errors:
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, ValueError))
                self.assertEqual(error.__module__, "shindo_est")


class StubTest(unittest.TestCase):

    def test_stub_names_match_the_module(self):
        stub_names = {
            name
            for name in _defined_names(_stub_tree().body)
            if _is_public(name) or name == "__version__"
        }
        self.assertEqual(stub_names, _module_names())

    def test_stub_members_match_the_classes(self):
        for node in _stub_tree().body:
            if not isinstance(node, ast.ClassDef):
                continue
            # A class whose body is `...` defines no members, so the class
            # must have no public members either.
            with self.subTest(cls=node.name):
                cls = getattr(shindo_est, node.name)
                stub_members = {
                    name
                    for name in _defined_names(node.body)
                    if _is_public(name)
                }
                members = {name for name in vars(cls) if _is_public(name)}
                self.assertEqual(stub_members, members)

    def test_public_objects_have_docstrings(self):
        for name in _module_names():
            value = getattr(shindo_est, name)
            if not callable(value):
                continue
            with self.subTest(name=name):
                self.assertTrue(value.__doc__, f"{name} has no docstring")
            if not inspect.isclass(value) or issubclass(value, Exception):
                continue
            for member in filter(_is_public, vars(value)):
                with self.subTest(name=f"{name}.{member}"):
                    doc = getattr(value, member).__doc__
                    self.assertTrue(doc, f"{name}.{member} has no docstring")


if __name__ == "__main__":
    unittest.main()

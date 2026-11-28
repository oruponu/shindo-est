# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests training a case on a split and measuring its predictions."""

import dataclasses
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from training import evaluate
from training import features
from training import hurdle
from training import runs
from training import selection
from training.tests import synthetic

_ROUNDS = 20
_INPUTS = {"case": "jshis", "split": "year"}


class _Split(unittest.TestCase):
    """Builds the synthetic dataset, its year split, and a temporary place."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = synthetic.grid_dataset()
        cls.table = synthetic.arv_table()
        cls.split = selection.splits(
            cls.data, selection.rows(cls.data, cls.table)
        )[0]
        cls.eligible = selection.stations_with_arv(
            cls.data, cls.table, selection.COMPARED_SOURCES
        )

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = pathlib.Path(temporary.name) / "out"


class TrainerTest(_Split):

    def setUp(self):
        super().setUp()
        self.trainer = runs.Trainer(
            self.data,
            self.table,
            features.read_slabs(),
            threads=2,
            eligible=self.eligible,
        )
        self.settings = hurdle.Settings(max_rounds=_ROUNDS)

    def _train(self, trainer):
        trainer.train("jshis", self.split, self.settings, self.out, _INPUTS)

    def test_trainer_writes_the_files_and_the_record_last(self):
        self._train(self.trainer)

        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            sorted(runs.FILES),
        )
        record = json.loads((self.out / runs.RECORD).read_text())
        self.assertEqual(record["settings"], self.settings.parameters())
        self.assertEqual(record["inputs"], _INPUTS)

        failing = mock.patch.object(
            runs.Predictions, "save", side_effect=OSError("full")
        )
        with failing, self.assertRaises(OSError):
            self._train(self.trainer)
        self.assertFalse((self.out / runs.RECORD).exists())

    def test_trainer_trains_with_the_given_settings(self):
        settings = hurdle.Settings(learning_rate=0.1, max_rounds=7)
        stopping = mock.patch.object(hurdle, "fit", side_effect=RuntimeError)

        with stopping as fit, self.assertRaises(RuntimeError):
            self.trainer.train("jshis", self.split, settings, self.out, {})

        self.assertIs(fit.call_args.kwargs["settings"], settings)

    def test_trainer_takes_candidates_from_the_eligible_stations(self):
        _, stations = selection.candidates(
            self.data, self.split.test, self.eligible
        )
        left_out = stations[0]
        eligible = self.eligible.copy()
        eligible[left_out] = False

        self._train(dataclasses.replace(self.trainer, eligible=eligible))

        predicted = runs.Predictions.load(self.out / runs.PREDICTIONS)
        self.assertNotIn(left_out, predicted.candidate_station)
        np.testing.assert_array_equal(
            predicted.candidate_station,
            selection.candidates(self.data, self.split.test, eligible)[1],
        )


class CompleteTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)
        for name in runs.FILES:
            (self.directory / name).write_text("", encoding="ascii")
        self._write_record(json.dumps({"inputs": _INPUTS}))

    def _write_record(self, text):
        (self.directory / runs.RECORD).write_text(text, encoding="ascii")

    def test_complete_needs_every_file_and_the_same_inputs(self):
        self.assertTrue(runs.complete(self.directory, _INPUTS))
        self.assertFalse(
            runs.complete(self.directory, {**_INPUTS, "case": "none"})
        )
        self.assertFalse(runs.complete(self.directory / "nowhere", _INPUTS))
        for name in runs.FILES:
            if name == runs.RECORD:
                continue
            with self.subTest(missing=name):
                path = self.directory / name
                path.unlink()
                self.assertFalse(runs.complete(self.directory, _INPUTS))
                path.write_text("", encoding="ascii")
        # A record cut off while writing counts as not trained.
        self._write_record("{")
        self.assertFalse(runs.complete(self.directory, _INPUTS))


class MeasureTest(_Split):

    def test_measure_uses_the_given_hurdle_values(self):
        # Every probability lies between the previous felt probability and
        # 0.9, so that 0.9 shows every station as unfelt.
        rows = np.flatnonzero(self.split.test)
        quakes, stations = selection.candidates(
            self.data, self.split.test, self.eligible
        )
        predicted = runs.Predictions(
            test_rows=rows,
            test_probability=np.full(len(rows), 0.5),
            test_regression=np.full(len(rows), 2.0),
            candidate_quake=quakes,
            candidate_station=stations,
            candidate_probability=np.full(len(quakes), 0.5),
            candidate_regression=np.full(len(quakes), 2.0),
        )
        chosen = np.ones(len(self.data.station_number), dtype=np.bool_)
        unfelt = evaluate.HurdleValues(felt_probability=0.9)

        previous, previous_errors = runs.measure(self.data, predicted, chosen)
        measured, errors = runs.measure(self.data, predicted, chosen, unfelt)

        row_quake = self.data.row_quake[rows]
        felt = self.data.felt[rows]
        observed = self.data.observed[rows]
        known = felt & ~np.isnan(observed)
        np.testing.assert_allclose(
            previous_errors[1] + observed[known], 2.0, rtol=1e-6
        )
        np.testing.assert_allclose(
            errors[1] + observed[known], 0.49, rtol=1e-6
        )
        self.assertEqual(
            measured["summary"],
            evaluate.summarize(
                row_quake,
                felt,
                observed,
                self.data.weight[rows],
                predicted.test_probability,
                predicted.test_regression,
                unfelt,
            ),
        )
        candidate_display = evaluate.display(
            predicted.candidate_probability,
            predicted.candidate_regression,
            unfelt,
        )
        self.assertEqual(
            measured["max"],
            evaluate.max_per_quake(
                quakes, candidate_display, row_quake, observed
            ),
        )
        self.assertNotEqual(measured["summary"], previous["summary"])
        self.assertNotEqual(measured["max"], previous["max"])


if __name__ == "__main__":
    unittest.main()

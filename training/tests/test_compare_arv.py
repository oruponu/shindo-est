# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests training and comparing the ARV cases."""

import contextlib
import dataclasses
import datetime
import io
import json
import math
import pathlib
import tempfile
import unittest
from unittest import mock

import lightgbm
import numpy as np

from training import arv
from training import bulletin
from training import compare_arv
from training import dataset
from training import evaluate
from training import features
from training import hurdle
from training import selection
from training import sources

_ROUNDS = 20
_NUMBERS = tuple(str(number) for number in range(7654321, 7654340))
_ONE_ARV = "7654323"
_TWO_CASES = ("cabinet_office", "jshis")


def _dataset():
    """Returns quakes from 2004 to 2020 recorded on a grid of stations.

    The stations stand 20 minutes of arc apart, five to a row, so that the
    small quakes leave far stations unfelt and some of those are sampled
    out.
    """
    period = bulletin.Period(datetime.datetime(1990, 1, 1), None)
    stations = [
        bulletin.Station(
            number=number,
            name="Test",
            latitude_minutes=35 * 60 + 20 * (at // 5),
            longitude_minutes=139 * 60 + 20 * (at % 5),
            period=period,
        )
        for at, number in enumerate(_NUMBERS)
    ]
    rng = np.random.default_rng(3)
    quakes = []
    for year in range(2004, 2021):
        for month in range(1, 13):
            latitude = 35 + rng.uniform(0, 1.2)
            longitude = 139 + rng.uniform(0, 1.4)
            magnitude = round(rng.uniform(3.5, 6.5), 1)
            records = []
            for station in stations:
                north = station.latitude_minutes / 60 - latitude
                east = (station.longitude_minutes / 60 - longitude) * 0.82
                distance = 111 * math.hypot(north, east)
                intensity = round(
                    1.3 * magnitude
                    - 2.2 * math.log10(distance + 10)
                    - 0.5
                    + rng.normal(scale=0.2),
                    1,
                )
                if intensity >= 0.5:
                    symbol = str(min(4, max(1, round(intensity))))
                    records.append(
                        bulletin.Record(station.number, symbol, intensity)
                    )
            time = datetime.datetime(year, month, 1, 12, 0)
            quakes.append(
                bulletin.Quake(
                    id=f"{time:%Y%m%d%H%M}0000",
                    origin_time=time,
                    latitude=latitude,
                    longitude=longitude,
                    depth=10.0,
                    magnitude=magnitude,
                    records=tuple(records),
                )
            )
    data, _ = dataset.build(stations, quakes)
    return data


def _table():
    table = {
        number: (1.0 + 0.05 * at, 1.2 + 0.03 * at)
        for at, number in enumerate(_NUMBERS)
    }
    table[_ONE_ARV] = (1.1, None)
    return table


class _Directory(unittest.TestCase):
    """Writes a dataset and an ARV table to a temporary directory.

    git takes seconds on a Windows drive under WSL, so the repository counts
    as changed unless a test asks git or says otherwise.
    """

    def setUp(self):
        super().setUp()
        self.revision = mock.patch.object(
            compare_arv, "_revision", return_value=("a" * 40, True)
        )
        self.revision.start()
        self.addCleanup(self.revision.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)
        self.data = _dataset()
        self.dataset_path = self.directory / "dataset.npz"
        self.data.save(self.dataset_path)
        self.arv_path = self.directory / "arv.csv"
        arv.write_table(self.arv_path, _table())
        self.out = self.directory / "out"

    def _run(self, cases=_TWO_CASES, max_rounds=_ROUNDS):
        compare_arv.run(
            self.dataset_path,
            self.arv_path,
            self.out,
            cases=cases,
            split_names=["year"],
            threads=2,
            max_rounds=max_rounds,
        )

    def _report(self):
        compare_arv.report(self.out, self.dataset_path, self.arv_path)
        return json.loads((self.out / "results.json").read_text())

    def _predictions(self, case):
        return self.out / case / "year" / "predictions.npz"

    def _saved(self, case):
        with np.load(self._predictions(case)) as arrays:
            return {name: arrays[name] for name in arrays.files}

    def _rewrite_predictions(self, case, **changed):
        np.savez(self._predictions(case), **{**self._saved(case), **changed})


class RunTest(_Directory):

    def test_run_writes_models_predictions_and_records(self):
        self.revision.stop()

        self._run()

        for case in _TWO_CASES:
            with self.subTest(case=case):
                directory = self.out / case / "year"
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()),
                    sorted(compare_arv.FILES),
                )
                record = json.loads((directory / "training.json").read_text())
                inputs = record["inputs"]
                self.assertEqual(
                    inputs["dataset"], sources.file_sha256(self.dataset_path)
                )
                self.assertEqual(
                    inputs["arv"], sources.file_sha256(self.arv_path)
                )
                self.assertEqual(
                    inputs["slabs"],
                    [sources.file_sha256(path) for path in features.SLAB_GRIDS],
                )
                self.assertEqual(
                    (inputs["case"], inputs["split"]), (case, "year")
                )
                self.assertEqual(
                    inputs["parameters"],
                    hurdle.Settings(max_rounds=_ROUNDS).parameters(),
                )
                self.assertRegex(inputs["head"], "^[0-9a-f]{40}$")
                self.assertIsInstance(inputs["dirty"], bool)
                split = selection.splits(
                    self.data, selection.rows(self.data, _table())
                )[0]
                self.assertEqual(record["train_rows"], split.train.sum())
                self.assertEqual(record["test_rows"], split.test.sum())
                for model in ("classifier", "regressor"):
                    shares = record["gain"][model]
                    self.assertEqual(list(shares), list(features.FEATURE_NAMES))
                    self.assertAlmostEqual(sum(shares.values()), 1.0)
                    self.assertGreater(record["trees"][model], 0)
                    self.assertLessEqual(record["trees"][model], _ROUNDS)

    def test_run_saves_what_it_predicts(self):
        batch = 97
        with mock.patch.object(compare_arv, "_PREDICTION_BATCH", batch):
            self._run(cases=("none",))

        directory = self.out / "none" / "year"
        table = _table()
        split = selection.splits(self.data, selection.rows(self.data, table))[0]
        rows = np.flatnonzero(split.test)
        quakes, stations = selection.candidates(
            self.data,
            split.test,
            selection.stations_with_arv(
                self.data, table, selection.COMPARED_SOURCES
            ),
        )
        self.assertGreater(min(len(rows), len(quakes)), 2 * batch)
        model = hurdle.Hurdle(
            *(
                lightgbm.Booster(model_file=directory / name)
                for name in ("classifier.txt", "regressor.txt")
            )
        )

        saved = self._saved("none")

        np.testing.assert_array_equal(saved["test_rows"], rows)
        np.testing.assert_array_equal(saved["candidate_quake"], quakes)
        np.testing.assert_array_equal(saved["candidate_station"], stations)
        for part, pair_quakes, pair_stations in (
            ("test", self.data.row_quake[rows], self.data.row_station[rows]),
            ("candidate", quakes, stations),
        ):
            with self.subTest(part=part):
                values, names = selection.feature_matrix(
                    self.data,
                    pair_quakes,
                    pair_stations,
                    table,
                    "none",
                    features.read_slabs(),
                    threads=2,
                )
                probability, regression = model.predict(values)
                self.assertNotIn("log_arv", names)
                np.testing.assert_array_equal(
                    saved[f"{part}_probability"], probability
                )
                np.testing.assert_array_equal(
                    saved[f"{part}_regression"], regression
                )

    def test_run_resumes(self):
        clean = mock.patch.object(
            compare_arv, "_revision", return_value=("a" * 40, False)
        )
        with clean, mock.patch.object(hurdle, "fit", wraps=hurdle.fit) as fit:
            self._run(cases=("cabinet_office",))
            self._run(cases=("cabinet_office",))
            self.assertEqual(fit.call_count, 1)

            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 2)

            self._predictions("cabinet_office").unlink()
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 3)

            table = _table()
            table[_NUMBERS[0]] = (2.0, 2.0)
            arv.write_table(self.arv_path, table)
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 4)

            observed = self.data.observed.copy()
            observed[np.flatnonzero(~np.isnan(observed))[0]] += 0.1
            dataclasses.replace(self.data, observed=observed).save(
                self.dataset_path
            )
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 5)

            record = self.out / "cabinet_office" / "year" / "training.json"
            record.write_text("{", encoding="ascii")
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 6)

        with mock.patch.object(hurdle, "fit", wraps=hurdle.fit) as fit:
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)
            self.assertEqual(fit.call_count, 2)

    def test_run_removes_the_record_before_training(self):
        self._run(cases=("cabinet_office",))
        record = self.out / "cabinet_office" / "year" / "training.json"

        with mock.patch.object(hurdle, "fit", side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                self._run(cases=("cabinet_office",), max_rounds=_ROUNDS + 1)

        self.assertFalse(record.exists())

    def test_run_checks_the_extension(self):
        failing = mock.patch.object(
            features,
            "check_extension",
            side_effect=features.StaleExtensionError("stale"),
        )
        with failing, self.assertRaises(features.StaleExtensionError):
            self._run()

        self.assertFalse(self.out.exists())

    def test_run_refuses_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "'other'"):
            self._run(cases=("other",))
        with self.assertRaisesRegex(ValueError, "'nowhere'"):
            compare_arv.run(
                self.dataset_path,
                self.arv_path,
                self.out,
                split_names=["nowhere"],
            )
        self.assertFalse(self.out.exists())


class ReportTest(_Directory):

    def setUp(self):
        super().setUp()
        self._run(cases=selection.CASES)

    def test_report_compares_the_cases(self):
        # Every other test row falls below the felt probability, so that its
        # displayed intensity is not its regression value.
        for case in selection.CASES:
            probability = self._saved(case)["test_probability"].copy()
            probability[::2] = 0.1
            self._rewrite_predictions(case, test_probability=probability)

        results = self._report()["year"]

        table = _table()
        split = selection.splits(self.data, selection.rows(self.data, table))[0]
        held_out = self.data.station_held_out
        groups = {
            "trained": ~held_out,
            "held_out": held_out,
            "all": np.ones_like(held_out),
        }
        self.assertEqual(list(results["cases"]), list(selection.CASES))
        for case in selection.CASES:
            saved = self._saved(case)
            rows = saved["test_rows"]
            probability = saved["test_probability"]
            regression = saved["test_regression"]
            for group, stations in groups.items():
                with self.subTest(case=case, group=group):
                    inside = stations[self.data.row_station[rows]]
                    chosen = rows[inside]
                    expected = evaluate.summarize(
                        self.data.row_quake[chosen],
                        self.data.felt[chosen],
                        self.data.observed[chosen],
                        self.data.weight[chosen],
                        probability[inside],
                        regression[inside],
                    )
                    measured = results["cases"][case][group]
                    self.assertEqual(
                        measured["summary"], json.loads(json.dumps(expected))
                    )
                    self.assertEqual(len(measured["calibration"]), 10)
                    self.assertIn("max_bias", measured["max"])
        self.assertTrue(np.any(held_out[self.data.row_station[split.test]]))
        for group in groups:
            for case in ("jshis", "none"):
                with self.subTest(group=group, case=case):
                    difference = results["differences"][group][
                        f"{case} - cabinet_office"
                    ]
                    rmse = {
                        name: results["cases"][name][group]["summary"][
                            "felt_rmse"
                        ]
                        for name in (case, "cabinet_office")
                    }
                    self.assertAlmostEqual(
                        difference["difference"],
                        rmse[case] - rmse["cabinet_office"],
                    )
                    self.assertLessEqual(difference["low"], difference["high"])
        markdown = (self.out / "results.md").read_text()
        self.assertIn("jshis - cabinet_office", markdown)
        self.assertIn("held_out", markdown)

    def test_report_takes_maxima_from_all_candidates(self):
        saved = self._saved("cabinet_office")
        quakes = saved["candidate_quake"]
        stations = saved["candidate_station"]
        test_rows = saved["test_rows"]
        row_quake = self.data.row_quake[test_rows]
        row_station = self.data.row_station[test_rows]
        rows = set(zip(row_quake.tolist(), row_station.tolist()))
        trained = ~self.data.station_held_out[row_station]
        observed = row_quake[trained & ~np.isnan(self.data.observed[test_rows])]
        sampled_out = [
            at
            for at, pair in enumerate(zip(quakes.tolist(), stations.tolist()))
            if pair not in rows
            and not self.data.station_held_out[pair[1]]
            and pair[0] in observed
        ]
        self.assertTrue(sampled_out)
        flat = {
            "test_probability": np.full(len(test_rows), 0.9),
            "test_regression": np.full(len(test_rows), 1.0),
            "candidate_probability": np.full(len(quakes), 0.9),
            "candidate_regression": np.full(len(quakes), 1.0),
        }
        self._rewrite_predictions("cabinet_office", **flat)
        before = self._report()["year"]["cases"]["cabinet_office"]
        raised = flat["candidate_regression"].copy()
        raised[sampled_out[0]] = 6.0
        self._rewrite_predictions("cabinet_office", candidate_regression=raised)

        after = self._report()["year"]["cases"]["cabinet_office"]

        for group in ("trained", "all"):
            with self.subTest(group=group):
                count = after[group]["max"]["max_quakes"]
                self.assertAlmostEqual(
                    after[group]["max"]["max_bias"]
                    - before[group]["max"]["max_bias"],
                    5.0 / count,
                )
        self.assertEqual(after["held_out"]["max"], before["held_out"]["max"])

    def test_report_refuses_stale_predictions(self):
        table = _table()
        table[_NUMBERS[0]] = (2.0, 2.0)
        arv.write_table(self.arv_path, table)
        with self.assertRaisesRegex(ValueError, "arv"):
            self._report()

        arv.write_table(self.arv_path, _table())
        self._report()
        observed = self.data.observed.copy()
        observed[np.flatnonzero(~np.isnan(observed))[0]] += 0.1
        dataclasses.replace(self.data, observed=observed).save(
            self.dataset_path
        )
        with self.assertRaisesRegex(ValueError, "dataset"):
            self._report()

    def test_report_refuses_mismatched_candidates(self):
        original = self._saved("jshis")
        for name in ("test_rows", "candidate_quake", "candidate_station"):
            with self.subTest(name=name):
                changed = original[name].copy()
                changed[0] += 1
                self._rewrite_predictions("jshis", **{name: changed})
                with self.assertRaisesRegex(ValueError, "jshis"):
                    self._report()
                self._rewrite_predictions("jshis", **original)
        self._report()


class PartialReportTest(_Directory):

    def test_report_without_the_first_case(self):
        self._run(cases=("jshis", "none"))

        results = self._report()["year"]

        self.assertEqual(list(results["cases"]), ["jshis", "none"])
        self.assertEqual(
            results["differences"], {group: {} for group in compare_arv.GROUPS}
        )

    def test_report_of_a_group_without_rows(self):
        # Without an ARV, no held-out station is a row or a candidate.
        held_out = {
            str(number)
            for number in self.data.station_number[self.data.station_held_out]
        }
        self.assertTrue(held_out)
        table = {
            number: value
            for number, value in _table().items()
            if number not in held_out
        }
        arv.write_table(self.arv_path, table)
        self._run()

        results = self._report()["year"]

        for case in _TWO_CASES:
            with self.subTest(case=case):
                measured = results["cases"][case]["held_out"]
                self.assertEqual(measured["summary"]["felt_rows"], 0)
                self.assertIsNone(measured["summary"]["felt_rmse"])
                self.assertEqual(measured["max"]["max_quakes"], 0)
        differences = results["differences"]
        self.assertIsNone(differences["held_out"]["jshis - cabinet_office"])
        self.assertIsNotNone(differences["trained"]["jshis - cabinet_office"])
        self.assertIn(
            "| jshis - cabinet_office | held_out | - | - | - |",
            (self.out / "results.md").read_text(),
        )


class MainTest(_Directory):

    def _arguments(self):
        return [
            f"--dataset={self.dataset_path}",
            f"--arv={self.arv_path}",
            f"--out={self.out}",
        ]

    def test_main_runs_and_reports(self):
        arguments = self._arguments()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            ran = compare_arv.main(
                ["run", *arguments, "--case", "jshis", "--split", "year"]
            )
            reported = compare_arv.main(["report", *arguments])

        self.assertEqual((ran, reported), (0, 0))
        self.assertEqual(
            sorted(path.name for path in self.out.iterdir()),
            ["jshis", "results.json", "results.md"],
        )
        self.assertEqual(
            output.getvalue(), (self.out / "results.md").read_text()
        )

    def test_main_reports_errors(self):
        # Nothing has been run, so there is nothing to report.
        with self.assertLogs(compare_arv.__name__, "ERROR"):
            status = compare_arv.main(["report", *self._arguments()])

        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()

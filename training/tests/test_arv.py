# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests that ARVs are read from the ground data at the station meshes."""

import contextlib
import fractions
import hashlib
import io
import pathlib
import tempfile
import unittest
import zipfile

from training import arv
from training import bulletin
from training import mesh

_MINUS_SIGMA = chr(0x2212) + chr(0x03C3)
_MEAN = chr(0x5E73) + chr(0x5747)
_HEADER = f'"MeshCode","{_MINUS_SIGMA}","{_MEAN}","Longuitude","Laitude"'
_JSHIS_COMMENTS = [
    "#",
    "# VER. = 1.0",
    "# DATE = 2022-05-30",
    "#",
    "# UPDATED",
    "#",
    "# CODE, JCODE, AVS, ARV, AVS_EB, AVS_REF",
]
_STATIONS = (
    "7654321\tTest\t3520\t13900\t199604011200\t\n"
    "7654322\tOther\t3540\t13945\t199604011200\t\n"
)
_FIRST = "5339000011"
_SECOND = "5339460011"


def _station(number, latitude_minutes, longitude_minutes):
    return bulletin.Station(
        number=number,
        name="Test",
        latitude_minutes=latitude_minutes,
        longitude_minutes=longitude_minutes,
        period=None,
    )


class ArvTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.root = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        )

    def _zip(self, name, entries):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for entry, text in entries.items():
                archive.writestr(entry, text.encode("utf-8"))
        return path

    def _cabinet_office(self, lines):
        text = "\r\n".join([_HEADER, *lines]) + "\r\n"
        return self._zip(arv.CABINET_OFFICE, {"to250mr3japan.csv": text})

    def _jshis(self, lines, comments=None):
        comments = _JSHIS_COMMENTS if comments is None else comments
        text = "\r\n".join([*comments, *lines]) + "\r\n"
        return self._zip(
            arv.JSHIS, {"Z-V4/Z-V4-JAPAN-AMP-VS400_M250.csv": text}
        )

    def test_cabinet_office_uses_the_minus_sigma_column(self):
        path = self._cabinet_office([
            f'"{_FIRST}",0.310,0.229,139.0,35.3',
            f'"{_SECOND}",-0.150,0.229,139.7,35.6',
        ])

        table = arv.read_cabinet_office(path, {_FIRST, _SECOND})

        self.assertEqual(
            table,
            {
                _FIRST: round(10 ** (0.310 / 1.72), 3),
                _SECOND: round(10 ** (-0.150 / 1.72), 3),
            },
        )
        self.assertEqual(table, {_FIRST: 1.514, _SECOND: 0.818})

    def test_cabinet_office_skips_missing_values(self):
        path = self._cabinet_office([
            f'"{_FIRST}",,0.229,139.0,35.3',
            f'"{_SECOND}",nan,0.229,139.7,35.6',
            '"5339000012",inf,0.229,139.0,35.3',
            '"5339000021",0.1,0.229,139.0,35.3',
            '"5339000022"',
            f'"{_FIRST}9",0.1,0.229,139.0,35.3',
        ])

        table = arv.read_cabinet_office(
            path, {_FIRST, _SECOND, "5339000012", "5339000022"}
        )

        self.assertEqual(table, {})

    def test_cabinet_office_rejects_repeated_meshes(self):
        path = self._cabinet_office([
            f'"{_FIRST}",0.310,0.229,139.0,35.3',
            f'"{_FIRST}",0.320,0.229,139.0,35.3',
        ])

        with self.assertRaisesRegex(ValueError, _FIRST):
            arv.read_cabinet_office(path, {_FIRST})

    def test_jshis_reads_the_arv_column(self):
        path = self._jshis([
            f"{_FIRST}, 4,510.4,   0.8125,-,0",
            f"{_SECOND}, 4,250.0,   1.2500,-,0",
        ])

        self.assertEqual(arv.read_jshis(path, {_FIRST}), {_FIRST: 0.8125})

    def test_jshis_skips_values_that_are_not_positive(self):
        path = self._jshis([
            f"{_FIRST}, 4,510.4,-,-,0",
            f"{_SECOND}, 0,  0.0,   0.0000,-,0",
            "5339000012, 4,510.4,  -1.2000,-,0",
            "5339000021, 4,510.4,      nan,-,0",
            f"5339000022, 4,510.4,   0.8{chr(0xB2)}25,-,0",
        ])

        table = arv.read_jshis(
            path, {_FIRST, _SECOND, "5339000012", "5339000021", "5339000022"}
        )

        self.assertEqual(table, {})

    def test_jshis_checks_the_columns(self):
        for comments in (
            [*_JSHIS_COMMENTS[:-1], "# CODE, JCODE, ARV, AVS"],
            [],
        ):
            with self.subTest(comments=comments):
                path = self._jshis([f"{_FIRST}, 4,510.4,0.8,-,0"], comments)
                with self.assertRaisesRegex(ValueError, "ARV"):
                    arv.read_jshis(path, {_FIRST})

    def test_jshis_rejects_repeated_meshes(self):
        path = self._jshis([
            f"{_FIRST}, 4,510.4,   0.8125,-,0",
            f"{_FIRST}, 4,510.4,   0.9000,-,0",
        ])

        with self.assertRaisesRegex(ValueError, _FIRST):
            arv.read_jshis(path, {_FIRST})

    def test_zip_needs_one_csv(self):
        for entries in ({}, {"a.csv": "", "b.CSV": ""}):
            for read in (arv.read_cabinet_office, arv.read_jshis):
                with self.subTest(entries=list(entries), read=read.__name__):
                    path = self._zip(
                        "ground.zip", {"readme.txt": "", **entries}
                    )
                    with self.assertRaisesRegex(ValueError, "ground.zip"):
                        read(path, {_FIRST})

    def test_station_arv_leaves_out_missing_meshes(self):
        stations = [
            _station("7654321", 2120, 8340),
            _station("7654322", 2160, 8400),
        ]
        code = mesh.code_250m(
            fractions.Fraction(2120, 60), fractions.Fraction(8340, 60)
        )

        self.assertEqual(
            arv.station_arv(stations, {code: 1.25}), {"7654321": 1.25}
        )

    def test_station_arv_rejects_repeated_numbers(self):
        stations = [_station("7654321", 2120, 8340)] * 2

        with self.assertRaisesRegex(ValueError, "7654321"):
            arv.station_arv(stations, {})

    def test_table_round_trips(self):
        table = {
            "7654322": (1.514, None),
            "7654321": (0.1 + 0.2, 0.8125),
            "7654323": (None, None),
        }
        path = self.root / "arv.csv"

        arv.write_table(path, table)

        self.assertEqual(arv.read_table(path), table)
        self.assertEqual(
            path.read_text(encoding="ascii"),
            "number,cabinet_office,jshis\n"
            "7654321,0.30000000000000004,0.8125\n"
            "7654322,1.514,\n"
            "7654323,,\n",
        )

    def test_read_table_rejects_other_files(self):
        path = self.root / "arv.csv"
        for text in (
            "number,jshis,cabinet_office\n",
            "number,cabinet_office,jshis\n7654321,1.0\n",
            "number,cabinet_office,jshis\n7654321,x,\n",
        ):
            with self.subTest(text=text):
                path.write_text(text, encoding="ascii")
                with self.assertRaises(ValueError):
                    arv.read_table(path)

    def _sources(self):
        jma = self.root / "jma"
        ground = self.root / "ground"
        jma.mkdir()
        ground.mkdir()
        with zipfile.ZipFile(jma / "code_p.zip", "w") as archive:
            archive.writestr("code_p.dat", _STATIONS.encode("ascii"))
        cabinet_office = self._cabinet_office(
            [f'"{_FIRST}",0.310,0.229,139.0,35.3']
        )
        jshis = self._jshis([
            f"{_FIRST}, 4,510.4,   0.8125,-,0",
            f"{_SECOND}, 4,250.0,   1.2500,-,0",
        ])
        cabinet_office.rename(ground / arv.CABINET_OFFICE)
        jshis.rename(ground / arv.JSHIS)
        paths = [
            jma / "code_p.zip",
            ground / arv.CABINET_OFFICE,
            ground / arv.JSHIS,
        ]
        sums = "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in paths
        )
        return jma, ground, sums

    def test_cli_writes_the_table(self):
        jma, ground, sums = self._sources()
        (self.root / "SHA256SUMS").write_text(sums, encoding="ascii")
        out = self.root / "arv.csv"
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            status = arv.main([
                "--sums", str(self.root / "SHA256SUMS"),
                "--jma", str(jma),
                "--ground", str(ground),
                "--out", str(out),
            ])

        self.assertEqual(status, 0)
        self.assertEqual(
            arv.read_table(out),
            {"7654321": (1.514, 0.8125), "7654322": (None, 1.25)},
        )
        self.assertIn(
            str(arv.Coverage(stations=2, cabinet_office=1, jshis=2, both=1)),
            output.getvalue(),
        )

    def test_cli_verifies_the_sources(self):
        jma, ground, sums = self._sources()
        lines = sums.splitlines(keepends=True)
        (self.root / "SHA256SUMS").write_text(
            "".join(lines[:2]), encoding="ascii"
        )
        out = self.root / "arv.csv"

        with self.assertLogs(level="ERROR") as logs:
            status = arv.main([
                "--sums", str(self.root / "SHA256SUMS"),
                "--jma", str(jma),
                "--ground", str(ground),
                "--out", str(out),
            ])

        self.assertEqual(status, 1)
        self.assertIn("SHA256SUMS", logs.output[0])
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

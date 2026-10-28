# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Run from the repository root.

python3 -m unittest tools/test_convert_slab2.py
"""

import pathlib
import tempfile
import unittest

from tools import convert_slab2


class ConvertRegionTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._temp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()
        super().tearDown()

    def convert(self, depth_lines, thickness_lines, newline="\n"):
        depth = self.dir / "dep.xyz"
        thickness = self.dir / "thk.xyz"
        depth.write_bytes("".join(s + newline for s in depth_lines).encode())
        thickness.write_bytes(
            "".join(s + newline for s in thickness_lines).encode()
        )
        return convert_slab2.convert_region(depth, thickness)

    def test_negates_depth_as_text_and_keeps_other_columns(self):
        lines = self.convert(
            ["140.05,35.1,-300.123456789", "140.1,35.1,5.5"],
            ["140.05,35.1,50.25", "140.1,35.1,40"],
        )

        self.assertEqual(
            lines, ["140.05,35.1,300.123456789,50.25", "140.1,35.1,-5.5,40"]
        )

    def test_skips_nan_depth(self):
        lines = self.convert(
            ["140,35,NaN", "140.05,35,-10"], ["140,35,NaN", "140.05,35,20"]
        )

        self.assertEqual(lines, ["140.05,35,10,20"])

    def test_crops_to_the_japan_window_with_tolerance(self):
        nodes = [
            "119.9,35",
            "119.95,35",
            "155.05,50.05",
            "155.1,35",
            "140,19.9",
            "140,50.1",
        ]
        lines = self.convert(
            [f"{node},-{i}" for i, node in enumerate(nodes, start=1)],
            [f"{node},1" for node in nodes],
        )

        self.assertEqual(lines, ["119.95,35,2,1", "155.05,50.05,3,1"])

    def test_reads_crlf_input(self):
        lines = self.convert(["140,35,-10"], ["140,35,20"], newline="\r\n")

        self.assertEqual(lines, ["140,35,10,20"])

    def test_rejects_mismatched_nodes(self):
        with self.assertRaises(ValueError):
            self.convert(["140,35,-10"], ["140.05,35,20"])

    def test_rejects_different_line_counts(self):
        with self.assertRaises(ValueError):
            self.convert(["140,35,-10", "140.05,35,-10"], ["140,35,20"])


class MainTest(unittest.TestCase):

    def test_writes_lf_csv_per_region(self):
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp) / "source"
            output = pathlib.Path(temp) / "output"
            source.mkdir()
            for region, version in convert_slab2.REGIONS.items():
                prefix = source / f"{region}_slab2"
                pathlib.Path(f"{prefix}_dep_{version}.xyz").write_text(
                    "140,35,-10\n", encoding="ascii"
                )
                pathlib.Path(f"{prefix}_thk_{version}.xyz").write_text(
                    "140,35,20\n", encoding="ascii"
                )

            self.assertEqual(convert_slab2.main([str(source), str(output)]), 0)

            for region in convert_slab2.REGIONS:
                self.assertEqual(
                    (output / f"{region}.csv").read_bytes(), b"140,35,10,20\n"
                )


if __name__ == "__main__":
    unittest.main()

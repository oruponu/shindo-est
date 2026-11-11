# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Tests that 250 m mesh codes are computed without rounding errors."""

import fractions
import unittest

from training import bulletin
from training import mesh


def _code(degrees_minutes_latitude, degrees_minutes_longitude):
    """Returns the code of a position given as (degrees, minutes) pairs."""
    latitude = fractions.Fraction(
        degrees_minutes_latitude[0] * 60 + degrees_minutes_latitude[1], 60
    )
    longitude = fractions.Fraction(
        degrees_minutes_longitude[0] * 60 + degrees_minutes_longitude[1], 60
    )
    return mesh.code_250m(latitude, longitude)


class MeshTest(unittest.TestCase):

    def test_codes_of_known_positions(self):
        for latitude, longitude, code in (
            ((35, 20), (139, 0), "5339000011"),
            ((35, 20), (139, 1), "5339000112"),
            ((35, 40), (139, 45), "5339460011"),
            ((43, 10), (141, 19), "6441620512"),
            ((24, 20), (123, 59), "3623470821"),
            ((45, 31), (148, 53), "6848272021"),
        ):
            with self.subTest(latitude=latitude, longitude=longitude):
                self.assertEqual(_code(latitude, longitude), code)

    def test_positions_on_boundaries_belong_to_the_northeast(self):
        self.assertEqual(_code((30, 0), (135, 6)), "4535000811")
        self.assertEqual(_code((36, 0), (140, 3)), "5440000411")
        self.assertEqual(_code((32, 3), (135, 6)), "4835006811")

    def test_half_and_quarter_digits(self):
        south_to_north = [
            ["11", "12", "21", "22"],
            ["13", "14", "23", "24"],
            ["31", "32", "41", "42"],
            ["33", "34", "43", "44"],
        ]
        south = fractions.Fraction(106, 3)
        west = fractions.Fraction(139)
        for row, digits in enumerate(south_to_north):
            for column, expected in enumerate(digits):
                with self.subTest(row=row, column=column):
                    code = mesh.code_250m(
                        south + fractions.Fraction(row, 480),
                        west + fractions.Fraction(column, 320),
                    )
                    self.assertEqual(code, "53390000" + expected)

    def test_station_code_uses_whole_minutes(self):
        station = bulletin.Station(
            number="7654321",
            name="Test",
            latitude_minutes=32 * 60 + 3,
            longitude_minutes=135 * 60 + 6,
            period=None,
        )

        self.assertEqual(mesh.station_code(station), "4835006811")

    def test_rejects_positions_outside_japan(self):
        for latitude, longitude in ((-1, 139), (35, 99), (67, 139), (35, 200)):
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaises(ValueError):
                    mesh.code_250m(
                        fractions.Fraction(latitude),
                        fractions.Fraction(longitude),
                    )


if __name__ == "__main__":
    unittest.main()

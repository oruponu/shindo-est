# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Computes 250 m mesh codes of the Japanese standard grid exactly.

A 250 m mesh is a third-order mesh divided into four each way: 7.5 seconds of
latitude by 11.25 seconds of longitude. Station positions are whole minutes,
which often lie on mesh boundaries, so the codes are computed with fractions
instead of floating point, and a position on a boundary belongs to the mesh
to its north and east.
"""

import fractions
import math

from training import bulletin

# Quarter meshes per degree, and per first-, second-, and third-order mesh.
_ROWS_PER_DEGREE = 480
_COLUMNS_PER_DEGREE = 320
_FIRST = 320
_SECOND = 40
_THIRD = 4
_WESTERN_LONGITUDE = 100


def code_250m(
    latitude: fractions.Fraction, longitude: fractions.Fraction
) -> str:
    """Returns the 10-digit code of the 250 m mesh holding a position.

    Args:
        latitude: The latitude in degrees.
        longitude: The longitude in degrees.

    Returns:
        The first-order code (4 digits), the second- and third-order digits
        (2 each), and the half and quarter digits (1 for south-west, 2 for
        south-east, 3 for north-west, and 4 for north-east).

    Raises:
        ValueError: The first-order code does not have two digits of
            latitude and two of longitude.
    """
    row = math.floor(latitude * _ROWS_PER_DEGREE)
    column = math.floor((longitude - _WESTERN_LONGITUDE) * _COLUMNS_PER_DEGREE)
    first_row, row = divmod(row, _FIRST)
    first_column, column = divmod(column, _FIRST)
    if not (0 <= first_row <= 99 and 0 <= first_column <= 99):
        raise ValueError(
            f"({latitude}, {longitude}) is outside the first-order meshes"
        )
    second_row, row = divmod(row, _SECOND)
    second_column, column = divmod(column, _SECOND)
    third_row, row = divmod(row, _THIRD)
    third_column, column = divmod(column, _THIRD)
    half = 1 + 2 * (row // 2) + column // 2
    quarter = 1 + 2 * (row % 2) + column % 2
    return (
        f"{first_row:02}{first_column:02}{second_row}{second_column}"
        f"{third_row}{third_column}{half}{quarter}"
    )


def station_code(station: bulletin.Station) -> str:
    """Returns the code of the 250 m mesh holding a station.

    Args:
        station: The station, whose position is in whole minutes.

    Returns:
        The 10-digit code, as code_250m returns it.

    Raises:
        ValueError: The station is outside the first-order meshes.
    """
    return code_250m(
        fractions.Fraction(station.latitude_minutes, 60),
        fractions.Fraction(station.longitude_minutes, 60),
    )

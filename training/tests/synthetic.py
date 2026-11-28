# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Builds a small synthetic dataset and ARV table for training tests."""

import datetime
import math

import numpy as np

from training import bulletin
from training import dataset

NUMBERS = tuple(str(number) for number in range(7654321, 7654340))
# The one station whose ARV is known from the first source only.
_ONE_ARV = "7654323"


def grid_dataset() -> dataset.Dataset:
    """Returns quakes from 2004 to 2020 recorded on a grid of stations.

    The stations, numbered NUMBERS, stand 20 minutes of arc apart, five to a
    row, so that the small quakes leave far stations unfelt and some of those
    are sampled out.
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
        for at, number in enumerate(NUMBERS)
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


def arv_table() -> dict[str, tuple[float | None, float | None]]:
    """Returns an ARV table of the stations of grid_dataset.

    Every station has both ARVs except one, which has only the first.
    """
    table: dict[str, tuple[float | None, float | None]] = {
        number: (1.0 + 0.05 * at, 1.2 + 0.03 * at)
        for at, number in enumerate(NUMBERS)
    }
    table[_ONE_ARV] = (1.1, None)
    return table

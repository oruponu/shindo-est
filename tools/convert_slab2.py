#!/usr/bin/env python3
# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Converts Slab2 depth and thickness grids into the CSV files in data/slab2.

Usage: python3 tools/convert_slab2.py <xyz directory> <output directory>

See data/slab2/SOURCE.md for where the .xyz files come from and what the
conversion does. Values are copied as text so that the output does not depend
on float formatting.
"""

import hashlib
import pathlib
import sys

REGIONS = {"kur": "02.24.18", "izu": "02.24.18"}

MIN_LATITUDE, MAX_LATITUDE = 19.95, 50.05
MIN_LONGITUDE, MAX_LONGITUDE = 119.95, 155.05
TOLERANCE = 1e-9


def _negate(text: str) -> str:
    return text[1:] if text.startswith("-") else "-" + text


def _inside(longitude: float, latitude: float) -> bool:
    lat_ok = MIN_LATITUDE - TOLERANCE <= latitude <= MAX_LATITUDE + TOLERANCE
    lon_ok = MIN_LONGITUDE - TOLERANCE <= longitude <= MAX_LONGITUDE + TOLERANCE
    return lat_ok and lon_ok


def convert_region(
    depth_path: pathlib.Path, thickness_path: pathlib.Path
) -> list[str]:
    """Converts one region of Slab2 into CSV lines.

    Args:
        depth_path: The `<region>_slab2_dep_<version>.xyz` file.
        thickness_path: The matching `<region>_slab2_thk_<version>.xyz` file.

    Returns:
        Lines of `longitude,latitude,depth,thickness` without line endings,
        for the nodes that have a depth and lie inside the cropping window.
        The depth is negated so that it is positive downwards.

    Raises:
        ValueError: The two files have different line counts, or a line
            refers to different nodes in them.
    """
    depth_lines = depth_path.read_text(encoding="ascii").splitlines()
    thickness_lines = thickness_path.read_text(encoding="ascii").splitlines()
    if len(depth_lines) != len(thickness_lines):
        raise ValueError(
            f"{depth_path.name} and {thickness_path.name} have different"
            " line counts"
        )
    output = []
    pairs = zip(depth_lines, thickness_lines)
    for number, (depth_line, thickness_line) in enumerate(pairs, start=1):
        depth = depth_line.split(",")
        thickness = thickness_line.split(",")
        if depth[:2] != thickness[:2]:
            raise ValueError(
                f"line {number}: depth and thickness are on different nodes"
            )
        if depth[2] == "NaN":
            continue
        if not _inside(float(depth[0]), float(depth[1])):
            continue
        longitude, latitude = depth[0], depth[1]
        output.append(
            f"{longitude},{latitude},{_negate(depth[2])},{thickness[2]}"
        )
    return output


def main(argv: list[str]) -> int:
    """Converts every region and prints the SHA-256 of each input file.

    Args:
        argv: The source directory and the output directory.

    Returns:
        The process exit code.
    """
    if len(argv) != 2:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    source, output = pathlib.Path(argv[0]), pathlib.Path(argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for region, version in REGIONS.items():
        depth = source / f"{region}_slab2_dep_{version}.xyz"
        thickness = source / f"{region}_slab2_thk_{version}.xyz"
        lines = convert_region(depth, thickness)
        text = "".join(line + "\n" for line in lines)
        (output / f"{region}.csv").write_bytes(text.encode("ascii"))
        for path in (depth, thickness):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{digest}  {path.name}")
        print(f"{region}.csv: {len(lines)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

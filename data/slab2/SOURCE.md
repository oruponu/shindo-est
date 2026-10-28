# Slab2 grids

`kur.csv` (Kuril) and `izu.csv` (Izu-Bonin) give the upper surface of the subducting Pacific plate. They are derived from Slab2.

## Source

- Data: Hayes, G., 2018, Slab2 - A Comprehensive Subduction Zone Geometry Model: U.S. Geological Survey data release, https://doi.org/10.5066/F7PV6JNV
- Paper: Hayes, G. P., Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., and Smoczyk, G. M., 2018, Slab2, a comprehensive subduction zone geometry model, Science, 362, 58-61
- License: U.S. public domain. The data release metadata (`Slab2.xml`) gives access constraints `none` and use constraints `none`. USGS asks to be acknowledged as the source.
- Obtained: 2026-10-29, `Slab2Distribute_Mar2018.tar.gz` downloaded by hand from ScienceBase (https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467)

The Ryukyu grid (`ryu`) is not included because the features use only the Pacific slab.

The overturned (dip > 90°) part of the Izu-Bonin slab is not included either. Slab2 cannot describe it as a depth grid and provides it separately as discrete nodes (`Slab2Supp/izu_slab2_sup_02.24.18.csv`, latitude 15 to 27, longitude 138.85 to 145.18, depth 321 to 648 km). Only the depth and thickness grids are used here.

## Original files

```text
3f53a835bfff6196eb89912deed131855fc525dc5e1332f2c5459b1e3b9f7b1d  Slab2Distribute_Mar2018.tar.gz
5521eb753cf7eaaedf0da61cc12c5ce1181c3e324c8f0779211c8107939f0c22  Slab2_TXT/kur_slab2_dep_02.24.18.xyz
f0b3823634ae6ead34bd35db814037174a22cfebb3e617e1ad9fad95e09356c1  Slab2_TXT/kur_slab2_thk_02.24.18.xyz
bcba836df32047a02d8bdb05fb9f1c3deff1166beadb4a76d8bfd1057a0b388a  Slab2_TXT/izu_slab2_dep_02.24.18.xyz
652beb84e8b05295dc0b29a726792cd121bf40b9e885c4e1e4c833a35a4d2441  Slab2_TXT/izu_slab2_thk_02.24.18.xyz
```

## Conversion

Extract the four `.xyz` files from `Slab2_TXT/` in the archive and run

```sh
python3 tools/convert_slab2.py path/to/Slab2_TXT data/slab2
```

Each region has a depth file (`<region>_slab2_dep_<version>.xyz`) and a thickness file (`<region>_slab2_thk_<version>.xyz`). Both are comma-separated `longitude,latitude,value` lines in the same order. For each line, the script:

1. Checks that the depth and thickness lines have the same longitude and latitude text, and stops with an error otherwise.
2. Skips the line if the depth is `NaN`.
3. Skips the line if it is outside latitude 19.95 to 50.05 and longitude 119.95 to 155.05 (with a tolerance of 1e-9).
4. Writes `longitude,latitude,depth,thickness`, where longitude, latitude and thickness are copied as text and depth is negated as text (a leading `-` is removed, otherwise one is added), so depths are positive downwards in km.

In the output, longitude and latitude are in degrees, and depth and thickness are in km with depth positive downwards. The output has no header, one line per grid node, ASCII, and LF line endings.

## Files in this directory

```text
3185248ace761693f342d7779018bb540d0e60f02aa6084e58e8ea2670640e59  kur.csv  (92,185 lines)
b45b484407f59817ef78bca16537ba80f24554dbbc4e57c6ec1f569d47a2cda7  izu.csv  (28,984 lines)
```

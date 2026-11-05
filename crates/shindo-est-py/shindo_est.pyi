# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Type stubs for the shindo_est extension module."""

from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

__version__: str
FEATURE_NAMES: tuple[str, ...]
FEATURE_VERSION: int
STATUS_OK: int
STATUS_OUT_OF_SCOPE: int
STATUS_INVALID_INPUT: int

class InputError(ValueError): ...
class ModelFileError(ValueError): ...
class SlabGridError(ValueError): ...

def intensity_class(instrumental: npt.ArrayLike) -> npt.NDArray[np.int32]: ...
def intensity_label(value: SupportsIndex) -> str | None: ...

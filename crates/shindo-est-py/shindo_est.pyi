# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Type stubs for the shindo_est extension module."""

import os
from typing import SupportsIndex, final

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

@final
class Hypocenter:
    def __new__(
        cls, latitude: float, longitude: float, depth: float, magnitude: float
    ) -> Hypocenter: ...
    @property
    def latitude(self) -> float: ...
    @property
    def longitude(self) -> float: ...
    @property
    def depth(self) -> float: ...
    @property
    def magnitude(self) -> float: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __getnewargs__(self) -> tuple[float, float, float, float]: ...

@final
class Prediction:
    @property
    def model_id(self) -> str: ...
    @property
    def offset(self) -> float: ...
    @property
    def status(self) -> npt.NDArray[np.int32]: ...
    @property
    def intensity_class(self) -> npt.NDArray[np.int32]: ...
    @property
    def instrumental(self) -> npt.NDArray[np.float64]: ...
    @property
    def probability(self) -> npt.NDArray[np.float64]: ...
    @property
    def regression(self) -> npt.NDArray[np.float64]: ...

@final
class Model:
    @staticmethod
    def open(path: str | os.PathLike[str]) -> Model: ...
    @staticmethod
    def from_bytes(data: bytes | bytearray) -> Model: ...
    @property
    def model_id(self) -> str: ...
    @property
    def manifest(self) -> str: ...
    def predict(
        self,
        hypocenter: Hypocenter,
        site_latitude: npt.ArrayLike,
        site_longitude: npt.ArrayLike,
        site_arv: npt.ArrayLike,
        *,
        offset: float = 0.0,
        max_threads: int | None = None,
    ) -> Prediction: ...

@final
class SlabGeometry:
    @staticmethod
    def read(
        kuril: str | os.PathLike[str], izu_bonin: str | os.PathLike[str]
    ) -> SlabGeometry: ...
    def features(
        self,
        *,
        hypocenter_latitude: npt.ArrayLike,
        hypocenter_longitude: npt.ArrayLike,
        hypocenter_depth: npt.ArrayLike,
        hypocenter_magnitude: npt.ArrayLike,
        site_latitude: npt.ArrayLike,
        site_longitude: npt.ArrayLike,
        site_arv: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]: ...

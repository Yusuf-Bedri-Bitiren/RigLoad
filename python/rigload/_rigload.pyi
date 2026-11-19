from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt

def hello() -> str: ...
def decode_rgb(path: str | PathLike[str]) -> npt.NDArray[np.uint8]:
    """(H, W, 3) uint8. Raises OSError if unreadable, ValueError if not decodable."""

def decode_depth(path: str | PathLike[str]) -> npt.NDArray[np.uint16]:
    """(H, W) uint16 from a 16-bit grayscale PNG; any other format raises ValueError."""

def load_sample(manifest: str | PathLike[str], index: int) -> dict[str, Any]:
    """{"id": str, "rgb": (H, W, 3) uint8, "depth": (H, W) uint16, "K": (3, 3) float64,
    "depth_scale": float}."""

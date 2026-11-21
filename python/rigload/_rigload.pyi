from collections.abc import Iterator
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

class EpochIterator(Iterator[dict[str, Any]]):
    """One epoch of batches: {"rgb": (B, H, W, 3) uint8, "depth": (B, H, W) uint16,
    "K": (B, 3, 3) float64, "ids": list[str], "indices": (B,) int64}."""

    def __iter__(self) -> EpochIterator: ...
    def __next__(self) -> dict[str, Any]: ...

class Loader:
    """Parallel, prefetching loader. Each `iter(loader)` starts the next epoch, shuffled
    from (seed, epoch). num_threads defaults to the number of logical CPUs."""

    def __init__(
        self,
        manifest: str | PathLike[str],
        batch_size: int = 32,
        num_threads: int | None = None,
        prefetch: int = 2,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> EpochIterator: ...

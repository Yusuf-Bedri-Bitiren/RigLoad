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

def augment_sample(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.uint16],
    k: npt.ArrayLike,
    params: dict[str, Any],
    out_size: tuple[int, int] = (640, 192),
) -> dict[str, Any]:
    """Crop -> resize -> flip -> colour jitter with explicit parameters (the keys of
    bench/common.py `draw_params`: x0, y0, w, h, flip, brightness, contrast, saturation).
    Returns {"rgb": (out_h, out_w, 3) uint8, "depth": (out_h, out_w) uint16, "K": (3, 3)}.
    Copies its inputs; for tests and comparisons."""

class EpochIterator(Iterator[dict[str, Any]]):
    """One epoch of batches: {"rgb": (B, H, W, 3) uint8, "depth": (B, H, W) uint16,
    "K": (B, 3, 3) float64, "depth_scale": (B,) float64, "ids": list[str],
    "indices": (B,) int64}; with lidar=True also "depth_sparse" (B, H, W) uint16,
    "points" (N, 4) float32, "point_offsets" (B + 1,) int64, "T_cam_lidar" (B, 4, 4)."""

    def __iter__(self) -> EpochIterator: ...
    def __next__(self) -> dict[str, Any]: ...

class Loader:
    """Parallel, prefetching loader. Each `iter(loader)` starts the next epoch, shuffled
    from (seed, epoch). num_threads defaults to the number of logical CPUs.

    augment=True: random crop (height fraction `crop_scale` of the largest crop with the
    output aspect) -> resize to out_size (width, height) -> horizontal flip with `flip_prob`
    -> brightness/contrast/saturation in [1 - jitter, 1 + jitter], drawn per sample from
    (seed, epoch, index). K is updated to match; depth is resized nearest-exact. A flipped
    sample is the projection of the mirrored scene (X -> -X in the camera frame).

    lidar=True: also load each sample's points (manifest `lidar`, `T_cam_lidar`). With
    augment=True they get a yaw in [-yaw_range, yaw_range] about the LiDAR z axis and a scale
    in `scale_range` about the camera centre, plus the image's flip as a mirror. Outputs are
    kept consistent: projecting batch["points"] with batch["T_cam_lidar"] and batch["K"]
    lands on the augmented image, at depth matching batch["depth"] / batch["depth_scale"].
    Batches then also hold "depth_sparse" (the points re-projected, nearest wins),
    "points" (N, 4) float32 for all samples, and "point_offsets" (B + 1,): sample j owns
    rows point_offsets[j]:point_offsets[j + 1]."""

    def __init__(
        self,
        manifest: str | PathLike[str],
        batch_size: int = 32,
        num_threads: int | None = None,
        prefetch: int = 2,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        augment: bool = False,
        out_size: tuple[int, int] = (640, 192),
        crop_scale: tuple[float, float] = (0.8, 1.0),
        flip_prob: float = 0.5,
        jitter: float = 0.2,
        lidar: bool = False,
        yaw_range: float = 0.7853981633974483,
        scale_range: tuple[float, float] = (0.95, 1.05),
    ) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> EpochIterator: ...

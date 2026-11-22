"""Baseline 1: PyTorch Dataset + torchvision.transforms.v2 (functional API).

Decode: torchvision.io (libjpeg-turbo for JPEG, libpng for 16-bit depth).
Augment: crop -> resize (RGB bilinear + antialias, depth nearest) -> hflip -> color jitter,
all from one `draw_params` call per sample, with K updated by `update_k`.

Why the functional API: random v2 transforms hide their parameters, so K could not be updated.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_jpeg, decode_png, read_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

from common import OUT_H, OUT_W, draw_params, read_manifest, update_k

# shortcut: torchvision.io image decode is deprecated since 0.29 (moving to TorchCodec) but still
# the fastest torch-native path; revisit at Step 9 or when it's removed.
warnings.filterwarnings("ignore", message="The image decoding", category=DeprecationWarning)


class TorchvisionRGBD(Dataset):
    def __init__(self, manifest: str, seed: int = 0, epoch: int = 0, augment: bool = True,
                 depth_interpolation: InterpolationMode = InterpolationMode.NEAREST_EXACT):
        # Mask resize defaults to legacy NEAREST, which is offset from the RGB's bilinear
        # resize (see docs/PRIOR_ART.md); NEAREST_EXACT matches the half-pixel convention.
        self.rows = read_manifest(manifest)
        self.seed, self.epoch, self.augment = seed, epoch, augment
        self.depth_interpolation = depth_interpolation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        rgb = decode_jpeg(read_file(row["rgb"]))  # 3xHxW uint8
        depth = tv_tensors.Mask(decode_png(read_file(row["depth"])))  # 1xHxW uint16
        k = np.array(row["K"])
        if self.augment:
            p = draw_params(self.seed, self.epoch, i, row["width"], row["height"])
            rgb, depth, k = augment(rgb, depth, k, p, self.depth_interpolation)
        return {"rgb": rgb, "depth": depth.as_subclass(torch.Tensor), "K": torch.from_numpy(k)}


def augment(rgb, depth, k, p: dict,
            depth_interpolation: InterpolationMode = InterpolationMode.NEAREST_EXACT):
    """The baseline's transforms for explicit parameters: rgb 3xHxW uint8, depth a 1xHxW
    `tv_tensors.Mask`. Also used by tests to compare rigload against the same ops."""
    rgb = F.crop(rgb, p["y0"], p["x0"], p["h"], p["w"])
    depth = F.crop(depth, p["y0"], p["x0"], p["h"], p["w"])
    rgb = F.resize(rgb, [OUT_H, OUT_W], antialias=True)
    depth = F.resize(depth, [OUT_H, OUT_W], interpolation=depth_interpolation)
    if p["flip"]:
        rgb, depth = F.horizontal_flip(rgb), F.horizontal_flip(depth)
    rgb = F.adjust_brightness(rgb, p["brightness"])
    rgb = F.adjust_contrast(rgb, p["contrast"])
    rgb = F.adjust_saturation(rgb, p["saturation"])
    return rgb, depth, update_k(k, p)

"""Baseline 2: PyTorch Dataset + AlbumentationsX kernels (OpenCV).

Decode: OpenCV (libjpeg-turbo for JPEG, 16-bit PNG unchanged).
Augment: the same `draw_params` draw as baseline 1, applied with Albumentations' functional
kernels: crop -> resize (RGB INTER_LINEAR, Albumentations' default; depth INTER_NEAREST_EXACT)
-> hflip -> color jitter.

Why not `A.Compose`: its applied-params record has no crop coordinates, so K could not be
updated, and building a Compose per sample costs as much as applying it (~14 ms measured).
Calling the kernels directly skips Compose overhead, so this baseline is if anything
optimistic for Albumentations.

AlbumentationsX is AGPL-3.0; it is only imported by these bench scripts, never by rigload.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.geometric import functional as fgeo
from albumentations.augmentations.pixel import functional as fpixel
from torch.utils.data import Dataset

from common import OUT_H, OUT_W, draw_params, read_manifest, update_k

# Common practice in DataLoader workers: no OpenCV-internal threads on top of the worker
# processes (oversubscription). Also makes num_workers=0 single-threaded.
cv2.setNumThreads(0)


class AlbumentationsRGBD(Dataset):
    def __init__(self, manifest: str, seed: int = 0, epoch: int = 0, augment: bool = True):
        self.rows = read_manifest(manifest)
        self.seed, self.epoch, self.augment = seed, epoch, augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        rgb = cv2.imread(row["rgb"], cv2.IMREAD_COLOR_RGB)  # HxWx3 uint8
        depth = cv2.imread(row["depth"], cv2.IMREAD_UNCHANGED)  # HxW uint16
        if rgb is None or depth is None:
            raise FileNotFoundError(f"sample {row['id']}: cannot read {row['rgb']} / {row['depth']}")
        k = np.array(row["K"])
        if self.augment:
            p = draw_params(self.seed, self.epoch, i, row["width"], row["height"])
            box = (p["x0"], p["y0"], p["x0"] + p["w"], p["y0"] + p["h"])
            rgb, depth = fcrops.crop(rgb, *box), fcrops.crop(depth, *box)
            rgb = fgeo.resize(rgb, (OUT_H, OUT_W), cv2.INTER_LINEAR)
            depth = fgeo.resize(depth, (OUT_H, OUT_W), cv2.INTER_NEAREST_EXACT)
            if p["flip"]:
                rgb, depth = fgeo.hflip(rgb), fgeo.hflip(depth)
            rgb = fpixel.apply_color_jitter(rgb, p["brightness"], p["contrast"], p["saturation"],
                                            0.0, ["brightness_contrast", "saturation"])
            k = update_k(k, p)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1),
            "depth": torch.from_numpy(np.ascontiguousarray(depth))[None],
            "K": torch.from_numpy(k),
        }

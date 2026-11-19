"""Shared by every loader: manifest reading, the per-sample parameter draw, and the K update.

The Rust pipeline (Step 6) must reproduce `draw_params` and `update_k` exactly; tests compare
against these.

Pixel convention: centres at integer coordinates, [u, v, 1]^T ~ K @ X_cam (see
tools/gen_synthetic.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_W, OUT_H = 640, 192
CROP_SCALE = (0.8, 1.0)  # crop height as a fraction of the largest crop with the output aspect
JITTER = 0.2  # brightness, contrast, saturation factors drawn from [1 - JITTER, 1 + JITTER]


def read_manifest(path: str | Path) -> list[dict]:
    """Rows of manifest.jsonl with file paths made absolute."""
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key in ("rgb", "depth", "depth_sparse", "lidar"):
            if key in row:
                row[key] = str(path.parent / row[key])
        rows.append(row)
    return rows


def draw_params(seed: int, epoch: int, index: int, width: int, height: int,
                out_w: int = OUT_W, out_h: int = OUT_H) -> dict:
    """One random draw per sample, shared by every modality.

    Seeded from (seed, epoch, index) only, so it doesn't depend on worker or thread scheduling.
    """
    rng = np.random.default_rng([seed, epoch, index])
    h_max = min(height, width * out_h / out_w)
    h = int(round(rng.uniform(*CROP_SCALE) * h_max))
    w = min(width, int(round(h * out_w / out_h)))
    b, c, s = rng.uniform(1 - JITTER, 1 + JITTER, 3)
    return {
        "x0": int(rng.integers(0, width - w + 1)),
        "y0": int(rng.integers(0, height - h + 1)),
        "w": w,
        "h": h,
        "flip": bool(rng.random() < 0.5),
        # Applied in this fixed order: brightness, contrast, saturation.
        "brightness": float(b),
        "contrast": float(c),
        "saturation": float(s),
    }


def update_k(k, p: dict, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    """Intrinsics after crop -> resize -> optional horizontal flip.

    Resize maps pixel *edges*, so with integer pixel centres: u' + 0.5 = (u + 0.5) * s.
    Flip: u' = W - 1 - u. With fx kept positive, the flipped image is the projection of the
    mirrored scene (X -> -X in the camera frame); 3D data must be mirrored to match.
    """
    k = np.array(k, dtype=np.float64)
    k[0, 2] -= p["x0"]
    k[1, 2] -= p["y0"]
    sx, sy = out_w / p["w"], out_h / p["h"]
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
    k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
    if p["flip"]:
        k[0, 2] = out_w - 1 - k[0, 2]
    return k

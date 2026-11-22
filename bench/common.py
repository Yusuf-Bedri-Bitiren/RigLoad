"""Shared by every loader: manifest reading, the per-sample parameter draw, and the K update.

The Rust pipeline (Step 6) uses its own RNG (SplitMix64), so its draws differ; it must match
`update_k` exactly and the pixel ops for identical parameters, which tests pass in explicitly.

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
YAW_RANGE = np.pi / 4  # LiDAR yaw in [-YAW_RANGE, YAW_RANGE] (OpenPCDet's KITTI default)
SCALE_RANGE = (0.95, 1.05)  # scale about the camera centre


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
        # 3D only; drawn last so the 2D draws above are unchanged
        "yaw": float(rng.uniform(-YAW_RANGE, YAW_RANGE)),
        "scale": float(rng.uniform(*SCALE_RANGE)),
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


def augment_lidar(points: np.ndarray, t_cam_lidar, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same math as crates/rigload-core/src/lidar.rs: points p' = R T^-1 (s F) T p (mirror
    and scale in the camera frame, then yaw about LiDAR z), extrinsic T' = T R^-1. The image
    and K are unaffected by yaw and scale; metric depth scales by s."""
    t = np.asarray(t_cam_lidar, dtype=np.float64)
    s = p["scale"]
    sf = np.diag([-s if p["flip"] else s, s, s, 1.0])
    c, si = np.cos(p["yaw"]), np.sin(p["yaw"])
    r = np.array([[c, -si, 0, 0], [si, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    a = r @ np.linalg.inv(t) @ sf @ t
    out = points.copy()
    out[:, :3] = points[:, :3].astype(np.float64) @ a[:3, :3].T + a[:3, 3]
    return out, t @ r.T


def project_sparse(points: np.ndarray, t_cam_lidar, k, width: int, height: int,
                   depth_scale: float) -> np.ndarray:
    """Sparse uint16 depth (nearest point wins), same rules as rigload's project_sparse."""
    t = np.asarray(t_cam_lidar, dtype=np.float64)
    xyz = points[:, :3].astype(np.float64) @ t[:3, :3].T + t[:3, 3]
    z = xyz[:, 2]
    front = z > 0.1
    xyz, z = xyz[front], z[front]
    u = np.rint((k[0, 0] * xyz[:, 0] + k[0, 1] * xyz[:, 1]) / z + k[0, 2]).astype(np.int64)
    v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2]).astype(np.int64)
    value = np.rint(z * depth_scale)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (value <= 65535)
    lin, value = v[ok] * width + u[ok], np.maximum(value[ok], 1)
    order = np.argsort(value, kind="stable")
    _, first = np.unique(lin[order], return_index=True)
    depth = np.zeros(height * width, dtype=np.uint16)
    depth[lin[order[first]]] = value[order[first]]
    return depth.reshape(height, width)

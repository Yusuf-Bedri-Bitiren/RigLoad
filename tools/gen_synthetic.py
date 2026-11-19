#!/usr/bin/env python3
"""Seeded synthetic RGB-D + LiDAR dataset generator.

Scene: a ground plane plus random axis-aligned boxes, defined in the camera frame
(OpenCV: x right, y down, z forward). Camera rays and LiDAR rays are cast against the
same scene, so all modalities agree by construction.

Output layout (paths in the manifest are relative to the output directory):

    rgb/NNNNNN.jpg           shaded, textured image; sky where a ray hits nothing
    depth/NNNNNN.png         16-bit dense z-depth; value / depth_scale = metres, 0 = invalid
    depth_sparse/NNNNNN.png  16-bit; the sample's LiDAR points projected into the image
                             (nearest point wins), like KITTI depth-completion input
    lidar/NNNNNN.bin         float32 (x, y, z, intensity) per point, KITTI velodyne layout,
                             LiDAR frame (x forward, y left, z up)
    manifest.jsonl           one sample per line: paths, K (3x3), T_cam_lidar (4x4), ...
    meta.json                generator arguments, to regenerate the same set

Conventions:
    - Pixel (u, v) has its centre at integer coordinates (OpenCV / KITTI):
      [u, v, 1]^T ~ K @ X_cam.
    - X_cam = T_cam_lidar @ [X_lidar, 1].
    - Each sample's RNG is seeded from (seed, index) only, so a sample does not depend
      on N, the worker count or the generation order.

Usage:
    python tools/gen_synthetic.py --out data/synthetic --n 10000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

# Bump whenever the output for a given seed changes, so stale datasets are detectable.
GENERATOR_VERSION = 1

CAM_HEIGHT = 1.65  # metres above ground (KITTI camera)
DEPTH_SCALE = 256.0  # KITTI depth PNG convention
MAX_DEPTH = 80.0  # depth beyond this is stored as 0 (invalid), like KITTI GT
LIDAR_MIN_RANGE = 1.0
LIDAR_MAX_RANGE = 120.0
LIDAR_ELEVATIONS_DEG = np.linspace(2.0, -24.8, 64)  # HDL-64E-like vertical FOV

# KITTI 2011_09_26 left colour camera, for 1242 x 375; scaled for other sizes.
BASE_W, BASE_H = 1242, 375
BASE_FX, BASE_FY, BASE_CX, BASE_CY = 721.5377, 721.5377, 609.5593, 172.854

# LiDAR -> camera axes (x_cam = -y_lidar, y_cam = -z_lidar, z_cam = x_lidar) and mount
# offset: LiDAR 8 cm above and 27 cm behind the camera, roughly KITTI.
R_CAM_LIDAR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T_CAM_LIDAR = np.array([0.0, -0.08, -0.27])

SUN = np.array([0.3, -1.0, -0.5]) / np.linalg.norm([0.3, -1.0, -0.5])  # towards the sun


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _sample_scene(rng: np.random.Generator, width: int, height: int) -> dict:
    """Draw every random quantity of one sample. Draw order is part of the format."""
    sx, sy = width / BASE_W, height / BASE_H
    fx = BASE_FX * sx * (1 + rng.uniform(-0.02, 0.02))
    fy = BASE_FY * sy * (1 + rng.uniform(-0.02, 0.02))
    cx = BASE_CX * sx + rng.uniform(-5, 5) * sx
    cy = BASE_CY * sy + rng.uniform(-5, 5) * sy
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # Small per-sample calibration perturbation, so code must use each sample's own K / T.
    rot = _rotation(*np.deg2rad(rng.normal(0.0, 0.3, 3))) @ R_CAM_LIDAR
    t_cam_lidar = np.eye(4)
    t_cam_lidar[:3, :3] = rot
    t_cam_lidar[:3, 3] = T_CAM_LIDAR + rng.normal(0.0, 0.01, 3)

    boxes = []  # (lo xyz, hi xyz, rgb, reflectivity)

    def add_box(x: float, z: float) -> None:
        w, h, d = rng.uniform(0.6, 3.0), rng.uniform(0.8, 3.5), rng.uniform(0.6, 4.5)
        lo = np.array([x - w / 2, CAM_HEIGHT - h, z - d / 2])
        hi = np.array([x + w / 2, CAM_HEIGHT, z + d / 2])
        boxes.append((lo, hi, rng.uniform(40, 230, 3), rng.uniform(0.1, 0.9)))

    for _ in range(rng.integers(3, 9)):  # in front of the camera, mostly in view
        z = rng.uniform(5, 45)
        add_box(rng.uniform(-0.7 * z, 0.7 * z), z)
    for _ in range(rng.integers(2, 7)):  # anywhere around, for the 360-degree LiDAR
        x, z = rng.uniform(-30, 30, 2)
        if abs(x) < 3 and abs(z) < 4:  # keep clear of the sensors
            x += 6.0
        add_box(x, z)

    return {
        "k": k,
        "t_cam_lidar": t_cam_lidar,
        "boxes": boxes,
        "ground_gray": (rng.uniform(70, 110), rng.uniform(110, 150)),
        "ground_tint": rng.uniform(0.9, 1.1, 3),
        "noise_seed": int(rng.integers(2**63)),
    }


def cast(origin: np.ndarray, dirs: np.ndarray, boxes: list) -> tuple:
    """Cast rays origin + t * dirs (camera frame) against the scene.

    Returns (t, obj, normal): t = inf on a miss; obj = -1 sky, 0 ground, i >= 1 box i - 1.
    """
    n = len(dirs)
    t = np.full(n, np.inf)
    obj = np.full(n, -1, dtype=np.int32)
    normal = np.zeros((n, 3))

    # Ground plane y = CAM_HEIGHT (y points down), normal points up.
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (CAM_HEIGHT - origin[1]) / dirs[:, 1]
    hit = (dirs[:, 1] > 1e-12) & (tg > 0)
    t[hit], obj[hit], normal[hit] = tg[hit], 0, (0.0, -1.0, 0.0)

    # Boxes: slab test. A zero direction component gives +-inf slabs, which is correct;
    # NaN would need the origin exactly on a face plane, which random boxes never produce.
    with np.errstate(divide="ignore"):
        inv = 1.0 / dirs
    for i, (lo, hi, _, _) in enumerate(boxes, start=1):
        t1, t2 = (lo - origin) * inv, (hi - origin) * inv
        near_axes = np.fmin(t1, t2)
        t_near = near_axes.max(axis=1)
        t_far = np.fmax(t1, t2).min(axis=1)
        h = (t_near <= t_far) & (t_near > 0) & (t_near < t)
        t[h], obj[h] = t_near[h], i
        axis = near_axes[h].argmax(axis=1)  # the slab entered last is the face hit
        nrm = np.zeros((len(axis), 3))
        nrm[np.arange(len(axis)), axis] = -np.sign(dirs[h, axis])
        normal[h] = nrm
    return t, obj, normal


def _ground_checker(points: np.ndarray) -> np.ndarray:
    """1 m checkerboard on the ground, in camera-frame x / z."""
    return (np.floor(points[:, 0]) + np.floor(points[:, 2])) % 2


def render_camera(scene: dict, width: int, height: int) -> tuple:
    """Return (rgb uint8 HxWx3, z-depth float64 HxW with inf on a miss)."""
    k = scene["k"]
    us, vs = np.meshgrid(np.arange(width), np.arange(height))  # pixel centres at integers
    dirs = np.stack(
        [(us - k[0, 2]) / k[0, 0], (vs - k[1, 2]) / k[1, 1], np.ones(us.shape)], axis=-1
    ).reshape(-1, 3)
    t, obj, normal = cast(np.zeros(3), dirs, scene["boxes"])
    z = t  # dirs have z = 1, so the ray parameter is the z-depth
    hit = obj >= 0
    points = dirs[hit] * t[hit, None]

    rgb = np.zeros((len(dirs), 3))
    # Sky: blue gradient, lighter towards the horizon.
    sky_mix = (vs.reshape(-1)[~hit] / height)[:, None]
    rgb[~hit] = (1 - sky_mix) * [110, 160, 220] + sky_mix * [190, 210, 235]

    shade = 0.35 + 0.65 * np.clip(normal[hit] @ SUN, 0, None)
    colors = np.zeros((hit.sum(), 3))
    ground = obj[hit] == 0
    g_dark, g_light = scene["ground_gray"]
    checker = _ground_checker(points[ground])
    fade = np.exp(-z[hit][ground] / 40.0)  # fade texture with distance to limit aliasing
    gray = (g_dark + g_light) / 2 + fade * (checker - 0.5) * (g_light - g_dark)
    colors[ground] = gray[:, None] * scene["ground_tint"]
    box_colors = np.array([b[2] for b in scene["boxes"]])
    colors[~ground] = box_colors[obj[hit][~ground] - 1]
    rgb[hit] = colors * shade[:, None]

    noise = np.random.default_rng(scene["noise_seed"]).normal(0.0, 2.0, rgb.shape)
    rgb = np.clip(np.rint(rgb + noise), 0, 255).astype(np.uint8)
    return rgb.reshape(height, width, 3), z.reshape(height, width)


def scan_lidar(scene: dict, hres_deg: float) -> np.ndarray:
    """Return float32 N x 4 (x, y, z, intensity) in the LiDAR frame."""
    el = np.deg2rad(LIDAR_ELEVATIONS_DEG)
    az = np.deg2rad(np.arange(0.0, 360.0, hres_deg))
    e, a = np.meshgrid(el, az, indexing="ij")
    dirs_lidar = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)
    dirs_lidar = dirs_lidar.reshape(-1, 3)

    rot, trans = scene["t_cam_lidar"][:3, :3], scene["t_cam_lidar"][:3, 3]
    dirs_cam = dirs_lidar @ rot.T
    t, obj, normal = cast(trans, dirs_cam, scene["boxes"])
    keep = (obj >= 0) & (t >= LIDAR_MIN_RANGE) & (t <= LIDAR_MAX_RANGE)
    t, obj, normal, dirs_cam = t[keep], obj[keep], normal[keep], dirs_cam[keep]

    reflect = np.array([0.0] + [b[3] for b in scene["boxes"]])[obj]
    ground = obj == 0
    reflect[ground] = 0.25 + 0.1 * _ground_checker(trans + dirs_cam[ground] * t[ground, None])
    intensity = reflect * np.abs(np.sum(normal * dirs_cam, axis=1))  # dirs are unit length

    points = dirs_lidar[keep] * t[:, None]  # unit dirs, so t is the range
    return np.column_stack([points, intensity]).astype("<f4")


def project_lidar(points: np.ndarray, k: np.ndarray, t_cam_lidar: np.ndarray, width: int,
                  height: int) -> np.ndarray:
    """Sparse uint16 depth from LiDAR points; the nearest point wins each pixel."""
    xyz = points[:, :3].astype(np.float64) @ t_cam_lidar[:3, :3].T + t_cam_lidar[:3, 3]
    z = xyz[:, 2]
    front = z > 0.1
    xyz, z = xyz[front], z[front]
    u = np.rint(k[0, 0] * xyz[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z <= MAX_DEPTH)
    lin, z = v[ok] * width + u[ok], z[ok]

    order = np.argsort(z, kind="stable")
    _, first = np.unique(lin[order], return_index=True)  # first = nearest per pixel
    nearest = order[first]
    depth = np.zeros(height * width, dtype=np.uint16)
    depth[lin[nearest]] = encode_depth(z[nearest])
    return depth.reshape(height, width)


def encode_depth(z: np.ndarray) -> np.ndarray:
    return np.minimum(np.rint(z * DEPTH_SCALE), 65535).astype(np.uint16)


def _generate_one(job: tuple) -> dict:
    out, index, seed, width, height, lidar_hres, jpeg_quality, resume = job
    rng = np.random.default_rng([seed, index])
    scene = _sample_scene(rng, width, height)
    sid = f"{index:06d}"

    row = {
        "id": sid,
        "rgb": f"rgb/{sid}.jpg",
        "depth": f"depth/{sid}.png",
        "depth_sparse": f"depth_sparse/{sid}.png",
        "lidar": f"lidar/{sid}.bin",
        "width": width,
        "height": height,
        "K": scene["k"].tolist(),
        "T_cam_lidar": scene["t_cam_lidar"].tolist(),
        "depth_scale": DEPTH_SCALE,
    }
    files = [out / row[key] for key in ("rgb", "depth", "depth_sparse", "lidar")]
    if resume and all(f.is_file() and f.stat().st_size > 0 for f in files):
        return row  # the row comes from the RNG alone; rendering is skipped

    rgb, z = render_camera(scene, width, height)
    dense = np.zeros(z.shape, dtype=np.uint16)
    valid = np.isfinite(z) & (z <= MAX_DEPTH)
    dense[valid] = encode_depth(z[valid])
    points = scan_lidar(scene, lidar_hres)
    sparse = project_lidar(points, scene["k"], scene["t_cam_lidar"], width, height)

    # Fixed encoder settings; Pillow writes no timestamps, so output bytes are reproducible.
    Image.fromarray(rgb).save(out / row["rgb"], quality=jpeg_quality, subsampling=0)
    Image.fromarray(dense).save(out / row["depth"])
    Image.fromarray(sparse).save(out / row["depth_sparse"])
    points.tofile(out / row["lidar"])
    return row


def generate(out: str | Path, n: int, seed: int = 0, width: int = BASE_W,
             height: int = BASE_H, lidar_hres: float = 0.2, jpeg_quality: int = 90,
             workers: int = 1, resume: bool = False) -> None:
    """resume=True skips samples whose files all exist and are non-empty (same seed and
    settings assumed); delete possibly truncated files from an interrupted run first."""
    out = Path(out)
    for sub in ("rgb", "depth", "depth_sparse", "lidar"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(out, i, seed, width, height, lidar_hres, jpeg_quality, resume) for i in range(n)]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_generate_one, jobs, chunksize=4)
    else:
        rows = [_generate_one(job) for job in jobs]

    # newline="\n": identical bytes on Windows and Linux.
    with open(out / "manifest.jsonl", "w", newline="\n", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    meta = {
        "generator_version": GENERATOR_VERSION,
        "n": n,
        "seed": seed,
        "width": width,
        "height": height,
        "lidar_hres": lidar_hres,
        "jpeg_quality": jpeg_quality,
    }
    with open(out / "meta.json", "w", newline="\n", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n", type=int, required=True, help="number of samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=BASE_W)
    p.add_argument("--height", type=int, default=BASE_H)
    p.add_argument("--lidar-hres", type=float, default=0.2,
                   help="LiDAR horizontal resolution in degrees (0.2 is HDL-64E-like)")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="processes; does not change the output")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose files exist (after an interrupted run)")
    a = p.parse_args()
    generate(a.out, a.n, a.seed, a.width, a.height, a.lidar_hres, a.jpeg_quality, a.workers,
             a.resume)


if __name__ == "__main__":
    main()

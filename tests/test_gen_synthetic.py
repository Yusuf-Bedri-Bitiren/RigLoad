"""Tests for tools/gen_synthetic.py: determinism, geometry conventions, fixture freshness."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"

# A regular import (not importlib by path) so multiprocessing workers can import it too.
sys.path.insert(0, str(ROOT / "tools"))
import gen_synthetic as gen  # noqa: E402

SMALL = dict(n=2, lidar_hres=1.0)  # full-size images, light LiDAR: ~1 s per sample


def hashes(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def load(root: Path):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    return rows


def read_depth(root: Path, rel: str, scale: float) -> np.ndarray:
    return np.asarray(Image.open(root / rel), dtype=np.float64) / scale


def read_lidar(root: Path, rel: str) -> np.ndarray:
    return np.fromfile(root / rel, dtype="<f4").reshape(-1, 4)


def test_same_seed_is_byte_identical(tmp_path):
    gen.generate(tmp_path / "a", seed=7, **SMALL)
    gen.generate(tmp_path / "b", seed=7, workers=2, **SMALL)  # worker count must not matter
    a, b = hashes(tmp_path / "a"), hashes(tmp_path / "b")
    assert len(a) == 2 * 4 + 2  # 4 files per sample + manifest + meta
    assert a == b


def test_resume_fills_in_missing_samples(tmp_path):
    gen.generate(tmp_path / "full", seed=4, **SMALL)
    gen.generate(tmp_path / "part", seed=4, n=1, lidar_hres=1.0)
    gen.generate(tmp_path / "part", seed=4, resume=True, **SMALL)
    assert hashes(tmp_path / "part") == hashes(tmp_path / "full")


def test_different_seed_differs(tmp_path):
    gen.generate(tmp_path / "a", seed=1, **SMALL)
    gen.generate(tmp_path / "b", seed=2, **SMALL)
    a, b = hashes(tmp_path / "a"), hashes(tmp_path / "b")
    assert a["rgb/000000.jpg"] != b["rgb/000000.jpg"]


def test_sample_independent_of_n(tmp_path):
    gen.generate(tmp_path / "a", seed=3, n=1, lidar_hres=1.0)
    gen.generate(tmp_path / "b", seed=3, n=2, lidar_hres=1.0)
    a, b = hashes(tmp_path / "a"), hashes(tmp_path / "b")
    for name in ("rgb/000000.jpg", "depth/000000.png", "depth_sparse/000000.png",
                 "lidar/000000.bin"):
        assert a[name] == b[name]


@pytest.mark.parametrize("row", load(FIXTURES)[:5], ids=lambda r: r["id"])
def test_ground_depth_uses_integer_pixel_centres(row):
    """Ground plane y = h gives z = fy * h / (v - cy) at pixel centre (u, v) = integers.

    A half-pixel convention error shifts z by more than one depth unit on the bottom row.
    """
    k = np.array(row["K"])
    depth = read_depth(FIXTURES, row["depth"], 1.0)  # raw units
    v = np.arange(row["height"] - 5, row["height"])  # bottom rows, mostly ground
    expected = gen.encode_depth(k[1, 1] * gen.CAM_HEIGHT / (v - k[1, 2]))
    match = np.abs(depth[v] - expected[:, None]) <= 1
    assert match.mean() > 0.3, "ground pixels don't match the integer-pixel-centre model"


@pytest.mark.parametrize("row", load(FIXTURES), ids=lambda r: r["id"])
def test_lidar_projects_onto_dense_depth(row):
    """LiDAR points, mapped with the manifest's T_cam_lidar and K, land on matching depth.

    Calibrated on the generator: correct calibration gives >= 98% of points within 2%;
    a 0.3 degree extrinsic error drops that to 40-78%.
    """
    k, t = np.array(row["K"]), np.array(row["T_cam_lidar"])
    depth = read_depth(FIXTURES, row["depth"], row["depth_scale"])
    pts = read_lidar(FIXTURES, row["lidar"])[:, :3].astype(np.float64)
    xyz = pts @ t[:3, :3].T + t[:3, 3]
    z = xyz[:, 2]
    front = z > 0.1
    xyz, z = xyz[front], z[front]
    u = np.rint(k[0, 0] * xyz[:, 0] / z + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2]).astype(int)
    ok = (u >= 0) & (u < row["width"]) & (v >= 0) & (v < row["height"]) & (z <= gen.MAX_DEPTH)
    d = depth[v[ok], u[ok]]
    valid = d > 0
    rel = np.abs(d[valid] - z[ok][valid]) / z[ok][valid]
    assert len(rel) > 500
    assert np.median(rel) < 0.004
    assert (rel < 0.02).mean() > 0.95


@pytest.mark.parametrize("row", load(FIXTURES)[:5], ids=lambda r: r["id"])
def test_sparse_depth_is_projected_lidar(row):
    sparse = read_depth(FIXTURES, row["depth_sparse"], 1.0).astype(np.uint16)
    pts = read_lidar(FIXTURES, row["lidar"])
    again = gen.project_lidar(pts, np.array(row["K"]), np.array(row["T_cam_lidar"]),
                              row["width"], row["height"])
    assert np.array_equal(sparse, again)
    assert 0 < (sparse > 0).mean() < 0.1


def test_fixtures_match_generator(tmp_path):
    """Committed fixtures are what the current generator produces.

    Bytes can differ across platforms (float rounding in numpy, encoder builds), so this
    compares decoded data with small tolerances; same-machine byte identity is tested above.
    """
    meta = json.loads((FIXTURES / "meta.json").read_text())
    assert meta["generator_version"] == gen.GENERATOR_VERSION, "regenerate the fixtures"
    params = {k: meta[k] for k in ("seed", "width", "height", "lidar_hres", "jpeg_quality")}
    gen.generate(tmp_path, n=2, **params)

    fresh, committed = load(tmp_path), load(FIXTURES)[:2]
    for a, b in zip(fresh, committed):
        assert a.keys() == b.keys()
        np.testing.assert_allclose(a["K"], b["K"], rtol=1e-12)
        np.testing.assert_allclose(a["T_cam_lidar"], b["T_cam_lidar"], atol=1e-12)
        for key in ("depth", "depth_sparse"):
            da = read_depth(tmp_path, a[key], 1.0)
            db = read_depth(FIXTURES, b[key], 1.0)
            assert (da != db).mean() < 1e-3
        la, lb = read_lidar(tmp_path, a["lidar"]), read_lidar(FIXTURES, b["lidar"])
        assert abs(len(la) - len(lb)) <= 5
        if len(la) == len(lb):
            np.testing.assert_allclose(la, lb, atol=1e-3)
        ra = np.asarray(Image.open(tmp_path / a["rgb"]), dtype=np.float64)
        rb = np.asarray(Image.open(FIXTURES / b["rgb"]), dtype=np.float64)
        assert np.abs(ra - rb).mean() < 0.5

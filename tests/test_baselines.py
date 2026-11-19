"""Correctness of the benchmark baselines: K update vs. the actual pixel operations.

Skipped when torch / albumentations aren't installed (CI installs only the test extras).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("albumentations")

from PIL import Image  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from baseline_albu import AlbumentationsRGBD  # noqa: E402
from baseline_torch import TorchvisionRGBD  # noqa: E402
from common import OUT_H, OUT_W, draw_params, read_manifest  # noqa: E402

FIXTURE_MANIFEST = ROOT / "tests" / "fixtures" / "synthetic" / "manifest.jsonl"
N_DRAWS = 16  # parameter draws per test, about half of them flipped


@pytest.fixture(scope="module")
def grid_manifests(tmp_path_factory):
    """Two one-sample datasets whose 'depth' stores the source column (u + 1) or row (v + 1)."""
    row = read_manifest(FIXTURE_MANIFEST)[0]
    w, h = row["width"], row["height"]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    out = {}
    for axis, grid in (("u", us), ("v", vs)):
        d = tmp_path_factory.mktemp(axis)
        Image.fromarray((grid + 1).astype(np.uint16)).save(d / "depth.png")
        Image.new("RGB", (w, h)).save(d / "rgb.jpg")
        entry = {"id": "0", "rgb": "rgb.jpg", "depth": "depth.png", "width": w, "height": h,
                 "K": row["K"], "depth_scale": 1.0}
        (d / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
        out[axis] = (d / "manifest.jsonl", np.array(row["K"]))
    return out


def max_source_error(make_ds, grid_manifests) -> float:
    """Largest |picked source pixel - source point implied by K and K'| over all draws.

    Output pixel (u', v') back-projects through K' to a ray; projecting that ray with the
    original K gives the continuous source location (mirrored in x when flipped).
    Nearest-neighbour resampling must pick a pixel within 0.5 of it.
    """
    worst = 0.0
    us, vs = np.meshgrid(np.arange(OUT_W), np.arange(OUT_H))
    for axis in ("u", "v"):
        manifest, k = grid_manifests[axis]
        for seed in range(N_DRAWS):
            sample = make_ds(manifest, seed)[0]
            k2 = sample["K"].numpy()
            picked = sample["depth"][0].numpy().astype(np.float64) - 1
            if axis == "u":
                x = (us - k2[0, 2]) / k2[0, 0]
                if draw_params(seed, 0, 0, 1242, 375)["flip"]:
                    x = -x  # flipped image = mirrored scene
                src = k[0, 0] * x + k[0, 2]
            else:
                src = k[1, 1] * (vs - k2[1, 2]) / k2[1, 1] + k[1, 2]
            worst = max(worst, float(np.abs(picked - src).max()))
    return worst


def test_torchvision_k_matches_pixels(grid_manifests):
    err = max_source_error(lambda m, s: TorchvisionRGBD(m, seed=s), grid_manifests)
    assert err <= 0.5 + 1e-6


def test_albumentations_k_matches_pixels(grid_manifests):
    err = max_source_error(lambda m, s: AlbumentationsRGBD(m, seed=s), grid_manifests)
    assert err <= 0.5 + 1e-6


def test_torchvision_default_mask_nearest_is_misaligned(grid_manifests):
    """Documents a finding: tv_tensors.Mask resize defaults to legacy NEAREST, which picks
    pixels up to ~1 source pixel away from where the RGB (and K) say they come from."""
    err = max_source_error(
        lambda m, s: TorchvisionRGBD(m, seed=s, depth_interpolation=InterpolationMode.NEAREST),
        grid_manifests)
    assert err > 0.75


def test_baselines_agree_on_depth_and_k():
    a = TorchvisionRGBD(FIXTURE_MANIFEST, seed=3)
    b = AlbumentationsRGBD(FIXTURE_MANIFEST, seed=3)
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert torch.equal(sa["depth"], sb["depth"])
        assert torch.equal(sa["K"], sb["K"])
        assert sa["rgb"].shape == sb["rgb"].shape == (3, OUT_H, OUT_W)


def test_lidar_lands_on_transformed_depth():
    """End to end on the fixtures: T_cam_lidar, K' and the flip mirror together put LiDAR
    points on matching depth (thresholds measured: correct code gives median 0.25 %,
    95.1 % within 2 %)."""
    ds = TorchvisionRGBD(FIXTURE_MANIFEST, seed=5)
    errs = []
    for i, row in enumerate(ds.rows):
        s = ds[i]
        k, t = s["K"].numpy(), np.array(row["T_cam_lidar"])
        depth = s["depth"][0].numpy() / row["depth_scale"]
        pts = np.fromfile(row["lidar"], dtype="<f4").reshape(-1, 4)[:, :3].astype(np.float64)
        xyz = pts @ t[:3, :3].T + t[:3, 3]
        if draw_params(5, 0, i, row["width"], row["height"])["flip"]:
            xyz[:, 0] *= -1
        z = xyz[:, 2]
        keep = z > 0.1
        xyz, z = xyz[keep], z[keep]
        u = np.rint(k[0, 0] * xyz[:, 0] / z + k[0, 2]).astype(int)
        v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2]).astype(int)
        ok = (u >= 0) & (u < OUT_W) & (v >= 0) & (v < OUT_H) & (z <= 80)
        d = depth[v[ok], u[ok]]
        errs.append(np.abs(d[d > 0] - z[ok][d > 0]) / z[ok][d > 0])
    rel = np.concatenate(errs)
    assert np.median(rel) < 0.0035
    assert (rel < 0.02).mean() > 0.93


def test_draw_params_bounds():
    for i in range(500):
        p = draw_params(0, 0, i, 1242, 375)
        assert 0 <= p["x0"] and p["x0"] + p["w"] <= 1242
        assert 0 <= p["y0"] and p["y0"] + p["h"] <= 375
        assert abs(p["w"] / p["h"] - OUT_W / OUT_H) < 0.02
        assert 0.8 <= p["brightness"] <= 1.2

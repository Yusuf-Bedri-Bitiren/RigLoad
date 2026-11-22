"""Step 11 end to end: with augmentation (crop, resize, flip, yaw, scale), LiDAR points in the
batch must still land on the augmented image at the depth the dense depth map says.

The dense depth comes from the image path (decode -> crop/resize/flip), the points from the
3D path (decode -> A, T'), so agreement is not by construction. The geometry itself is proven
by the Rust property tests (crates/rigload-core/src/lidar.rs)."""

from pathlib import Path

import numpy as np
import pytest

import rigload

MANIFEST = Path(__file__).resolve().parent / "fixtures" / "synthetic" / "manifest.jsonl"


def project(points, k, t):
    xyz = points[:, :3].astype(np.float64) @ t[:3, :3].T + t[:3, 3]
    z = xyz[:, 2]
    keep = z > 0.1
    xyz, z = xyz[keep], z[keep]
    u = np.rint((k[0, 0] * xyz[:, 0] + k[0, 1] * xyz[:, 1]) / z + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * xyz[:, 1] / z + k[1, 2]).astype(int)
    return u, v, z


@pytest.fixture(scope="module")
def batches():
    loader = rigload.Loader(MANIFEST, batch_size=5, num_threads=4, seed=3, augment=True,
                            lidar=True)
    return list(loader)


def test_shapes_and_offsets(batches):
    for b in batches:
        n = len(b["ids"])
        assert b["depth_sparse"].shape == (n, 192, 640)
        assert b["T_cam_lidar"].shape == (n, 4, 4) and b["depth_scale"].shape == (n,)
        offs = b["point_offsets"]
        assert offs[0] == 0 and offs[-1] == len(b["points"]) and np.all(np.diff(offs) > 1000)
    # the scale augmentation shows up as a per-sample depth_scale around 256
    scales = np.concatenate([b["depth_scale"] for b in batches])
    assert np.all((256 / 1.05 <= scales) & (scales <= 256 / 0.95)) and scales.std() > 0


def test_points_land_on_matching_dense_depth(batches):
    """Thresholds as for the baselines in test_baselines.py (median 0.25 % there)."""
    errs = []
    for b in batches:
        for j in range(len(b["ids"])):
            pts = b["points"][b["point_offsets"][j]:b["point_offsets"][j + 1]]
            u, v, z = project(pts, b["K"][j], b["T_cam_lidar"][j])
            ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 192)
            depth = b["depth"][j] / b["depth_scale"][j]
            d, zz = depth[v[ok], u[ok]], z[ok]
            valid = d > 0
            errs.append(np.abs(d[valid] - zz[valid]) / zz[valid])
    rel = np.concatenate(errs)
    print(f"{rel.size} points, median rel err {np.median(rel):.4%}, "
          f"within 2 %: {(rel < 0.02).mean():.1%}")
    assert rel.size > 10_000
    assert np.median(rel) < 0.0035
    assert (rel < 0.02).mean() > 0.93


def test_sparse_depth_agrees_with_dense(batches):
    for b in batches:
        s, d = b["depth_sparse"].astype(float), b["depth"].astype(float)
        both = (s > 0) & (d > 0)
        assert both.sum() > 1000
        assert np.median(np.abs(s[both] - d[both]) / d[both]) < 0.0035


def test_torchvision_lidar_baseline_is_consistent_too():
    """The NumPy LiDAR path in the benchmark baseline must be as correct as rigload's, or the
    benchmark compares against something broken."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import sys

    sys.path.insert(0, str(MANIFEST.parents[3] / "bench"))
    from baseline_torch import TorchvisionRGBD, collate_lidar

    ds = TorchvisionRGBD(str(MANIFEST), seed=3, lidar=True)
    b = {k: v.numpy() for k, v in collate_lidar([ds[i] for i in range(len(ds))]).items()}
    errs, sparse_errs = [], []
    for j in range(len(ds)):
        pts = b["points"][b["point_offsets"][j]:b["point_offsets"][j + 1]]
        u, v, z = project(pts, b["K"][j], b["T_cam_lidar"][j])
        ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 192)
        d = b["depth"][j, 0][v[ok], u[ok]] / b["depth_scale"][j]
        errs.append(np.abs(d[d > 0] - z[ok][d > 0]) / z[ok][d > 0])
        s, dd = b["depth_sparse"][j].astype(float), b["depth"][j, 0].astype(float)
        both = (s > 0) & (dd > 0)
        sparse_errs.append(np.abs(s[both] - dd[both]) / dd[both])
    assert np.median(np.concatenate(errs)) < 0.0035
    assert np.median(np.concatenate(sparse_errs)) < 0.0035


def test_lidar_needs_manifest_fields(tmp_path):
    rows = MANIFEST.read_text().splitlines()[:2]
    import json

    stripped = []
    for line in rows:
        row = json.loads(line)
        row.pop("lidar")
        row["rgb"] = str(MANIFEST.parent / row["rgb"])
        row["depth"] = str(MANIFEST.parent / row["depth"])
        row.pop("depth_sparse", None)
        stripped.append(json.dumps(row))
    (tmp_path / "m.jsonl").write_text("\n".join(stripped))
    with pytest.raises(ValueError, match="no lidar path"):
        next(iter(rigload.Loader(tmp_path / "m.jsonl", batch_size=2, lidar=True)))

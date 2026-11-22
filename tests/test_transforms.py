"""Step 6: rigload's transforms vs. the torchvision baseline for identical parameters, and the
augmenting Loader from Python. Geometry itself is proven by the Rust property tests
(crates/rigload-core/src/transform.rs)."""

import sys
from pathlib import Path

import numpy as np
import pytest

import rigload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "synthetic" / "manifest.jsonl"


def test_loader_augments_to_output_size():
    loader = rigload.Loader(MANIFEST, batch_size=4, num_threads=2, augment=True,
                            out_size=(320, 96))
    b = next(iter(loader))
    assert b["rgb"].shape == (4, 96, 320, 3) and b["depth"].shape == (4, 96, 320)
    assert b["K"].shape == (4, 3, 3)


def test_bad_augment_config_is_value_error():
    with pytest.raises(ValueError, match="augmentation config"):
        rigload.Loader(MANIFEST, augment=True, crop_scale=(0.0, 1.0))
    with pytest.raises(ValueError, match="does not fit"):
        s = rigload.load_sample(MANIFEST, 0)
        p = dict(x0=10_000, y0=0, w=10, h=10, flip=False, brightness=1, contrast=1, saturation=1)
        rigload.augment_sample(s["rgb"], s["depth"], s["K"], p)


@pytest.fixture(scope="module")
def pairs():
    """(params, rigload output, torchvision baseline output) per fixture sample, both from
    the same decoded input and the same parameters."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(ROOT / "bench"))
    from torchvision import tv_tensors

    from baseline_torch import augment
    from common import draw_params, read_manifest

    out = []
    for i, row in enumerate(read_manifest(MANIFEST)):
        s = rigload.load_sample(MANIFEST, i)  # same decoded input for both
        p = draw_params(seed=5, epoch=0, index=i, width=row["width"], height=row["height"])
        ours = rigload.augment_sample(s["rgb"], s["depth"], s["K"], p)
        rgb_t = torch.from_numpy(s["rgb"]).permute(2, 0, 1).contiguous()
        depth_t = tv_tensors.Mask(torch.from_numpy(s["depth"].astype(np.int32))[None])
        rgb2, depth2, k2 = augment(rgb_t, depth_t, s["K"], p)
        theirs = {
            "rgb": rgb2.permute(1, 2, 0).numpy(),
            "depth": depth2.as_subclass(torch.Tensor)[0].numpy().astype(np.uint16),
            "K": k2,
        }
        out.append((p, ours, theirs))
    return out


def test_k_identical(pairs):
    for p, ours, theirs in pairs:
        np.testing.assert_allclose(ours["K"], theirs["K"], rtol=0, atol=1e-9, err_msg=str(p))


def test_depth_identical(pairs):
    for p, ours, theirs in pairs:
        np.testing.assert_array_equal(ours["depth"], theirs["depth"], err_msg=str(p))


def test_rgb_within_tolerance(pairs):
    diffs = np.concatenate([
        np.abs(ours["rgb"].astype(int) - theirs["rgb"].astype(int)).ravel()
        for _, ours, theirs in pairs
    ])
    print(f"rgb |diff| max {diffs.max()}, mean {diffs.mean():.4f}, "
          f"share > 1: {(diffs > 1).mean():.5f}")
    # Resize rounding differs by at most 1 (fast_image_resize vs torch, both antialiased
    # bilinear); the three truncating jitter steps (gains up to 1.2) can grow that to 4.
    assert diffs.max() <= 4 and diffs.mean() < 0.5

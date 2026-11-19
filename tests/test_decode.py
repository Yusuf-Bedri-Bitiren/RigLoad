"""Rust decode (via the Python binding) vs Pillow on the committed fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rigload

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
MANIFEST = FIXTURES / "manifest.jsonl"
ROWS = [json.loads(line) for line in MANIFEST.read_text().splitlines()]


@pytest.mark.parametrize("key", ["depth", "depth_sparse"])
def test_png16_identical_to_pillow(key):
    for row in ROWS:
        ours = rigload.decode_depth(FIXTURES / row[key])
        ref = np.asarray(Image.open(FIXTURES / row[key]))
        assert ours.dtype == np.uint16
        assert np.array_equal(ours, ref), row["id"]


def test_jpeg_within_tolerance_of_pillow():
    """zune-jpeg vs libjpeg-turbo: IDCT rounding differs slightly. Measured on the fixtures
    (4:4:4, quality 90): 99.4 % of values identical, max |diff| 3, mean 0.006."""
    for row in ROWS:
        ours = rigload.decode_rgb(FIXTURES / row["rgb"]).astype(np.int16)
        ref = np.asarray(Image.open(FIXTURES / row["rgb"])).astype(np.int16)
        assert ours.shape == ref.shape == (row["height"], row["width"], 3)
        diff = np.abs(ours - ref)
        assert diff.max() <= 4, row["id"]
        assert diff.mean() < 0.05, row["id"]


def test_load_sample_matches_manifest():
    for i in (0, 7, 19):
        s = rigload.load_sample(MANIFEST, i)
        row = ROWS[i]
        assert s["id"] == row["id"]
        assert s["rgb"].shape == (row["height"], row["width"], 3)
        assert np.array_equal(s["depth"], np.asarray(Image.open(FIXTURES / row["depth"])))
        assert np.array_equal(s["K"], np.array(row["K"]))
        assert s["depth_scale"] == row["depth_scale"]


def test_arrays_wrap_rust_buffers_without_copy():
    s = rigload.load_sample(MANIFEST, 0)
    for key in ("rgb", "depth", "K"):
        arr = s[key]
        assert not arr.flags["OWNDATA"] and arr.base is not None  # view over the Rust Vec
        assert arr.flags["C_CONTIGUOUS"] and arr.flags["WRITEABLE"]


def test_errors_carry_the_path():
    with pytest.raises(OSError, match="nope.jpg"):
        rigload.decode_rgb(FIXTURES / "rgb" / "nope.jpg")
    with pytest.raises(ValueError, match="16-bit"):
        rigload.decode_depth(FIXTURES / ROWS[0]["rgb"])
    with pytest.raises(IndexError):
        rigload.load_sample(MANIFEST, len(ROWS))

"""rigload.Loader from Python: completeness, determinism, contents, shutdown, errors."""

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rigload

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
MANIFEST = FIXTURES / "manifest.jsonl"


def ids_of(loader):
    return [b["ids"] for b in loader]


def test_every_sample_once_per_epoch():
    loader = rigload.Loader(MANIFEST, batch_size=6, num_threads=4)
    batches = ids_of(loader)
    assert len(loader) == len(batches) == 4
    flat = [i for b in batches for i in b]
    assert sorted(flat) == [f"{i:06d}" for i in range(20)]


def test_drop_last():
    loader = rigload.Loader(MANIFEST, batch_size=6, drop_last=True)
    assert len(loader) == 3 and len(ids_of(loader)) == 3


def test_same_seed_same_order_and_new_epoch_reshuffles():
    a = rigload.Loader(MANIFEST, batch_size=5, seed=3, num_threads=2)
    b = rigload.Loader(MANIFEST, batch_size=5, seed=3, num_threads=8)  # threads don't matter
    first = ids_of(a)
    assert first == ids_of(b)
    assert ids_of(a) != first  # epoch 1


def test_shuffle_false_keeps_manifest_order():
    loader = rigload.Loader(MANIFEST, batch_size=20, shuffle=False)
    assert ids_of(loader) == [[f"{i:06d}" for i in range(20)]]


def test_batch_contents_match_single_sample_decode():
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    batch = next(iter(rigload.Loader(MANIFEST, batch_size=4, seed=9)))
    assert batch["rgb"].shape == (4, 375, 1242, 3) and batch["rgb"].dtype == np.uint8
    assert batch["depth"].shape == (4, 375, 1242) and batch["depth"].dtype == np.uint16
    assert batch["K"].shape == (4, 3, 3)
    for j, i in enumerate(batch["indices"]):
        row = rows[i]
        assert batch["ids"][j] == row["id"]
        assert np.array_equal(batch["depth"][j], np.asarray(Image.open(FIXTURES / row["depth"])))
        assert np.array_equal(batch["rgb"][j], rigload.decode_rgb(FIXTURES / row["rgb"]))
        assert np.array_equal(batch["K"][j], np.array(row["K"]))
    for key in ("rgb", "depth"):  # moved from Rust, not copied
        assert not batch[key].flags["OWNDATA"] and batch[key].flags["C_CONTIGUOUS"]


def test_break_early_then_iterate_again():
    loader = rigload.Loader(MANIFEST, batch_size=2, prefetch=2)
    t = time.perf_counter()
    for i, _ in enumerate(loader):
        if i == 1:
            break
    assert time.perf_counter() - t < 10
    assert len(ids_of(loader)) == 10  # next epoch runs normally


def test_gil_released_while_waiting():
    """Another Python thread keeps running while the main thread waits on a batch."""
    ticks = 0
    stop = threading.Event()

    def spin():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        for _ in rigload.Loader(MANIFEST, batch_size=10, num_threads=1, shuffle=False):
            pass
    finally:
        stop.set()
        thread.join()
    assert ticks > 1000


def test_missing_file_raises_with_path(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "data")
    (tmp_path / "data" / "rgb" / "000007.jpg").unlink()
    loader = rigload.Loader(tmp_path / "data" / "manifest.jsonl", batch_size=20)
    with pytest.raises(OSError, match="000007.jpg"):
        for _ in loader:
            pass


def test_bad_arguments():
    with pytest.raises(RuntimeError, match="batch_size"):
        rigload.Loader(MANIFEST, batch_size=0)
    with pytest.raises(OSError):
        rigload.Loader(FIXTURES / "nope.jsonl")

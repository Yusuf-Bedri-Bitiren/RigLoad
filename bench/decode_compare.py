"""Per-image decode time, single-threaded: rigload (Rust) vs Pillow, torchvision.io, OpenCV.

Same files for every decoder (the first --n samples of the bench set), page cache warmed
first, --passes passes; reports the median (and min-max) of the per-pass mean ms per image.
The decoded array is materialized in every case (Pillow decodes lazily otherwise).

Build rigload in release mode first: `maturin develop --release` (debug builds are ~10x slower).

Usage: python bench/decode_compare.py [--n 300] [--passes 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.io import decode_jpeg, decode_png, read_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import rigload  # noqa: E402
from bench import environment  # noqa: E402
from common import read_manifest  # noqa: E402

warnings.filterwarnings("ignore", message="The image decoding", category=DeprecationWarning)
cv2.setNumThreads(0)
torch.set_num_threads(1)

DECODERS = {
    "rgb_jpeg": {
        "rigload": lambda p: rigload.decode_rgb(p),
        "pillow": lambda p: np.asarray(Image.open(p)),  # files are RGB already
        "torchvision": lambda p: decode_jpeg(read_file(p)),
        "opencv": lambda p: cv2.imread(p, cv2.IMREAD_COLOR_RGB),
    },
    "depth_png16_dense": {
        "rigload": lambda p: rigload.decode_depth(p),
        "pillow": lambda p: np.asarray(Image.open(p)),
        "torchvision": lambda p: decode_png(read_file(p)),
        "opencv": lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED),
    },
    "depth_png16_sparse": {
        "rigload": lambda p: rigload.decode_depth(p),
        "pillow": lambda p: np.asarray(Image.open(p)),
        "torchvision": lambda p: decode_png(read_file(p)),
        "opencv": lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED),
    },
}
FILE_KEY = {"rgb_jpeg": "rgb", "depth_png16_dense": "depth", "depth_png16_sparse": "depth_sparse"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", default=str(ROOT / "data" / "synthetic" / "manifest.jsonl"))
    p.add_argument("--n", type=int, default=300, help="files per kind")
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "bench" / "results" / "decode.json"))
    a = p.parse_args()

    rows = read_manifest(a.manifest)[: a.n]
    results = {}
    for kind, decoders in DECODERS.items():
        paths = [row[FILE_KEY[kind]] for row in rows]
        for path in paths:  # warm the page cache
            Path(path).read_bytes()
        results[kind] = {}
        for name, fn in decoders.items():
            fn(paths[0])  # first-call overhead (imports, lazy init) outside the timing
            per_pass = []
            for _ in range(a.passes):
                t0 = time.perf_counter()
                for path in paths:
                    fn(path)
                per_pass.append((time.perf_counter() - t0) * 1e3 / len(paths))
            results[kind][name] = {
                "ms_per_image_median": float(np.median(per_pass)),
                "ms_per_image_min": float(np.min(per_pass)),
                "ms_per_image_max": float(np.max(per_pass)),
            }
            print(f"{kind:20s} {name:12s} {np.median(per_pass):7.3f} ms/image", flush=True)

    report = {
        "env": environment(),
        "config": {"manifest": a.manifest, "files_per_kind": len(rows), "passes": a.passes,
                   "threads": "single (cv2.setNumThreads(0), torch.set_num_threads(1))",
                   "rigload_build": "maturin develop --release"},
        "results": results,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n-> {out}\n")
    names = list(DECODERS["rgb_jpeg"])
    print("| file | " + " | ".join(f"{n} ms" for n in names) + " | rigload vs fastest other |")
    print("|---" * (len(names) + 2) + "|")
    for kind, res in results.items():
        med = {n: res[n]["ms_per_image_median"] for n in names}
        best_other = min(v for n, v in med.items() if n != "rigload")
        cells = " | ".join(f"{med[n]:.2f}" for n in names)
        print(f"| {kind} | {cells} | {best_other / med['rigload']:.2f}x |")


if __name__ == "__main__":
    main()

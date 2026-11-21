"""Benchmark harness: throughput, batch latency, CPU and peak memory of a data loader.

Measures any iterable of batches (the Rust loader plugs in at Step 5). For the baselines it
sweeps DataLoader num_workers and writes one results JSON with the environment.

Protocol (CLAUDE.md section 6): fresh loader per repeat; max(`--warmup`, 2 * workers) batches
discarded (covers worker start-up); `--batches` timed; >= 5 repeats; median and min/max reported; page cache
warmed first unless --no-warm-cache.

Metrics:
    samples_per_s   timed batches * batch size / wall time
    batch_p50/p99   wall time between consecutive batches as seen by the consumer
    cpu_percent     CPU seconds of this process + all child processes during the timed window,
                    / wall time (100 = one logical core fully busy)
    peak_rss_mb     peak sum of RSS over the process tree (sampled every 50 ms; an upper
                    bound, since pages shared between worker processes are counted per process)

Usage:
    python bench/bench.py --loader torchvision
    python bench/bench.py --loader albumentations
    python bench/bench.py --loader torchvision --no-augment
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from common import OUT_H, OUT_W, read_manifest  # noqa: E402


def make_dataset(name: str, manifest: str, augment: bool, seed: int):
    if name == "torchvision":
        from baseline_torch import TorchvisionRGBD

        return TorchvisionRGBD(manifest, seed=seed, augment=augment)
    if name == "albumentations":
        from baseline_albu import AlbumentationsRGBD

        return AlbumentationsRGBD(manifest, seed=seed, augment=augment)
    raise ValueError(f"unknown loader {name!r}")


def make_dataloader(dataset, workers: int, batch_size: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=True,
    )


def make_loader(a, dataset, workers: int):
    """A fresh loader for one repeat. For rigload, `workers` is the number of Rust threads,
    all in this process; for the baselines it is DataLoader worker processes."""
    if a.loader == "rigload":
        import rigload

        return rigload.Loader(a.manifest, batch_size=a.batch_size, num_threads=workers,
                              prefetch=2, seed=a.seed, shuffle=True, drop_last=True)
    return make_dataloader(dataset, workers, a.batch_size, a.seed)


def _tree() -> list[psutil.Process]:
    me = psutil.Process()
    return [me, *me.children(recursive=True)]


def _cpu_seconds() -> dict[int, float]:
    out = {}
    for p in _tree():
        try:
            t = p.cpu_times()
            out[p.pid] = t.user + t.system
        except psutil.Error:
            pass
    return out


# Each DataLoader worker is a process that loads torch (~1.5 GB committed with the CUDA build on
# Windows). Running out of RAM makes the whole desktop freeze, so runs abort below this.
MIN_AVAILABLE_GB = 1.5


class MemoryPressure(RuntimeError):
    pass


class _PeakRss(threading.Thread):
    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self.low_memory = False

    def run(self) -> None:
        while not self._stop.is_set():
            total = 0
            for p in _tree():
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            if psutil.virtual_memory().available < MIN_AVAILABLE_GB * 2**30:
                self.low_memory = True
            self._stop.wait(self.interval)

    def check(self) -> None:
        if self.low_memory:
            raise MemoryPressure(f"available RAM fell below {MIN_AVAILABLE_GB} GB")

    def stop(self) -> int:
        self._stop.set()
        self.join()
        return self.peak


def measure(iterable, batch_size: int, warmup: int, batches: int) -> dict:
    """Time `batches` batches from `iterable` after discarding `warmup` batches."""
    rss = _PeakRss()
    rss.start()
    try:
        it = iter(iterable)
        for _ in range(warmup):
            next(it)
            rss.check()
        cpu0 = _cpu_seconds()
        latencies = []
        t0 = last = time.perf_counter()
        for _ in range(batches):
            next(it)
            rss.check()
            now = time.perf_counter()
            latencies.append(now - last)
            last = now
        cpu1 = _cpu_seconds()
    finally:
        peak = rss.stop()
    wall = last - t0
    cpu = sum(cpu1[pid] - cpu0[pid] for pid in cpu1 if pid in cpu0)
    lat_ms = np.array(latencies) * 1e3
    return {
        "samples_per_s": batches * batch_size / wall,
        "batch_p50_ms": float(np.percentile(lat_ms, 50)),
        "batch_p99_ms": float(np.percentile(lat_ms, 99)),
        "cpu_percent": 100.0 * cpu / wall,
        "peak_rss_mb": peak / 2**20,
    }


def warm_page_cache(manifest: str) -> float:
    """Read every RGB and depth file once so runs see a warm page cache. Returns GB read."""
    total = 0
    for row in read_manifest(manifest):
        for key in ("rgb", "depth"):
            total += len(Path(row[key]).read_bytes())
    return total / 1e9


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _git() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      text=True).strip()
        # Untracked files (e.g. new results JSONs) don't change the code under test.
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict:
    versions = {}
    for pkg in ("torch", "torchvision", "numpy", "pillow", "albumentationsx",
                "opencv-python-headless", "psutil"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    battery = psutil.sensors_battery()
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git(),
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "power_plugged": None if battery is None else battery.power_plugged,
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "versions": versions,
    }


def summarize(repeats: list[dict]) -> dict:
    keys = repeats[0].keys()
    return {
        stat: {k: float(fn([r[k] for r in repeats])) for k in keys}
        for stat, fn in (("median", np.median), ("min", np.min), ("max", np.max))
    }


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| workers | samples/s median (min–max) | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {r['workers']} | skipped: {r['skipped']} | | | | |")
            continue
        m, lo, hi = r["median"], r["min"], r["max"]
        label = r["workers"]
        if "stopped_early" in r:
            label = f"{label} ({r['repeats_completed']} repeats, then low memory)"
        lines.append(
            f"| {label} | {m['samples_per_s']:.0f} ({lo['samples_per_s']:.0f}–"
            f"{hi['samples_per_s']:.0f}) | {m['batch_p50_ms']:.0f} | {m['batch_p99_ms']:.0f} | "
            f"{m['cpu_percent']:.0f} | {m['peak_rss_mb']:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark a baseline DataLoader.")
    p.add_argument("--loader", choices=["torchvision", "albumentations", "rigload"], required=True)
    p.add_argument("--manifest", default=str(ROOT / "data" / "synthetic" / "manifest.jsonl"))
    p.add_argument("--workers", default="0,2,4,6,8", help="comma-separated num_workers sweep")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10, help="batches discarded per repeat")
    p.add_argument("--batches", type=int, default=100, help="batches timed per repeat")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-augment", action="store_true", help="decode + collate only")
    p.add_argument("--no-warm-cache", action="store_true")
    p.add_argument("--out", help="results JSON (default bench/results/<loader>_<mode>.json)")
    a = p.parse_args()

    mode = "noaug" if a.no_augment else "aug"
    if a.loader == "rigload" and not a.no_augment:
        p.error("rigload augmentation arrives in Step 6; use --no-augment")
    out = Path(a.out or ROOT / "bench" / "results" / f"{a.loader}_{mode}.json")
    workers = [int(w) for w in a.workers.split(",")]
    gb = 0.0 if a.no_warm_cache else warm_page_cache(a.manifest)

    dataset = None if a.loader == "rigload" else make_dataset(
        a.loader, a.manifest, augment=not a.no_augment, seed=a.seed)
    n_samples = len(read_manifest(a.manifest))
    env = environment()
    env["available_ram_gib_at_start"] = round(psutil.virtual_memory().available / 2**30, 1)
    results = []
    for w in workers:
        repeats = []
        # Every worker must have delivered batches before timing starts (spawn start-up).
        warmup = max(a.warmup, 2 * w)
        try:
            for r in range(a.repeats):
                loader = make_loader(a, dataset, w)
                try:
                    repeats.append(measure(loader, a.batch_size, warmup, a.batches))
                finally:
                    del loader
                    gc.collect()
                print(f"workers={w} repeat={r} {repeats[-1]['samples_per_s']:.0f} samples/s",
                      flush=True)
        except MemoryPressure as e:
            print(f"workers={w}: stopped, {e}; not trying more workers", flush=True)
            # Repeats that finished before the stop are valid measurements: keep them.
            entry = {"workers": w, "stopped_early": str(e), "repeats_completed": len(repeats)}
            if repeats:
                entry.update(repeats=repeats, **summarize(repeats))
            else:
                entry["skipped"] = "low memory"
            results.append(entry)
            write_report(out, env, a, n_samples, gb, results)
            break
        results.append({"workers": w, "repeats": repeats, **summarize(repeats)})
        write_report(out, env, a, n_samples, gb, results)  # keep partial sweeps

    print(f"\n{a.loader} ({'noaug' if a.no_augment else 'aug'}) -> {out}\n")
    print(markdown_table(results))


def write_report(out: Path, env: dict, a, n_samples: int, gb: float, results: list) -> None:
    report = {
        "env": env,
        "config": {
            "loader": a.loader,
            "augment": not a.no_augment,
            "manifest": a.manifest,
            "n_samples": n_samples,
            "batch_size": a.batch_size,
            "output_size": None if a.no_augment else [OUT_W, OUT_H],
            "warmup_batches": f"max({a.warmup}, 2 * workers)",
            "timed_batches": a.batches,
            "repeats": a.repeats,
            "seed": a.seed,
            "page_cache": "unknown" if a.no_warm_cache else f"warm ({gb:.2f} GB read first)",
        },
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

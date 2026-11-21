# CLAUDE.md — rigload: fast, calibration-aware multi-modal data loader (Rust + Python)

> Individual portfolio project of Yusuf (separate from the CS491 capstone). Written 2026-11-19.
> **Work one step at a time** (§9). Markers: **OPEN** = not decided, ask the user;
> **VERIFY** = an unconfirmed belief, check it before relying on it.
> Working name `rigload` ("sensor rig"). If the repo got another name, rename everywhere.

---

## 1. Goal

Build a pip-installable Python library with a Rust core that loads and augments
**synchronized multi-modal samples** (RGB + depth first, LiDAR later) for training
perception models, faster than standard PyTorch pipelines, with transforms that keep
all modalities **and the camera calibration** consistent.

Why this project: the user is aiming at ML engineering / ML systems roles. The project
must demonstrate Rust, PyO3/FFI, concurrency and rigorous benchmarking, and produce
concrete numbers for a resume.

Definition of success:

1. **Correct:** property tests prove cross-modal consistency (§5).
2. **Measured:** speedup vs strong baselines on the same hardware, reported honestly,
   including cases where we lose.
3. **Usable:** `pip install` works; a Python iterator API that drops into a PyTorch
   training loop.
4. **Presented:** README with a results table, plots and a design write-up.

---

## 2. Honest positioning (read before writing any README claim)

| Prior work | What it already does | Implication for us |
|---|---|---|
| `torch.utils.data.DataLoader` | Multi-process workers | Baseline |
| `torchvision.transforms.v2` | Applies the same random transform to image + mask + boxes (`tv_tensors`) | "Synchronized image + depth augmentation" alone is **not novel**. Use it as the baseline, with depth as `tv_tensors.Mask` (nearest resampling) |
| NVIDIA DALI | GPU-offloaded decode and augmentation, C++ core | Strongest general-purpose speed |
| FFCV | Pre-converted `.beton` format, fast loading | Different trade-off (conversion step) |
| WebDataset | Tar shards, sequential I/O | Different trade-off (sequential access) |
| Meta SPDL | Thread-based loading; ImageNet 74% faster than DataLoader with 38% less CPU (paper v1 abstract) | "Threads instead of processes is faster" is already published |
| `dataio-rs` (v0.5.6) | Rust/PyO3 loader, native worker pools, cloud IO, DLPack, free-threaded; no documented depth/intrinsics | Closest stack to ours: read it, don't copy it |
| Albumentations | Joint image + mask (depth as mask); `user_data` lets custom transforms shift cx, cy | Closest functionally, but Python-level and K is user-written |
| nvTorchCam | PyTorch camera-model library; crop/flip update intrinsics | Geometry library, not a loader |
| MMDetection3D | Python multi-modal 3D pipelines (points + images), `sync_2d` flips | Python-level speed |

Full details and links: `docs/PRIOR_ART.md` (checked 2026-11-19).

**Our angle (confirmed by the user 2026-11-19):** a native-speed pipeline whose geometric
transforms are **calibration-aware**: RGB, depth, camera intrinsics **and LiDAR points +
camera↔LiDAR extrinsics** stay mutually consistent, with modality-correct resampling
(never interpolate depth across invalid pixels). **LiDAR ↔ camera consistency is the main
differentiator**: existing tools are weakest there. Claim only what `bench/results/` shows.

**Data decides the claim:** build the engine first, benchmark it, and if the numbers don't
support this angle, pivot to a different one with the user instead of forcing the claim.

Step 1 re-verifies this table. If an existing tool already does calibration-aware
multi-modal augmentation at native speed, **stop and discuss with the user** before
continuing.

---

## 3. Scope

**In scope (v0.1):**

- RGB (JPEG/PNG) + 16-bit depth PNG + per-sample intrinsics, described by a manifest.
- LiDAR point clouds (KITTI-style `.bin`) + camera↔LiDAR extrinsics, with augmentations
  that keep point ↔ image projection consistent (moved from stretch, 2026-11-19).
- Rust decode, a `rayon` worker pool, bounded prefetch.
- Transforms: random crop, resize, horizontal flip, color jitter (RGB only), small
  rotation (**OPEN**: now or later), with intrinsics updated.
- Batch collation into contiguous buffers → NumPy → `torch.from_numpy`.
- Benchmarks, tests, wheels, docs.

**Stretch:** DLPack; free-threaded
Python builds; GPU decode; remote storage (S3); distributed sharding.

**Out of scope:** model research, video, any LLM feature.

---

## 4. Architecture

```
rigload/
├── Cargo.toml                 # workspace
├── crates/
│   ├── rigload-core/          # pure Rust: manifest, decode, transforms, sampler, pipeline, collate (no PyO3)
│   └── rigload-py/            # PyO3 bindings (cdylib), built by maturin
├── python/rigload/            # Python package: thin wrappers, type hints (.pyi)
├── tools/gen_synthetic.py     # seeded synthetic RGB-D dataset generator
├── bench/                     # baseline_torch.py, bench.py, results/*.json, plots/
├── tests/                     # pytest; tests/fixtures/ holds a tiny committed dataset
├── docs/                      # SETUP.md, PRIOR_ART.md, DECISIONS.md, PROGRESS.md, DESIGN.md
├── pyproject.toml             # maturin build backend
└── .github/workflows/ci.yml
```

Data path:

```
manifest ─► sampler (seeded shuffle) ─► rayon pool [ read → decode → transform ] ─► collate
        ─► bounded channel (prefetch depth P) ─► Python __next__ (GIL released while waiting)
        ─► torch.from_numpy(...) ─► model
```

Rules:

- `rigload-core` has **no Python dependency**: testable with `cargo test`,
  benchmarkable with `criterion`.
- Release the GIL during all Rust work (`Python::detach` in PyO3 0.30, formerly
  `allow_threads`; verified 2026-11-19).
- **Determinism independent of thread scheduling:** each sample's RNG is derived from
  `(seed, epoch, sample_index)`, never from a shared RNG.
- One random-parameter draw per sample, shared by every modality.
- Errors carry context (file path, sample id) and surface as Python exceptions; worker
  threads never panic silently; shutdown never hangs.

---

## 5. Transform correctness rules

| Transform | RGB | Depth | Intrinsics K = (fx, fy, cx, cy) |
|---|---|---|---|
| Crop (x0, y0, w, h) | crop | same crop | cx −= x0, cy −= y0 |
| Resize (sx, sy) | bilinear / area | nearest-exact, or sparse-aware (**OPEN**) | fx ×= sx, fy ×= sy; cx′ = (cx + 0.5)·sx − 0.5, cy′ likewise (integer pixel centres; verified against pixel ops in `tests/test_baselines.py`) |
| Horizontal flip | flip | flip | cx′ = W − 1 − cx, fx stays positive, and 3D data is **mirrored** (X → −X in the camera frame): a flipped image is the projection of the mirrored scene |
| Small rotation | bilinear, fill 0 | nearest, fill 0 (invalid) | not expressible in K alone: rotate the camera frame about the optical axis, or defer to the LiDAR step |
| Color jitter | yes | never | unchanged |

- Depth value 0 means invalid. Never average valid and invalid pixels.
- **Central test (property test with `proptest`):** for random 3D points X in the camera
  frame and a random transform T, projecting X with the updated K′ must equal applying T
  to the original projection (within ε), and the depth at the transformed pixel must equal
  the original depth at the source pixel. This test is what makes "calibration-aware" a
  defensible claim.

---

## 6. Benchmark protocol

- **Metrics:** samples/s; batch latency p50/p99; CPU % (`psutil`); peak RSS. GPU
  utilization (`pynvml`) only in GPU steps.
- **Baselines:** PyTorch DataLoader + `torchvision.transforms.v2` with the same transforms
  (depth as `tv_tensors.Mask`, intrinsics updated manually), sweeping
  `num_workers ∈ {0, 2, 4, 8, …}`, `persistent_workers=True`. Later: SPDL and
  `dataio-rs` where comparable.
- **Hygiene:** same machine and data for every run; discard warm-up batches; ≥ 5 repeats,
  report median and spread; state warm vs. cold page cache; write hardware, OS and library
  versions into every results JSON.
- **Never** state a number that isn't in `bench/results/`.
- **Known risk:** Pillow decodes JPEG with libjpeg-turbo, so the pure-Rust `image` decoder
  may be *slower* per image. Measure in Step 4. Alternatives: `zune-jpeg` (pure Rust),
  `turbojpeg` (libjpeg-turbo bindings). Our speedup may come from parallelism and less
  per-sample Python overhead rather than from decode.

---

## 7. Data

- Steps 2–7 use a **synthetic** dataset: no downloads, no license issues.
- Real data (**OPEN**, Step 8): KITTI depth completion (RGB + 16-bit PNG depth;
  value / 256 = meters, 0 = invalid, **VERIFY** in the devkit) or nuScenes mini
  (multi-camera + LiDAR). Both have non-commercial licenses: provide download
  instructions only, never commit data, and keep data directories in `.gitignore`.

---

## 8. Working style (for Claude Code)

- **One step at a time.** At the start of a step: restate its goal, ask its
  "Ask first" questions, propose a short plan, and wait for the user.
- At the end of a step: run `cargo fmt --check`, `cargo clippy -- -D warnings`,
  `cargo test`, `pytest`; show results as a table; update `docs/PROGRESS.md`; propose a
  commit message; wait for approval before starting the next step.
- The user knows C++, C, Java and Python but is **new to Rust**. Explain ownership,
  borrowing, `Send`/`Sync`, lifetimes and PyO3 GIL handling briefly when they first
  appear. The user must be able to explain every design choice in an interview; offer
  to let him write designated small pieces himself.
- Record every decision in `docs/DECISIONS.md` (date, options, choice, reason).
- Don't add a dependency without saying why. Don't invent numbers. Don't overclaim
  novelty.
- Keep commits small and focused.

---

## 9. Steps

Steps 0–5 are fixed. From Step 6 on, each step starts with decisions the user makes.

**Order (decided 2026-11-19):** 0 → 5 (engine) → 6 (camera transforms) → **11 (LiDAR)** →
7 → 8 → 9 → 10 → 12. Step numbers are kept so references stay valid.

### Step 0 — Environment (fixed)

- **Ask first:** OS; CPU core count and RAM; GPU access (**OPEN**: own GPU, Colab, cloud,
  or a university cluster under its usage policy); Python version.
- **Tasks:** install Rust via `rustup` (stable); create a Python venv (3.13, decided 2026-11-19);
  install `maturin`, `pytest`, `numpy`, `torch` (CPU build is fine for now),
  `torchvision`, `pillow`, `psutil`; record versions in `docs/SETUP.md`.
- **Done when:** `rustc --version`, `maturin --version` and
  `python -c "import torch, torchvision"` all succeed, and `SETUP.md` is committed.

### Step 1 — Repo skeleton + prior-art check (fixed)

- **Tasks:**
  - Cargo workspace with `rigload-core` and `rigload-py`; `pyproject.toml` using maturin;
    `rigload.hello()` returns a string from Rust.
  - CI on GitHub Actions (Ubuntu): `cargo fmt --check`, `cargo clippy -- -D warnings`,
    `cargo test`, `maturin develop` + `pytest`.
  - `.gitignore` (including `data/`), README stub with a one-line pitch, LICENSE
    (**OPEN**: MIT / Apache-2.0 dual license is the Rust norm).
  - Read the docs of every tool in §2 and write `docs/PRIOR_ART.md` with links. Report
    anything that already covers our angle.
- **Done when:** CI is green, `PRIOR_ART.md` is written, and the user confirms the
  positioning.

### Step 2 — Synthetic multi-modal dataset (fixed)

- **Tasks:** `tools/gen_synthetic.py`, seeded:
  - N samples of RGB JPEG (default 1242×375, KITTI-like) and matching 16-bit depth PNG,
    rendered from simple 3D geometry (planes, boxes) with a known K, so pixels
    back-project to real 3D points.
  - Optional sparsity mask on depth (LiDAR-like), 0 = invalid.
  - A LiDAR point cloud per sample sampled from the same geometry (KITTI `.bin` layout),
    with a known camera↔LiDAR extrinsic, so Step 11 has data from day one. Steps 3–5
    may ignore it.
  - `manifest.jsonl`: sample id, RGB path, depth path, LiDAR path, K, extrinsic,
    depth scale.
  - Sizes: a tiny set (~20 samples) committed in `tests/fixtures/`; a bench set
    (e.g. 10k samples) generated locally, never committed.
- **Done when:** the same seed produces byte-identical files (hash check in a test).

### Step 3 — Baseline + benchmark harness (fixed)

- **Tasks:**
  - `bench/baseline_torch.py`: a PyTorch `Dataset` reading the manifest, decoding with
    Pillow/torchvision, applying `transforms.v2` crop + resize + flip + color jitter
    jointly to image and depth, updating K manually; wrapped in a `DataLoader`.
  - Second baseline: the same pipeline with Albumentations (depth as an
    `additional_targets` mask, K updated manually), since it is the closest existing
    tool functionally (see `docs/PRIOR_ART.md`).
  - `bench/bench.py`: one harness that measures §6 metrics for any iterable loader and
    writes a results JSON; `num_workers` sweep.
  - Defaults: batch size 32, output size 640×192 (user may change).
- **Done when:** a baseline results table for the user's machine is committed. This is
  the number to beat.

### Step 4 — Rust core: manifest + decode, single-threaded (fixed)

- **Tasks:**
  - Manifest parser (`serde_json`); RGB decode (start with the `image` crate) and 16-bit
    depth PNG decode into owned buffers (HWC `u8`, HW `u16`).
  - Python binding returning NumPy arrays for one sample.
  - Tests: PNG pixels identical to Pillow's; JPEG within a documented tolerance (decoders
    can differ slightly).
  - `criterion` benchmark of per-image decode, compared with Pillow timing.
- **Done when:** tests pass and a decode-time table (Rust vs. Pillow) is shown.
  **Gate:** if Rust decode is slower, decide on the decoder with the user.

### Step 5 — Parallel pipeline + prefetch (fixed)

- **Tasks:**
  - Seeded sampler with per-epoch shuffle.
  - `rayon` pool decoding a batch's samples in parallel; collation into contiguous batch
    buffers.
  - Bounded `crossbeam` channel with configurable prefetch depth, fed by a background
    producer thread.
  - Python iterator `rigload.Loader(manifest, batch_size, num_threads, prefetch, seed)`
    whose `__next__` releases the GIL while waiting.
  - Clean shutdown on drop and on early `break`; errors surface as Python exceptions with
    the file path.
- **Tests:** no sample lost or duplicated in an epoch; same seed → same order; shutdown
  doesn't hang.
- **Benchmark:** throughput without augmentation vs. the no-augmentation baseline, across
  thread counts.
- **Done when:** the first honest comparison table exists, whatever it shows.

### Step 6 — Calibration-aware transforms

- **Ask first:** transform set and parameter ranges; output resolution; depth resize
  policy (nearest vs. sparse-aware min/max pooling); rotation now or deferred.
- **Tasks:** a `Transform` trait over a `Sample { rgb, depth, k, .. }` and a per-sample
  RNG; crop, resize, flip, color jitter (rotation if chosen); a pipeline configurable
  from Python.
- **Tests:** the §5 property test (≥ 10,000 cases); comparison with the baseline's
  transforms for identical parameters (document resize tolerance; torchvision
  antialiasing may differ).
- **Done when:** property tests pass and the with-augmentation benchmark vs. baseline is
  committed.

### Step 7 — Handoff to PyTorch

- **Ask first:** output dtype and layout (`uint8` NCHW vs. normalized `float32`;
  normalize in Rust or on the GPU).
- **Tasks:** expose batches as NumPy arrays over Rust-owned buffers without extra copies
  (`rust-numpy` / buffer protocol); count and document every copy; `torch.from_numpy`;
  optional `pin_memory`. Stretch: DLPack.
- **Done when:** `docs/DESIGN.md` states exactly which steps are zero-copy and which
  aren't, backed by a measurement.

### Step 8 — Real dataset adapter

- **Ask first:** KITTI depth completion vs. nuScenes mini vs. other; available disk
  space; subset size.
- **Tasks:** download instructions (no data in the repo); manifest builder including
  calibration parsing; a script saving overlays (depth colormap over RGB, before and
  after transforms) for visual checks.
- **Done when:** the user confirms the overlays line up, and a real-data benchmark is
  committed.

### Step 9 — Full benchmark suite

- **Ask first:** which extra baselines (SPDL, `dataio-rs`, DALI if a GPU is available);
  which machines.
- **Tasks:** sweeps over threads/workers, prefetch depth, batch size and image size;
  ablation of which optimization contributes what; plots; README results table with
  hardware stated; one command to reproduce.
- **Done when:** someone else could rerun the suite from the README.

### Step 10 — End-to-end GPU training test (needs a GPU)

- **Ask first:** model (e.g. a small monocular depth network) and dataset.
- **Tasks:** train N steps with each loader; sample GPU utilization with `pynvml`; report
  step time, GPU utilization and samples/s; include a CPU-limited setting (few cores),
  where the loader matters most.
- **Done when:** a GPU-utilization comparison chart is committed.

### Step 11 — LiDAR point clouds (core; runs right after Step 6)

- **Before starting:** LiDAR-side prior-art pass (OpenPCDet, MMDetection3D, Open3D-ML,
  nuScenes devkit) and update `docs/PRIOR_ART.md`; same stop-and-discuss rule as Step 1.
- **Ask first:** dataset (KITTI raw velodyne `.bin`: float32 x, y, z, intensity,
  **VERIFY**); which augmentations (global rotation, flip, scale).
- **Tasks:** point-cloud loading; transforms keeping point ↔ image projection consistent
  through the extrinsics; property tests; variable-length collation (offsets or padding).

### Step 12 — Packaging, docs, release

- **Tasks:** wheels via GitHub Actions for Linux and macOS (Windows **OPEN**); version
  0.1.0; PyPI publish from the user's account; README quickstart, API reference,
  `DESIGN.md`, benchmark methodology; CHANGELOG; tagged release.
- Draft a resume bullet from measured numbers only, e.g.:
  "Built rigload, a Rust/PyO3 data loader for multi-modal (RGB-D) training: X× samples/s
  vs. PyTorch DataLoader on N cores, with calibration-aware augmentation verified by
  property tests; pip-installable."

---

## 10. Open decisions (initial)

| Decision | Options | When |
|---|---|---|
| GPU source | **Decided:** RTX 3050 Laptop first; university HPC (H100/L40/H200) for scale, with permission | Step 0 |
| License | **Decided:** MIT OR Apache-2.0 | Step 1 |
| JPEG decoder | **Decided:** pure Rust `image` (= zune-jpeg); revisit `turbojpeg` in the Step 9 ablation | Step 4 |
| Depth resize policy | Nearest / sparse-aware pooling | Step 6 |
| Output dtype and normalization | `uint8` / `float32`; Rust / GPU | Step 7 |
| Real dataset | KITTI depth completion / nuScenes mini | Step 8 |
| Windows wheels | Yes / no | Step 12 |

---

## 11. References

- PyO3: https://pyo3.rs · maturin: https://www.maturin.rs
- Crates: `numpy` (rust-numpy), `rayon`, `crossbeam-channel`, `image`, `zune-jpeg`,
  `turbojpeg`, `serde_json`, `criterion`, `proptest` (docs on https://docs.rs)
- NVIDIA DALI: https://github.com/NVIDIA/DALI
- FFCV: https://github.com/libffcv/ffcv
- WebDataset: https://github.com/webdataset/webdataset
- SPDL: https://github.com/facebookresearch/spdl · paper: https://arxiv.org/abs/2504.20067
- dataio-rs: https://pypi.org/project/dataio-rs/
- ai-dataloader (small Rust DataLoader port): https://docs.rs/ai-dataloader/
- torchvision transforms v2: https://pytorch.org/vision/stable/transforms.html
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- KITTI: https://www.cvlibs.net/datasets/kitti/

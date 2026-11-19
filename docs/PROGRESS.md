# Progress

| Step | Status | Date | Notes |
|---|---|---|---|
| 0 — Environment | Done | 2026-11-19 | Rust 1.100.0 (MSVC), Python 3.13.5 venv, torch 2.14.1+cu130 with CUDA working. See `SETUP.md`. |
| 1 — Skeleton + prior art | Done | 2026-11-19 | Workspace, `rigload.hello()` from Rust, CI (green), dual license, `PRIOR_ART.md`. Positioning confirmed (LiDAR ↔ camera angle). |
| 2 — Synthetic dataset | Done | 2026-11-19 | `tools/gen_synthetic.py`: RGB JPEG, dense + sparse 16-bit depth, LiDAR `.bin`, manifest with K and T_cam_lidar. 20 fixtures (8.1 MB). Byte-identical regeneration tested; `--resume` for interrupted runs. 10k bench set generated locally (16.5 GB). Caveat for benchmarks: synthetic depth PNGs compress far better than real ones (planes give constant z per row), so depth decode is cheaper than on real data. |
| 3 — Baselines + harness | Code done; sweep pending | 2026-11-20 | torchvision + Albumentations baselines with K updates verified against pixel ops; `bench/bench.py` (memory guard, partial saves). First sweep stopped: 12 workers exhausted RAM. Results table still to commit. |

# Decisions

| Date | Decision | Options | Choice | Reason |
|---|---|---|---|---|
| 2026-11-19 | Python version | 3.11 / 3.12 / 3.13 | 3.13 | Already installed; torch, torchvision, PyO3 and maturin all support it. |
| 2026-11-19 | GPU source | Own GPU / Colab / cloud / university cluster | Own RTX 3050 Laptop (4 GB) to start; university HPC (H100, L40, possibly H200) for larger runs | Local GPU is enough for early GPU steps; HPC jobs need permission and get scheduled when scale requires it. |
| 2026-11-19 | Torch build | CPU / CUDA | CUDA (cu130) | Ready for the GPU steps; driver 616.92 supports CUDA 13. |
| 2026-11-19 | License | MIT / Apache-2.0 / MIT OR Apache-2.0 | MIT OR Apache-2.0 | Rust ecosystem norm (PyO3, serde, rayon); Apache adds a patent grant, MIT keeps it simple for users. |
| 2026-11-19 | PyO3 version | — | 0.30 (latest) | `Python::detach` releases the GIL (renamed from `allow_threads`). |
| 2026-11-19 | `extension-module` feature | Always on in Cargo.toml / only in maturin builds | Only via `[tool.maturin] features` | With it on, `cargo test` can't link libpython. Maturin turns it on for wheels; plain cargo builds link normally. |
| 2026-11-19 | Python package layout | Pure Rust module / `python/` source + `rigload._rigload` native module | Mixed layout | Python wrappers and `.pyi` type hints live next to the native module; the native part stays private (`_rigload`). |
| 2026-11-19 | Positioning | Camera-only calibration-aware / LiDAR ↔ camera consistency as the main angle | LiDAR ↔ camera, built on the camera pipeline | Prior art (`PRIOR_ART.md`) is weakest on keeping LiDAR, image and calibration consistent at native speed. Claims follow benchmarks; pivot if data doesn't support it. |
| 2026-11-19 | Step order | Original (LiDAR last, stretch) / LiDAR right after camera transforms | 0–5 → 6 → 11 → 7–10 → 12 | Engine first, then the main angle early. Synthetic data includes LiDAR from Step 2. |
| 2026-11-19 | Second baseline | torchvision v2 only / + Albumentations | + Albumentations | It is the closest existing tool functionally; beating only torchvision would be a weak comparison. |
| 2026-11-19 | Sparse depth source | Random mask on dense depth / projected LiDAR | Projected LiDAR, nearest point per pixel | KITTI-like scan-line structure, and it ties depth, LiDAR, K and T together so calibration bugs show up. |
| 2026-11-19 | Pixel-centre convention | Integer centres (OpenCV/KITTI) / half-integer | Integer: `[u, v, 1]^T ~ K X` | Matches KITTI and OpenCV calibration files. A test detects a half-pixel error. |
| 2026-11-19 | Synthetic calibration | Fixed K and T / per-sample jitter | K ±2 % focal, ±5 px centre; T ±0.3° and ±1 cm | Code that uses the wrong sample's calibration fails the tests. |
| 2026-11-19 | Fixtures | 20 full / 8 full / 20 half-res | 20 samples, full-res images, LiDAR at 1° (8.1 MB) | Full-size images like KITTI; LiDAR density is the only size driver. |
| 2026-11-19 | Determinism scope | Bytes identical everywhere / same machine | Bytes identical on the same machine; fixtures compared with tolerances across machines | numpy transcendental functions and encoder builds may differ by platform in the last bit. |
| 2026-11-19 | Rust toolchain | MSVC / GNU on Windows | stable-x86_64-pc-windows-msvc | Rust's default on Windows; matches the CPython build ABI. Requires VS C++ tools + Windows SDK. |

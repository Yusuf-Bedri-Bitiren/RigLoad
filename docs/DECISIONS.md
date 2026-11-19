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
| 2026-11-19 | Rust toolchain | MSVC / GNU on Windows | stable-x86_64-pc-windows-msvc | Rust's default on Windows; matches the CPython build ABI. Requires VS C++ tools + Windows SDK. |

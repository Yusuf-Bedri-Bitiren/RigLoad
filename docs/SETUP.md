# Setup

Recorded 2026-11-19. Every benchmark result JSON must also record these versions.

## Machine

| Item | Value |
|---|---|
| OS | Windows 11 Home, build 26200 |
| CPU | Intel Core i5-12500H, 12 cores / 16 threads (4 P-cores + 8 E-cores) |
| RAM | 15.7 GB |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB, driver 616.92 (plus Intel Iris Xe iGPU) |
| Larger GPUs | University HPC (H100, L40, possibly H200), with permission, when scale needs it |

The 12500H mixes P-cores and E-cores, so thread-count sweeps may not scale linearly.
Benchmarks should state this.

## Toolchain

| Tool | Version |
|---|---|
| rustup | 1.29.3 |
| rustc / cargo | 1.100.0 (stable-x86_64-pc-windows-msvc) |
| clippy / rustfmt | 0.1.100 / 1.11.0 |
| MSVC | Visual Studio Community 2022, C++ tools |
| Windows SDK | 10.0.26100 (needed by the MSVC linker) |
| Python | 3.13.5 (venv in `.venv/`) |
| maturin | 1.16.0 |
| torch | 2.14.1+cu130 (CUDA 13.0, `torch.cuda.is_available()` = True) |
| torchvision | 0.29.1+cu130 |
| numpy | 2.5.4 |
| pillow | 13.0.0 (libjpeg-turbo 3.2.0) |
| psutil | 7.2.2 |
| pytest | 9.2.0 |

## Reproduce

```powershell
winget install --id Rustlang.Rustup -e
winget install --id Microsoft.WindowsSDK.10.0.26100 -e
rustup default stable
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install maturin pytest numpy pillow psutil
```

Check:

```powershell
rustc --version
.\.venv\Scripts\maturin.exe --version
.\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.cuda.is_available())"
```

## Gotcha

rustup warned "installing msvc toolchain without its prerequisites": VS 2022 had the C++
tools but no Windows SDK, so linking failed until the SDK was installed. An interrupted
rustup install also left the toolchain without a manifest; fixed with
`rustup toolchain uninstall stable` then `rustup toolchain install stable`.

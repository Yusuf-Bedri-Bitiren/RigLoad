# rigload

A Rust-core Python data loader for synchronized RGB-D samples, with geometric augmentations that keep RGB, depth and camera intrinsics consistent.

**Status:** early development (v0.0.x). Nothing is benchmarked yet; see [docs/PROGRESS.md](docs/PROGRESS.md).

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install maturin
.\.venv\Scripts\maturin.exe develop --extras test
.\.venv\Scripts\python.exe -m pytest
```

Synthetic RGB-D + LiDAR data (seeded; a 20-sample set is committed in `tests/fixtures/synthetic/`):

```powershell
.\.venv\Scripts\python.exe tools\gen_synthetic.py --out data\synthetic --n 1000 --seed 0
```

Full environment details: [docs/SETUP.md](docs/SETUP.md).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

# rigload

A Rust-core Python data loader for synchronized RGB-D samples, with geometric augmentations that keep RGB, depth and camera intrinsics consistent.

**Status:** early development (v0.0.x). Nothing is benchmarked yet; see [docs/PROGRESS.md](docs/PROGRESS.md).

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install maturin pytest
.\.venv\Scripts\maturin.exe develop
.\.venv\Scripts\python.exe -m pytest
```

Full environment details: [docs/SETUP.md](docs/SETUP.md).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

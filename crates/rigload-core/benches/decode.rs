//! Per-image decode time, single-threaded. Compare with bench/decode_compare.py (Pillow etc.).
//!
//! Run: cargo bench -p rigload-core --bench decode
//! Uses the committed fixtures; the files are read once by the OS cache, so this measures
//! decode, not disk.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use rigload_core::{decode_depth, decode_rgb, read_manifest};

fn decode(c: &mut Criterion) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/synthetic/manifest.jsonl");
    let rows = read_manifest(&manifest).expect("fixtures manifest");

    // Cycle over all 20 fixtures so one easy image doesn't dominate.
    let mut i = 0;
    c.bench_function("decode_rgb_jpeg_1242x375", |b| {
        b.iter(|| {
            i = (i + 1) % rows.len();
            black_box(decode_rgb(&rows[i].rgb).unwrap())
        })
    });
    c.bench_function("decode_depth_png16_dense_1242x375", |b| {
        b.iter(|| {
            i = (i + 1) % rows.len();
            black_box(decode_depth(&rows[i].depth).unwrap())
        })
    });
    c.bench_function("decode_depth_png16_sparse_1242x375", |b| {
        b.iter(|| {
            i = (i + 1) % rows.len();
            black_box(decode_depth(rows[i].depth_sparse.as_ref().unwrap()).unwrap())
        })
    });
}

criterion_group!(benches, decode);
criterion_main!(benches);

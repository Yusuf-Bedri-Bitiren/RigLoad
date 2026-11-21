# Benchmark results

Every number here comes from the JSON files in this folder (2026-11-20, `bench/run_overnight.ps1`,
code at commit `af5d013`; the JSONs say `-dirty` only because the run's own new result files
were untracked).

**Machine:** Intel i5-12500H (4 P + 8 E cores, 16 threads), 15.7 GiB RAM, Windows 11,
plugged in, 6.7–7.3 GiB RAM available at start. Python 3.13.5, torch 2.14.1+cu130 (CPU only
here), torchvision 0.29.1, albumentationsx 2.6.6, OpenCV 5.1.
**Data:** synthetic bench set, 10,000 samples, 1242×375 RGB JPEG (q90, 4:4:4) + 16-bit depth
PNG, warm page cache.

## Decode, single-threaded (`decode.json`, Step 4)

300 files per kind, 5 passes, median ms per image (lower is better).

| file | rigload (Rust) | Pillow | torchvision.io | OpenCV | rigload vs fastest other |
|---|---:|---:|---:|---:|---:|
| RGB JPEG | 2.77 | 3.63 | **2.22** | 3.05 | 0.80× (slower) |
| depth PNG16, dense | **1.41** | 1.71 | 2.41 | 2.55 | 1.21× |
| depth PNG16, sparse | **0.52** | 0.97 | 1.78 | 1.83 | 1.85× |
| **RGB + dense depth** | **4.18** | 5.34 | 4.63 | 5.60 | 1.11× |

Criterion (pure Rust, fixtures): JPEG 2.81 ms, dense PNG16 1.47 ms, sparse 0.34 ms. The
Python-binding numbers match these, so the binding adds no measurable overhead.

## Baseline loaders: DataLoader `num_workers` sweep (Step 3)

Batch 32, 7 repeats × 200 timed batches, warm-up max(10, 2 × workers). samples/s is the median
(min–max). CPU %: 100 = one logical core. Peak RSS: summed over all processes (upper bound).

### With augmentation (crop → resize 640×192 → hflip → colour jitter, K updated)

**torchvision v2** (`torchvision_aug.json`)

| workers | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 0 | 108 (100–109) | 294 | 349 | 1196 | 687 |
| 2 | 207 (206–208) | 154 | 305 | 212 | 1992 |
| 4 | 389 (386–391) | 39 | 300 | 414 | 3318 |
| 6 | 454 (431–458) | 3 | 402 | 619 | 4622 |
| 8 | 464 (437–477) | 3 | 548 | 817 | 5914 |

**Albumentations kernels** (`albumentations_aug.json`)

| workers | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 0 | 76 (76–77) | 423 | 455 | 631 | 640 |
| 2 | 202 (195–204) | 159 | 318 | 212 | 1851 |
| 4 | 338 (313–361) | 25 | 386 | 414 | 3061 |
| 6 | 430 (428–445) | 3 | 438 | 614 | 4282 |
| 8 | **532 (509–536)** | 3 | 439 | 817 | 5505 |

### Without augmentation: decode + collate at full resolution (`torchvision_noaug.json`)

| workers | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 0 | 119 (118–122) | 276 | 303 | 949 | 785 |
| 2 | 297 (267–300) | 109 | 233 | 221 | 2270 |
| 4 | 483 (453–502) | 14 | 284 | 436 | 3723 |
| 6 | 601 (562–647) | 8 | 312 | 646 | 5113 |
| 8 | see note: 6 of 7 repeats at **725–757**, then RAM < 1.5 GB | | | | |

**Note on 8 workers:** `overnight.log` shows 6 completed repeats (740, 757, 744, 740, 725, 754
samples/s) before the memory guard stopped the 7th. The harness then discarded them, a bug
fixed since: completed repeats are now kept and marked. Latency, CPU and RSS for this setting
were lost. A re-run on 2026-11-21 (`torchvision_noaug_w8.json`, `step5_torch_w8.log`) failed the
same way: RAM fell below 1.5 GB during warm-up, Windows reported "paging file too small"
(WinError 1455), and the Claude Code session running on the laptop crashed. So 8 workers
stays log-only on this machine; a full run needs the laptop with nothing else open.

## Step 5: rigload `Loader`, no augmentation (`rigload_noaug.json`, 2026-11-21)

Same data, batch size and protocol as above; `num_threads` rayon threads + 1 producer thread,
`prefetch=2`, `drop_last=True`. Code: Step 5 working tree on top of `af5d013`.

| threads | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 1 | 146 (76–147) | 218 | 288 | 119 | 667 |
| 2 | 338 (336–339) | 95 | 104 | 189 | 669 |
| 4 | 513 (472–518) | 63 | 73 | 290 | 673 |
| 8 | 560 (512–614) | 55 | 69 | 427 | 677 |
| 12 | 624 (617–627) | 51 | 66 | 525 | 666 |
| 16 | **686 (665–694)** | 47 | 68 | 578 | 669 |

**Comparison with the torchvision no-augmentation baseline:**

| | rigload, 16 threads | torchvision, 6 workers | torchvision, 8 workers |
|---|---:|---:|---:|
| samples/s | **686** | 601 | ~745 (log only) |
| rigload relative | — | 1.14× faster | **0.92× (8% slower)** |
| CPU % | 578 | 646 | not measured |
| samples/s per 100 % CPU | **119** | 93 | not measured |
| peak RSS | **0.67 GB** | 5.1 GB | > 5.1 GB |
| batch p99 ms | **68** | 312 | not measured |

**Reading the Step 5 numbers honestly:**

- **Raw throughput: rigload does not win yet.** Its best (686) is below torchvision's best
  on this laptop (~745 at 8 workers). It beats every torchvision setting that completes with
  full metrics (≤ 6 workers).
- **What rigload does win:** 7.6× less memory, since threads share one process instead of each
  worker loading torch. That is also why rigload can use 16 threads where torchvision crashes
  above 8 workers. It also uses less CPU per sample and has 4.6× lower p99 batch latency.
- **p50 is higher than torchvision's** (47 vs 8 ms). rigload's consumer is faster than its
  producer, so it waits for each batch. torchvision's p50 is low because its workers prefetch,
  while its slow batches show up in p99 instead.
- **Scaling is uneven.** 1 → 2 threads more than doubles throughput, while 4 → 8 adds only 9%.
  At 1 thread, 6.8 ms per sample is well above the 4.18 ms decode cost from Step 4, so
  about 2.7 ms per sample goes somewhere other than decode. Unverified suspects: a fresh
  ~2.3 MB allocation per sample (Windows zero-fills each new page), the copy in `collate`, and
  the 12500H's E-cores. Not profiled yet.
- The 1-thread repeats 4–6 (76–109) are outliers; the other repeats are 146–147.

## Numbers to beat

| setting | best baseline | at | cost |
|---|---|---|---|
| with augmentation | **532 samples/s** (Albumentations) | 8 workers | 817 % CPU, 5.5 GB RSS |
| with augmentation, torchvision | 464 samples/s | 8 workers | 817 % CPU, 5.9 GB RSS |
| no augmentation | **~745 samples/s** (median of 6 repeats, log only; re-run pending) | 8 workers | near this laptop's RAM limit |
| no augmentation, full metrics | 601 samples/s | 6 workers | 646 % CPU, 5.1 GB RSS |
| *rigload so far, no augmentation* | *686 samples/s* | *16 threads* | *578 % CPU, 0.67 GB RSS* |

## Reading notes

- **Memory grows ~650 MB per worker**, because each worker is a process that loads torch. On this
  15.7 GiB laptop that caps the sweep at 6–8 workers. 8 workers without augmentation reached the
  limit on its last repeat (full-resolution batches are ~75 MB each). An earlier 12-worker
  attempt froze the desktop.
- **`num_workers=0` uses 6–12 cores anyway** (631–1196 % CPU): torch's intra-op threads and
  OpenMP spin-waiting in the main process.
- **Batch p50 drops to 3 ms at 6–8 workers** because batches are prefetched. The consumer then
  mostly waits on the slowest worker, which shows up as p99 of 400–550 ms.
- Synthetic depth PNGs compress much better than real ones, so depth decode is cheaper here
  than on real data (Step 8 checks this).
- The Albumentations baseline calls the kernels directly (no `Compose` overhead), so it is if
  anything optimistic.

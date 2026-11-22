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
`prefetch=2`, `drop_last=True`. Code: Step 5 working tree on top of `af5d013`. Windows power
throttling disabled for the process (`power_throttling_disabled: true` in the JSON); 3.8 GiB
RAM available at start (rigload needs < 0.7 GB).

| threads | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 1 | 185 (181–194) | 171 | 207 | 115 | 603 |
| 2 | 376 (372–384) | 84 | 103 | 212 | 609 |
| 4 | 691 (647–697) | 46 | 60 | 401 | 621 |
| 8 | 990 (961–998) | 32 | 46 | 740 | 638 |
| 12 | **1006 (943–1084)** | 31 | 54 | 1022 | 654 |
| 16 | 978 (927–1095) | 29 | 104 | 1139 | 697 |

**Comparison with the torchvision no-augmentation baseline:**

| | rigload, 8 threads | torchvision, 6 workers | torchvision, 8 workers |
|---|---:|---:|---:|
| samples/s | **990** | 601 | ~745 (log only) |
| rigload relative | — | **1.65×** | **1.33×** |
| CPU % | 740 | 646 | not measured |
| peak RSS | **0.64 GB** | 5.1 GB | > 5.1 GB |
| batch p99 ms | **46** | 312 | not measured |

**How we got here: the first run lost, profiling found why.** The first sweep
(`rigload_noaug_serial_collate.json`) peaked at 686 samples/s at 16 threads, 8% *below*
torchvision's ~745. Timing inside `load_batch` showed decode taking 19 ms per batch at
16 threads, and the copy into the batch buffer (`collate`) taking **25 ms on a single thread
at every thread count**. That copy is 75 MB per batch, most of the cost being first-touch page
faults on a fresh buffer. The fix: each worker copies its own sample into its slot of the
batch buffer, so the copy runs in parallel. Same machine, before → after:

| threads | 1 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| serial collate | 146 | 513 | 560 | 686 |
| parallel copy | 185 | 691 | 990 | 978 |

**Reading the numbers honestly:**

- **At equal parallelism (8), rigload is 1.33× torchvision's best**, using 8× less memory.
  The 8-worker torchvision number comes from the log of a run the memory guard stopped
  (6 of 7 repeats), so its CPU and latency are unknown.
- **It plateaus at about 1000 samples/s from 8 threads up.** The 12500H has 4 P-cores (8
  hyper-threads) and 8 slower E-cores, so threads 9–16 add little. 16 threads has a worse p99
  (104 ms) than 8 (46 ms).
- **Memory is where processes vs. threads shows most:** about 0.64 GB at every thread count,
  vs. ~0.65 GB per extra DataLoader worker.
- **Caveats:** no augmentation yet (Step 6 adds it; torchvision's augmented best is 464).
  The torchvision runs did not have the power-throttling opt-out, but they ran in a visible
  console window, which Windows does not throttle. Not re-verified.
- **A run we discarded:** a sweep launched without a visible window and without the opt-out
  got 94 samples/s at 1 thread, then jumped mid-run (580 → 1116 at 12 threads). Windows 11
  power throttling had moved it onto E-cores. `bench.py` now opts out on Windows for itself
  and its DataLoader workers.

## Step 6: rigload with augmentation (`rigload_aug.json`, 2026-11-22)

Same transforms and parameter ranges as the baselines: crop (scale 0.8–1) → resize to
640×192 (RGB antialiased bilinear, depth nearest-exact) → flip (p = 0.5) → brightness,
contrast, saturation ±0.2, K updated. Plugged in, power throttling disabled.

| threads | samples/s | batch p50 ms | batch p99 ms | CPU % | peak RSS MB |
|---:|---:|---:|---:|---:|---:|
| 1 | 145 (144–146) | 220 | 244 | 112 | 548 |
| 2 | 299 (298–299) | 107 | 114 | 211 | 554 |
| 4 | 564 (533–576) | 57 | 64 | 400 | 565 |
| 8 | 811 (801–827) | 40 | 51 | 743 | 577 |
| 12 | 977 (972–979) | 33 | 44 | 1024 | 590 |
| 16 | **1121 (1062–1142)** | 28 | 52 | 1380 | 605 |

**Comparison with the augmented baselines (all at their best setting, 8 workers):**

| | rigload, 8 threads | rigload, 16 threads | Albumentations, 8 workers | torchvision, 8 workers |
|---|---:|---:|---:|---:|
| samples/s | 811 | **1121** | 532 | 464 |
| vs Albumentations | 1.52× | **2.11×** | — | 0.87× |
| vs torchvision | 1.75× | **2.42×** | 1.15× | — |
| CPU % | 743 | 1380 | 817 | 817 |
| samples/s per 100 % CPU | **109** | 81 | 65 | 57 |
| peak RSS | **0.58 GB** | 0.61 GB | 5.5 GB | 5.9 GB |
| batch p99 ms | **51** | 52 | 439 | 548 |

**Reading the numbers honestly:**

- **At equal parallelism (8), rigload is 1.52× Albumentations and 1.75× torchvision**, with
  about 10× less memory and 1.7–1.9× the throughput per unit of CPU.
- **The 16-thread figure (2.11×) is not a like-for-like comparison:** the baselines could not
  run more than 8 workers on this laptop (memory), so it partly shows what threads allow on a
  memory-limited machine, not a faster kernel.
- **The gap is wider with augmentation than without** (1.33× → 1.52–1.75× at 8). The Python
  baselines pay per-sample interpreter and tensor overhead for every transform step; rigload
  runs them in one native pass per sample.
- **Augmented is faster than non-augmented at 16 threads** (1121 vs 978): the 640×192 output
  is 3.8× smaller than the full-resolution batch, so there is less to copy and hand over.
- **Correctness for these same parameters** (`tests/test_transforms.py`): K and depth are
  identical to the torchvision baseline; RGB differs by at most 4 levels (mean 0.095).
- A first run of this sweep on battery power (`rigload_aug_battery.json`, not used) gave
  similar numbers from 2 threads up, but 76–125 at 1 thread.

## Step 11: augmentation + LiDAR (2026-11-22)

Same augmentation as Step 6, plus per sample: read the KITTI `.bin` (~106k points, 1.7 MB),
flip/yaw/scale the points, update `T_cam_lidar`, re-project sparse depth with K′ and T′.
torchvision baseline: the same math in NumPy (`bench/common.py`), tested for the same
consistency (`tests/test_lidar.py`). `--samples 3000` (5.2 GB incl. LiDAR, warmed first, so it
stays in RAM), 100 timed batches × 7 repeats, plugged in, throttling off. Files:
`rigload_aug_lidar.json`, `torchvision_aug_lidar.json`.

| workers / threads | rigload samples/s | torchvision samples/s | rigload RSS | torchvision RSS |
|---:|---:|---:|---:|---:|
| 0 / 1 | **83** (78–86) | 31 (30–32) | 0.61 GB | 0.82 GB |
| 2 | **180** (166–195) | 65 (60–68) | 0.61 GB | 2.35 GB |
| 4 | **339** | stopped: RAM below 1.5 GB | 0.62 GB | – |
| 8 | **470** (438–515) | – | 0.64 GB | – |
| 16 | **581** (515–620) | – | 0.66 GB | – |

**Reading the numbers honestly:**

- **At equal parallelism, rigload is 2.7–2.8× torchvision** (1 vs 0, 2 vs 2), at 3.8× less
  memory and ~3.3× the throughput per unit of CPU (2 vs 2: 192 % vs 229 % CPU).
- **torchvision could not go past 2 workers in this run:** only 4.5 GB RAM was free at start
  (other apps open), and the memory guard stopped the 4-worker sweep. On a machine with more
  free RAM it would scale further, so **581 vs 65 (8.9×) is not a like-for-like speedup**;
  what it does show is that rigload's threads need ~0.6 GB where torchvision's processes
  need ~0.75 GB each.
- **LiDAR costs rigload 42–48 %** of its Step 6 camera-only throughput (470 vs 811 at 8
  threads, 581 vs 1121 at 16). Each sample's points are 1.7 MB, against ~0.1 MB for its RGB
  and depth files together.
- **Two fixes came out of this step's first run** (kept as `*_10k_unfused.json`,
  `*_10k_partial.json`): (1) the full 10k set is 16.5 GB, more than this laptop's RAM, so it
  measured the disk, and only rigload was fast enough to be held back; (2) points went through
  5 passes per sample (read, parse copy, transform, project, concat copy). Reading each `.bin`
  straight into its slice of the batch buffer removed two passes and all per-sample buffers.
  Together: 194 → 470 samples/s at 8 threads, 275 → 581 at 16.
- The NumPy projection was checked for a faster form (`np.minimum.at`): identical output, no
  faster, so the baseline is not handicapped by it.

## Numbers to beat

| setting | best baseline | at | cost |
|---|---|---|---|
| with augmentation | **532 samples/s** (Albumentations) | 8 workers | 817 % CPU, 5.5 GB RSS |
| with augmentation, torchvision | 464 samples/s | 8 workers | 817 % CPU, 5.9 GB RSS |
| no augmentation | **~745 samples/s** (median of 6 repeats, log only; re-run pending) | 8 workers | near this laptop's RAM limit |
| no augmentation, full metrics | 601 samples/s | 6 workers | 646 % CPU, 5.1 GB RSS |
| *rigload (Step 5), no augmentation* | *990 samples/s* | *8 threads* | *740 % CPU, 0.64 GB RSS* |
| *rigload (Step 6), with augmentation* | *811 / 1121 samples/s* | *8 / 16 threads* | *743 / 1380 % CPU, 0.58 / 0.61 GB RSS* |
| with augmentation + LiDAR, torchvision | 65 samples/s | 2 workers (4 stopped: RAM) | 229 % CPU, 2.35 GB RSS |
| *rigload (Step 11), augmentation + LiDAR* | *180 / 470 / 581 samples/s* | *2 / 8 / 16 threads* | *0.61–0.66 GB RSS* |

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

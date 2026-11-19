# Prior art

Checked 2026-11-19 (Step 1). Sources are public docs, READMEs and papers. Speed figures
below are **the projects' own claims**, not measured by us.

## Verdict

No tool found that combines all four:

1. a native-code (non-Python) multi-threaded loader,
2. joint RGB + depth geometric augmentation with **one shared random draw**,
3. **built-in** camera-intrinsics updates (K follows crop / resize / flip),
4. depth-correct resampling (never mixing valid and invalid pixels), with tests that
   prove projection consistency.

Each piece exists somewhere. The gap is the combination, plus verified correctness.
**Updating K by hand in a custom dataset is common practice in research code**, so
"calibration-aware" is not a new idea. The defensible claim is *native speed + built-in,
property-tested consistency*, not novelty of the geometry.

## Per tool

| Tool | What it does (as documented) | Overlap with our angle | Gap left for us |
|---|---|---|---|
| [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) | Multi-process workers | Baseline | — |
| [torchvision transforms v2](https://pytorch.org/vision/stable/transforms.html) | Joint transforms over `tv_tensors`: Image, Video, BoundingBoxes, Mask, KeyPoints | Same random params for image + mask (depth as Mask) | No depth or intrinsics type; K must be updated manually. Depth was listed only as a *future* use case in the v2 announcement |
| [Albumentations](https://albumentations.ai/docs/releases/2.0.20) / [AlbumentationsX paper](https://arxiv.org/pdf/2608.11123) | Joint image + mask + bbox + keypoint pipelines (OpenCV core) | **Closest functionally.** Depth via `additional_targets={"depth": "mask"}`; 2.0.20 added `user_data` so a custom transform can shift cx, cy after a crop | Python-level orchestration; K handling is user-written per transform. The paper recommends *linear* interpolation for depth, which mixes valid and invalid pixels on sparse (LiDAR) depth. **VERIFY** AlbumentationsX license (believed AGPL/commercial) |
| [nvTorchCam](https://github.com/NVlabs/nvTorchCam) ([paper](https://arxiv.org/abs/2410.12074)) | Differentiable camera-model library (pinhole, fisheye, equirect) in PyTorch, Apache-2.0 | Crop updates intrinsics; flip via negated focal or flipped extrinsics; random-crop utilities | A geometry library, not a loader; GPU tensor ops. Resize helper not confirmed |
| [Kornia](https://kornia.readthedocs.io/en/stable/augmentation.container.html) | GPU augmentation; `AugmentationSequential` with `input`, `mask`, `bbox`, `keypoints`; returns the chained transform matrix | K could be derived from the returned matrix | No depth or intrinsics data key; runs after loading, on tensors |
| [kornia-rs](https://arxiv.org/abs/2505.12425) | Rust CV library, PyO3 bindings, Apache-2.0; image I/O, SIMD resize | Same stack (Rust + PyO3) | Low-level ops, not a loader; no calibration-aware augmentation found |
| [dataio-rs](https://pypi.org/project/dataio-rs/) (v0.5.6) | Rust/PyO3 loader: native worker pools, S3/GCS/HF IO, slot-based decode/transform/batch, DLPack/NumPy/torch output, free-threaded CPython, Apache-2.0 | **Closest stack.** Strong on general IO and throughput | README documents no depth, intrinsics, or cross-slot shared random params. Wheels Linux x86_64 only. The "~1.85–2x vs DataLoader" figure in our notes came from an older version; the current README doesn't state it. The GitHub repo returned 404 on 2026-11-19 |
| [SPDL](https://facebookresearch.github.io/spdl/main) ([paper](https://arxiv.org/abs/2504.20067)) | Thread-based pipeline; GIL work kept to the main + scheduler threads, GIL-releasing ops in a pool. BSD-2 | "Threads beat processes" is published: 74% faster than DataLoader on ImageNet, 38% less CPU, 50 GB less memory (v1 abstract); +33% on 3.13t (v2) | General-purpose; no calibration logic |
| [NVIDIA DALI](https://github.com/NVIDIA/DALI) | GPU decode + augmentation, C++ core, auto-augment | Strongest general speed | Nothing found on depth or intrinsics handling |
| [MMDetection3D](https://mmdetection3d.readthedocs.io/en/v0.17.1/_modules/mmdet3d/datasets/pipelines/transforms_3d.html) | Python 3D pipelines; `RandomFlip3D(sync_2d=...)` syncs point/image flips; "transformation flow" + MoCa ([paper](https://arxiv.org/abs/2012.12741)) for multi-modal consistency | Points ↔ image consistency (relevant to Step 11) | Python-level speed; detection-specific |
| [FFCV](https://github.com/libffcv/ffcv), [WebDataset](https://github.com/webdataset/webdataset) | Pre-converted format / tar shards | Different trade-offs (conversion step, sequential I/O) | Not re-read in depth; no calibration features known |

## Positioning (confirmed by the user 2026-11-19)

Main angle: **LiDAR ↔ camera consistency** (points, image, depth, K, extrinsics) at native
speed, built on the camera pipeline. Claims follow measured data; pivot if they don't hold.

**Not yet checked:** LiDAR-side tools (OpenPCDet, Open3D-ML, nuScenes devkit, MMDetection3D
in depth). Do this pass before Step 11, with the same stop-and-discuss rule.

## Findings from building the baselines (Step 3, 2026-11-20)

- **torchvision `tv_tensors.Mask` resize uses legacy `NEAREST` by default** (any other mode
  is silently coerced to it; `NEAREST_EXACT` is allowed if asked for). Legacy nearest picks
  pixels up to ~1 source pixel away from where the RGB's bilinear resize and the updated K
  say they come from. On our fixtures it doubles the median LiDAR-on-depth reprojection error
  (0.25 % → 0.48 %). Documented by `tests/test_baselines.py`. Evidence that "depth as a mask"
  is not calibration-correct out of the box.
- **torchvision.io image decoding is deprecated since 0.29** in favour of TorchCodec.
- **AlbumentationsX** (the maintained fork) is AGPL-3.0; its `Compose` defaults to
  `telemetry=True`; its applied-params record has no crop coordinates, so K can't be
  recovered from a random `Compose`.

## Implications

- **README claim** (draft): "A Rust data loader for RGB-D training where crop, resize and
  flip keep RGB, depth and camera intrinsics consistent by construction, verified by
  property tests." Don't call it the first.
- **Baselines** for credibility: DataLoader + torchvision v2 (planned). Consider adding
  DataLoader + Albumentations, since it is the most likely tool a practitioner already
  uses for RGB-D augmentation.
- **Depth policy** is a real differentiator: Albumentations' published advice (linear)
  is wrong for sparse depth. Our nearest / sparse-aware choice (Step 6) should be shown
  with a small visual or numeric example.
- **dataio-rs** is the speed competitor to watch at Step 9; it only ships Linux wheels,
  so compare on Linux (HPC or WSL).

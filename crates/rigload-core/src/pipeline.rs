//! Parallel loading: a producer thread walks the epoch's batches, decodes each batch's
//! samples in parallel on a rayon pool, collates them into contiguous buffers, and sends the
//! batch through a bounded channel. The channel's capacity is the prefetch depth: the producer
//! runs ahead of the consumer by at most that many batches, then blocks.
//!
//! Ownership across threads: the sample list and the pool are shared read-only through `Arc`
//! (reference-counted pointer; the last owner frees it). Each batch is created on the
//! producer thread and *moved* through the channel, so exactly one thread owns it at a time.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::decode::{Sample, load_sample};
use crate::error::{Error, Result};
use crate::lidar::{
    Mat4, augment_lidar, point_count, project_sparse, read_points_into, transform_points,
};
use crate::manifest::SampleMeta;
use crate::rng::epoch_order;
use crate::transform::{AugmentConfig, AugmentParams, Intrinsics, augment_into};

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    /// Batches buffered ahead of the consumer (>= 1).
    pub prefetch: usize,
    pub seed: u64,
    pub shuffle: bool,
    pub drop_last: bool,
    /// None: full-resolution decode only. Some: crop/resize/flip/jitter with K updated.
    pub augment: Option<AugmentConfig>,
    /// Load LiDAR points, apply the 3D augmentation (yaw, scale, flip mirror) and re-project
    /// them into a sparse depth image. Needs `lidar` and `T_cam_lidar` in the manifest.
    pub lidar: bool,
}

/// One collated batch. Buffers are contiguous, row-major:
/// rgb `[B, H, W, 3]`, depth `[B, H, W]`, k `[B][3][3]`, depth_scale `[B]`.
/// With LiDAR: depth_sparse `[B, H, W]`, points `[N, 4]` (x, y, z, intensity, LiDAR frame)
/// for all samples back to back, sample j owning rows `point_offsets[j]..point_offsets[j+1]`,
/// and t_cam_lidar `[B][4][4]`. Otherwise those are empty.
#[derive(Debug)]
pub struct Batch {
    pub indices: Vec<usize>,
    pub ids: Vec<String>,
    pub height: u32,
    pub width: u32,
    pub rgb: Vec<u8>,
    pub depth: Vec<u16>,
    pub k: Vec<Intrinsics>,
    /// Metres = stored depth / depth_scale (per sample: the 3D scale augmentation changes it).
    pub depth_scale: Vec<f64>,
    pub depth_sparse: Vec<u16>,
    pub points: Vec<f32>,
    pub point_offsets: Vec<usize>,
    pub t_cam_lidar: Vec<Mat4>,
}

pub struct Loader {
    samples: Arc<Vec<SampleMeta>>,
    pool: Arc<ThreadPool>,
    cfg: LoaderConfig,
    next_epoch: u64,
}

impl Loader {
    pub fn new(samples: Vec<SampleMeta>, cfg: LoaderConfig) -> Result<Self> {
        if cfg.batch_size == 0 || cfg.num_threads == 0 || cfg.prefetch == 0 {
            return Err(Error::Config(
                "batch_size, num_threads and prefetch must be >= 1".into(),
            ));
        }
        if let Some(a) = &cfg.augment {
            a.validate()?;
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(cfg.num_threads)
            .thread_name(|i| format!("rigload-{i}"))
            .build()
            .map_err(|e| Error::Loader(format!("cannot start thread pool: {e}")))?;
        Ok(Self {
            samples: Arc::new(samples),
            pool: Arc::new(pool),
            cfg,
            next_epoch: 0,
        })
    }

    /// Batches per epoch.
    pub fn len(&self) -> usize {
        let (n, b) = (self.samples.len(), self.cfg.batch_size);
        if self.cfg.drop_last {
            n / b
        } else {
            n.div_ceil(b)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start the next epoch (0, 1, 2, ...), like iterating a PyTorch DataLoader again.
    pub fn epoch(&mut self) -> EpochIter {
        let epoch = self.next_epoch;
        self.next_epoch += 1;
        EpochIter::start(
            Arc::clone(&self.samples),
            Arc::clone(&self.pool),
            self.cfg.clone(),
            epoch,
        )
    }
}

/// Iterator over one epoch's batches. Dropping it (e.g. Python `break`) stops the producer
/// and waits for it to exit, which takes at most one in-flight batch.
pub struct EpochIter {
    // Option so `drop` can drop the receiver *before* joining: that unblocks a producer
    // waiting on a full channel.
    rx: Option<Receiver<Result<Batch>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl EpochIter {
    fn start(
        samples: Arc<Vec<SampleMeta>>,
        pool: Arc<ThreadPool>,
        cfg: LoaderConfig,
        epoch: u64,
    ) -> Self {
        let (tx, rx) = sync_channel(cfg.prefetch);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        // `move` hands ownership of samples, pool, cfg, tx and the flag to the new thread.
        // The compiler only allows this because all of them are `Send` (safe to move
        // between threads); the Arc'd pool and samples are also `Sync` (safe to share).
        let handle = std::thread::Builder::new()
            .name(format!("rigload-producer-e{epoch}"))
            .spawn(move || produce(&samples, &pool, &cfg, epoch, &tx, &stop_flag))
            .expect("failed to spawn producer thread");
        Self {
            rx: Some(rx),
            stop,
            handle: Some(handle),
        }
    }

    fn join_producer(&mut self) -> Result<()> {
        match self.handle.take().map(JoinHandle::join) {
            Some(Err(panic)) => Err(Error::Loader(format!(
                "loader thread panicked: {}",
                panic_message(&*panic)
            ))),
            _ => Ok(()),
        }
    }
}

fn produce(
    samples: &[SampleMeta],
    pool: &ThreadPool,
    cfg: &LoaderConfig,
    epoch: u64,
    tx: &SyncSender<Result<Batch>>,
    stop: &AtomicBool,
) {
    let order = if cfg.shuffle {
        epoch_order(samples.len(), cfg.seed, epoch)
    } else {
        (0..samples.len()).collect()
    };
    for chunk in order.chunks(cfg.batch_size) {
        if stop.load(Ordering::Relaxed) || (cfg.drop_last && chunk.len() < cfg.batch_size) {
            return;
        }
        // `install` runs the closure inside our pool, so par_iter uses its threads.
        let batch = pool.install(|| load_batch(samples, chunk, cfg, epoch));
        let failed = batch.is_err();
        // send blocks while the channel is full (= prefetch); it fails once the consumer
        // has gone away, which is our signal to stop.
        if tx.send(batch).is_err() || failed {
            return;
        }
    }
}

/// Decode a batch in parallel, each worker writing its sample straight into its slot of the
/// contiguous batch buffers. Collating on one thread afterwards cost ~25 ms per batch
/// (75 MB of copies and first-touch page faults), more than the parallel decode at 16 threads.
// shortcut: still one memcpy per sample (decoder buffer -> batch slot); decoding directly into
// the slot needs a decoder API that writes into a caller's buffer (Step 7 / Step 9 ablation).
fn load_batch(
    samples: &[SampleMeta],
    indices: &[usize],
    cfg: &LoaderConfig,
    epoch: u64,
) -> Result<Batch> {
    let first = &samples[indices[0]];
    let (w, h) = match &cfg.augment {
        Some(a) => (a.out_width, a.out_height),
        None => (first.width, first.height),
    };
    // load_sample checks decoded size == manifest size, so equal manifest sizes suffice.
    // With augmentation every sample is resized to the output size, so any size works.
    for &i in indices.iter().filter(|_| cfg.augment.is_none()) {
        let meta = &samples[i];
        if (meta.width, meta.height) != (w, h) {
            return Err(Error::Decode {
                path: meta.rgb.clone(),
                msg: format!(
                    "sample {} is {}x{}, but its batch is {w}x{h}; without a resize \
                     transform all samples must have the same size",
                    meta.id, meta.width, meta.height
                ),
            });
        }
    }
    let (n, px) = (indices.len(), (w * h) as usize);
    let mut rgb = vec![0u8; px * 3 * n];
    let mut depth = vec![0u16; px * n];
    // Points: sized from the file sizes first, so each worker reads its sample's points
    // straight into its slice of the batch buffer (no per-sample buffer, no concat copy).
    let point_lens: Vec<usize> = if cfg.lidar {
        indices
            .par_iter()
            .map(|&i| {
                let meta = &samples[i];
                let path = meta.lidar.as_ref().filter(|_| meta.t_cam_lidar.is_some());
                Ok(point_count(path.ok_or_else(|| missing_lidar(meta))?)? * 4)
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    let mut points = vec![0f32; point_lens.iter().sum()];
    let mut depth_sparse = vec![0u16; if cfg.lidar { px * n } else { 0 }];
    let mut lidar_slots: Vec<Option<LidarSlots>> = (0..n).map(|_| None).collect();
    let mut rest = points.as_mut_slice();
    for ((slot, sparse), &len) in lidar_slots
        .iter_mut()
        .zip(depth_sparse.chunks_mut(px))
        .zip(&point_lens)
    {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        *slot = Some((sparse, head));
        rest = tail;
    }
    let outs: Vec<SampleOut> = rgb
        .par_chunks_mut(px * 3)
        .zip(depth.par_chunks_mut(px))
        .zip(lidar_slots.into_par_iter())
        .zip(indices.par_iter())
        .map(|(((rgb_slot, depth_slot), lidar_slot), &i)| {
            load_one(
                &samples[i],
                i,
                cfg,
                epoch,
                (w, h),
                (rgb_slot, depth_slot, lidar_slot),
            )
        })
        .collect::<Result<_>>()?; // first error wins; it names its file
    let mut point_offsets = Vec::new();
    if cfg.lidar {
        point_offsets.push(0);
        for len in &point_lens {
            point_offsets.push(point_offsets.last().unwrap() + len / 4);
        }
    }
    Ok(Batch {
        indices: indices.to_vec(),
        ids: indices.iter().map(|&i| samples[i].id.clone()).collect(),
        height: h,
        width: w,
        rgb,
        depth,
        k: outs.iter().map(|o| o.k).collect(),
        depth_scale: outs.iter().map(|o| o.depth_scale).collect(),
        depth_sparse,
        points,
        point_offsets,
        t_cam_lidar: outs.iter().filter_map(|o| o.t_cam_lidar).collect(),
    })
}

/// A sample's outputs that don't live in a fixed-size batch slot.
struct SampleOut {
    k: Intrinsics,
    depth_scale: f64,
    t_cam_lidar: Option<Mat4>,
}

/// A sample's sparse depth image and its points, inside the batch buffers.
type LidarSlots<'a> = (&'a mut [u16], &'a mut [f32]);
type Slots<'a> = (&'a mut [u8], &'a mut [u16], Option<LidarSlots<'a>>);

fn missing_lidar(meta: &SampleMeta) -> Error {
    Error::Config(format!(
        "sample {} has no lidar path or T_cam_lidar in the manifest, but the loader was asked \
         for LiDAR",
        meta.id
    ))
}

fn load_one(
    meta: &SampleMeta,
    index: usize,
    cfg: &LoaderConfig,
    epoch: u64,
    (w, h): (u32, u32),
    (rgb_slot, depth_slot, lidar_slot): Slots<'_>,
) -> Result<SampleOut> {
    let s: Sample = load_sample(meta)?; // an error names its file
    let (sw, sh) = (s.rgb.width, s.rgb.height);
    let mut p = match &cfg.augment {
        Some(a) => AugmentParams::draw(a, cfg.seed, epoch, index as u64, sw, sh),
        None => AugmentParams::identity(sw, sh),
    };
    if !cfg.lidar {
        (p.yaw, p.scale) = (0.0, 1.0); // 3D augmentation only applies with points
    }
    let k = if cfg.augment.is_some() {
        augment_into(s, &p, w, h, rgb_slot, depth_slot).map_err(|e| Error::Decode {
            path: meta.rgb.clone(),
            msg: e.to_string(),
        })?
    } else {
        rgb_slot.copy_from_slice(&s.rgb.data);
        depth_slot.copy_from_slice(&s.depth.data);
        s.k
    };
    // Metric depth scales by s; the stored values stay, so the scale factor changes instead.
    let depth_scale = meta.depth_scale / p.scale;
    let t_cam_lidar = match lidar_slot {
        None => None,
        Some((sparse, points)) => {
            // load_batch checked both fields when it sized the buffers
            let (path, t) = (meta.lidar.as_ref(), meta.t_cam_lidar);
            let (path, t) = path.zip(t).ok_or_else(|| missing_lidar(meta))?;
            read_points_into(path, points)?;
            let (a, t2) = augment_lidar(&t, &p);
            transform_points(points, &a);
            project_sparse(points, &t2, &k, w, h, depth_scale, sparse);
            Some(t2)
        }
    };
    Ok(SampleOut {
        k,
        depth_scale,
        t_cam_lidar,
    })
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

impl Iterator for EpochIter {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.as_ref()?.recv() {
            Ok(batch) => Some(batch),
            // Channel closed: the producer finished, stopped after an error, or panicked.
            // A panic must not end the epoch silently.
            Err(_) => {
                self.rx = None;
                self.join_producer().err().map(Err)
            }
        }
    }
}

impl Drop for EpochIter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.rx = None; // unblocks a producer stuck in `send`
        let _ = self.join_producer();
    }
}

/// `EpochIter` holds a `Receiver`, which is `Send` but not `Sync` (one thread may use it at a
/// time). Python objects can be touched from several threads, so the binding needs `Sync`:
/// a `Mutex` provides it by allowing one user at a time.
pub type SharedEpochIter = Mutex<EpochIter>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::read_manifest;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    fn fixtures() -> Vec<SampleMeta> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/synthetic/manifest.jsonl");
        read_manifest(&p).unwrap()
    }

    fn cfg(batch_size: usize, drop_last: bool) -> LoaderConfig {
        LoaderConfig {
            batch_size,
            num_threads: 4,
            prefetch: 2,
            seed: 7,
            shuffle: true,
            drop_last,
            augment: None,
            lidar: false,
        }
    }

    fn epoch_indices(loader: &mut Loader) -> Vec<Vec<usize>> {
        loader.epoch().map(|b| b.unwrap().indices).collect()
    }

    #[test]
    fn every_sample_once_per_epoch() {
        let mut loader = Loader::new(fixtures(), cfg(6, false)).unwrap();
        assert_eq!(loader.len(), 4);
        let batches = epoch_indices(&mut loader);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            [6, 6, 6, 2]
        );
        let mut all: Vec<usize> = batches.concat();
        all.sort_unstable();
        assert_eq!(all, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn drop_last_skips_the_partial_batch() {
        let mut loader = Loader::new(fixtures(), cfg(6, true)).unwrap();
        assert_eq!(loader.len(), 3);
        assert_eq!(epoch_indices(&mut loader).len(), 3);
    }

    #[test]
    fn same_seed_same_order_new_epoch_new_order() {
        let mut a = Loader::new(fixtures(), cfg(5, false)).unwrap();
        let mut b = Loader::new(fixtures(), cfg(5, false)).unwrap();
        let (a0, b0) = (epoch_indices(&mut a), epoch_indices(&mut b));
        assert_eq!(a0, b0);
        assert_ne!(a0, epoch_indices(&mut a)); // epoch 1
    }

    #[test]
    fn batch_buffers_hold_the_decoded_samples() {
        let samples = fixtures();
        let mut loader = Loader::new(samples.clone(), cfg(3, false)).unwrap();
        let b = loader.epoch().next().unwrap().unwrap();
        let px = (b.width * b.height) as usize;
        assert_eq!(b.rgb.len(), 3 * px * 3);
        for (j, &i) in b.indices.iter().enumerate() {
            let s = load_sample(&samples[i]).unwrap();
            assert_eq!(b.ids[j], samples[i].id);
            assert_eq!(&b.rgb[j * px * 3..(j + 1) * px * 3], &s.rgb.data[..]);
            assert_eq!(&b.depth[j * px..(j + 1) * px], &s.depth.data[..]);
            assert_eq!(b.k[j], samples[i].k);
        }
    }

    #[test]
    fn dropping_mid_epoch_does_not_hang() {
        let mut loader = Loader::new(fixtures(), cfg(2, false)).unwrap();
        let t = Instant::now();
        {
            let mut it = loader.epoch();
            it.next().unwrap().unwrap();
            // producer is now blocked on a full channel (prefetch 2)
            std::thread::sleep(Duration::from_millis(200));
        } // drop here
        assert!(t.elapsed() < Duration::from_secs(5));
        // the loader still works afterwards
        assert_eq!(epoch_indices(&mut loader).len(), 10);
    }

    #[test]
    fn missing_file_surfaces_as_error_with_path() {
        let mut samples = fixtures();
        samples[3].rgb = samples[3].rgb.with_file_name("missing.jpg");
        let mut c = cfg(20, false);
        c.shuffle = false;
        let mut loader = Loader::new(samples, c).unwrap();
        let mut it = loader.epoch();
        let err = it.next().unwrap().unwrap_err();
        assert!(err.to_string().contains("missing.jpg"), "{err}");
        assert!(it.next().is_none(), "epoch must end after an error");
    }

    #[test]
    fn mixed_sizes_in_a_batch_are_an_error() {
        let mut samples = fixtures();
        samples[1].width += 2;
        let mut c = cfg(4, false);
        c.shuffle = false;
        let err = Loader::new(samples, c).unwrap().epoch().next().unwrap();
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("without a resize"), "{msg}");
    }

    #[test]
    fn augmented_batches_match_single_sample_transform() {
        let samples = fixtures();
        let mut c = cfg(5, false);
        c.augment = Some(AugmentConfig::default());
        c.seed = 11;
        let mut loader = Loader::new(samples.clone(), c.clone()).unwrap();
        let mut it = loader.epoch();
        let _ = it.next(); // epoch 0, batch 0
        let b = it.next().unwrap().unwrap(); // batch 1: indices depend on the shuffle
        assert_eq!((b.width, b.height), (640, 192));
        let px = 640 * 192;
        let a = c.augment.as_ref().unwrap();
        for (j, &i) in b.indices.iter().enumerate() {
            let m = &samples[i];
            let p = AugmentParams::draw(a, 11, 0, i as u64, m.width, m.height);
            let (mut rgb, mut depth) = (vec![0; px * 3], vec![0; px]);
            let k =
                augment_into(load_sample(m).unwrap(), &p, 640, 192, &mut rgb, &mut depth).unwrap();
            assert_eq!(b.rgb[j * px * 3..(j + 1) * px * 3], rgb[..]);
            assert_eq!(b.depth[j * px..(j + 1) * px], depth[..]);
            assert_eq!(b.k[j], k);
        }
    }

    /// Without augmentation the re-projected sparse depth must reproduce the generator's stored
    /// depth_sparse PNG (Python, tools/gen_synthetic.py), up to its 80 m cutoff.
    #[test]
    fn lidar_without_augmentation_matches_the_stored_sparse_depth() {
        let samples = fixtures();
        let mut c = cfg(4, false);
        (c.shuffle, c.lidar) = (false, true);
        let b = Loader::new(samples.clone(), c)
            .unwrap()
            .epoch()
            .next()
            .unwrap()
            .unwrap();
        let px = (b.width * b.height) as usize;
        assert_eq!(b.point_offsets.len(), 5);
        for (j, &i) in b.indices.iter().enumerate() {
            let m = &samples[i];
            let raw = crate::lidar::decode_points(m.lidar.as_ref().unwrap()).unwrap();
            let (lo, hi) = (b.point_offsets[j] * 4, b.point_offsets[j + 1] * 4);
            assert_eq!(hi - lo, raw.len());
            // T⁻¹·T is identity only to ~1e-16, so values may move by one float32 ulp
            for (&a, &r) in b.points[lo..hi].iter().zip(&raw) {
                assert!((a - r).abs() <= 1e-6 * r.abs().max(1.0), "{a} vs {r}");
            }
            assert_eq!(b.t_cam_lidar[j], m.t_cam_lidar.unwrap());
            let stored = crate::decode::decode_depth(m.depth_sparse.as_ref().unwrap()).unwrap();
            let ours = &b.depth_sparse[j * px..(j + 1) * px];
            let cutoff = (80.0 * m.depth_scale) as u16;
            let mut compared = 0;
            for (&o, &s) in ours.iter().zip(&stored.data) {
                if o <= cutoff || s > 0 {
                    assert_eq!(o, s);
                    compared += (s > 0) as usize;
                }
            }
            assert!(compared > 1000, "only {compared} LiDAR pixels compared");
        }
    }

    #[test]
    fn missing_lidar_is_a_config_error() {
        let mut samples = fixtures();
        samples[0].lidar = None;
        let mut c = cfg(2, false);
        (c.shuffle, c.lidar) = (false, true);
        let err = Loader::new(samples, c)
            .unwrap()
            .epoch()
            .next()
            .unwrap()
            .unwrap_err();
        assert!(err.to_string().contains("no lidar path"), "{err}");
    }

    #[test]
    fn invalid_config_is_rejected() {
        assert!(Loader::new(fixtures(), cfg(0, false)).is_err());
        let mut c = cfg(4, false);
        c.augment = Some(AugmentConfig {
            out_width: 0,
            ..AugmentConfig::default()
        });
        assert!(Loader::new(fixtures(), c).is_err());
    }
}

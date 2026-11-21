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
use crate::manifest::SampleMeta;
use crate::rng::epoch_order;

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    /// Batches buffered ahead of the consumer (>= 1).
    pub prefetch: usize,
    pub seed: u64,
    pub shuffle: bool,
    pub drop_last: bool,
}

/// One collated batch. Buffers are contiguous, row-major:
/// rgb `[B, H, W, 3]`, depth `[B, H, W]`, k `[B][3][3]`.
#[derive(Debug)]
pub struct Batch {
    pub indices: Vec<usize>,
    pub ids: Vec<String>,
    pub height: u32,
    pub width: u32,
    pub rgb: Vec<u8>,
    pub depth: Vec<u16>,
    pub k: Vec<[[f64; 3]; 3]>,
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
            return Err(Error::Loader(
                "batch_size, num_threads and prefetch must be >= 1".into(),
            ));
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
        let batch = pool.install(|| load_batch(samples, chunk));
        let failed = batch.is_err();
        // send blocks while the channel is full (= prefetch); it fails once the consumer
        // has gone away, which is our signal to stop.
        if tx.send(batch).is_err() || failed {
            return;
        }
    }
}

fn load_batch(samples: &[SampleMeta], indices: &[usize]) -> Result<Batch> {
    let decoded: Vec<Sample> = indices
        .par_iter()
        .map(|&i| load_sample(&samples[i]))
        .collect::<Result<_>>()?; // first error wins; it names its file
    collate(samples, indices, decoded)
}

/// Copy the samples into one contiguous buffer per modality.
// shortcut: one extra memcpy per sample (decode buffer -> batch buffer); Step 7 may decode
// straight into the batch buffer if this shows up in profiles.
fn collate(samples: &[SampleMeta], indices: &[usize], decoded: Vec<Sample>) -> Result<Batch> {
    let (w, h) = (decoded[0].rgb.width, decoded[0].rgb.height);
    let px = (w * h) as usize;
    let mut batch = Batch {
        indices: indices.to_vec(),
        ids: Vec::with_capacity(indices.len()),
        height: h,
        width: w,
        rgb: Vec::with_capacity(px * 3 * indices.len()),
        depth: Vec::with_capacity(px * indices.len()),
        k: Vec::with_capacity(indices.len()),
    };
    for (&i, s) in indices.iter().zip(&decoded) {
        let meta = &samples[i];
        if (s.rgb.width, s.rgb.height) != (w, h) {
            return Err(Error::Decode {
                path: meta.rgb.clone(),
                msg: format!(
                    "sample {} is {}x{}, but its batch is {w}x{h}; without a resize \
                     transform all samples must have the same size",
                    meta.id, s.rgb.width, s.rgb.height
                ),
            });
        }
        batch.ids.push(meta.id.clone());
        batch.rgb.extend_from_slice(&s.rgb.data);
        batch.depth.extend_from_slice(&s.depth.data);
        batch.k.push(s.k);
    }
    Ok(batch)
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
    fn invalid_config_is_rejected() {
        assert!(Loader::new(fixtures(), cfg(0, false)).is_err());
    }
}

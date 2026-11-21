//! PyO3 bindings. Keep this layer thin: logic lives in rigload-core.
//!
//! GIL rule: all Rust work runs inside `py.detach(...)`, which releases the GIL so other
//! Python threads keep running while we read and decode files.

use std::path::PathBuf;
use std::sync::Mutex;

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rigload_core::{Batch, Error, LoaderConfig, SharedEpochIter};

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::Io { .. } => PyIOError::new_err(e.to_string()),
        Error::Loader(_) => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Parallel, prefetching loader over a manifest. Iterating it starts the next epoch
/// (0, 1, 2, ...), like a PyTorch DataLoader.
#[pyclass(module = "rigload")]
struct Loader {
    inner: rigload_core::Loader,
}

#[pymethods]
impl Loader {
    #[new]
    #[pyo3(signature = (manifest, batch_size=32, num_threads=None, prefetch=2, seed=0, shuffle=true, drop_last=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        manifest: PathBuf,
        batch_size: usize,
        num_threads: Option<usize>,
        prefetch: usize,
        seed: u64,
        shuffle: bool,
        drop_last: bool,
    ) -> PyResult<Self> {
        let num_threads = num_threads
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let samples = py
            .detach(|| rigload_core::read_manifest(&manifest))
            .map_err(to_py_err)?;
        let cfg = LoaderConfig {
            batch_size,
            num_threads,
            prefetch,
            seed,
            shuffle,
            drop_last,
        };
        let inner = rigload_core::Loader::new(samples, cfg).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Batches per epoch.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __iter__(&mut self) -> EpochIterator {
        EpochIterator {
            inner: Mutex::new(self.inner.epoch()),
        }
    }
}

/// One epoch. Dropping it (end of a `for` loop, `break`, garbage collection) stops the
/// background threads.
#[pyclass(module = "rigload")]
struct EpochIterator {
    inner: SharedEpochIter,
}

#[pymethods]
impl EpochIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        // shortcut: waiting holds no GIL but ignores Ctrl+C until the next batch arrives
        // (one batch time); add recv_timeout + check_signals if that becomes annoying.
        let next = py.detach(|| {
            self.inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .next()
        });
        match next {
            None => Ok(None), // StopIteration
            Some(Err(e)) => Err(to_py_err(e)),
            Some(Ok(batch)) => batch_dict(py, batch).map(Some),
        }
    }
}

/// {"rgb": (B,H,W,3) uint8, "depth": (B,H,W) uint16, "K": (B,3,3) float64,
///  "ids": list[str], "indices": (B,) int64}. Image buffers are moved, not copied.
fn batch_dict(py: Python<'_>, b: Batch) -> PyResult<Bound<'_, PyDict>> {
    let (n, h, w) = (b.indices.len(), b.height as usize, b.width as usize);
    let k: Vec<f64> = b.k.iter().flatten().flatten().copied().collect();
    let indices: Vec<i64> = b.indices.iter().map(|&i| i as i64).collect();
    let out = PyDict::new(py);
    out.set_item("rgb", PyArray1::from_vec(py, b.rgb).reshape([n, h, w, 3])?)?;
    out.set_item("depth", PyArray1::from_vec(py, b.depth).reshape([n, h, w])?)?;
    out.set_item("K", PyArray1::from_vec(py, k).reshape([n, 3, 3])?)?;
    out.set_item("ids", b.ids)?;
    out.set_item("indices", PyArray1::from_vec(py, indices))?;
    Ok(out)
}

/// Move a Vec into a NumPy array of the given shape. `from_vec` takes ownership of the
/// buffer (no copy); `reshape` returns a view of it.
fn rgb_array<'py>(py: Python<'py>, img: rigload_core::Rgb) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let shape = [img.height as usize, img.width as usize, 3];
    PyArray1::from_vec(py, img.data).reshape(shape)
}

fn depth_array<'py>(
    py: Python<'py>,
    d: rigload_core::Depth,
) -> PyResult<Bound<'py, PyArray2<u16>>> {
    let shape = [d.height as usize, d.width as usize];
    PyArray1::from_vec(py, d.data).reshape(shape)
}

/// Decode a JPEG/PNG to an (H, W, 3) uint8 array.
#[pyfunction]
fn decode_rgb(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyArray3<u8>>> {
    let img = py
        .detach(|| rigload_core::decode_rgb(&path))
        .map_err(to_py_err)?;
    rgb_array(py, img)
}

/// Decode a 16-bit PNG to an (H, W) uint16 array.
#[pyfunction]
fn decode_depth(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyArray2<u16>>> {
    let d = py
        .detach(|| rigload_core::decode_depth(&path))
        .map_err(to_py_err)?;
    depth_array(py, d)
}

/// Load sample `index` of a manifest: {"id", "rgb" (H,W,3) uint8, "depth" (H,W) uint16,
/// "K" (3,3) float64, "depth_scale"}.
#[pyfunction]
fn load_sample(py: Python<'_>, manifest: PathBuf, index: usize) -> PyResult<Bound<'_, PyDict>> {
    // shortcut: parses the whole manifest per call; fine for tests and one-off use, the
    // Loader (Step 5) parses it once.
    let (meta, sample) = py
        .detach(|| {
            let rows = rigload_core::read_manifest(&manifest)?;
            let meta = rows.into_iter().nth(index);
            match meta {
                Some(m) => rigload_core::load_sample(&m).map(|s| Some((m, s))),
                None => Ok(None),
            }
        })
        .map_err(to_py_err)?
        .ok_or_else(|| PyIndexError::new_err(format!("sample index {index} out of range")))?;

    let k: Vec<f64> = sample.k.iter().flatten().copied().collect();
    let out = PyDict::new(py);
    out.set_item("id", meta.id)?;
    out.set_item("rgb", rgb_array(py, sample.rgb)?)?;
    out.set_item("depth", depth_array(py, sample.depth)?)?;
    out.set_item("K", PyArray1::from_vec(py, k).reshape([3, 3])?)?;
    out.set_item("depth_scale", meta.depth_scale)?;
    Ok(out)
}

#[pyfunction]
fn hello() -> String {
    rigload_core::hello()
}

/// Module init. Function name must match the last part of
/// `module-name` in pyproject.toml (`rigload._rigload`).
#[pymodule]
fn _rigload(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(decode_depth, m)?)?;
    m.add_function(wrap_pyfunction!(load_sample, m)?)?;
    m.add_class::<Loader>()?;
    m.add_class::<EpochIterator>()?;
    Ok(())
}

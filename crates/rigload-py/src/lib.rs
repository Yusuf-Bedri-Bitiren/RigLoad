//! PyO3 bindings. Keep this layer thin: logic lives in rigload-core.
//!
//! GIL rule: all Rust work runs inside `py.detach(...)`, which releases the GIL so other
//! Python threads keep running while we read and decode files.

use std::path::PathBuf;
use std::sync::Mutex;

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyIOError, PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rigload_core::{AugmentConfig, AugmentParams, Batch, Error, LoaderConfig, SharedEpochIter};

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
    #[pyo3(signature = (manifest, batch_size=32, num_threads=None, prefetch=2, seed=0, shuffle=true,
                        drop_last=false, augment=false, out_size=(640, 192), crop_scale=(0.8, 1.0),
                        flip_prob=0.5, jitter=0.2, lidar=false,
                        yaw_range=std::f64::consts::FRAC_PI_4, scale_range=(0.95, 1.05)))]
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
        augment: bool,
        out_size: (u32, u32),
        crop_scale: (f64, f64),
        flip_prob: f64,
        jitter: f64,
        lidar: bool,
        yaw_range: f64,
        scale_range: (f64, f64),
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
            augment: augment.then_some(AugmentConfig {
                out_width: out_size.0,
                out_height: out_size.1,
                crop_scale,
                flip_prob,
                jitter,
                yaw_range,
                scale_range,
            }),
            lidar,
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
///  "depth_scale": (B,) float64, "ids": list[str], "indices": (B,) int64}, and with LiDAR
/// "depth_sparse": (B,H,W) uint16, "points": (N,4) float32, "point_offsets": (B+1,) int64,
/// "T_cam_lidar": (B,4,4) float64. Large buffers are moved, not copied.
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
    out.set_item("depth_scale", PyArray1::from_vec(py, b.depth_scale))?;
    if !b.point_offsets.is_empty() {
        let n_points = b.points.len() / 4;
        let offsets: Vec<i64> = b.point_offsets.iter().map(|&o| o as i64).collect();
        let t: Vec<f64> = b.t_cam_lidar.iter().flatten().flatten().copied().collect();
        out.set_item(
            "depth_sparse",
            PyArray1::from_vec(py, b.depth_sparse).reshape([n, h, w])?,
        )?;
        out.set_item(
            "points",
            PyArray1::from_vec(py, b.points).reshape([n_points, 4])?,
        )?;
        out.set_item("point_offsets", PyArray1::from_vec(py, offsets))?;
        out.set_item("T_cam_lidar", PyArray1::from_vec(py, t).reshape([n, 4, 4])?)?;
    }
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

/// The keys of `bench/common.py:draw_params`, so tests can pass the same dict to both.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct Params {
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    flip: bool,
    brightness: f64,
    contrast: f64,
    saturation: f64,
}

/// Apply crop -> resize -> flip -> jitter with explicit parameters to one decoded sample:
/// {"rgb" (out_h,out_w,3) uint8, "depth" (out_h,out_w) uint16, "K" (3,3)}. For tests and
/// comparisons; the Loader draws parameters itself. Copies the inputs.
#[pyfunction]
#[pyo3(signature = (rgb, depth, k, params, out_size=(640, 192)))]
fn augment_sample<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, u8>,
    depth: PyReadonlyArray2<'py, u16>,
    k: [[f64; 3]; 3],
    params: Params,
    out_size: (u32, u32),
) -> PyResult<Bound<'py, PyDict>> {
    let (rgb, depth) = (rgb.as_array(), depth.as_array());
    let (h, w, c) = rgb.dim();
    if c != 3 || depth.dim() != (h, w) {
        return Err(PyValueError::new_err(format!(
            "need rgb (H,W,3) and depth (H,W) of the same size, got {:?} and {:?}",
            (h, w, c),
            depth.dim()
        )));
    }
    let sample = rigload_core::Sample {
        rgb: rigload_core::Rgb {
            width: w as u32,
            height: h as u32,
            data: rgb.iter().copied().collect(),
        },
        depth: rigload_core::Depth {
            width: w as u32,
            height: h as u32,
            data: depth.iter().copied().collect(),
        },
        k,
    };
    let p = AugmentParams {
        x0: params.x0,
        y0: params.y0,
        w: params.w,
        h: params.h,
        flip: params.flip,
        brightness: params.brightness,
        contrast: params.contrast,
        saturation: params.saturation,
        yaw: 0.0,
        scale: 1.0,
    };
    let (ow, oh) = out_size;
    let px = ow as usize * oh as usize;
    let (mut rgb_out, mut depth_out) = (vec![0u8; px * 3], vec![0u16; px]);
    let k2 = py
        .detach(|| rigload_core::augment_into(sample, &p, ow, oh, &mut rgb_out, &mut depth_out))
        .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "rgb",
        PyArray1::from_vec(py, rgb_out).reshape([oh as usize, ow as usize, 3])?,
    )?;
    out.set_item(
        "depth",
        PyArray1::from_vec(py, depth_out).reshape([oh as usize, ow as usize])?,
    )?;
    out.set_item(
        "K",
        PyArray1::from_vec(py, k2.iter().flatten().copied().collect()).reshape([3, 3])?,
    )?;
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
    m.add_function(wrap_pyfunction!(augment_sample, m)?)?;
    m.add_class::<Loader>()?;
    m.add_class::<EpochIterator>()?;
    Ok(())
}

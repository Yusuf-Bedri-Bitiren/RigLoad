//! PyO3 bindings. Keep this layer thin: logic lives in rigload-core.
//!
//! GIL rule: all Rust work runs inside `py.detach(...)`, which releases the GIL so other
//! Python threads keep running while we read and decode files.

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rigload_core::Error;

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::Io { .. } => PyIOError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
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
    Ok(())
}

//! PyO3 bindings. Keep this layer thin: logic lives in rigload-core.

use pyo3::prelude::*;

/// `#[pyfunction]` generates the glue that converts Python args/returns.
/// Returning `String` (owned) is fine: PyO3 copies it into a new Python `str`.
#[pyfunction]
fn hello() -> String {
    rigload_core::hello()
}

/// Module init. Function name must match the last part of
/// `module-name` in pyproject.toml (`rigload._rigload`).
#[pymodule]
fn _rigload(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}

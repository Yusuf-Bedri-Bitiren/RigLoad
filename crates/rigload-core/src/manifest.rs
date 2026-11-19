use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

/// One line of `manifest.jsonl` (format: tools/gen_synthetic.py). Paths are absolute
/// after `read_manifest`.
#[derive(Debug, Clone, Deserialize)]
pub struct SampleMeta {
    pub id: String,
    pub rgb: PathBuf,
    pub depth: PathBuf,
    #[serde(default)]
    pub depth_sparse: Option<PathBuf>,
    #[serde(default)]
    pub lidar: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    /// Camera intrinsics, row-major 3x3; pixel centres at integer coordinates.
    #[serde(rename = "K")]
    pub k: [[f64; 3]; 3],
    /// X_cam = T_cam_lidar * [X_lidar, 1].
    #[serde(rename = "T_cam_lidar", default)]
    pub t_cam_lidar: Option<[[f64; 4]; 4]>,
    /// depth metres = stored value / depth_scale; 0 = invalid.
    pub depth_scale: f64,
}

/// Parse a manifest; relative file paths are resolved against the manifest's directory.
pub fn read_manifest(path: &Path) -> Result<Vec<SampleMeta>> {
    let text = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let base = path.parent().unwrap_or(Path::new(""));
    let bad = |line: usize, msg: String| Error::Manifest {
        path: path.to_path_buf(),
        line,
        msg,
    };

    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut m: SampleMeta =
            serde_json::from_str(line).map_err(|e| bad(i + 1, e.to_string()))?;
        if m.width == 0 || m.height == 0 || m.depth_scale <= 0.0 || !m.depth_scale.is_finite() {
            return Err(bad(
                i + 1,
                "width, height and depth_scale must be finite and > 0".into(),
            ));
        }
        if m.k.iter().flatten().any(|v| !v.is_finite()) {
            return Err(bad(i + 1, "K has a non-finite value".into()));
        }
        m.rgb = base.join(&m.rgb);
        m.depth = base.join(&m.depth);
        m.depth_sparse = m.depth_sparse.map(|p| base.join(p));
        m.lidar = m.lidar.map(|p| base.join(p));
        rows.push(m);
    }
    Ok(rows)
}

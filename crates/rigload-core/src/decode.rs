use std::path::Path;

use image::{DynamicImage, ImageError};

use crate::error::{Error, Result};
use crate::manifest::SampleMeta;

/// 8-bit RGB, row-major HWC: pixel (u, v) channel c is `data[(v * width + u) * 3 + c]`.
#[derive(Debug, Clone)]
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// 16-bit depth, row-major HW. Metres = value / depth_scale; 0 = invalid.
#[derive(Debug, Clone)]
pub struct Depth {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
}

/// One decoded sample. Owns its buffers.
#[derive(Debug, Clone)]
pub struct Sample {
    pub rgb: Rgb,
    pub depth: Depth,
    pub k: [[f64; 3]; 3],
}

fn open(path: &Path) -> Result<DynamicImage> {
    image::open(path).map_err(|e| match e {
        ImageError::IoError(source) => Error::Io {
            path: path.to_path_buf(),
            source,
        },
        other => Error::Decode {
            path: path.to_path_buf(),
            msg: other.to_string(),
        },
    })
}

/// Decode a JPEG or PNG into 8-bit RGB (grey or RGBA input is converted).
pub fn decode_rgb(path: &Path) -> Result<Rgb> {
    let img = open(path)?.into_rgb8(); // no copy when the file is already 8-bit RGB
    Ok(Rgb {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    })
}

/// Decode a 16-bit single-channel PNG. Anything else is an error, never a silent conversion:
/// rescaling 8-bit data would corrupt metric depth.
pub fn decode_depth(path: &Path) -> Result<Depth> {
    match open(path)? {
        DynamicImage::ImageLuma16(img) => Ok(Depth {
            width: img.width(),
            height: img.height(),
            data: img.into_raw(),
        }),
        other => Err(Error::Decode {
            path: path.to_path_buf(),
            msg: format!("expected a 16-bit grayscale PNG, got {:?}", other.color()),
        }),
    }
}

/// Decode a sample's RGB and dense depth, and check both match the manifest's size.
pub fn load_sample(meta: &SampleMeta) -> Result<Sample> {
    let rgb = decode_rgb(&meta.rgb)?;
    let depth = decode_depth(&meta.depth)?;
    for (path, w, h) in [
        (&meta.rgb, rgb.width, rgb.height),
        (&meta.depth, depth.width, depth.height),
    ] {
        if (w, h) != (meta.width, meta.height) {
            return Err(Error::Decode {
                path: path.clone(),
                msg: format!(
                    "size {w}x{h} does not match manifest {}x{} (sample {})",
                    meta.width, meta.height, meta.id
                ),
            });
        }
    }
    Ok(Sample {
        rgb,
        depth,
        k: meta.k,
    })
}

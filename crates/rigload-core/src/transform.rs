//! Calibration-aware augmentation: crop -> resize -> optional horizontal flip -> colour jitter.
//!
//! One parameter draw per sample, from (seed, epoch, index), shared by RGB, depth and K.
//! Pixel convention: centres at integer coordinates, [u, v, 1]^T ~ K X_cam.
//!
//! - RGB: antialiased bilinear resize (like torchvision `antialias=True`), then jitter.
//! - Depth: nearest-exact resize, never jittered, so a valid depth is never averaged with
//!   an invalid 0.
//! - K: crop shifts the centre, resize maps pixel *edges* (u' + 0.5 = (u + 0.5) s), flip maps
//!   u' = W - 1 - u. With fx kept positive, the flipped image is the projection of the mirrored
//!   scene (X -> -X in the camera frame); 3D data must be mirrored to match.

use fast_image_resize::images::{CroppedImage, Image};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::decode::Sample;
use crate::error::{Error, Result};
use crate::rng::SplitMix64;

pub type Intrinsics = [[f64; 3]; 3];

#[derive(Debug, Clone, PartialEq)]
pub struct AugmentConfig {
    pub out_width: u32,
    pub out_height: u32,
    /// Crop height as a fraction of the largest crop with the output's aspect ratio.
    pub crop_scale: (f64, f64),
    pub flip_prob: f64,
    /// Brightness, contrast and saturation factors are drawn from [1 - jitter, 1 + jitter].
    pub jitter: f64,
}

/// The baselines' values (bench/common.py), so benchmarks compare like with like.
impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            out_width: 640,
            out_height: 192,
            crop_scale: (0.8, 1.0),
            flip_prob: 0.5,
            jitter: 0.2,
        }
    }
}

impl AugmentConfig {
    pub fn validate(&self) -> Result<()> {
        let (lo, hi) = self.crop_scale;
        // Written so that NaN fails every check.
        let ok = self.out_width > 0
            && self.out_height > 0
            && 0.0 < lo
            && lo <= hi
            && hi <= 1.0
            && (0.0..=1.0).contains(&self.flip_prob)
            && (0.0..1.0).contains(&self.jitter);
        if ok {
            Ok(())
        } else {
            Err(Error::Config(format!(
                "invalid augmentation config {self:?}: need output size >= 1, \
                 0 < crop_scale.0 <= crop_scale.1 <= 1, 0 <= flip_prob <= 1, 0 <= jitter < 1"
            )))
        }
    }
}

/// One sample's random draw. The crop (x0, y0, w, h) is in source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AugmentParams {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
    pub flip: bool,
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
}

fn uniform(rng: &mut SplitMix64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.next_f64()
}

impl AugmentParams {
    /// Same logic and draw order as `draw_params` in bench/common.py; only the RNG differs.
    pub fn draw(
        cfg: &AugmentConfig,
        seed: u64,
        epoch: u64,
        index: u64,
        width: u32,
        height: u32,
    ) -> Self {
        let mut rng = SplitMix64::from_keys(&[seed, epoch, index]);
        let (ow, oh) = (cfg.out_width as f64, cfg.out_height as f64);
        let h_max = (height as f64).min(width as f64 * oh / ow);
        let scale = uniform(&mut rng, cfg.crop_scale.0, cfg.crop_scale.1);
        let h = ((scale * h_max).round() as u32).clamp(1, height);
        let w = ((h as f64 * ow / oh).round() as u32).clamp(1, width);
        let (lo, hi) = (1.0 - cfg.jitter, 1.0 + cfg.jitter);
        let brightness = uniform(&mut rng, lo, hi);
        let contrast = uniform(&mut rng, lo, hi);
        let saturation = uniform(&mut rng, lo, hi);
        Self {
            x0: rng.below((width - w + 1) as u64) as u32,
            y0: rng.below((height - h + 1) as u64) as u32,
            w,
            h,
            flip: rng.next_f64() < cfg.flip_prob,
            brightness,
            contrast,
            saturation,
        }
    }
}

/// Intrinsics after crop -> resize to (out_w, out_h) -> optional flip.
pub fn update_k(k: &Intrinsics, p: &AugmentParams, out_w: u32, out_h: u32) -> Intrinsics {
    let (sx, sy) = (out_w as f64 / p.w as f64, out_h as f64 / p.h as f64);
    let mut k = *k;
    k[0][0] *= sx;
    k[0][1] *= sx; // skew
    k[1][1] *= sy;
    k[0][2] = (k[0][2] - p.x0 as f64 + 0.5) * sx - 0.5;
    k[1][2] = (k[1][2] - p.y0 as f64 + 0.5) * sy - 0.5;
    if p.flip {
        k[0][1] = -k[0][1]; // mirrored X flips the skew term's sign
        k[0][2] = out_w as f64 - 1.0 - k[0][2];
    }
    k
}

/// Nearest-exact (PyTorch `NEAREST_EXACT`): the source pixel containing the output pixel's
/// centre, floor((dst + 0.5) * src_len / dst_len).
fn nearest(dst: u32, src_len: u32, dst_len: u32) -> usize {
    let src = ((dst as f64 + 0.5) * src_len as f64 / dst_len as f64) as usize;
    src.min(src_len as usize - 1)
}

fn resize_err(e: impl std::fmt::Display) -> Error {
    Error::Config(format!("resize failed: {e}"))
}

/// Augment one sample into caller-provided buffers (`out_w * out_h * 3` and `out_w * out_h`,
/// e.g. its slot in a batch buffer). Returns the updated K.
pub fn augment_into(
    s: Sample,
    p: &AugmentParams,
    out_w: u32,
    out_h: u32,
    rgb_out: &mut [u8],
    depth_out: &mut [u16],
) -> Result<Intrinsics> {
    let (sw, sh) = (s.rgb.width, s.rgb.height);
    let fits = |start: u32, len: u32, max: u32| {
        len > 0 && start.checked_add(len).is_some_and(|end| end <= max)
    };
    if !fits(p.x0, p.w, sw) || !fits(p.y0, p.h, sh) || (s.depth.width, s.depth.height) != (sw, sh) {
        return Err(Error::Config(format!(
            "augmentation crop {p:?} does not fit the {sw}x{sh} image (depth {}x{})",
            s.depth.width, s.depth.height
        )));
    }
    let px = out_w as usize * out_h as usize;
    assert_eq!((rgb_out.len(), depth_out.len()), (px * 3, px));

    // RGB: crop + antialiased bilinear resize in one pass, written straight into rgb_out.
    {
        let src = Image::from_vec_u8(sw, sh, s.rgb.data, PixelType::U8x3).map_err(resize_err)?;
        let mut dst =
            Image::from_slice_u8(out_w, out_h, rgb_out, PixelType::U8x3).map_err(resize_err)?;
        // A cropped *view*, not a crop box: the filter then treats the crop border as the
        // image border (like crop-then-resize in the baselines) instead of reading pixels
        // outside the crop.
        let src = CroppedImage::new(&src, p.x0, p.y0, p.w, p.h).map_err(resize_err)?;
        let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
        // shortcut: a new Resizer per sample (small scratch allocation); keep one per thread
        // if profiles show it.
        Resizer::new()
            .resize(&src, &mut dst, &opts)
            .map_err(resize_err)?;
    }
    if p.flip {
        for row in rgb_out.chunks_exact_mut(out_w as usize * 3) {
            row.as_chunks_mut::<3>().0.reverse(); // pixel order, channels unchanged
        }
    }
    color_jitter(rgb_out, p);

    // Depth: nearest-exact; the flip is folded into the column lookup table.
    let cols: Vec<usize> = (0..out_w)
        .map(|x| {
            let x = if p.flip { out_w - 1 - x } else { x };
            p.x0 as usize + nearest(x, p.w, out_w)
        })
        .collect();
    for (y, row) in depth_out.chunks_exact_mut(out_w as usize).enumerate() {
        let sy = p.y0 as usize + nearest(y as u32, p.h, out_h);
        let src_row = &s.depth.data[sy * sw as usize..(sy + 1) * sw as usize];
        for (d, &c) in row.iter_mut().zip(&cols) {
            *d = src_row[c];
        }
    }

    Ok(update_k(&s.k, p, out_w, out_h))
}

// Colour jitter, matching torchvision.transforms.v2.functional on uint8 images: each step
// blends in float32, clamps to [0, 255] and truncates back to uint8; grey is
// floor(0.2989 R + 0.587 G + 0.114 B). Order: brightness, contrast, saturation.

fn blend(a: u8, b: f32, ratio: f32) -> u8 {
    (a as f32 * ratio + b * (1.0 - ratio)).clamp(0.0, 255.0) as u8
}

fn grey(px: &[u8; 3]) -> f32 {
    (px[0] as f32 * 0.2989 + px[1] as f32 * 0.587 + px[2] as f32 * 0.114).floor()
}

fn color_jitter(rgb: &mut [u8], p: &AugmentParams) {
    let b = p.brightness as f32;
    rgb.iter_mut().for_each(|v| *v = blend(*v, 0.0, b));

    let pixels = rgb.as_chunks_mut::<3>().0;
    let mean = (pixels.iter().map(|px| grey(px) as f64).sum::<f64>() / pixels.len() as f64) as f32;
    let c = p.contrast as f32;
    pixels
        .iter_mut()
        .flatten()
        .for_each(|v| *v = blend(*v, mean, c));

    let s = p.saturation as f32;
    for px in pixels.iter_mut() {
        let g = grey(px);
        px.iter_mut().for_each(|v| *v = blend(*v, g, s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{Depth, Rgb};
    use proptest::prelude::*;

    fn project(k: &Intrinsics, x: [f64; 3]) -> (f64, f64) {
        let u = k[0][0] * x[0] + k[0][1] * x[1] + k[0][2] * x[2];
        let v = k[1][1] * x[1] + k[1][2] * x[2];
        (u / x[2], v / x[2])
    }

    /// K with random focal lengths, skew and centre for a width x height image.
    fn intrinsics(w: u32, h: u32) -> impl Strategy<Value = Intrinsics> {
        (
            50.0..3000.0f64,
            50.0..3000.0f64,
            -5.0..5.0f64,
            0.0..1.0f64,
            0.0..1.0f64,
        )
            .prop_map(move |(fx, fy, skew, cx, cy)| {
                [
                    [fx, skew, cx * w as f64],
                    [0.0, fy, cy * h as f64],
                    [0.0, 0.0, 1.0],
                ]
            })
    }

    fn config() -> impl Strategy<Value = AugmentConfig> {
        (1u32..1024, 1u32..512, 0.05..1.0f64, 0.0..1.0f64).prop_map(|(ow, oh, lo, t)| {
            AugmentConfig {
                out_width: ow,
                out_height: oh,
                crop_scale: (lo, lo + (1.0 - lo) * t),
                ..AugmentConfig::default()
            }
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        /// §5 geometry: projecting a 3D point with the updated K' (and the scene mirrored if
        /// flipped) lands exactly where the crop/resize/flip pixel map sends its original
        /// projection.
        #[test]
        fn updated_k_matches_the_pixel_map(
            (w, h, k) in (16u32..4000, 16u32..2000).prop_flat_map(|(w, h)| (Just(w), Just(h), intrinsics(w, h))),
            cfg in config(),
            seed in any::<u64>(),
            x in (-50.0..50.0f64, -20.0..20.0f64, 0.5..120.0f64),
        ) {
            let p = AugmentParams::draw(&cfg, seed, 0, 0, w, h);
            let (ow, oh) = (cfg.out_width, cfg.out_height);
            let x = [x.0, x.1, x.2];
            let (u, v) = project(&k, x);
            // the pixel map applied to the original projection
            let mut u_map = (u - p.x0 as f64 + 0.5) * ow as f64 / p.w as f64 - 0.5;
            let v_map = (v - p.y0 as f64 + 0.5) * oh as f64 / p.h as f64 - 0.5;
            let mut x_cam = x;
            if p.flip {
                u_map = ow as f64 - 1.0 - u_map;
                x_cam[0] = -x_cam[0];
            }
            let (u2, v2) = project(&update_k(&k, &p, ow, oh), x_cam);
            let tol = |a: f64| 1e-9 * a.abs().max(1.0);
            prop_assert!((u2 - u_map).abs() <= tol(u_map), "u: {u2} vs {u_map}, {p:?}");
            prop_assert!((v2 - v_map).abs() <= tol(v_map), "v: {v2} vs {v_map}, {p:?}");
        }

        /// §5 pixels: for every output pixel, back-project its centre with K' to a 3D point,
        /// project that point with the original K: the depth value written there came from the
        /// source pixel containing that location. This ties K' to the actual resampling.
        #[test]
        fn depth_pixels_come_from_where_k_says(
            (w, h, k) in (2u32..64, 2u32..64).prop_flat_map(|(w, h)| (Just(w), Just(h), intrinsics(w, h))),
            (ow, oh) in (1u32..96, 1u32..96),
            seed in any::<u64>(),
            z in 0.5..100.0f64,
        ) {
            let cfg = AugmentConfig { out_width: ow, out_height: oh, ..AugmentConfig::default() };
            let p = AugmentParams::draw(&cfg, seed, 0, 0, w, h);
            // Each source pixel stores its own id (< 65536 since w, h < 64).
            let ids: Vec<u16> = (0..w * h).map(|i| i as u16 + 1).collect();
            let s = Sample {
                rgb: Rgb { width: w, height: h, data: vec![128; (w * h * 3) as usize] },
                depth: Depth { width: w, height: h, data: ids },
                k,
            };
            let (mut rgb, mut depth) = (vec![0; (ow * oh * 3) as usize], vec![0; (ow * oh) as usize]);
            let k2 = augment_into(s, &p, ow, oh, &mut rgb, &mut depth).unwrap();
            for y in 0..oh {
                for x in 0..ow {
                    let id = depth[(y * ow + x) as usize] as u32 - 1;
                    let (us, vs) = ((id % w) as f64, (id / w) as f64);
                    // back-project the output pixel centre at depth z with K'
                    let yy = (y as f64 - k2[1][2]) / k2[1][1];
                    let xx = (x as f64 - k2[0][2] - k2[0][1] * yy) / k2[0][0];
                    let mut pt = [xx * z, yy * z, z];
                    if p.flip { pt[0] = -pt[0]; } // undo the mirror to get the original scene
                    let (u, v) = project(&k, pt);
                    prop_assert!((us - u).abs() <= 0.5 + 1e-6, "x={x} y={y}: source u {us} vs K {u}, {p:?}");
                    prop_assert!((vs - v).abs() <= 0.5 + 1e-6, "x={x} y={y}: source v {vs} vs K {v}, {p:?}");
                }
            }
        }
    }

    #[test]
    fn draw_is_deterministic_and_in_range() {
        let cfg = AugmentConfig::default();
        let a = AugmentParams::draw(&cfg, 3, 1, 42, 1242, 375);
        assert_eq!(a, AugmentParams::draw(&cfg, 3, 1, 42, 1242, 375));
        assert_ne!(a, AugmentParams::draw(&cfg, 3, 1, 43, 1242, 375));
        for i in 0..1000 {
            let p = AugmentParams::draw(&cfg, 0, 0, i, 1242, 375);
            assert!(p.x0 + p.w <= 1242 && p.y0 + p.h <= 375);
            assert!((0.8..=1.2).contains(&p.brightness));
            // crop keeps the output aspect ratio up to rounding
            assert!((p.w as f64 / p.h as f64 - 640.0 / 192.0).abs() < 0.02);
        }
    }

    #[test]
    fn identity_jitter_and_full_crop_keep_pixels() {
        let (w, h) = (8u32, 4u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        let s = Sample {
            rgb: Rgb {
                width: w,
                height: h,
                data: rgb.clone(),
            },
            depth: Depth {
                width: w,
                height: h,
                data: (0..w * h).map(|i| i as u16).collect(),
            },
            k: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let p = AugmentParams {
            x0: 0,
            y0: 0,
            w,
            h,
            flip: true,
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
        };
        let (mut out, mut depth) = (vec![0; rgb.len()], vec![0; (w * h) as usize]);
        augment_into(s, &p, w, h, &mut out, &mut depth).unwrap();
        // same size, no jitter: just the flip
        for y in 0..h as usize {
            for x in 0..w as usize {
                let (o, i) = ((y * 8 + x) * 3, (y * 8 + 7 - x) * 3);
                assert_eq!(out[o..o + 3], rgb[i..i + 3]);
                assert_eq!(depth[y * 8 + x] as usize, y * 8 + 7 - x);
            }
        }
    }

    #[test]
    fn bad_crop_and_config_are_errors() {
        let s = Sample {
            rgb: Rgb {
                width: 4,
                height: 4,
                data: vec![0; 48],
            },
            depth: Depth {
                width: 4,
                height: 4,
                data: vec![0; 16],
            },
            k: [[0.0; 3]; 3],
        };
        let p = AugmentParams {
            x0: 2,
            y0: 0,
            w: 3,
            h: 4,
            flip: false,
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
        };
        assert!(augment_into(s, &p, 2, 2, &mut [0; 12], &mut [0; 4]).is_err());
        let bad = AugmentConfig {
            crop_scale: (0.0, 1.0),
            ..AugmentConfig::default()
        };
        assert!(bad.validate().is_err());
        let nan = AugmentConfig {
            jitter: f64::NAN,
            ..AugmentConfig::default()
        };
        assert!(nan.validate().is_err());
        assert!(AugmentConfig::default().validate().is_ok());
    }
}

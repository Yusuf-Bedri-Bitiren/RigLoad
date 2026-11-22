//! LiDAR points, kept consistent with the augmented image, depth and K.
//!
//! Frames: p_cam = T_cam_lidar · p_lidar, pixel ~ K · p_cam. The camera augmentation
//! (crop/resize/flip, `transform.rs`) yields K' and, if flipped, shows the scene mirrored in
//! the camera frame (F = diag(-1, 1, 1)). The 3D augmentation adds a scale s about the camera
//! centre (S = s·I; the image is unchanged, depth values scale) and a yaw rotation R about the
//! LiDAR z axis (the camera rides along; the image is unchanged). Chosen so that:
//!
//! - points:  p' = A · p,  A = R · T⁻¹ · S · F · T  (mirror and scale happen in the camera frame)
//! - extrinsic: T' = T · R⁻¹, still rigid (rotation + translation)
//! - then T' · p' = S · F · T · p = s · F · p_cam: the mirrored camera-frame point, scaled,
//!   which K' projects to exactly where the image transform moved the original pixel.
//! - depth: metric depth scales by s, stored values stay: depth_scale' = depth_scale / s.
//!
//! This folds the augmentation into the outputs: consumers use the plain `K' · T' · p'`
//! projection, with no augmentation matrices to chain (contrast: BEVFusion's
//! `img_aug_matrix` / `lidar_aug_matrix`, see docs/PRIOR_ART.md).

use std::path::Path;

use crate::error::{Error, Result};
use crate::transform::{AugmentParams, Intrinsics};

pub type Mat4 = [[f64; 4]; 4];

// Points are flat float32 slices, 4 values per point: x, y, z, intensity (KITTI velodyne
// layout), in the LiDAR frame.

fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Number of points in a KITTI-style `.bin` (little-endian float32 x, y, z, intensity per
/// point), from its size alone, so the batch buffer can be sized before reading.
pub fn point_count(path: &Path) -> Result<usize> {
    let bytes = std::fs::metadata(path).map_err(io_err(path))?.len();
    if bytes % 16 != 0 {
        return Err(Error::Decode {
            path: path.to_path_buf(),
            msg: format!(
                "{bytes} bytes is not a whole number of (x, y, z, intensity) float32 points"
            ),
        });
    }
    Ok((bytes / 16) as usize)
}

/// Read a `.bin` straight into `out` (4 floats per point; `out.len()` must match the file), with
/// no intermediate buffer: the file's bytes land in their final place in the batch.
pub fn read_points_into(path: &Path, out: &mut [f32]) -> Result<()> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(io_err(path))?;
    let bytes = file.metadata().map_err(io_err(path))?.len();
    if bytes != (out.len() * 4) as u64 {
        return Err(Error::Decode {
            path: path.to_path_buf(),
            msg: format!(
                "file is {bytes} bytes, expected {} (changed while loading?)",
                out.len() * 4
            ),
        });
    }
    // SAFETY: a [f32] is a valid [u8] of 4x its length (same allocation, u8 has alignment 1,
    // every byte pattern is a valid u8 and every 32-bit pattern a valid f32).
    let raw =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
    file.read_exact(raw).map_err(io_err(path))?;
    for v in out.iter_mut() {
        *v = f32::from_bits(u32::from_le(v.to_bits())); // no-op on little-endian CPUs
    }
    Ok(())
}

/// Read a whole `.bin` into a new buffer (tests and one-off use; the loader reads in place).
pub fn decode_points(path: &Path) -> Result<Vec<f32>> {
    let mut out = vec![0.0; point_count(path)? * 4];
    read_points_into(path, &mut out)?;
    Ok(out)
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn diag(x: f64, y: f64, z: f64) -> Mat4 {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Inverse of a rigid transform [R t; 0 1]: [Rᵀ −Rᵀt; 0 1].
fn rigid_inverse(t: &Mat4) -> Mat4 {
    let mut out = diag(1.0, 1.0, 1.0);
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = t[j][i];
        }
        out[i][3] = -(0..3).map(|k| t[k][i] * t[k][3]).sum::<f64>();
    }
    out
}

fn yaw(angle: f64) -> Mat4 {
    let (s, c) = angle.sin_cos();
    [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// The point transform A and the new extrinsic T' (see the module docs).
pub fn augment_lidar(t_cam_lidar: &Mat4, p: &AugmentParams) -> (Mat4, Mat4) {
    let t = t_cam_lidar;
    let mirror = if p.flip { -1.0 } else { 1.0 };
    let sf = diag(mirror * p.scale, p.scale, p.scale);
    let a = mul(&yaw(p.yaw), &mul(&rigid_inverse(t), &mul(&sf, t)));
    let t2 = mul(t, &yaw(-p.yaw));
    (a, t2)
}

/// Apply a 4x4 transform to every point's (x, y, z); intensity is unchanged.
pub fn transform_points(points: &mut [f32], a: &Mat4) {
    for pt in points.as_chunks_mut::<4>().0 {
        let (x, y, z) = (pt[0] as f64, pt[1] as f64, pt[2] as f64);
        for (i, v) in pt.iter_mut().take(3).enumerate() {
            *v = (a[i][0] * x + a[i][1] * y + a[i][2] * z + a[i][3]) as f32;
        }
    }
}

/// Sparse depth image (row-major, width x height, written into `depth`) from points: each
/// point goes to the pixel nearest its projection (pixel centres at integers), the nearest
/// point wins, depth is stored as round(z · depth_scale) like the dense depth. Points with
/// z <= 0.1 m or depth beyond what uint16 holds are dropped; 0 = no point.
pub fn project_sparse(
    points: &[f32],
    t_cam_lidar: &Mat4,
    k: &Intrinsics,
    width: u32,
    height: u32,
    depth_scale: f64,
    depth: &mut [u16],
) {
    assert_eq!(depth.len(), width as usize * height as usize);
    depth.fill(0);
    let t = t_cam_lidar;
    for pt in points.as_chunks::<4>().0 {
        let (x, y, z) = (pt[0] as f64, pt[1] as f64, pt[2] as f64);
        let c = |i: usize| t[i][0] * x + t[i][1] * y + t[i][2] * z + t[i][3];
        let (xc, yc, zc) = (c(0), c(1), c(2));
        if zc <= 0.1 {
            continue;
        }
        let u = ((k[0][0] * xc + k[0][1] * yc) / zc + k[0][2]).round_ties_even();
        let v = (k[1][1] * yc / zc + k[1][2]).round_ties_even();
        let value = (zc * depth_scale).round_ties_even();
        if !(0.0..width as f64).contains(&u)
            || !(0.0..height as f64).contains(&v)
            || value > 65535.0
        {
            continue;
        }
        let d = &mut depth[v as usize * width as usize + u as usize];
        let value = value.max(1.0) as u16; // 0 means "no point"
        if *d == 0 || value < *d {
            *d = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{AugmentConfig, update_k};
    use proptest::prelude::*;

    fn apply(a: &Mat4, p: [f64; 3]) -> [f64; 3] {
        let c = |i: usize| a[i][0] * p[0] + a[i][1] * p[1] + a[i][2] * p[2] + a[i][3];
        [c(0), c(1), c(2)]
    }

    fn project(k: &Intrinsics, x: [f64; 3]) -> (f64, f64) {
        (
            (k[0][0] * x[0] + k[0][1] * x[1]) / x[2] + k[0][2],
            k[1][1] * x[1] / x[2] + k[1][2],
        )
    }

    /// Rigid transform from a rotation vector (Rodrigues) and a translation.
    fn rigid(rv: [f64; 3], tr: [f64; 3]) -> Mat4 {
        let th = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
            .sqrt()
            .max(1e-12);
        let (x, y, z) = (rv[0] / th, rv[1] / th, rv[2] / th);
        let (s, c) = th.sin_cos();
        let t = 1.0 - c;
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, tr[0]],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, tr[1]],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, tr[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10_000))]

        /// §5 for LiDAR: after any crop/resize/flip/yaw/scale, projecting the augmented point
        /// with K' and T' lands where the image transform moved the original projection, its
        /// camera depth is scaled by s, and T' is still rigid.
        #[test]
        fn augmented_points_project_where_the_image_moved(
            (w, h) in (16u32..4000, 16u32..2000),
            (fx, fy, skew, cxf, cyf) in (50.0..3000.0f64, 50.0..3000.0f64, -5.0..5.0f64, 0.0..1.0f64, 0.0..1.0f64),
            rv in (-3.0..3.0f64, -3.0..3.0f64, -3.0..3.0f64),
            tr in (-3.0..3.0f64, -3.0..3.0f64, -3.0..3.0f64),
            (ow, oh) in (1u32..1024, 1u32..512),
            seed in any::<u64>(),
            pc in (-50.0..50.0f64, -20.0..20.0f64, 0.5..120.0f64),
        ) {
            let k = [[fx, skew, cxf * w as f64], [0.0, fy, cyf * h as f64], [0.0, 0.0, 1.0]];
            let t = rigid([rv.0, rv.1, rv.2], [tr.0, tr.1, tr.2]);
            let cfg = AugmentConfig { out_width: ow, out_height: oh, ..AugmentConfig::default() };
            let p = AugmentParams::draw(&cfg, seed, 0, 0, w, h);
            let p_cam = [pc.0, pc.1, pc.2];
            let p_lidar = apply(&rigid_inverse(&t), p_cam);

            let (a, t2) = augment_lidar(&t, &p);
            let k2 = update_k(&k, &p, ow, oh);
            let p_cam2 = apply(&t2, apply(&a, p_lidar));

            // depth scales by s
            prop_assert!(close(p_cam2[2], p.scale * p_cam[2]), "z {} vs {}", p_cam2[2], p.scale * p_cam[2]);
            // pixel: the Step 6 image map applied to the original projection
            let (u, v) = project(&k, p_cam);
            let mut u_map = (u - p.x0 as f64 + 0.5) * ow as f64 / p.w as f64 - 0.5;
            let v_map = (v - p.y0 as f64 + 0.5) * oh as f64 / p.h as f64 - 0.5;
            if p.flip {
                u_map = ow as f64 - 1.0 - u_map;
            }
            let (u2, v2) = project(&k2, p_cam2);
            prop_assert!((u2 - u_map).abs() <= 1e-7 * u_map.abs().max(1.0), "u {u2} vs {u_map}, {p:?}");
            prop_assert!((v2 - v_map).abs() <= 1e-7 * v_map.abs().max(1.0), "v {v2} vs {v_map}, {p:?}");
            // T' rigid: rotation part orthonormal with det +1, bottom row (0 0 0 1)
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f64 = (0..3).map(|r| t2[r][i] * t2[r][j]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    prop_assert!(close(dot, want), "RᵀR[{}][{}] = {}", i, j, dot);
                }
            }
            let r = |i: usize, j: usize| t2[i][j];
            let det = r(0, 0) * (r(1, 1) * r(2, 2) - r(1, 2) * r(2, 1))
                - r(0, 1) * (r(1, 0) * r(2, 2) - r(1, 2) * r(2, 0))
                + r(0, 2) * (r(1, 0) * r(2, 1) - r(1, 1) * r(2, 0));
            prop_assert!(close(det, 1.0));
            prop_assert_eq!(t2[3], [0.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn identity_params_leave_points_and_extrinsic_unchanged() {
        let t = rigid([0.3, -1.2, 0.4], [0.1, -0.5, 1.0]);
        let (a, t2) = augment_lidar(&t, &AugmentParams::identity(100, 50));
        for i in 0..4 {
            for j in 0..4 {
                assert!(close(a[i][j], if i == j { 1.0 } else { 0.0 }));
                assert!(close(t2[i][j], t[i][j]));
            }
        }
    }

    #[test]
    fn sparse_projection_keeps_the_nearest_point() {
        let k = [[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]];
        let t = diag(1.0, 1.0, 1.0);
        // two points on the same ray (pixel (2, 2)), one behind the camera, one off-image
        let points = [
            0.0, 0.0, 5.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
            0.0, 0.0, -3.0, 0.0, //
            10.0, 0.0, 1.0, 0.0,
        ];
        let mut d = vec![7u16; 25]; // stale values must be cleared
        project_sparse(&points, &t, &k, 5, 5, 256.0, &mut d);
        assert_eq!(d[2 * 5 + 2], 768);
        assert_eq!(d.iter().filter(|&&v| v > 0).count(), 1);
    }

    #[test]
    fn bin_must_hold_whole_points() {
        let dir = std::env::temp_dir().join(format!("rigload-bin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (good, bad) = (dir.join("good.bin"), dir.join("bad.bin"));
        let vals: Vec<u8> = [1.0f32, 2.0, 3.0, 0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&good, &vals).unwrap();
        std::fs::write(&bad, &vals[..10]).unwrap();
        assert_eq!(decode_points(&good).unwrap(), [1.0, 2.0, 3.0, 0.5]);
        // the loader sizes its buffer first; a file that changed size since must not be read
        let mut short = [0.0f32; 2];
        assert!(read_points_into(&good, &mut short).is_err());
        assert!(
            decode_points(&bad)
                .unwrap_err()
                .to_string()
                .contains("bad.bin")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}

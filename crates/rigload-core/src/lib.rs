//! rigload-core: pure Rust, no Python dependency.

mod decode;
mod error;
mod lidar;
mod manifest;
mod pipeline;
mod rng;
mod transform;

pub use decode::{Depth, Rgb, Sample, decode_depth, decode_rgb, load_sample};
pub use error::{Error, Result};
pub use lidar::{
    Mat4, augment_lidar, decode_points, point_count, project_sparse, read_points_into,
    transform_points,
};
pub use manifest::{SampleMeta, read_manifest};
pub use pipeline::{Batch, EpochIter, Loader, LoaderConfig, SharedEpochIter};
pub use rng::{SplitMix64, epoch_order};
pub use transform::{AugmentConfig, AugmentParams, Intrinsics, augment_into, update_k};

/// Greeting used to check the Rust -> Python path end to end.
pub fn hello() -> String {
    format!("hello from rigload-core {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/synthetic")
    }

    #[test]
    fn hello_mentions_crate() {
        assert!(hello().starts_with("hello from rigload-core"));
    }

    #[test]
    fn reads_fixture_manifest() {
        let rows = read_manifest(&fixtures().join("manifest.jsonl")).unwrap();
        assert_eq!(rows.len(), 20);
        let r = &rows[0];
        assert_eq!(r.id, "000000");
        assert!(r.rgb.is_file() && r.depth.is_file());
        assert!(r.lidar.as_ref().is_some_and(|p| p.is_file()));
        assert_eq!((r.width, r.height), (1242, 375));
        assert_eq!(r.k[2], [0.0, 0.0, 1.0]);
        assert_eq!(r.depth_scale, 256.0);
    }

    #[test]
    fn loads_every_fixture_sample() {
        for meta in read_manifest(&fixtures().join("manifest.jsonl")).unwrap() {
            let s = load_sample(&meta).unwrap();
            assert_eq!(s.rgb.data.len(), 1242 * 375 * 3);
            assert_eq!(s.depth.data.len(), 1242 * 375);
            assert!(
                s.depth.data.iter().any(|&d| d > 0),
                "{}: no valid depth",
                meta.id
            );
            assert_eq!(s.k, meta.k);
        }
    }

    #[test]
    fn missing_file_error_names_the_path() {
        let err = decode_rgb(&fixtures().join("rgb/does_not_exist.jpg")).unwrap_err();
        assert!(matches!(err, Error::Io { .. }));
        assert!(err.to_string().contains("does_not_exist.jpg"));
    }

    #[test]
    fn rgb_file_is_not_accepted_as_depth() {
        let err = decode_depth(&fixtures().join("rgb/000000.jpg")).unwrap_err();
        assert!(matches!(err, Error::Decode { .. }), "{err}");
    }

    #[test]
    fn size_mismatch_is_an_error() {
        let mut meta = read_manifest(&fixtures().join("manifest.jsonl")).unwrap()[0].clone();
        meta.width = 100;
        let err = load_sample(&meta).unwrap_err();
        assert!(err.to_string().contains("does not match manifest"), "{err}");
    }

    #[test]
    fn bad_manifest_line_reports_line_number() {
        let dir = std::env::temp_dir().join(format!("rigload-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.jsonl");
        std::fs::write(&path, "\n{\"id\": \"0\"}\n").unwrap();
        let err = read_manifest(&path).unwrap_err();
        assert!(
            matches!(err, Error::Manifest { line: 2, .. }),
            "unexpected: {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

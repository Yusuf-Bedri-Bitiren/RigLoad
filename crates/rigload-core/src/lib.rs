//! rigload-core: pure Rust, no Python dependency.

/// Greeting used to check the Rust -> Python path end to end.
pub fn hello() -> String {
    format!("hello from rigload-core {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_mentions_crate() {
        assert!(hello().starts_with("hello from rigload-core"));
    }
}

//! SplitMix64 (Steele, Lea, Flood 2014): a tiny, fast, well-mixed 64-bit generator.
//! Not cryptographic; good for shuffles and augmentation parameters.
//!
//! Determinism rule: every random stream is derived from integers like (seed, epoch) or
//! (seed, epoch, sample_index), never shared between threads, so results don't depend on
//! thread scheduling.

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// An independent stream for a tuple of keys, e.g. `&[seed, epoch]`.
    pub fn from_keys(keys: &[u64]) -> Self {
        let state = keys
            .iter()
            .fold(0u64, |s, &k| mix(s.wrapping_add(GOLDEN) ^ k));
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        mix(self.state)
    }

    /// Uniform integer in 0..n (n > 0), by multiply-shift. Bias is at most n / 2^64,
    /// negligible for any dataset size.
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform float in [0, 1) with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The sample order for one epoch: a Fisher-Yates shuffle seeded from (seed, epoch).
pub fn epoch_order(n: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::from_keys(&[seed, epoch]);
    for i in (1..n).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        order.swap(i, j);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_splitmix64() {
        // First outputs of the reference C implementation with state 0.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn floats_and_ranges_stay_in_bounds() {
        let mut r = SplitMix64::new(7);
        for _ in 0..10_000 {
            let f = r.next_f64();
            assert!((0.0..1.0).contains(&f));
            assert!(r.below(5) < 5);
        }
    }

    #[test]
    fn epoch_order_is_a_deterministic_permutation() {
        let a = epoch_order(1000, 42, 0);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert_eq!(a, epoch_order(1000, 42, 0));
        assert_ne!(a, epoch_order(1000, 42, 1));
        assert_ne!(a, epoch_order(1000, 43, 0));
        assert_ne!(a, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_is_roughly_uniform() {
        // Over many epochs, each position should hold each of 4 items ~25 % of the time.
        let mut counts = [[0u32; 4]; 4];
        for epoch in 0..20_000 {
            for (pos, &item) in epoch_order(4, 1, epoch).iter().enumerate() {
                counts[pos][item] += 1;
            }
        }
        for row in counts {
            for c in row {
                assert!((4_500..5_500).contains(&c), "{counts:?}");
            }
        }
    }
}

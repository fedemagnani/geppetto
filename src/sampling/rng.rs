//! Seeded pseudo-random numbers for the categorical draw.
//!
//! xoshiro256** with SplitMix64 seeding: small, well-studied, and enough for
//! sampling. Hand-rolled rather than pulled from `rand`, whose only use here
//! would be one uniform draw per token.

/// A seeded xoshiro256** generator. The seed is always explicit -- nothing in
/// geppetto draws entropy from the environment, so a run is reproducible from
/// its config alone.
#[derive(Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // SplitMix64 spreads a single seed word over the four state words;
        // seeding xoshiro directly from a small integer starts it poorly.
        let mut z = seed;
        let mut split_mix = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            state: [split_mix(), split_mix(), split_mix(), split_mix()],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);

        result
    }

    /// A uniform value in `[0, 1)`. The top 53 bits fill an f64 mantissa
    /// exactly, so every representable value in the range is reachable and
    /// the result is never 1.0.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let diverged = (0..8).any(|_| a.next_u64() != b.next_u64());
        assert!(diverged);
    }

    #[test]
    fn uniform_values_stay_in_range() {
        let mut rng = Rng::new(7);
        let mut sum = 0.0;
        const N: usize = 100_000;
        for _ in 0..N {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "{x} out of range");
            sum += x;
        }
        // the mean of a uniform [0,1) stream is 0.5; a loose band is enough to
        // catch a broken shift or scale without being flaky
        let mean = sum / N as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} is not near 0.5");
    }
}

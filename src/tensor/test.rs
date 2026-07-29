//! Deterministic helpers shared by the op unit tests. The RNG is a tiny
//! xorshift, not the `rand` crate: the plan introduces `rand` in epoch 5 for
//! sampling, and tests want reproducibility without a runtime dependency.

/// xorshift64* -- adequate for filling test buffers, not for sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `[-range, range)`.
    pub fn next_f32(&mut self, range: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A buffer of `len` values in `[-range, range)`.
    pub fn vec(&mut self, len: usize, range: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32(range)).collect()
    }
}

/// Matmul written straight from the definition with explicit flat indexing --
/// an independent reference for the convention `out[i, j] = dot(a row i,
/// b row j)` over `a` `[m, k]` and `b` `[n, k]`, both row-major.
pub fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[j * k + l];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Asserts two buffers have the same length and elementwise-close data.
pub fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "element {i}: {x} vs {y} exceeds tol {tol}"
        );
    }
}

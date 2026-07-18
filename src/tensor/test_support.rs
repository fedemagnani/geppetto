//! Deterministic helpers shared by the op unit tests. The RNG is a tiny
//! xorshift, not the `rand` crate: the plan introduces `rand` in epoch 5 for
//! sampling, and tests want reproducibility without a runtime dependency.

use crate::tensor::{Shape, Tensor};

/// xorshift64* -- adequate for filling test tensors, not for sampling.
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

    pub fn tensor(&mut self, rows: usize, cols: usize, range: f32) -> Tensor {
        let data = (0..rows * cols).map(|_| self.next_f32(range)).collect();
        Tensor::new(Shape::new(rows, cols), data)
    }
}

/// Matmul written straight from the definition with explicit flat indexing --
/// an independent reference for the convention `out[i, j] = dot(input row i,
/// weight row j)`.
pub fn naive_matmul(input: &Tensor, weight: &Tensor) -> Tensor {
    let (m, k, n) = (input.rows(), input.cols(), weight.rows());
    assert_eq!(k, weight.cols());
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += input.data()[i * k + l] * weight.data()[j * k + l];
            }
            out[i * n + j] = acc;
        }
    }
    Tensor::new(Shape::new(m, n), out)
}

/// Asserts two tensors have the same shape and elementwise-close data.
pub fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    for (i, (&x, &y)) in a.data().iter().zip(b.data()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "element {i}: {x} vs {y} exceeds tol {tol}"
        );
    }
}

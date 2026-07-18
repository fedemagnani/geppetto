use std::mem;
use std::ops::{Mul, MulAssign};

use crate::tensor::{Shape, Tensor};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Matrix multiply in the ggml/GGUF weight convention, exposed as `*`.
///
/// The left operand `self` is `[m, k]` row-major: `m` vectors of length `k`
/// (e.g. `m` tokens, `k` features). The right operand `rhs` is `[n, k]`
/// row-major: `n` output rows, each the `k` input weights of one output
/// feature. This is how GGUF stores a linear weight -- "transposed" relative
/// to a math `[k, n]` matrix -- so no transpose happens at runtime and both
/// operands' rows are contiguous, the inner loop being a dot product of two
/// contiguous slices.
///
/// Returns `[m, n]` with `out[i, j] = dot(self row i, rhs row j)`, i.e.
/// `self @ rhs^T`, which is ggml's `mul_mat(rhs, self)`. Note this is not the
/// textbook `A @ B`: `*` follows the weight convention, not math layout.
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        let (m, k, n) = (self.rows(), self.cols(), rhs.rows());
        assert_eq!(
            k,
            rhs.cols(),
            "matmul: contracted dim mismatch, {} vs {}",
            self.shape(),
            rhs.shape(),
        );

        let mut out = vec![0.0f32; m * n];
        for (i, out_row) in out.chunks_mut(n).enumerate() {
            let in_row = self.row(i);
            for (j, slot) in out_row.iter_mut().enumerate() {
                *slot = dot(in_row, rhs.row(j));
            }
        }
        Tensor::new(Shape::new(m, n), out)
    }
}

/// `self *= rhs` computes `self * rhs` (the same convention as [`Mul`]) into
/// `self`'s own buffer, avoiding the fresh `[m, n]` allocation the borrowing
/// `*` makes.
///
/// One output row `[m, n]` moves through a reused `n`-wide scratch, because a
/// row's dest slice overlaps its source slice whenever `n != k`. When
/// `n <= k` the product fits in the existing buffer, so no reallocation
/// happens (the common shrink is the FFN-down projection); when `n > k` the
/// buffer must grow once, and rows are filled back-to-front so a write never
/// lands on a source row still to be read.
impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: &Tensor) {
        let (m, k, n) = (self.rows(), self.cols(), rhs.rows());
        assert_eq!(
            k,
            rhs.cols(),
            "matmul: contracted dim mismatch, {} vs {}",
            self.shape(),
            rhs.shape(),
        );

        // own the buffer (mem::take leaves an empty Vec, no allocation) so a
        // row's source read and dest write are plain sequential borrows
        let mut data = mem::take(&mut self.data);
        let mut scratch = vec![0.0f32; n];

        let mut fill_row = |data: &mut [f32], i: usize| {
            for (j, slot) in scratch.iter_mut().enumerate() {
                *slot = dot(&data[i * k..i * k + k], rhs.row(j));
            }
            data[i * n..i * n + n].copy_from_slice(&scratch);
        };

        if n > k {
            data.resize(m * n, 0.0);
            for i in (0..m).rev() {
                fill_row(&mut data, i);
            }
        } else {
            for i in 0..m {
                fill_row(&mut data, i);
            }
            data.truncate(m * n);
        }

        self.data = data;
        self.shape = Shape::new(m, n);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::test_support::{Rng, assert_close, naive_matmul};
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn hand_computed_pins_the_convention() {
        // input: 2 tokens x 3 features
        let input = Tensor::new(Shape::new(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // weight: 2 outputs, each 3 input weights
        let weight = Tensor::new(Shape::new(2, 3), vec![1.0, 0.0, -1.0, 1.0, 1.0, 1.0]);
        let out = &input * &weight;
        assert_eq!(out.shape(), Shape::new(2, 2));
        // out[t, o] = dot(input[t], weight[o])
        assert_eq!(out.data(), &[-2.0, 6.0, -2.0, 15.0]);
    }

    #[test]
    fn agrees_with_naive_reference_on_random_shapes() {
        let mut rng = Rng::new(0xA11CE);
        for &(m, k, n) in &[(1, 1, 1), (3, 4, 2), (5, 5, 5), (2, 7, 3), (8, 1, 4)] {
            let input = rng.tensor(m, k, 3.0);
            let weight = rng.tensor(n, k, 3.0);
            assert_close(&(&input * &weight), &naive_matmul(&input, &weight), 1e-4);
        }
    }

    #[test]
    fn mul_assign_hand_computed() {
        let mut input = Tensor::new(Shape::new(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weight = Tensor::new(Shape::new(2, 3), vec![1.0, 0.0, -1.0, 1.0, 1.0, 1.0]);
        input *= &weight;
        assert_eq!(input.shape(), Shape::new(2, 2));
        assert_eq!(input.data(), &[-2.0, 6.0, -2.0, 15.0]);
    }

    #[test]
    fn mul_assign_matches_borrowed_mul_for_shrink_same_and_grow() {
        // n < k (shrink, reuses buffer), n == k, and n > k (grow)
        let mut rng = Rng::new(0xBEEF);
        for &(m, k, n) in &[
            (3, 6, 2),
            (3, 4, 4),
            (3, 2, 8),
            (1, 5, 1),
            (5, 1, 5),
            (4, 4, 4),
        ] {
            let a = rng.tensor(m, k, 2.0);
            let b = rng.tensor(n, k, 2.0);
            let expected = &a * &b;
            let mut got = a.clone();
            got *= &b;
            assert_eq!(got.shape(), expected.shape(), "shape for ({m},{k},{n})");
            assert_close(&got, &expected, 1e-5);
        }
    }

    #[test]
    fn mul_assign_shrink_keeps_the_original_capacity() {
        // n <= k: the product fits, so the backing allocation is not grown
        let mut t = Tensor::new(Shape::new(4, 4), vec![1.0; 16]);
        let cap_before = t.data().len();
        let w = Tensor::new(Shape::new(2, 4), vec![1.0; 8]);
        t *= &w;
        assert_eq!(t.shape(), Shape::new(4, 2));
        assert!(t.data().len() <= cap_before);
    }

    #[test]
    #[should_panic(expected = "contracted dim mismatch")]
    fn mismatched_inner_dim_panics() {
        let input = Tensor::new(Shape::new(1, 3), vec![1.0, 2.0, 3.0]);
        let weight = Tensor::new(Shape::new(1, 2), vec![1.0, 2.0]);
        let _ = &input * &weight;
    }
}

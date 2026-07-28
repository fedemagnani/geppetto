#[cfg(test)]
use std::mem;
#[cfg(test)]
use std::ops::{Mul, MulAssign};

#[cfg(test)]
use crate::tensor::Tensor;
use crate::tensor::{Shape, TensorView, TensorViewMut};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Matrix multiply in the ggml/GGUF weight convention.
///
/// `a` is `[m, k]` row-major: `m` vectors of length `k` (e.g. `m` tokens, `k`
/// features). `b` is `[n, k]` row-major: `n` output rows, each the `k` input
/// weights of one output feature. This is how GGUF stores a linear weight --
/// "transposed" relative to a math `[k, n]` matrix -- so no transpose happens
/// at runtime and both operands' rows are contiguous, the inner loop being a
/// dot product of two contiguous slices.
///
/// Writes `[m, n]` into `out`, fully overwriting it:
/// `out[i, j] = dot(a row i, b row j)`, i.e. `a @ b^T`, which is ggml's
/// `mul_mat(b, a)`. Note this is not the textbook `A @ B`: for that, see
/// [`matmul_nn`].
#[hotpath::measure]
pub fn matmul(a: TensorView, b: TensorView, mut out: TensorViewMut) {
    let (m, k, n) = (a.rows(), a.cols(), b.rows());
    assert_eq!(
        k,
        b.cols(),
        "matmul: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    for i in 0..m {
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        for (j, slot) in out_row.iter_mut().enumerate() {
            *slot = dot(a_row, b.row(j));
        }
    }
}

/// Textbook matrix multiply `out = a @ b`: `a` is `[m, k]`, `b` is `[k, n]`,
/// `out` is `[m, n]`. Delegates to [`matmul_nn_causal`] with a saturating
/// bound so every row accumulates over all of `k`.
pub fn matmul_nn(a: TensorView, b: TensorView, out: TensorViewMut) {
    let k = a.cols();
    matmul_nn_causal(a, b, out, k);
}

/// [`matmul_nn`] with the causal accumulation bound: `out` row `i` sums
/// `a[i, t] * b row t` only over `t < n_past + i + 1` (clamped to `k`), the
/// same valid prefix [`softmax_causal`] normalizes over. Columns of `a`
/// beyond the bound are never read, so they may hold garbage (or NaN
/// poison).
///
/// Runs as a row-axpy over `b`'s contiguous rows. Fully overwrites `out`:
/// each output row is zero-initialized before accumulating, so the `+=` is
/// internal and never observable by callers.
///
/// [`softmax_causal`]: crate::tensor::softmax_causal
#[hotpath::measure]
pub fn matmul_nn_causal(a: TensorView, b: TensorView, mut out: TensorViewMut, n_past: usize) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    assert_eq!(
        k,
        b.rows(),
        "matmul_nn: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nn: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    for i in 0..m {
        let bound = (n_past + i + 1).min(k);
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        out_row.fill(0.0);
        for (t, &coeff) in a_row[..bound].iter().enumerate() {
            let b_row = b.row(t);
            for (slot, &v) in out_row.iter_mut().zip(b_row) {
                *slot += coeff * v;
            }
        }
    }
}

/// The [`matmul`] weight convention, exposed as `*` on owned tensors:
/// allocates the `[m, n]` output and delegates.
#[cfg(test)]
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        let mut out = Tensor::zeros(Shape::new(self.rows(), rhs.rows()));
        matmul(self.as_view(), rhs.as_view(), out.as_view_mut());
        out
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
#[cfg(test)]
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
    use crate::tensor::{Shape, Tensor, TensorView, matmul, matmul_nn, matmul_nn_causal};

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
    fn strided_operand_matches_a_copied_column_block() {
        // a [2, 6] fused activation; its width-2 block at column 2 used as
        // the matmul input, once via a strided view, once via a copy
        let mut rng = Rng::new(0xFACE);
        let fused = rng.tensor(2, 6, 2.0);
        let weight = rng.tensor(3, 2, 2.0);

        let copied_rows: Vec<f32> = fused
            .data()
            .chunks(6)
            .flat_map(|row| row[2..4].to_vec())
            .collect();
        let copied = Tensor::new(Shape::new(2, 2), copied_rows);
        let expected = &copied * &weight;

        let block = &fused.data()[2..10];
        let a = TensorView::strided(block, Shape::new(2, 2), 6);
        let mut out = Tensor::zeros(Shape::new(2, 3));
        matmul(a, weight.as_view(), out.as_view_mut());
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn matmul_nn_hand_computed() {
        // A [2, 2] @ B [2, 3], textbook layout
        let a = Tensor::new(Shape::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(Shape::new(2, 3), vec![1.0, 0.0, 2.0, 0.0, 1.0, 3.0]);
        let mut out = Tensor::zeros(Shape::new(2, 3));
        matmul_nn(a.as_view(), b.as_view(), out.as_view_mut());
        assert_eq!(out.data(), &[1.0, 2.0, 8.0, 3.0, 4.0, 18.0]);
    }

    #[test]
    fn matmul_nn_agrees_with_the_transpose_trick() {
        // out = probs @ v equals the `a @ b^T` convention applied to v^T
        let mut rng = Rng::new(0xD07);
        let probs = rng.tensor(3, 4, 1.0);
        let v = rng.tensor(4, 2, 2.0);

        let mut v_t = Tensor::zeros(Shape::new(2, 4));
        for r in 0..4 {
            for c in 0..2 {
                v_t.row_mut(c)[r] = v.row(r)[c];
            }
        }
        let expected = &probs * &v_t;

        let mut out = Tensor::zeros(Shape::new(3, 2));
        matmul_nn(probs.as_view(), v.as_view(), out.as_view_mut());
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn causal_bound_never_reads_past_the_prefix() {
        // row i may only touch a[i, ..n_past + i + 1]; columns beyond hold
        // NaN, so any out-of-bound read would poison the output
        let n_past = 0;
        let a = Tensor::new(
            Shape::new(3, 3),
            vec![2.0, f32::NAN, f32::NAN, 1.0, 3.0, f32::NAN, 0.5, -1.0, 2.0],
        );
        let b = Tensor::new(Shape::new(3, 2), vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0]);
        let mut out = Tensor::zeros(Shape::new(3, 2));
        matmul_nn_causal(a.as_view(), b.as_view(), out.as_view_mut(), n_past);
        let expected = [
            2.0, 4.0, // 2 * b0
            31.0, 62.0, // 1 * b0 + 3 * b1
            190.5, 381.0, // 0.5 * b0 - 1 * b1 + 2 * b2
        ];
        assert_eq!(out.data(), &expected);
    }

    #[test]
    fn causal_bound_overwrites_stale_output() {
        // the driver's contract: out is fully overwritten, never accumulated
        // into across calls
        let a = Tensor::new(Shape::new(1, 1), vec![3.0]);
        let b = Tensor::new(Shape::new(1, 2), vec![1.0, 2.0]);
        let mut out = Tensor::new(Shape::new(1, 2), vec![f32::NAN, 7.0]);
        matmul_nn_causal(a.as_view(), b.as_view(), out.as_view_mut(), 0);
        assert_eq!(out.data(), &[3.0, 6.0]);
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

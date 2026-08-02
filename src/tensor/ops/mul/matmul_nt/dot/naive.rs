use crate::tensor::{TensorView, TensorViewMut};

use super::{DotMatMulNt, DotProduct, matmul_nt_dot};

/// The serial dot: one accumulator, one dependency chain, the order the
/// source spells. Slow on purpose -- it is the semantic baseline every
/// research dot is compared against.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveDotProduct;

impl DotProduct for NaiveDotProduct {
    #[inline(always)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

/// The baseline kernel: raw weights, no scratch, the triple loop of
/// [`matmul_nt`]. Every research kernel benchmarks against this.
pub type NaiveMatMulNt = DotMatMulNt<NaiveDotProduct>;

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
///
/// The raw-view driver attention's Q@K^T score matmul calls directly; in
/// hotpath reports it shows under the family driver's label.
///
/// [`matmul_nn`]: crate::tensor::matmul_nn()
pub fn matmul_nt(a: TensorView, b: TensorView, out: TensorViewMut) {
    matmul_nt_dot::<NaiveDotProduct>(a, b, out);
}

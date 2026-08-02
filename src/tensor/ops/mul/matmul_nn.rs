use crate::tensor::{TensorView, TensorViewMut, matmul_nn_causal};

/// Textbook matrix multiply `out = a @ b`: `a` is `[m, k]`, `b` is `[k, n]`,
/// `out` is `[m, n]`. Delegates to [`matmul_nn_causal`] with a saturating
/// bound so every row accumulates over all of `k`.
pub fn matmul_nn(a: TensorView, b: TensorView, out: TensorViewMut) {
    let k = a.cols();
    matmul_nn_causal(a, b, out, k);
}

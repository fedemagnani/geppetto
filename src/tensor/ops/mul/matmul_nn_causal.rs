use crate::tensor::{Shape, TensorView, TensorViewMut};

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
/// [`matmul_nn`]: crate::tensor::matmul_nn
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

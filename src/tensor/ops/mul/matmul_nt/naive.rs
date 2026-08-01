use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// The baseline kernel: raw weights, no scratch, the triple loop of
/// [`matmul_nt`]. Every research kernel benchmarks against this.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveMatMulNt;

impl MatmulNtKernel for NaiveMatMulNt {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt(a, b.view(), out);
    }
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
///
/// [`matmul_nn`]: crate::tensor::matmul_nn
#[hotpath::measure]
pub fn matmul_nt(a: TensorView, b: TensorView, mut out: TensorViewMut) {
    let (m, k, n) = (a.rows(), a.cols(), b.rows());
    assert_eq!(
        k,
        b.cols(),
        "matmul_nt: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nt: out must be [{m}, {n}], got {}",
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

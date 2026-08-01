use crate::tensor::{MatmulNtKernel, TensorView, TensorViewMut, WeightTensor, matmul_nt};

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

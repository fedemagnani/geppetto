use crate::tensor::{MatmulKernel, TensorView, TensorViewMut, WeightTensor, matmul};

/// The baseline kernel: raw weights, no scratch, the triple loop of
/// [`matmul`]. Every research kernel benchmarks against this.
#[derive(Debug, Default, Clone, Copy)]
pub struct NaiveMatMul;

impl MatmulKernel for NaiveMatMul {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul(a, b.view(), out);
    }
}

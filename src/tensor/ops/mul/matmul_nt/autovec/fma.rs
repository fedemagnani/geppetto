use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

use super::{ACC_VECTORS, FLOATS_IN_REGISTER};

/// [`AutoVecMatMulNt`] with the multiply and add fused explicitly: the lane
/// update is [`f32::mul_add`], which compiles to one `fmla.4s` where
/// autovec emits a `fmul.4s` + `fadd.4s` pair.
///
/// The fusion must be spelled out because Rust never contracts
/// `acc + a * b` on its own: an FMA rounds once where mul-then-add rounds
/// twice, and the language will not change your program's numerics for
/// speed. Explicit fusion halves the FP micro-ops per accumulation --
/// headroom where the FP pipes are the wall (prefill), invisible where
/// bandwidth is (decode). One structural difference from autovec: the
/// accumulator dependency chain now runs through the fmla's full ~4-cycle
/// latency instead of the bare add's, so its optimal chain depth may sit
/// deeper -- the bench sweeps both.
///
/// Fusing also shifts results by rounding relative to both the serial
/// reference and autovec, so comparisons stay tolerance-based.
///
/// [`AutoVecMatMulNt`]: super::AutoVecMatMulNt
#[derive(Debug, Default, Clone, Copy)]
pub struct FmaMatMulNt<const LANES: usize = { FLOATS_IN_REGISTER * ACC_VECTORS }>;

impl<const LANES: usize> MatmulNtKernel for FmaMatMulNt<LANES> {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt_fma::<LANES>(a, b.view(), out);
    }
}

/// [`matmul_nt`](crate::tensor::matmul_nt) with the fused dot: same
/// convention, same asserts, same loop nest.
#[hotpath::measure]
fn matmul_nt_fma<const LANES: usize>(a: TensorView, b: TensorView, mut out: TensorViewMut) {
    let (m, k, n) = (a.rows(), a.cols(), b.rows());
    assert_eq!(
        k,
        b.cols(),
        "matmul_nt_fma: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nt_fma: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    for i in 0..m {
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        for (j, slot) in out_row.iter_mut().enumerate() {
            *slot = dot_fma::<LANES>(a_row, b.row(j));
        }
    }
}

fn dot_fma<const LANES: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let a_tail = a_chunks.remainder();
    let b_tail = b_chunks.remainder();

    let mut acc = [0.0f32; LANES];

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        for lane in 0..LANES {
            acc[lane] = a_chunk[lane].mul_add(b_chunk[lane], acc[lane]);
        }
    }

    let mut sum = 0.0f32;
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum = x.mul_add(*y, sum);
    }
    for lane in acc {
        sum += lane;
    }
    sum
}

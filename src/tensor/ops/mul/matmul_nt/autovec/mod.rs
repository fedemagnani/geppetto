use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

mod fma;

pub use fma::FmaMatMulNt;

/// Bytes of one aarch64 NEON vector register: 128 bits wide.
const BYTES_IN_VECTOR_REGISTER: usize = 128 / 8;

/// f32 lanes one vector register holds.
const FLOATS_IN_REGISTER: usize = BYTES_IN_VECTOR_REGISTER / size_of::<f32>();

/// How many vector-wide partial sums the accumulator splits into -- the
/// shape of the `acc` array, not an op count (total FMAs are fixed by
/// `k`). Each partial sum is an independent dependency chain, so while one
/// chain's multiply-add is in flight the others issue: the analytic target
/// is FP latency x issue rate, capped by what fits the 32-register NEON
/// file alongside the loaded operands. Register residency is a request,
/// not a guarantee -- the allocator honors it only while everything fits
/// (at 16 vectors it demotes the array to the stack). The bench sweep pins
/// 4 as the peak: 2 vectors leaves latency exposed (~26 GFLOP/s), 8
/// regresses (~30), 16 spills and halves throughput (~18).
const ACC_VECTORS: usize = 4;

/// [`NaiveMatMulNt`] with the serial dependency chain broken: each dot
/// product accumulates into `LANES` independent partial sums, combined at
/// the end.
///
/// The naive `sum += a[l] * b[l]` is one serial chain: floating-point
/// addition is not associative, so LLVM must honor the order -- no
/// vectorization, and every FMA waits out the previous one. Splitting the
/// reduction re-associates it in the source, which hides FMA latency by
/// itself and licenses the autovectorizer to map the lane loop onto SIMD
/// registers. Memory access is identical to naive (contiguous row-row
/// dots): raw weights, zero scratch, a pure microkernel experiment.
///
/// `LANES` is a const generic because it shapes inner-loop codegen: the
/// accumulators only live in SIMD registers when the lane loop has a
/// compile-time trip count (a runtime field would spill them to memory and
/// re-serialize the loop). The default is no free choice: it is
/// [`FLOATS_IN_REGISTER`] `x` [`ACC_VECTORS`], both derived from the
/// machine; other
/// widths are one turbofish away for the bench to sweep. GPT-2's
/// contracted dims (768..5120, all multiples of 64) never hit the scalar
/// tail.
///
/// The changed summation order shifts results by rounding error relative to
/// [`NaiveMatMulNt`], so cross-kernel comparisons are tolerance-based,
/// never bitwise.
///
/// [`NaiveMatMulNt`]: crate::tensor::NaiveMatMulNt
#[derive(Debug, Default, Clone, Copy)]
pub struct AutoVecMatMulNt<const LANES: usize = { FLOATS_IN_REGISTER * ACC_VECTORS }>;

impl<const LANES: usize> MatmulNtKernel for AutoVecMatMulNt<LANES> {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt_unrolled::<LANES>(a, b.view(), out);
    }
}

/// [`matmul_nt`](crate::tensor::matmul_nt) with the unrolled dot: same
/// convention, same asserts, same loop nest.
#[hotpath::measure]
fn matmul_nt_unrolled<const LANES: usize>(a: TensorView, b: TensorView, mut out: TensorViewMut) {
    let (m, k, n) = (a.rows(), a.cols(), b.rows());
    assert_eq!(
        k,
        b.cols(),
        "matmul_nt_unrolled: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nt_unrolled: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    for i in 0..m {
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        for (j, slot) in out_row.iter_mut().enumerate() {
            *slot = dot::<LANES>(a_row, b.row(j));
        }
    }
}

fn dot<const LANES: usize>(a: &[f32], b: &[f32]) -> f32 {
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let a_tail = a_chunks.remainder();
    let b_tail = b_chunks.remainder();

    // this is likely going to use lanes/4 registries to write the output of the
    // SIMD operation, since with `f32` a `v*`nregister can hold up to 4 `f32`.
    // Thus, the first register is going to manage the first 4 floats, the second
    // register the next 4 floats and so on.
    let mut acc = [0.0f32; LANES];

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        for lane in 0..LANES {
            acc[lane] += a_chunk[lane] * b_chunk[lane];
        }
    }

    let mut sum = 0.0f32;
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += x * y;
    }
    for lane in acc {
        sum += lane;
    }
    sum
}

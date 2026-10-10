//! The hand-vectorized tile: the register-tiled GEMV of `tiled` with the
//! vectors spelled as NEON intrinsics instead of hinted to the
//! autovectorizer.
//!
//! Exists because the autovectorized tile falls off a cliff past 2 rows:
//! the emitted assembly shows LLVM keeping a `[[f32; 16]; 2]` accumulator
//! block in registers (the ideal 8-`fmla.4s` loop) but compiling the same
//! code at 4 rows to fully scalar `fmadd`s round-tripping every lane
//! through the stack. Intrinsics make the register intent explicit:
//! `float32x4_t` accumulators, `vld1q_f32` loads, one `vfmaq_f32` per
//! lane vector. Whether that recovers the wider tiles is exactly what the
//! bench and the emitted assembly are asked to verify.
//!
//! Like the rest of the crate's constants, this module assumes the aarch64
//! NEON machine it is researched on; it is not `cfg`-gated.

use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

use super::{DotProduct, DotProductAutoVec};
use crate::tensor::ops::mul_add::Fma;
use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// f32 lanes in one 128-bit NEON vector.
const FLOATS_IN_VECTOR: usize = 4;

/// The intrinsics tile: `ROWS` weight rows per loaded `a` chunk, `VECS`
/// accumulator vectors (of 4 lanes each) per row. Raw weights, zero
/// scratch, fused lanes by construction -- `fmla` is what the intrinsic
/// is. The register budget is exact rather than hoped-for:
/// `ROWS x VECS` accumulators + `VECS` loaded `a` vectors must stay
/// within the 32-register NEON file, `(ROWS + 1) x VECS <= 32`.
#[derive(Debug, Default, Clone, Copy)]
pub struct MatMulNtNeonTiled<const ROWS: usize = 4, const VECS: usize = 4>;

impl<const ROWS: usize, const VECS: usize> MatmulNtKernel for MatMulNtNeonTiled<ROWS, VECS> {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt_neon::<ROWS, VECS>(a, b.view(), out);
    }
}

/// The driver: identical convention, asserts and tail policy to the
/// autovectorized tiled driver -- remainder rows fall back to the best
/// single-row dot.
#[hotpath::measure]
fn matmul_nt_neon<const ROWS: usize, const VECS: usize>(
    a: TensorView,
    b: TensorView,
    mut out: TensorViewMut,
) {
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

    let n_tiled = n / ROWS * ROWS;
    for i in 0..m {
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        for j in (0..n_tiled).step_by(ROWS) {
            let b_rows: [&[f32]; ROWS] = std::array::from_fn(|r| b.row(j + r));
            let sums = tile::<ROWS, VECS>(a_row, &b_rows);
            out_row[j..j + ROWS].copy_from_slice(&sums);
        }
        let tail = out_row[n_tiled..].iter_mut();
        for (j, slot) in tail.enumerate() {
            *slot = DotProductAutoVec::<Fma>::dot(a_row, b.row(n_tiled + j));
        }
    }
}

/// One tile, registers held by hand: load `VECS` `a` vectors once, fold
/// them into all `ROWS` accumulator blocks with one `vfmaq_f32` per lane
/// vector, `1 + 1/ROWS` loads per FMA. Const trip counts unroll both
/// register loops fully. The `k % (4 x VECS)` tail is scalar per row.
#[inline(always)]
fn tile<const ROWS: usize, const VECS: usize>(
    a_row: &[f32],
    b_rows: &[&[f32]; ROWS],
) -> [f32; ROWS] {
    const {
        assert!(ROWS > 0 && VECS > 0);
    }
    let k = a_row.len();
    for b_row in b_rows {
        assert_eq!(b_row.len(), k);
    }
    let lanes = FLOATS_IN_VECTOR * VECS;
    let k_tiled = k / lanes * lanes;

    // SAFETY: every `vld1q_f32` reads 4 floats at `base + 4 * v` with
    // `base + lanes <= k_tiled <= k`; the length asserts above extend
    // that bound to every `b` row.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); VECS]; ROWS];
        let a_ptr = a_row.as_ptr();
        let mut base = 0;
        while base < k_tiled {
            let mut a_vecs = [vdupq_n_f32(0.0); VECS];
            for (v, a_vec) in a_vecs.iter_mut().enumerate() {
                *a_vec = vld1q_f32(a_ptr.add(base + FLOATS_IN_VECTOR * v));
            }
            for (row, acc_row) in acc.iter_mut().enumerate() {
                let b_ptr = b_rows[row].as_ptr();
                for (v, acc_vec) in acc_row.iter_mut().enumerate() {
                    let b_vec = vld1q_f32(b_ptr.add(base + FLOATS_IN_VECTOR * v));
                    *acc_vec = vfmaq_f32(*acc_vec, a_vecs[v], b_vec);
                }
            }
            base += lanes;
        }

        let a_tail = &a_row[k_tiled..];
        let mut sums = [0.0f32; ROWS];
        for (row, sum) in sums.iter_mut().enumerate() {
            let acc_row = acc[row];
            let mut vec_sum = acc_row[0];
            for acc_vec in &acc_row[1..] {
                vec_sum = vaddq_f32(vec_sum, *acc_vec);
            }
            let mut tail_sum = vaddvq_f32(vec_sum);
            let b_tail = &b_rows[row][k_tiled..];
            for (x, y) in a_tail.iter().zip(b_tail) {
                tail_sum = x.mul_add(*y, tail_sum);
            }
            *sum = tail_sum;
        }
        sums
    }
}

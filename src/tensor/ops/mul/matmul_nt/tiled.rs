//! The register-tiled GEMV kernel: one `a` row against `ROWS` weight rows
//! at a time, each loaded `a` vector reused `ROWS` times.
//!
//! The dot family measured itself into the load-port ceiling: two loads
//! feed every FMA, so the FP pipes idle waiting on loads (see [`Fma`]).
//! Tiling attacks exactly that ratio -- a tile loads one `a` chunk and
//! `ROWS` `b` chunks per `ROWS` accumulations, `1 + 1/ROWS` loads per FMA
//! instead of 2. Latency hiding shifts sources too: the dot needed 4
//! accumulator vectors on one row to keep the pipes busy; here `ROWS`
//! independent rows provide chains for free, so `LANES` can shrink and
//! the register file spend moves from depth to breadth.
//!
//! Breaks the one-dot-per-output shape, so it implements
//! [`MatmulNtKernel`] directly rather than riding [`DotMatMulNt`].
//!
//! [`DotMatMulNt`]: super::DotMatMulNt

use std::marker::PhantomData;

use super::{DotProduct, DotProductAutoVec};
use crate::tensor::ops::mul_add::{Fma, MulAdd, Unfused};
use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// The row-tiled kernel: `out[i, j..j+ROWS] = tile(a row i, b rows j..)`,
/// raw weights (identity pack), zero scratch. `ROWS` is the reuse factor
/// per loaded `a` chunk; `LANES` the accumulator width per row. Both must
/// be compile-time so the `ROWS x LANES` accumulator block stays in
/// registers: on 128-bit NEON that block plus one `LANES`-wide `a` chunk
/// occupies `(ROWS + 1) x LANES / 4` of the 32 vector registers.
///
/// The default tile is the bench sweep's peak, 2 rows x 16 lanes: the dot
/// family's proven width, reused once. The naive register math promised
/// more -- (4, 16) fits in 20 vectors on paper -- but every ROWS > 2 tile
/// measured at or below the untiled dot, (4, 16) and (8, 8) collapsing to
/// ~7 GFLOP/s: LLVM keeps a 2-row accumulator block in registers but
/// spills wider ones to the stack, and a spilled tile pays its loads back
/// with interest.
#[derive(Debug, Default, Clone, Copy)]
pub struct MatMulNtTiled<A, const ROWS: usize = 2, const LANES: usize = 16> {
    _strategy: PhantomData<A>,
}

/// The tiled kernel with plain mul-then-add lanes.
pub type MatMulNtTiledUnfused<const ROWS: usize = 2, const LANES: usize = 16> =
    MatMulNtTiled<Unfused, ROWS, LANES>;

/// The tiled kernel with explicitly fused lanes.
pub type MatMulNtTiledFma<const ROWS: usize = 2, const LANES: usize = 16> =
    MatMulNtTiled<Fma, ROWS, LANES>;

impl<A: MulAdd, const ROWS: usize, const LANES: usize> MatmulNtKernel
    for MatMulNtTiled<A, ROWS, LANES>
{
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt_tiled::<A, ROWS, LANES>(a, b.view(), out);
    }
}

/// The driver: same convention and asserts as the dot driver, but the
/// inner loop walks `n` in steps of `ROWS`. The `n % ROWS` remainder rows
/// fall back to the family's best single-row dot at its own default lane
/// count -- a tail of at most `ROWS - 1` outputs per call, not worth a
/// second tile shape.
#[hotpath::measure]
fn matmul_nt_tiled<A: MulAdd, const ROWS: usize, const LANES: usize>(
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
            let sums = tile::<A, ROWS, LANES>(a_row, &b_rows);
            out_row[j..j + ROWS].copy_from_slice(&sums);
        }
        let tail = out_row[n_tiled..].iter_mut();
        for (j, slot) in tail.enumerate() {
            *slot = DotProductAutoVec::<A>::dot(a_row, b.row(n_tiled + j));
        }
    }
}

/// One tile: `ROWS` dots sharing every `a` load. The chunk loop loads one
/// `LANES`-wide `a` chunk, then folds it into all `ROWS` accumulator
/// blocks before the next load -- the reuse the whole kernel exists for.
/// The `k % LANES` tail is scalar per row, same trade as the dot family.
#[inline(always)]
fn tile<A: MulAdd, const ROWS: usize, const LANES: usize>(
    a_row: &[f32],
    b_rows: &[&[f32]; ROWS],
) -> [f32; ROWS] {
    let k = a_row.len();
    for b_row in b_rows {
        assert_eq!(b_row.len(), k);
    }
    let k_tiled = k / LANES * LANES;

    let mut acc = [[0.0f32; LANES]; ROWS];
    let a_chunks = a_row[..k_tiled].chunks_exact(LANES);
    for (c, a_chunk) in a_chunks.enumerate() {
        let base = c * LANES;
        for (row, acc_row) in acc.iter_mut().enumerate() {
            let b_chunk = &b_rows[row][base..base + LANES];
            for lane in 0..LANES {
                acc_row[lane] = A::mul_add(a_chunk[lane], b_chunk[lane], acc_row[lane]);
            }
        }
    }

    let a_tail = &a_row[k_tiled..];
    let mut sums = [0.0f32; ROWS];
    for (row, sum) in sums.iter_mut().enumerate() {
        let b_tail = &b_rows[row][k_tiled..];
        let mut tail_sum = 0.0f32;
        for (x, y) in a_tail.iter().zip(b_tail) {
            tail_sum = A::mul_add(*x, *y, tail_sum);
        }
        for lane in acc[row] {
            tail_sum += lane;
        }
        *sum = tail_sum;
    }
    sums
}

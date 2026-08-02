//! The chunked-accumulator dots: swappable lane arithmetic under one
//! chunk/tail/reduce scaffold.
//!
//! [`AutoVecDotProduct`] implements [`DotProduct`] for the whole family; a
//! submodule contributes only a [`MulAdd`] impl, the single expression
//! saying how one lane folds a product into its running total. That
//! expression is the entire experiment.

use std::marker::PhantomData;

use super::{DotMatMulNt, DotProduct};

mod fma;
mod unfused;

pub use fma::Fma;
pub use unfused::Unfused;

/// Bytes of one aarch64 NEON vector register: 128 bits wide.
const BYTES_IN_VECTOR_REGISTER: usize = 128 / 8;

/// f32 lanes one vector register holds.
const FLOATS_IN_REGISTER: usize = BYTES_IN_VECTOR_REGISTER / size_of::<f32>();

/// Independent accumulator vectors kept in flight -- the latency-hiding
/// depth, not an op count (total FMAs are fixed by `k`). While one chain's
/// ~3-4 cycle FMA is in progress the others issue, so the analytic target
/// is latency x issue rate, capped by what fits the 32-register NEON file
/// alongside the loaded operands. The bench sweep pins 4 as the peak: 2
/// chains leaves latency exposed (~26 GFLOP/s), 8 regresses (~30), 16
/// spills accumulators to the stack and halves throughput (~18).
const ACC_VECTORS: usize = 4;

/// The default lane count: every accumulator vector full, no more.
const DEFAULT_LANES: usize = FLOATS_IN_REGISTER * ACC_VECTORS;

/// How one lane folds a product into its running total, in the argument
/// order of [`f32::mul_add`]: `x * y + acc`. The only thing that
/// distinguishes dots of the chunked family.
pub trait MulAdd {
    fn mul_add(x: f32, y: f32, acc: f32) -> f32;
}

/// The [`NaiveDotProduct`] with its serial dependency chain broken:
/// `LANES` independent partial sums via the strategy `A`, combined at the
/// end. The re-association hides FMA latency and licenses the
/// autovectorizer (LLVM may not reorder a serial float sum itself), at the
/// price of rounding drift vs the serial reference -- comparisons are
/// tolerance-based, never bitwise.
///
/// `LANES` must be a const generic: accumulators only stay in SIMD
/// registers when the lane loop has a compile-time trip count. The default
/// is `FLOATS_IN_REGISTER x ACC_VECTORS`, both machine-derived;
/// other widths are one turbofish away for the bench to sweep.
///
/// [`NaiveDotProduct`]: super::NaiveDotProduct
#[derive(Debug, Default, Clone, Copy)]
pub struct AutoVecDotProduct<A, const LANES: usize = DEFAULT_LANES> {
    _strategy: PhantomData<A>,
}

/// The chunked kernel with plain mul-then-add lanes.
pub type AutoVecMatMulNt<const LANES: usize = DEFAULT_LANES> =
    DotMatMulNt<AutoVecDotProduct<Unfused, LANES>>;

/// The chunked kernel with explicitly fused lanes.
pub type FmaMatMulNt<const LANES: usize = DEFAULT_LANES> =
    DotMatMulNt<AutoVecDotProduct<Fma, LANES>>;

impl<A: MulAdd, const LANES: usize> DotProduct for AutoVecDotProduct<A, LANES> {
    #[inline(always)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
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
                acc[lane] = A::mul_add(a_chunk[lane], b_chunk[lane], acc[lane]);
            }
        }

        let mut sum = 0.0f32;
        for (x, y) in a_tail.iter().zip(b_tail) {
            sum = A::mul_add(*x, *y, sum);
        }
        for lane in acc {
            sum += lane;
        }
        sum
    }
}

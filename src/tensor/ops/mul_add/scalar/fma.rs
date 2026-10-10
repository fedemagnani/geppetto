use crate::tensor::ops::mul_add::MulAdd;

/// Explicitly fused multiply-add: [`f32::mul_add`], one `fmla.4s` per
/// accumulator vector where [`Unfused`] emits a `fmul.4s` + `fadd.4s`
/// pair.
///
/// The fusion must be spelled out because Rust never contracts
/// `acc + x * y` on its own: an FMA rounds once where mul-then-add rounds
/// twice, and the language will not change your program's numerics for
/// speed. Halves the FP micro-ops per accumulation -- headroom where the
/// FP pipes are the wall, invisible where loads or bandwidth are (which is
/// what the bench showed: parity with [`Unfused`] at the optimal lane
/// count, both sitting on the load-port ceiling).
///
/// [`Unfused`]: super::Unfused
#[derive(Debug, Default, Clone, Copy)]
pub struct Fma;

impl MulAdd for Fma {
    #[inline(always)]
    fn mul_add(x: f32, y: f32, acc: f32) -> f32 {
        x.mul_add(y, acc)
    }
}

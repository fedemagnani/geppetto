use crate::tensor::ops::mul_add::MulAdd;

/// Plain mul-then-add: two IEEE operations, two roundings, compiling to a
/// `fmul.4s` + `fadd.4s` pair per accumulator vector. The accumulator
/// dependency chain runs through the `fadd` only, so latency to hide is
/// the add's, not the FMA's.
#[derive(Debug, Default, Clone, Copy)]
pub struct Unfused;

impl MulAdd for Unfused {
    #[inline(always)]
    fn mul_add(x: f32, y: f32, acc: f32) -> f32 {
        x * y + acc
    }
}

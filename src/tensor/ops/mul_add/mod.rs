//! The lane-arithmetic seam: [`MulAdd`] is the single expression saying
//! how one lane folds a product into its running total. The chunked dots
//! and the register-tiled kernels are generic over it; `scalar` holds the
//! one-expression strategies.

mod scalar;

pub use scalar::{Fma, Unfused};

/// How one lane folds a product into its running total, in the argument
/// order of [`f32::mul_add`]: `x * y + acc`. The only thing that
/// distinguishes kernels of one chunked family.
pub trait MulAdd {
    fn mul_add(x: f32, y: f32, acc: f32) -> f32;
}

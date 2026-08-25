//! Matmul and elementwise-add drivers, one per file: free functions over
//! views ([`matmul_nt`], [`matmul_nn`], [`matmul_nn_causal`], [`add`]).
//! The lane-arithmetic strategies the kernels share live in `mul_add`.
//! Nonlinear functions live in `activation`, parametrized layers in `layer`.

mod add;
mod mul;
mod mul_add;

pub use add::add;
pub use mul::{
    DotAutoVecFma, DotAutoVecUnfused, DotProduct, DotProductAutoVec, DotProductNaive,
    MatMulNtAutoVecFma, MatMulNtAutoVecUnfused, MatMulNtNaive, MatMulNtNeonTiled,
    MatMulNtPackedNeon, MatMulNtTiled, MatMulNtTiledFma, MatMulNtTiledUnfused, MatmulNtKernel,
    PackedPanels, matmul_nn, matmul_nn_causal, matmul_nt,
};
pub use mul_add::{Fma, Unfused};

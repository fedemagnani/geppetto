//! Matmul and elementwise-add drivers, one per file: free functions over
//! views ([`matmul_nt`], [`matmul_nn`], [`matmul_nn_causal`], [`add`]).
//! Nonlinear functions live in `activation`, parametrized layers in `layer`.

mod add;
mod mul;

pub use add::add;
pub use mul::{
    DotAutoVecFma, DotAutoVecUnfused, DotProduct, DotProductAutoVec, DotProductNaive, Fma,
    MatMulNtAutoVecFma, MatMulNtAutoVecUnfused, MatMulNtNaive, MatMulNtNeonTiled,
    MatMulNtPackedNeon, MatMulNtTiled, MatMulNtTiledFma, MatMulNtTiledUnfused, MatmulNtKernel,
    PackedPanels, Unfused, matmul_nn, matmul_nn_causal, matmul_nt,
};

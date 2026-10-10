//! Matrix multiplication: the textbook `a @ b` drivers as one free function
//! per file, and the `a @ b^T` weight path -- [`MatmulNtKernel`] seam plus
//! its research kernels -- in `matmul_nt`.

// a `use mul::matmul_nn` downstream resolves the name in both namespaces
// (module and function), so these modules must be as visible as the
// functions they re-export
pub mod matmul_nn;
pub mod matmul_nn_causal;
pub mod matmul_nt;

#[cfg(test)]
mod tests;

pub use matmul_nn::matmul_nn;
pub use matmul_nn_causal::matmul_nn_causal;
pub use matmul_nt::{
    DotAutoVecFma, DotAutoVecUnfused, DotProduct, DotProductAutoVec, DotProductNaive,
    MatMulNtAutoVecFma, MatMulNtAutoVecUnfused, MatMulNtNaive, MatMulNtNeonTiled,
    MatMulNtPackedNeon, MatMulNtTiled, MatMulNtTiledFma, MatMulNtTiledUnfused, MatmulNtKernel,
    PackedPanels, matmul_nt,
};

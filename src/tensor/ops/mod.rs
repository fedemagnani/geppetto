//! Matmul and elementwise-add drivers, one per file: free functions over
//! views ([`matmul`], [`matmul_nn`], [`matmul_nn_causal`], [`add`]) plus the
//! operator overloads on [`crate::tensor::Tensor`] (`Add`/`AddAssign`,
//! `Mul`/`MulAssign`) that wrap them for owned operands. Nonlinear functions
//! live in `activation`, parametrized layers in `layer`.

mod add;
mod mul;

pub use add::add;
pub use mul::{matmul, matmul_nn, matmul_nn_causal};

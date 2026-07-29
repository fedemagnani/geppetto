//! Matmul and elementwise-add drivers, one per file: free functions over
//! views ([`matmul`], [`matmul_nn`], [`matmul_nn_causal`], [`add`]).
//! Nonlinear functions live in `activation`, parametrized layers in `layer`.

mod add;
mod mul;

pub use add::add;
pub use mul::{matmul, matmul_nn, matmul_nn_causal};

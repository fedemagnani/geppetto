//! Arithmetic operator overloads on [`crate::tensor::Tensor`], one per file:
//! `Add`/`AddAssign` (elementwise with single-row bias broadcast) and
//! `Mul`/`MulAssign` (the ggml matmul convention, `self @ rhs^T`). Nonlinear
//! functions live in `activation`, parametrized layers in `layer`.
//!
//! These modules hold only `impl` blocks for `Tensor`, so there is nothing to
//! re-export: the operators are reachable wherever `Tensor` is.

mod add;
mod mul;

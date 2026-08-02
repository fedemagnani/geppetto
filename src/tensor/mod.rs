//! CPU tensors and the ops the GPT-2 forward pass is built from.
//!
//! Everything works over borrowed views: ops are free functions consuming
//! [`TensorView`]/[`TensorViewMut`] mounted over the arena, and model weights
//! are [`WeightTensor`]s backed by the mmap'd GGUF file. Nothing here owns
//! activation memory -- the arena does.

mod activation;
mod dtype;
mod error;
pub mod layer;
mod ops;
mod shape;
mod view;
mod weight;

// unit tests see this unconditionally; bench targets are separate crates,
// so they reach it through the `test-utils` feature and the public path
#[cfg(any(test, feature = "test-utils"))]
pub mod test;

pub use activation::{gelu, softmax, softmax_causal};
pub use dtype::DType;
pub use error::TensorError;
pub use ops::{
    AutoVecMatMulNt, FmaMatMulNt, MatmulNtKernel, NaiveMatMulNt, add, matmul_nn, matmul_nn_causal,
    matmul_nt,
};
pub use shape::Shape;
pub use view::{TensorView, TensorViewMut};
pub use weight::WeightTensor;

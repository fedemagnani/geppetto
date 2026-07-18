//! Activation functions on [`crate::tensor::Tensor`], one per file: `gelu`
//! (the tanh approximation ggml uses for GPT-2's FFN) and `softmax` (the
//! attention distribution). Both are inherent methods; the modules hold only
//! `impl Tensor` blocks, so there is nothing to re-export.

mod gelu;
mod softmax;

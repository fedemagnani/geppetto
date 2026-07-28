//! Activation functions, one per file: `gelu` (the tanh approximation ggml
//! uses for GPT-2's FFN) and `softmax` (the attention distribution), each an
//! in-place free function over a view.

mod gelu;
mod softmax;

pub use gelu::gelu;
pub use softmax::{softmax, softmax_causal};

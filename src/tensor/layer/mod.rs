//! Neural-network layers, one per file: `layernorm` (ggml's `ggml_norm` plus
//! weight-scale and bias-shift) and `get_rows` (the embedding lookup, also
//! used to select output positions), each a free function over views plus a
//! thin allocating wrapper ([`NormLayer`], `Tensor::get_rows`).

mod get_rows;
mod norm;
mod simple;

pub use get_rows::get_rows;
pub use norm::{NormLayer, norm};
pub use simple::SimpleLayer;

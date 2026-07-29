//! Neural-network layers, one per file: `layernorm` (ggml's `ggml_norm` plus
//! weight-scale and bias-shift) and `get_rows` (the embedding lookup, also
//! used to select output positions), each a free function over views.

mod get_rows;
mod norm;

pub use get_rows::get_rows;
pub use norm::norm;

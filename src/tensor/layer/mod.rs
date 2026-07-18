//! Neural-network layers on [`crate::tensor::Tensor`], one per file:
//! `layernorm` (ggml's `ggml_norm` plus weight-scale and bias-shift) and
//! `get_rows` (the embedding lookup, also used to select output positions).
//! Both are inherent methods; the modules hold only `impl Tensor` blocks, so
//! there is nothing to re-export.

mod get_rows;
mod layernorm;

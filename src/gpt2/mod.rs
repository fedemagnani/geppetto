//! The GPT-2 model: hyperparameters, weights, and the eager forward pass.
//!
//! Reference: `src/models/gpt2.cpp` in llama.cpp. This is a behavioral port --
//! a direct forward pass over [`crate::tensor::Tensor`] ops -- not a
//! reimplementation of ggml's compute graph.

mod attention;
mod error;
mod forward;
mod hparams;
mod weights;

#[cfg(test)]
mod test;

pub use error::ModelError;
pub use hparams::HParams;
pub use weights::{Gpt2Transformer, Gpt2Weights};

use crate::gguf::GgufFile;
use crate::kv_cache::KvCache;

pub struct Gpt2Model {
    /// Hyperparameters
    hparams: HParams,
    /// Model weights
    weights: Gpt2Weights,
}

impl Gpt2Model {
    pub fn from_gguf(file: &GgufFile) -> Result<Gpt2Model, ModelError> {
        let hparams = HParams::from_gguf(file)?;
        let weights = Gpt2Weights::from_gguf(file, &hparams)?;
        Ok(Gpt2Model { hparams, weights })
    }

    pub fn hparams(&self) -> &HParams {
        &self.hparams
    }

    pub fn weights(&self) -> &Gpt2Weights {
        &self.weights
    }

    /// A fresh, empty KV cache sized for this model.
    pub fn new_kv_cache(&self) -> KvCache {
        self.hparams.new_kv_cache()
    }
}

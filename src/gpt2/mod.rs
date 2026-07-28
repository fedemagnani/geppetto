//! The GPT-2 model: hyperparameters, weights, and the eager forward pass.
//!
//! Reference: `src/models/gpt2.cpp` in llama.cpp. This is a behavioral port --
//! a direct forward pass over [`crate::tensor::Tensor`] ops -- not a
//! reimplementation of ggml's compute graph.

mod attention;
mod error;
mod forward;
mod hparams;
mod layout;
mod state;
mod weights;

#[cfg(test)]
pub(crate) mod test;

pub use error::ModelError;
pub use hparams::HParams;
pub use layout::{Gpt2Layout, Gpt2Persistent, Gpt2Scratch, Gpt2Views};
pub use state::Gpt2State;
pub use weights::{Gpt2TransformerWeights, Gpt2Weights};

use crate::gguf::GgufFile;

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

    /// A fresh inference state (arena + KV cache) sized for this model.
    pub fn new_state(&self) -> Gpt2State {
        Gpt2State::new(self.hparams, self.weights.vocabulary_size())
    }
}

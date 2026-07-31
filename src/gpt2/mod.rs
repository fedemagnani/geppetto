//! The GPT-2 model: hyperparameters, weights, and the eager forward pass.
//!
//! Reference: `src/models/gpt2.cpp` in llama.cpp. This is a behavioral port --
//! a direct forward pass over view ops and a pre-allocated arena -- not a
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
use crate::tensor::{MatmulKernel, NaiveMatMul};

/// The model, generic over the matmul kernel its five weight matmuls run
/// through (the Q@K^T score matmul in attention stays on the raw
/// [`matmul`](crate::tensor::matmul)). Weights are packed into the kernel's
/// format once at load; swapping kernels is a type parameter, not a rewire.
pub struct Gpt2Model<K: MatmulKernel = NaiveMatMul> {
    /// Hyperparameters
    hparams: HParams,
    /// Model weights, linear ones in the kernel's packed format
    weights: Gpt2Weights<K>,
    /// The matmul implementation; owns its own knobs (tiles, threads, ...)
    matmul_kernel: K,
}

impl Gpt2Model<NaiveMatMul> {
    /// Loads with the baseline [`Naive`] kernel.
    pub fn from_gguf(file: &GgufFile) -> Result<Gpt2Model, ModelError> {
        Gpt2Model::with_kernel(file, NaiveMatMul)
    }
}

impl<K: MatmulKernel> Gpt2Model<K> {
    /// Loads the model and packs its linear weights through `kernel`.
    pub fn with_kernel(file: &GgufFile, kernel: K) -> Result<Gpt2Model<K>, ModelError> {
        let hparams = HParams::from_gguf(file)?;
        let weights = Gpt2Weights::from_gguf(file, &hparams, &kernel)?;
        Ok(Gpt2Model {
            hparams,
            weights,
            matmul_kernel: kernel,
        })
    }

    pub fn hparams(&self) -> &HParams {
        &self.hparams
    }

    pub fn weights(&self) -> &Gpt2Weights<K> {
        &self.weights
    }

    pub fn kernel(&self) -> &K {
        &self.matmul_kernel
    }

    /// A fresh inference state (arena + KV cache) sized for this model.
    pub fn new_state(&self) -> Gpt2State {
        let kernel_scratch = self.kernel_scratch_len();
        Gpt2State::new(self.hparams, self.weights.vocabulary_size(), kernel_scratch)
    }

    /// Worst-case kernel scratch for one forward pass: the max of
    /// [`MatmulKernel::scratch_len`] over the five weight-matmul shapes at
    /// `m = n_ctx` (a full-context prefill). A plain float count, so the
    /// layout stays kernel-agnostic.
    fn kernel_scratch_len(&self) -> usize {
        let hp = &self.hparams;
        let n_embd = hp.n_embd;
        let n_vocab = self.weights.vocabulary_size();
        // (k, n) of the qkv, attention-out, ffn-up, ffn-down and unembedding
        // matmuls; mirrors the call sites in forward.rs
        let shapes = [
            (n_embd, 3 * n_embd),
            (n_embd, n_embd),
            (n_embd, hp.n_ff),
            (hp.n_ff, n_embd),
            (n_embd, n_vocab),
        ];
        let mut worst = 0;
        for (k, n) in shapes {
            let need = self.matmul_kernel.scratch_len(hp.n_ctx, k, n);
            worst = worst.max(need);
        }
        worst
    }
}

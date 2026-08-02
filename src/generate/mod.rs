//! The generation loop: prefill a prompt, then decode one token at a time.
//!
//! Reference: the loop shape of `tools/main` and `llama-context`. This works
//! in token ids and knows nothing about text, so it drives a model with no
//! tokenizer (the tiny test model) as readily as a real one; detokenization is
//! the caller's job, with [`Utf8Stream`] for the streaming case.

mod error;
mod stream;

#[cfg(test)]
mod test;

pub use error::GenerateError;
pub use stream::Utf8Stream;

use crate::gpt2::{Gpt2Model, Gpt2State};
use crate::sampling::Sampler;
use crate::tensor::{AutoVecMatMulNt, MatmulNtKernel};
use crate::tokenizer::TokenId;

/// Drives one sequence: owns its inference state and sampler, and forwards
/// the prompt on the first step and a single token on each step after.
/// Generic over the model's matmul kernel, defaulting to the model's
/// default.
pub struct Generator<'a, K: MatmulNtKernel = AutoVecMatMulNt> {
    model: &'a Gpt2Model<K>,
    sampler: Sampler,
    state: Gpt2State,
    config: GenerationConfig,
    /// Tokens to feed on the next step: the whole prompt first, then the token
    /// just sampled. Reused rather than reallocated per step.
    input: Vec<TokenId>,
    generated: usize,
    stop: Option<StopReason>,
}

impl<'a, K: MatmulNtKernel> Generator<'a, K> {
    /// Runs to completion, collecting every emitted token.
    pub fn generate_all(&mut self) -> Result<Vec<TokenId>, GenerateError> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Produces the next token, or `None` once generation has stopped; check
    /// [`Self::stop_reason`] to learn why.
    ///
    /// The first call forwards the entire prompt (the prefill) and samples
    /// from its final position; later calls forward only the previous token,
    /// which is what makes decoding incremental.
    pub fn next_token(&mut self) -> Result<Option<TokenId>, GenerateError> {
        if self.stop.is_some() {
            return Ok(None);
        }
        if self.generated >= self.config.max_new_tokens {
            return Ok(self.stop_with(StopReason::MaxTokens));
        }
        if self.state.n_past() + self.input.len() > self.model.hparams().n_ctx {
            return Ok(self.stop_with(StopReason::ContextFull));
        }

        // the logits are the last position's row, borrowed straight from the
        // state's arena -- earlier prefill positions predict nothing
        let logits = self.model.forward(&mut self.state, &self.input)?;
        let token = self.sampler.sample(logits)?;

        self.generated += 1;
        self.input.clear();
        self.input.push(token);

        if self.config.eos == Some(token) {
            return Ok(self.stop_with(StopReason::Eos));
        }
        Ok(Some(token))
    }

    pub fn new(
        model: &'a Gpt2Model<K>,
        sampler: Sampler,
        config: GenerationConfig,
        prompt: &[TokenId],
    ) -> Result<Generator<'a, K>, GenerateError> {
        if prompt.is_empty() {
            return Err(GenerateError::EmptyPrompt);
        }
        let n_ctx = model.hparams().n_ctx;
        if prompt.len() > n_ctx {
            return Err(GenerateError::PromptTooLong {
                n_prompt: prompt.len(),
                n_ctx,
            });
        }

        Ok(Generator {
            state: model.new_state(),
            model,
            sampler,
            config,
            input: prompt.to_vec(),
            generated: 0,
            stop: None,
        })
    }

    pub const fn stop_reason(&self) -> Option<StopReason> {
        self.stop
    }

    /// Tokens sampled so far, including a consumed end-of-sequence token.
    pub const fn generated(&self) -> usize {
        self.generated
    }

    pub const fn state(&self) -> &Gpt2State {
        &self.state
    }

    fn stop_with(&mut self, reason: StopReason) -> Option<TokenId> {
        self.stop = Some(reason);
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    /// Token that ends generation. `None` runs to `max_new_tokens`.
    pub eos: Option<TokenId>,
}

impl Default for GenerationConfig {
    fn default() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 64,
            eos: None,
        }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_new_tokens` were produced.
    MaxTokens,
    /// The end-of-sequence token was sampled. It is consumed, not emitted.
    Eos,
    /// The context window is full, so no further token could be forwarded.
    ContextFull,
}

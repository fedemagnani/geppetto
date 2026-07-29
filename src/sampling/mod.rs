//! Token sampling: a chain of filters over a logits row, then a draw.
//!
//! Reference: `src/llama-sampler.cpp`. The chain runs in llama.cpp's order --
//! temperature, top-k, top-p -- and ends in a categorical draw. Greedy
//! decoding is not a separate path: temperature `<= 0` collapses the set onto
//! the argmax, leaving the draw nothing to choose.

mod candidates;
mod error;
pub mod filters;
mod rng;

#[cfg(test)]
mod test;

pub use candidates::{Candidate, Candidates};
pub use error::SamplingError;
pub use rng::Rng;

use crate::sampling::filters::{Filter, Temperature, TopK, TopP};
use crate::tokenizer::TokenId;

#[derive(Debug)]
pub struct Sampler {
    filters: Vec<Box<dyn Filter>>,
    rng: Rng,
    /// Reused across calls, so steady-state sampling does not allocate.
    candidates: Candidates,
}

impl Sampler {
    /// Samples one token from a row of logits, one entry per vocab id.
    #[hotpath::measure]
    pub fn sample(&mut self, logits: &[f32]) -> Result<TokenId, SamplingError> {
        self.candidates.reset_from_logits(logits)?;
        for filter in &self.filters {
            filter.apply(&mut self.candidates);
        }
        // renormalize over whatever survived; the filters left `p` stale
        self.candidates.softmax();

        let r = self.rng.next_f64();
        self.candidates.draw(r).ok_or(SamplingError::EmptyLogits)
    }

    /// Builds the chain once. Disabled filters are left out entirely rather
    /// than short-circuiting per token.
    pub fn new(config: SamplerConfig) -> Result<Sampler, SamplingError> {
        config.validate()?;

        let mut filters: Vec<Box<dyn Filter>> = Vec::new();
        filters.push(Box::new(Temperature::new(config.temperature)));
        if config.top_k > 0 {
            filters.push(Box::new(TopK::new(config.top_k)));
        }
        if config.top_p < 1.0 {
            filters.push(Box::new(TopP::new(config.top_p, config.min_keep)));
        }

        Ok(Sampler {
            filters,
            rng: Rng::new(config.seed),
            candidates: Candidates::new(),
        })
    }

    pub fn greedy() -> Sampler {
        Sampler::new(SamplerConfig::greedy()).expect("the greedy config is valid")
    }

    /// The candidate set left by the last [`Self::sample`], for inspection.
    pub fn candidates(&self) -> &Candidates {
        &self.candidates
    }
}

/// How to turn logits into a token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// Logit scaling; `<= 0` means greedy.
    pub temperature: f32,
    /// Keep the `k` highest-logit candidates; `0` disables.
    pub top_k: usize,
    /// Keep the shortest prefix whose probabilities reach `p`; `>= 1` disables.
    pub top_p: f32,
    /// Floor on the candidate count [`TopP`] may leave.
    pub min_keep: usize,
    /// Seed for the draw. Required, never taken from the environment, so a run
    /// is reproducible from its config alone.
    pub seed: u64,
}

impl SamplerConfig {
    /// Always take the most probable token.
    pub const fn greedy() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_keep: 1,
            seed: 0,
        }
    }

    fn validate(&self) -> Result<(), SamplingError> {
        if self.temperature.is_nan() {
            return Err(SamplingError::InvalidTemperature(self.temperature));
        }
        if self.top_p.is_nan() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(SamplingError::InvalidTopP(self.top_p));
        }
        Ok(())
    }
}

impl Default for SamplerConfig {
    /// Unmodified model distribution: temperature 1, no truncation.
    fn default() -> SamplerConfig {
        SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_keep: 1,
            seed: 0,
        }
    }
}

//! The stages of the sampler chain. Each narrows or rescales the candidate
//! set in place, in the order llama.cpp applies them: temperature, top-k,
//! top-p.

mod temperature;
mod top_k;
mod top_p;

pub use temperature::Temperature;
pub use top_k::TopK;
pub use top_p::TopP;

use crate::sampling::Candidates;

/// One stage of a sampler chain. `Debug` is required so an assembled chain can
/// be printed as configured.
pub trait Filter: std::fmt::Debug {
    fn apply(&self, candidates: &mut Candidates);
}

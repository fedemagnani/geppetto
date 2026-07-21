use crate::gpt2::ModelError;
use crate::sampling::SamplingError;

#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error("the prompt is empty; seed generation with at least one token")]
    EmptyPrompt,
    #[error("the prompt is {n_prompt} tokens, longer than the context window of {n_ctx}")]
    PromptTooLong { n_prompt: usize, n_ctx: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum SamplingError {
    #[error("logits buffer is empty")]
    EmptyLogits,
    #[error("logit at index {index} is NaN")]
    NaNLogit { index: usize },
    #[error("temperature {0} is NaN")]
    InvalidTemperature(f32),
    #[error("top_p {0} is not in [0, 1]")]
    InvalidTopP(f32),
}

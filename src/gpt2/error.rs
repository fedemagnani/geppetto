use crate::gguf::GgufError;
use crate::tensor::TensorError;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("unsupported architecture '{0}', expected 'gpt2'")]
    UnsupportedArch(String),
    #[error("invalid hyperparameters: {0}")]
    InvalidHParams(String),
    #[error("missing tensor '{0}'")]
    MissingTensor(String),
    #[error("tensor '{name}' has unsupported ggml type id {type_id}")]
    UnsupportedTensorType { name: String, type_id: u32 },
    #[error("token id {token} is out of range for a vocab of {n_vocab}")]
    TokenOutOfRange { token: u32, n_vocab: usize },
    #[error("context overflow: {n_past} cached + {n_new} new exceeds n_ctx {n_ctx}")]
    ContextOverflow {
        n_past: usize,
        n_new: usize,
        n_ctx: usize,
    },
}

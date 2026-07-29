use crate::gguf::GgufError;
use crate::tokenizer::TokenId;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("pre-tokenizer regex: {0}")]
    Regex(#[from] Box<fancy_regex::Error>),
    #[error("unsupported tokenizer model '{0}', expected 'gpt2'")]
    UnsupportedModel(String),
    #[error("unsupported pre-tokenizer '{0}', expected 'gpt-2'")]
    UnsupportedPretokenizer(String),
    #[error("merge #{index} '{merge}' is not two space-separated symbols")]
    BadMerge { index: usize, merge: String },
    #[error("merge #{index} '{merge}': its parts or their concatenation are not vocab tokens")]
    MergeNotInVocab { index: usize, merge: String },
    #[error("duplicate token '{0}' in vocab")]
    DuplicateToken(String),
    #[error("byte {byte:#04x} maps to '{token}', which is not a vocab token")]
    ByteNotInVocab { byte: u8, token: String },
    #[error("token id {id} is out of range for a vocab of {n_tokens}")]
    IdOutOfRange { id: TokenId, n_tokens: usize },
    #[error("token {id} contains '{ch}', which is outside the byte map")]
    UnmappableChar { id: TokenId, ch: char },
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(err: fancy_regex::Error) -> TokenizerError {
        TokenizerError::Regex(Box::new(err))
    }
}

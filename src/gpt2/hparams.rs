use crate::gguf::GgufFile;
use crate::gpt2::ModelError;
use crate::kv_cache::KvCache;

/// GPT-2 hyperparameters read from the `gpt2.*` metadata keys. `n_vocab` is not
/// here: it is derived from the token-embedding tensor when weights load, since
/// a model file always carries that tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HParams {
    /// Context length: the maximal amount of tokens which can processed at once by the model.
    pub n_ctx: usize,
    /// Extended embedding dimension. To get the embedding dimension of each token, it must be
    /// divided by `n_head`.
    pub n_embd: usize,
    /// Number of rows of the feed-forward weight matrix.
    pub n_ff: usize,
    /// Number of heads in the multi-head self-attention mechanism.
    pub n_head: usize,
    /// Number of transformer blocks used by the model.
    pub n_layer: usize,
    /// Machine precision (used during normalization step).
    pub eps: f32,
}

impl HParams {
    pub const ARCH: &'static str = "gpt2";

    pub const ARCH_KEY: &'static str = "general.architecture";
    pub const CONTEXT_LENGTH_KEY: &'static str = "gpt2.context_length";
    pub const EMBEDDING_LENGTH_KEY: &'static str = "gpt2.embedding_length";
    pub const FEED_FORWARD_LENGTH_KEY: &'static str = "gpt2.feed_forward_length";
    pub const HEAD_COUNT_KEY: &'static str = "gpt2.attention.head_count";
    pub const BLOCK_COUNT_KEY: &'static str = "gpt2.block_count";
    pub const LAYER_NORM_EPS_KEY: &'static str = "gpt2.attention.layer_norm_epsilon";

    pub fn from_gguf(file: &GgufFile) -> Result<HParams, ModelError> {
        let md = file.metadata();
        let arch = md.get_str(Self::ARCH_KEY)?;
        if arch != Self::ARCH {
            return Err(ModelError::UnsupportedArch(arch.into()));
        }

        let hparams = HParams {
            n_ctx: md.get_u32(Self::CONTEXT_LENGTH_KEY)? as usize,
            n_embd: md.get_u32(Self::EMBEDDING_LENGTH_KEY)? as usize,
            n_ff: md.get_u32(Self::FEED_FORWARD_LENGTH_KEY)? as usize,
            n_head: md.get_u32(Self::HEAD_COUNT_KEY)? as usize,
            n_layer: md.get_u32(Self::BLOCK_COUNT_KEY)? as usize,
            eps: md.get_f32(Self::LAYER_NORM_EPS_KEY)?,
        };
        hparams.validate()?;
        Ok(hparams)
    }

    /// Head dimension; `n_embd / n_head`, which [`Self::validate`] guarantees is
    /// exact.
    ///
    /// This corresponds to `d_Q` and `d_K`, that is the number of columns for the query weight and key weight for each matrix
    pub const fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    fn validate(&self) -> Result<(), ModelError> {
        let invalid = |msg: String| Err(ModelError::InvalidHParams(msg));
        if self.n_head == 0 {
            return invalid("head count is zero".into());
        }
        if self.n_embd == 0 || self.n_layer == 0 || self.n_ff == 0 || self.n_ctx == 0 {
            return invalid("a dimension is zero".into());
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            return invalid(format!(
                "n_embd {} is not divisible by n_head {}",
                self.n_embd, self.n_head
            ));
        }
        Ok(())
    }

    /// A fresh, empty KV cache sized for this model.
    pub fn new_kv_cache(&self) -> KvCache {
        KvCache::new(self.n_layer, self.n_embd, self.n_ctx)
    }
}

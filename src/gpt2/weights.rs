use crate::gguf::GgufFile;
use crate::gpt2::{HParams, ModelError};
use crate::tensor::Tensor;

/// The weights of one transformer block, all converted to f32 tensors. Linear
/// weights are `[n_out, n_in]` (the ggml convention `Tensor`'s `*` expects);
/// norm weights and biases are single rows.
pub struct Gpt2Transformer {
    pub attn_norm_w: Tensor,
    pub attn_norm_b: Tensor,
    pub attn_qkv_w: Tensor,
    pub attn_qkv_b: Tensor,
    pub attn_out_w: Tensor,
    pub attn_out_b: Tensor,
    pub ffn_norm_w: Tensor,
    pub ffn_norm_b: Tensor,
    pub ffn_up_w: Tensor,
    pub ffn_up_b: Tensor,
    pub ffn_down_w: Tensor,
    pub ffn_down_b: Tensor,
}

/// All model weights. `output` is the unembedding; when the file omits
/// `output.weight` it is tied to `token_embd` (GPT-2 shares them).
pub struct Gpt2Weights {
    /// A matrix [vocabulary_size, embedding_dimension]: each row represents a vector embedding
    /// associated with a specific token.
    pub token_embd: Tensor,
    /// A matrix [context_length, embedding_dimension]: positional embeddings added to the \
    /// initial input embedding vectors.
    pub pos_embd: Tensor,
    pub output_norm_w: Tensor,
    pub output_norm_b: Tensor,
    /// A matrix [embedding_dimension, vocabulary_size]: this is needed to map each embedding vector
    /// into a vector having the same length of vocabulary size, so that it can be later turned into
    /// a discrete probability distribution to pick the next predicted token
    pub output: Tensor,
    /// The transformer blocks used to transform embedding vectors
    pub layers: Vec<Gpt2Transformer>,
}

impl Gpt2Weights {
    pub fn from_gguf(file: &GgufFile, hp: &HParams) -> Result<Gpt2Weights, ModelError> {
        let e = hp.n_embd;
        let ff = hp.n_ff;

        // n_vocab is whatever the token embedding carries; only its row width
        // (n_embd) is known up front.
        let token_embd = file.load_cols("token_embd.weight", e)?;
        // the number of rows of the embedding matrix corresponds with the vocabulary size
        let n_vocab = token_embd.rows();

        let pos_embd = file.load("position_embd.weight", hp.n_ctx, e)?;
        let output_norm_w = file.load("output_norm.weight", 1, e)?;
        let output_norm_b = file.load("output_norm.bias", 1, e)?;
        let output = match file.tensor("output.weight") {
            Some(_) => file.load("output.weight", n_vocab, e)?,
            None => token_embd.clone(), // tied unembedding
        };

        let mut layers = Vec::with_capacity(hp.n_layer);
        for il in 0..hp.n_layer {
            let name = |suffix: &str| format!("blk.{il}.{suffix}");
            layers.push(Gpt2Transformer {
                attn_norm_w: file.load(&name("attn_norm.weight"), 1, e)?,
                attn_norm_b: file.load(&name("attn_norm.bias"), 1, e)?,
                // the query, key, value weight matrices are stacked one above the other
                attn_qkv_w: file.load(&name("attn_qkv.weight"), 3 * e, e)?,
                attn_qkv_b: file.load(&name("attn_qkv.bias"), 1, 3 * e)?,
                attn_out_w: file.load(&name("attn_output.weight"), e, e)?,
                attn_out_b: file.load(&name("attn_output.bias"), 1, e)?,
                ffn_norm_w: file.load(&name("ffn_norm.weight"), 1, e)?,
                ffn_norm_b: file.load(&name("ffn_norm.bias"), 1, e)?,
                ffn_up_w: file.load(&name("ffn_up.weight"), ff, e)?,
                ffn_up_b: file.load(&name("ffn_up.bias"), 1, ff)?,
                ffn_down_w: file.load(&name("ffn_down.weight"), e, ff)?,
                ffn_down_b: file.load(&name("ffn_down.bias"), 1, e)?,
            });
        }

        Ok(Gpt2Weights {
            token_embd,
            pos_embd,
            output_norm_w,
            output_norm_b,
            output,
            layers,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.token_embd.rows()
    }
}

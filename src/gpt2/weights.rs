use crate::gguf::GgufFile;
use crate::gpt2::{HParams, ModelError};
use crate::tensor::{MatmulNtKernel, WeightTensor};

/// The weights of one transformer block, with the four linear weights held in
/// the kernel's packed format ([`MatmulNtKernel::Weights`], produced once at
/// load). Raw linear weights are `[n_out, n_in]` (the ggml convention
/// [`crate::tensor::matmul_nt`] expects); norm weights and biases stay
/// [`WeightTensor`] single rows. F32 tensors borrow the file mapping, F16
/// ones are widened at load -- see [`WeightTensor`].
pub struct Gpt2TransformerWeights<K: MatmulNtKernel> {
    pub attn_norm_w: WeightTensor,
    pub attn_norm_b: WeightTensor,
    pub attn_qkv_w: K::Weights,
    pub attn_qkv_b: WeightTensor,
    pub attn_out_w: K::Weights,
    pub attn_out_b: WeightTensor,
    pub ffn_norm_w: WeightTensor,
    pub ffn_norm_b: WeightTensor,
    pub ffn_up_w: K::Weights,
    pub ffn_up_b: WeightTensor,
    pub ffn_down_w: K::Weights,
    pub ffn_down_b: WeightTensor,
}

/// All model weights. `output` is the unembedding; when the file omits
/// `output.weight` it is tied to `token_embd` (GPT-2 shares them, and the
/// clone shares storage rather than copying -- though a kernel whose pack is
/// not the identity then holds its own packed copy). `token_embd` stays raw:
/// the embedding lookup reads its rows directly, only the five matmul
/// weights go through [`MatmulNtKernel::pack`].
pub struct Gpt2Weights<K: MatmulNtKernel> {
    /// A matrix [vocabulary_size, embedding_dimension]: each row represents a vector embedding
    /// associated with a specific token.
    pub token_embd: WeightTensor,
    /// A matrix [context_length, embedding_dimension]: positional embeddings added to the \
    /// initial input embedding vectors.
    pub pos_embd: WeightTensor,
    pub output_norm_w: WeightTensor,
    pub output_norm_b: WeightTensor,
    /// A matrix [embedding_dimension, vocabulary_size]: this is needed to map each embedding vector
    /// into a vector having the same length of vocabulary size, so that it can be later turned into
    /// a discrete probability distribution to pick the next predicted token
    pub output: K::Weights,
    /// The transformer blocks used to transform embedding vectors
    pub layers: Vec<Gpt2TransformerWeights<K>>,
}

impl<K: MatmulNtKernel> Gpt2Weights<K> {
    #[hotpath::measure]
    pub fn from_gguf(
        file: &GgufFile,
        hp: &HParams,
        kernel: &K,
    ) -> Result<Gpt2Weights<K>, ModelError> {
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
        let output = kernel.pack(output);

        let mut layers = Vec::with_capacity(hp.n_layer);
        for il in 0..hp.n_layer {
            let name = |suffix: &str| format!("blk.{il}.{suffix}");
            let attn_norm_w = file.load(&name("attn_norm.weight"), 1, e)?;
            let attn_norm_b = file.load(&name("attn_norm.bias"), 1, e)?;
            // the query, key, value weight matrices are stacked one above the other
            let attn_qkv_w = file.load(&name("attn_qkv.weight"), 3 * e, e)?;
            let attn_qkv_b = file.load(&name("attn_qkv.bias"), 1, 3 * e)?;
            let attn_out_w = file.load(&name("attn_output.weight"), e, e)?;
            let attn_out_b = file.load(&name("attn_output.bias"), 1, e)?;
            let ffn_norm_w = file.load(&name("ffn_norm.weight"), 1, e)?;
            let ffn_norm_b = file.load(&name("ffn_norm.bias"), 1, e)?;
            let ffn_up_w = file.load(&name("ffn_up.weight"), ff, e)?;
            let ffn_up_b = file.load(&name("ffn_up.bias"), 1, ff)?;
            let ffn_down_w = file.load(&name("ffn_down.weight"), e, ff)?;
            let ffn_down_b = file.load(&name("ffn_down.bias"), 1, e)?;
            layers.push(Gpt2TransformerWeights {
                attn_norm_w,
                attn_norm_b,
                attn_qkv_w: kernel.pack(attn_qkv_w),
                attn_qkv_b,
                attn_out_w: kernel.pack(attn_out_w),
                attn_out_b,
                ffn_norm_w,
                ffn_norm_b,
                ffn_up_w: kernel.pack(ffn_up_w),
                ffn_up_b,
                ffn_down_w: kernel.pack(ffn_down_w),
                ffn_down_b,
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

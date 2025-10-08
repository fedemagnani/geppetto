use super::*;

pub struct GptConfig {
    pub vocab_size: usize, // input of the embedding layer (number of rows of the embedding matrix) = number of token ids (as each token id maps to a row of the matrix)
    pub emb_dim: usize, // columns of the embedding matrix: each token id is mapped to a vector having this dimensions
    pub context_length: usize, //the max number of tokens processed together
    pub drop_p: f32,    //dropout probability
    pub num_trf: usize, //number of transformers invovled in the model
    pub num_heads: usize, //number of heads of multi-head attention module in transformer block
    pub bias: bool,     // whether to add bias or not in the output module
}

impl GptConfig {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            num_trf: 12,
            num_heads: 12,
            drop_p: 0.1,
            bias: true,
        }
    }

    pub fn adapt_gpt2_weights(
        weights: &mut HashMap<String, candle_core::Tensor>,
    ) -> eyre::Result<()> {
        let mut i = 0;
        // query-key-value dimensions must be unwrapped, and all the weights associated with linear layers must be transposed due to bad convention
        while let (Some(w), Some(b)) = (
            weights.remove(&format!("h.{i}.attn.c_attn.weight")),
            weights.remove(&format!("h.{i}.attn.c_attn.bias")),
        ) {
            let dim = b.dims()[0] / 3_usize;
            let (q_w, q_b) = (w.i((.., ..dim))?, b.i(..dim)?);
            let (k_w, k_b) = (w.i((.., dim..2 * dim))?, b.i(dim..2 * dim)?);
            let (v_w, v_b) = (w.i((.., 2 * dim..))?, b.i(2 * dim..)?);

            weights.insert(format!("h.{i}.attn.query.weight"), q_w.t()?.contiguous()?);
            weights.insert(format!("h.{i}.attn.key.weight"), k_w.t()?.contiguous()?);
            weights.insert(format!("h.{i}.attn.value.weight"), v_w.t()?.contiguous()?);
            weights.insert(format!("h.{i}.attn.query.bias"), q_b);
            weights.insert(format!("h.{i}.attn.key.bias"), k_b);
            weights.insert(format!("h.{i}.attn.value.bias"), v_b);

            i += 1
        }

        for (k, v) in weights.iter_mut() {
            if !k.contains("c_proj.weight") && !k.contains("c_fc.weight") {
                continue;
            }
            *v = v.t()?;
        }

        let t = weights.get("wte.weight").context("wte.weight not found")?;
        weights.insert("out_head.weight".to_string(), t.clone());

        Ok(())
    }
}

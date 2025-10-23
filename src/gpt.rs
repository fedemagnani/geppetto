use std::collections::HashMap;

use eyre::ContextCompat;

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

pub struct GPTModel {
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub drop_emb: Dropout,
    pub trf_blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub out_head: Linear,
}

impl GPTModel {
    pub fn new(vb: &VarBuilder, c: &GptConfig) -> candle_core::Result<Self> {
        let tok_emb = embedding(c.vocab_size, c.emb_dim, vb.pp("wte"))?;
        let pos_emb = embedding(c.context_length, c.emb_dim, vb.pp("wpe"))?;
        let drop_emb = Dropout::new(c.drop_p);
        let vbh = vb.pp("h");
        let mut trf_blocks = Vec::with_capacity(c.num_trf);
        for i in 0..c.num_trf {
            trf_blocks.push(TransformerBlock::from_config(&vbh.pp(i), c)?)
        }

        let final_norm = layer_norm(c.emb_dim, LayerNormConfig::default(), vb.pp("ln_f"))?;

        let out_head = linear_b(c.emb_dim, c.vocab_size, false, vb.pp("out_head"))?;
        let out = Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        };
        Ok(out)
    }

    pub fn generate_text_simple(
        &self,
        mut idx: Tensor,
        max_new_tokens: usize,
        context_length: usize,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        for _ in 0..max_new_tokens {
            // Limit the context window
            let (_b, seq_len) = idx.dims2()?;
            let start = seq_len.saturating_sub(context_length);
            let idx_cond = idx.i((.., start..seq_len))?;

            // Forward pass
            let logits = self.forward_t(&idx_cond, train)?;

            // Get last logits
            let (_b, c, _vocab_size) = logits.dims3()?;
            let logits = logits.i((.., c - 1, ..))?;

            // Greedy sampling
            let probas = softmax(&logits, 1)?;
            let idx_next = probas.argmax_keepdim(D::Minus1)?;

            // Append new token
            idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
        }
        Ok(idx)
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (_, seq_len) = xs.dims2()?;
        let tok_emb = self.tok_emb.forward_t(xs, train)?;

        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_emb = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let x = tok_emb.broadcast_add(&pos_emb)?;

        let mut x = self.drop_emb.forward_t(&x, train)?;

        for t in &self.trf_blocks {
            x = t.forward_t(&x, train)?;
        }

        let logits = self.final_norm.forward_t(&x, train)?;

        let logits = self.out_head.forward_t(&logits, train)?;

        Ok(logits)
    }
}

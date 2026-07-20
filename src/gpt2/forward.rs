use crate::gpt2::attention::multi_head_attention;
use crate::gpt2::{Gpt2Model, ModelError};
use crate::kv_cache::KvCache;
use crate::tensor::layer::LayerNorm;
use crate::tensor::{Shape, Tensor};

impl Gpt2Model {
    /// Runs `tokens` through the network on top of the cache's `n_past`
    /// positions, appending their keys and values, and returns the logits
    /// `[tokens.len(), n_vocab]` (one row per new position). Mirrors the graph
    /// in `src/models/gpt2.cpp`.
    pub fn forward(&self, cache: &mut KvCache, tokens: &[u32]) -> Result<Tensor, ModelError> {
        let hp = self.hparams();
        let w = self.weights();
        let n_vocab = w.vocabulary_size();

        // verifies that each token is included in the vocabulary size
        for &token in tokens {
            if token as usize >= n_vocab {
                return Err(ModelError::TokenOutOfRange { token, n_vocab });
            }
        }

        // verifies that the overall processed tokens don't overshoot the context window.
        let n_new = tokens.len();
        let n_past = cache.len();
        if n_past + n_new > hp.n_ctx {
            return Err(ModelError::ContextOverflow {
                n_past,
                n_new,
                n_ctx: hp.n_ctx,
            });
        }

        if n_new == 0 {
            return Ok(Tensor::zeros(Shape::new(0, n_vocab)));
        }

        let head_dim = hp.head_dim();
        let n_kv = n_past + n_new;
        // define the position of the new tokens, just appended to the tail.
        // this is needed to
        let positions: Vec<u32> = (n_past..n_kv).map(|p| p as u32).collect();

        /////////////////////////////
        // Compute embeddings
        // returns the row of the embedding vector indexed by the token index
        let mut inp = w.token_embd.get_rows(tokens);

        /////////////////////////////
        // Apply positional encodings
        inp += &w.pos_embd.get_rows(&positions);

        // evaluate each transformer block
        for (il, layer) in w.layers.iter().enumerate() {
            /////////////////////////////
            // Normalization layer
            let layer_norm = LayerNorm::new(&layer.attn_norm_w, &layer.attn_norm_b, hp.eps);
            let normed = layer_norm.eval(&inp);

            /////////////////////////////
            // Q,K,V computation
            let qkv = (&normed * &layer.attn_qkv_w) + &layer.attn_qkv_b; // [n_new, 3*n_embd]
            let (q, k, v) = split_qkv(&qkv, hp.n_embd);
            cache.append(il, &k, &v);

            /////////////////////////////
            // Multi-Head attention layer
            let attn = multi_head_attention(
                q.data(),
                cache.keys(il),
                cache.values(il),
                n_new,
                n_kv,
                hp.n_head,
                head_dim,
            );
            let attn = (&attn * &layer.attn_out_w) + &layer.attn_out_b;

            /////////////////////////////
            // Residual connections
            let ffn_inp = attn + &inp;

            /////////////////////////////
            // Feed-Forward layer
            let ff_layer_norm = LayerNorm::new(&layer.ffn_norm_w, &layer.ffn_norm_b, hp.eps);
            let ff = ff_layer_norm.eval(&ffn_inp);
            let ff = (&ff * &layer.ffn_up_w) + &layer.ffn_up_b;
            let ff = ff.gelu();
            let ff = (&ff * &layer.ffn_down_w) + &layer.ffn_down_b;

            /////////////////////////////
            // Residual connections
            inp = ff + &ffn_inp;
        }

        /////////////////////////////
        // Normalization layer
        let out_layer_norm = LayerNorm::new(&w.output_norm_w, &w.output_norm_b, hp.eps);
        let normed = out_layer_norm.eval(&inp);

        Ok(&normed * &w.output) // [n_new, n_vocab]
    }
}

/// Splits a fused QKV activation `[n, 3*n_embd]` into Q, K, V, each
/// `[n, n_embd]`, from the contiguous column blocks `[0, n_embd)`,
/// `[n_embd, 2*n_embd)`, `[2*n_embd, 3*n_embd)`.
fn split_qkv(qkv: &Tensor, n_embd: usize) -> (Tensor, Tensor, Tensor) {
    let n = qkv.rows();
    let mut q = vec![0.0f32; n * n_embd];
    let mut k = vec![0.0f32; n * n_embd];
    let mut v = vec![0.0f32; n * n_embd];
    for i in 0..n {
        let row = qkv.row(i);
        let dst = i * n_embd..(i + 1) * n_embd;
        q[dst.clone()].copy_from_slice(&row[0..n_embd]);
        k[dst.clone()].copy_from_slice(&row[n_embd..2 * n_embd]);
        v[dst].copy_from_slice(&row[2 * n_embd..3 * n_embd]);
    }
    let shape = Shape::new(n, n_embd);
    (
        Tensor::new(shape, q),
        Tensor::new(shape, k),
        Tensor::new(shape, v),
    )
}

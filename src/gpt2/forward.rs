use crate::arena::poison;
use crate::gpt2::attention::multi_head_attention;
use crate::gpt2::{Gpt2Model, Gpt2State, ModelError};
use crate::tensor::layer::{get_rows, norm};
use crate::tensor::{Shape, TensorView, TensorViewMut, add, gelu, matmul};

/// Mounts the `[rows, cols]` active prefix of a scratch region; never the
/// whole worst-case carve.
fn prefix(region: &[f32], rows: usize, cols: usize) -> TensorView<'_> {
    TensorView::new(&region[..rows * cols], Shape::new(rows, cols))
}

fn prefix_mut(region: &mut [f32], rows: usize, cols: usize) -> TensorViewMut<'_> {
    TensorViewMut::new(&mut region[..rows * cols], Shape::new(rows, cols))
}

impl Gpt2Model {
    /// Runs `tokens` through the network on top of the state's `n_past`
    /// positions, appending their keys and values, and returns the last
    /// position's logits (`n_vocab` floats) -- earlier positions predict
    /// nothing a caller can use. Mirrors the graph in `src/models/gpt2.cpp`.
    ///
    /// The logits live in the state's arena: they stay valid until the next
    /// forward call overwrites them, which the borrow checker enforces.
    #[hotpath::measure]
    pub fn forward<'s>(
        &self,
        state: &'s mut Gpt2State,
        tokens: &[u32],
    ) -> Result<&'s [f32], ModelError> {
        let hp = self.hparams();
        let w = self.weights();
        let n_vocab = w.vocabulary_size();
        let n_embd = hp.n_embd;

        // verifies that each token is included in the vocabulary size
        for &token in tokens {
            if token as usize >= n_vocab {
                return Err(ModelError::TokenOutOfRange { token, n_vocab });
            }
        }

        // verifies that the overall processed tokens don't overshoot the context window.
        let n_new = tokens.len();
        let n_past = state.n_past();
        if n_past + n_new > hp.n_ctx {
            return Err(ModelError::ContextOverflow {
                n_past,
                n_new,
                n_ctx: hp.n_ctx,
            });
        }

        if n_new == 0 {
            return Ok(&[]);
        }

        let n_kv = n_past + n_new;
        let head_dim = hp.head_dim();
        let qkv_width = 3 * n_embd;

        // nothing can fail past this point, so the bookkeeping happens before
        // the arena borrow starts
        state.advance(n_new);
        let (views, mut kv) = state.begin_pass();

        /////////////////////////////
        // Compute embeddings + positional encodings
        get_rows(
            w.token_embd.as_view(),
            tokens,
            prefix_mut(views.x, n_new, n_embd),
        );
        // consecutive positions are consecutive rows of pos_embd: the gather
        // collapses to one contiguous block
        let pos_block = &w.pos_embd.data()[n_past * n_embd..n_kv * n_embd];
        let pos = TensorView::new(pos_block, Shape::new(n_new, n_embd));
        add(prefix_mut(views.x, n_new, n_embd), pos);

        // evaluate each transformer block
        for (il, layer) in w.layers.iter().enumerate() {
            /////////////////////////////
            // Normalization layer
            poison(views.norm_out);
            norm(
                prefix(views.x, n_new, n_embd),
                layer.attn_norm_w.as_view(),
                layer.attn_norm_b.as_view(),
                hp.eps,
                prefix_mut(views.norm_out, n_new, n_embd),
            );

            /////////////////////////////
            // Q,K,V computation
            poison(views.qkv);
            let mut qkv = prefix_mut(views.qkv, n_new, qkv_width);
            matmul(
                prefix(views.norm_out, n_new, n_embd),
                layer.attn_qkv_w.as_view(),
                qkv.reborrow(),
            );
            add(qkv, layer.attn_qkv_b.as_view());

            for (i, row) in views.qkv.chunks_exact(qkv_width).take(n_new).enumerate() {
                let k_row = &row[n_embd..2 * n_embd];
                let v_row = &row[2 * n_embd..];
                kv.append_row(il, n_past + i, k_row, v_row);
            }

            /////////////////////////////
            // Multi-Head attention layer
            multi_head_attention(
                views.qkv,
                kv.keys(il, n_kv),
                kv.values(il, n_kv),
                views.scores,
                views.attn_out,
                n_new,
                n_kv,
                hp.n_head,
                head_dim,
            );

            poison(views.proj_out);
            let mut proj = prefix_mut(views.proj_out, n_new, n_embd);
            matmul(
                prefix(views.attn_out, n_new, n_embd),
                layer.attn_out_w.as_view(),
                proj.reborrow(),
            );
            add(proj, layer.attn_out_b.as_view());

            /////////////////////////////
            // Residual connections
            add(
                prefix_mut(views.x, n_new, n_embd),
                prefix(views.proj_out, n_new, n_embd),
            );

            /////////////////////////////
            // Feed-Forward layer
            poison(views.norm_out);
            norm(
                prefix(views.x, n_new, n_embd),
                layer.ffn_norm_w.as_view(),
                layer.ffn_norm_b.as_view(),
                hp.eps,
                prefix_mut(views.norm_out, n_new, n_embd),
            );

            poison(views.ffn_up);
            let mut up = prefix_mut(views.ffn_up, n_new, hp.n_ff);
            matmul(
                prefix(views.norm_out, n_new, n_embd),
                layer.ffn_up_w.as_view(),
                up.reborrow(),
            );
            add(up.reborrow(), layer.ffn_up_b.as_view());
            gelu(up);

            poison(views.proj_out);
            let mut down = prefix_mut(views.proj_out, n_new, n_embd);
            matmul(
                prefix(views.ffn_up, n_new, hp.n_ff),
                layer.ffn_down_w.as_view(),
                down.reborrow(),
            );
            add(down, layer.ffn_down_b.as_view());

            /////////////////////////////
            // Residual connections
            add(
                prefix_mut(views.x, n_new, n_embd),
                prefix(views.proj_out, n_new, n_embd),
            );
        }

        /////////////////////////////
        // Normalization layer, on the last position only
        poison(views.norm_out);
        let last_row = &views.x[(n_new - 1) * n_embd..n_new * n_embd];
        norm(
            TensorView::new(last_row, Shape::new(1, n_embd)),
            w.output_norm_w.as_view(),
            w.output_norm_b.as_view(),
            hp.eps,
            prefix_mut(views.norm_out, 1, n_embd),
        );

        /////////////////////////////
        // Unembedding into the logits region
        matmul(
            prefix(views.norm_out, 1, n_embd),
            w.output.as_view(),
            prefix_mut(views.logits, 1, n_vocab),
        );

        let logits: &'s [f32] = views.logits;
        let logits = &logits[..n_vocab];
        debug_assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logits: a scratch region was read before being written"
        );
        Ok(logits)
    }
}

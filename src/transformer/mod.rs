pub use super::*;
mod attention;
pub use attention::*;

mod feed_forward;
pub use feed_forward::*;

pub struct TransformerBlock {
    pub norm_1: LayerNorm,
    pub multi_head: MultiHeadAttention,
    pub norm_2: LayerNorm,
    pub feed_forward: FeedForward,
    pub dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(
        vbhi: &VarBuilder,
        num_heads: usize,
        emb_dim: usize,
        bias: bool,
        drop_p: f32,
    ) -> candle_core::Result<Self> {
        let norm_1 = layer_norm(emb_dim, LayerNormConfig::default(), vbhi.pp("ln_1"))?;
        let multi_head = MultiHeadAttention::new(
            &vbhi.pp("attn"),
            num_heads,
            emb_dim,
            emb_dim / num_heads,
            bias,
            drop_p,
        )?;

        let norm_2 = layer_norm(emb_dim, LayerNormConfig::default(), vbhi.pp("ln_2"))?;
        let feed_forward = FeedForward::new(&vbhi.pp("mlp"), emb_dim, bias)?;
        let dropout = Dropout::new(drop_p);
        let out = Self {
            norm_1,
            multi_head,
            norm_2,
            feed_forward,
            dropout,
        };
        Ok(out)
    }

    pub fn from_config(vb: &VarBuilder, c: &GptConfig) -> candle_core::Result<Self> {
        Self::new(vb, c.num_heads, c.emb_dim, c.bias, c.drop_p)
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let shortcut_1 = xs;
        let x = self.norm_1.forward_t(xs, train)?;

        let x = self.multi_head.forward_t(&x, train)?;

        let x = self.dropout.forward_t(&x, train)?;

        let x = (x + shortcut_1)?; // shortcut connection

        let shortcut_2 = &x;
        let x = self.norm_2.forward_t(&x, train)?;
        let x = self.feed_forward.forward_t(&x, train)?;
        let x = self.dropout.forward_t(&x, train)?;

        x + shortcut_2 // shortcut connection
    }
}

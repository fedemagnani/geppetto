//! A tiny GPT-2 with seeded random weights, held in memory and serialized to
//! GGUF bytes via the epoch-1 fixture writer. This exercises the whole
//! load-from-GGUF path end to end without shipping a real model.

use crate::gguf::Value;
use crate::gguf::test::fixtures::FixtureBuilder;
use crate::tensor::test::Rng;

const F32_TYPE_ID: u32 = 0;

#[derive(Clone, Copy)]
pub struct TinyConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub eps: f32,
}

impl TinyConfig {
    pub fn small() -> TinyConfig {
        TinyConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            n_ff: 16,
            n_ctx: 16,
            n_vocab: 11,
            eps: 1e-5,
        }
    }
}

struct LayerData {
    attn_norm_w: Vec<f32>,
    attn_norm_b: Vec<f32>,
    attn_qkv_w: Vec<f32>,
    attn_qkv_b: Vec<f32>,
    attn_out_w: Vec<f32>,
    attn_out_b: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    ffn_norm_b: Vec<f32>,
    ffn_up_w: Vec<f32>,
    ffn_up_b: Vec<f32>,
    ffn_down_w: Vec<f32>,
    ffn_down_b: Vec<f32>,
}

/// All tensor data, generated once so it can be reused (e.g. `output.weight`
/// tied to `token_embd`).
pub struct TinyModelData {
    pub cfg: TinyConfig,
    token_embd: Vec<f32>,
    pos_embd: Vec<f32>,
    output_norm_w: Vec<f32>,
    output_norm_b: Vec<f32>,
    layers: Vec<LayerData>,
}

impl TinyModelData {
    pub fn generate(cfg: TinyConfig, seed: u64) -> TinyModelData {
        let mut rng = Rng::new(seed);
        let mut g = |n: usize| (0..n).map(|_| rng.next_f32(0.2)).collect::<Vec<f32>>();
        let (e, ff) = (cfg.n_embd, cfg.n_ff);

        let token_embd = g(cfg.n_vocab * e);
        let pos_embd = g(cfg.n_ctx * e);
        let output_norm_w = g(e);
        let output_norm_b = g(e);
        let layers = (0..cfg.n_layer)
            .map(|_| LayerData {
                attn_norm_w: g(e),
                attn_norm_b: g(e),
                attn_qkv_w: g(3 * e * e),
                attn_qkv_b: g(3 * e),
                attn_out_w: g(e * e),
                attn_out_b: g(e),
                ffn_norm_w: g(e),
                ffn_norm_b: g(e),
                ffn_up_w: g(ff * e),
                ffn_up_b: g(ff),
                ffn_down_w: g(e * ff),
                ffn_down_b: g(e),
            })
            .collect();

        TinyModelData {
            cfg,
            token_embd,
            pos_embd,
            output_norm_w,
            output_norm_b,
            layers,
        }
    }

    /// Serializes to GGUF bytes. When `include_output` the file carries an
    /// explicit `output.weight` equal to `token_embd`, so a loaded model must
    /// produce the same logits as the tied path that omits it.
    pub fn to_gguf(&self, include_output: bool) -> Vec<u8> {
        let c = self.cfg;
        let (e, ff) = (c.n_embd, c.n_ff);
        let mut b = FixtureBuilder::new()
            .kv("general.architecture", Value::String("gpt2".into()))
            .kv("gpt2.block_count", Value::U32(c.n_layer as u32))
            .kv("gpt2.context_length", Value::U32(c.n_ctx as u32))
            .kv("gpt2.embedding_length", Value::U32(e as u32))
            .kv("gpt2.feed_forward_length", Value::U32(ff as u32))
            .kv("gpt2.attention.head_count", Value::U32(c.n_head as u32))
            .kv("gpt2.attention.layer_norm_epsilon", Value::F32(c.eps));

        b = weight(b, "token_embd.weight", c.n_vocab, e, &self.token_embd);
        b = weight(b, "position_embd.weight", c.n_ctx, e, &self.pos_embd);
        b = weight(b, "output_norm.weight", 1, e, &self.output_norm_w);
        b = weight(b, "output_norm.bias", 1, e, &self.output_norm_b);
        if include_output {
            b = weight(b, "output.weight", c.n_vocab, e, &self.token_embd);
        }
        for (i, l) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            b = weight(b, &n("attn_norm.weight"), 1, e, &l.attn_norm_w);
            b = weight(b, &n("attn_norm.bias"), 1, e, &l.attn_norm_b);
            b = weight(b, &n("attn_qkv.weight"), 3 * e, e, &l.attn_qkv_w);
            b = weight(b, &n("attn_qkv.bias"), 1, 3 * e, &l.attn_qkv_b);
            b = weight(b, &n("attn_output.weight"), e, e, &l.attn_out_w);
            b = weight(b, &n("attn_output.bias"), 1, e, &l.attn_out_b);
            b = weight(b, &n("ffn_norm.weight"), 1, e, &l.ffn_norm_w);
            b = weight(b, &n("ffn_norm.bias"), 1, e, &l.ffn_norm_b);
            b = weight(b, &n("ffn_up.weight"), ff, e, &l.ffn_up_w);
            b = weight(b, &n("ffn_up.bias"), 1, ff, &l.ffn_up_b);
            b = weight(b, &n("ffn_down.weight"), e, ff, &l.ffn_down_w);
            b = weight(b, &n("ffn_down.bias"), 1, e, &l.ffn_down_b);
        }
        b.build()
    }
}

/// Adds a weight tensor. GGUF dims are `[cols, rows]` (`dims[0]` is the
/// contiguous row length); a single-row tensor is written 1D as `[cols]`.
fn weight(b: FixtureBuilder, name: &str, rows: usize, cols: usize, data: &[f32]) -> FixtureBuilder {
    assert_eq!(data.len(), rows * cols, "tiny weight {name} size");
    let dims = if rows == 1 {
        vec![cols as u64]
    } else {
        vec![cols as u64, rows as u64]
    };
    let bytes = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    b.tensor(name, &dims, F32_TYPE_ID, bytes)
}

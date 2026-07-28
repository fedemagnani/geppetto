use super::tiny::{TinyConfig, TinyModelData};
use crate::gguf::GgufFile;
use crate::gpt2::{Gpt2Model, ModelError};

fn tiny_model(include_output: bool) -> Gpt2Model {
    let data = TinyModelData::generate(TinyConfig::small(), 0xC0FFEE);
    let bytes = data.to_gguf(include_output);
    let file = GgufFile::from_bytes(bytes).expect("tiny gguf parses");
    Gpt2Model::from_gguf(&file).expect("tiny gguf loads")
}

/// Last-position logits of a full-prompt forward pass from a fresh state.
fn logits_full(model: &Gpt2Model, tokens: &[u32]) -> Vec<f32> {
    let mut state = model.new_state();
    let logits = model.forward(&mut state, tokens).expect("forward");
    logits.to_vec()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "element {i}: {x} vs {y} (tol {tol})");
    }
}

#[test]
fn loads_hparams_and_shapes() {
    let model = tiny_model(true);
    let hp = model.hparams();
    assert_eq!(hp.n_layer, 2);
    assert_eq!(hp.n_embd, 8);
    assert_eq!(hp.n_head, 2);
    assert_eq!(hp.head_dim(), 4);
    assert_eq!(model.weights().vocabulary_size(), 11);
    assert_eq!(model.weights().layers.len(), 2);
}

#[test]
fn forward_returns_finite_logits_of_the_right_shape() {
    let model = tiny_model(true);
    let logits = logits_full(&model, &[1, 2, 3, 4]);
    assert_eq!(logits.len(), 11);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn forward_is_deterministic() {
    let model = tiny_model(true);
    let a = logits_full(&model, &[5, 1, 9, 2]);
    let b = logits_full(&model, &[5, 1, 9, 2]);
    assert_eq!(a, b);
}

#[test]
fn incremental_decoding_matches_a_full_forward() {
    // the strongest correctness lever without golden outputs: feeding tokens
    // one at a time through the cache must reproduce, at every step, the
    // logits of a fresh full pass over the same prefix. This is also the
    // causality check: the full prefix pass cannot see the later tokens.
    let model = tiny_model(true);
    let tokens = [7, 2, 10, 0, 5, 3];

    let mut state = model.new_state();
    for (i, &token) in tokens.iter().enumerate() {
        let step = model.forward(&mut state, &[token]).unwrap().to_vec();
        let full = logits_full(&model, &tokens[..=i]);
        assert_close(&step, &full, 1e-4);
    }
    assert_eq!(state.n_past(), tokens.len());
}

#[test]
fn chunked_prefill_matches_a_full_forward() {
    // splitting the prompt into two chunks (batch > 1 on top of n_past) must
    // also match the single-shot forward.
    let model = tiny_model(true);
    let tokens = [4, 8, 1, 6, 2];
    let full = logits_full(&model, &tokens);

    let mut state = model.new_state();
    model.forward(&mut state, &tokens[..2]).unwrap();
    let tail = model.forward(&mut state, &tokens[2..]).unwrap();
    assert_close(tail, &full, 1e-4);
}

#[test]
fn clear_resets_the_state_for_reuse() {
    // after a clear, the same arena must reproduce a fresh run exactly: no
    // stale KV position may leak into the new sequence
    let model = tiny_model(true);
    let mut state = model.new_state();
    let a = model.forward(&mut state, &[1, 2, 3]).unwrap().to_vec();
    state.clear();
    let b = model.forward(&mut state, &[1, 2, 3]).unwrap().to_vec();
    assert_eq!(a, b);
}

#[test]
fn tied_output_matches_explicit_output_weight() {
    // omitting output.weight (tied to token_embd) must equal a file that
    // carries output.weight == token_embd.
    let tokens = [1, 5, 2, 8];
    let tied = logits_full(&tiny_model(false), &tokens);
    let explicit = logits_full(&tiny_model(true), &tokens);
    assert_close(&tied, &explicit, 0.0);
}

#[test]
fn missing_tensor_is_a_typed_error() {
    // a metadata-complete file with the top-level tensors but no blocks: the
    // first tensor the loader cannot find is blk.0's attn_norm weight.
    let file = GgufFile::from_bytes(top_level_only_gguf(TinyConfig::small())).unwrap();
    assert!(matches!(
        Gpt2Model::from_gguf(&file),
        Err(ModelError::Gguf(crate::gguf::GgufError::TensorNotFound(name))) if name == "blk.0.attn_norm.weight"
    ));
}

#[test]
fn wrong_architecture_is_rejected() {
    use crate::gguf::Value;
    use crate::gguf::test::fixtures::FixtureBuilder;
    let bytes = FixtureBuilder::new()
        .kv("general.architecture", Value::String("llama".into()))
        .build();
    let file = GgufFile::from_bytes(bytes).unwrap();
    assert!(matches!(
        Gpt2Model::from_gguf(&file),
        Err(ModelError::UnsupportedArch(arch)) if arch == "llama"
    ));
}

#[test]
fn token_out_of_range_is_a_typed_error() {
    let model = tiny_model(true);
    let mut state = model.new_state();
    assert!(matches!(
        model.forward(&mut state, &[999]),
        Err(ModelError::TokenOutOfRange {
            token: 999,
            n_vocab: 11
        })
    ));
}

/// A metadata-complete GGUF carrying only the top-level tensors (no blocks), so
/// the loader fails on the first per-block tensor it requests.
fn top_level_only_gguf(c: TinyConfig) -> Vec<u8> {
    use crate::gguf::Value;
    use crate::gguf::test::fixtures::FixtureBuilder;
    let e = c.n_embd;
    let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    FixtureBuilder::new()
        .kv("general.architecture", Value::String("gpt2".into()))
        .kv("gpt2.block_count", Value::U32(c.n_layer as u32))
        .kv("gpt2.context_length", Value::U32(c.n_ctx as u32))
        .kv("gpt2.embedding_length", Value::U32(e as u32))
        .kv("gpt2.feed_forward_length", Value::U32(c.n_ff as u32))
        .kv("gpt2.attention.head_count", Value::U32(c.n_head as u32))
        .kv("gpt2.attention.layer_norm_epsilon", Value::F32(c.eps))
        .tensor(
            "token_embd.weight",
            &[e as u64, c.n_vocab as u64],
            0,
            f32_bytes(&vec![0.1; c.n_vocab * e]),
        )
        .tensor(
            "position_embd.weight",
            &[e as u64, c.n_ctx as u64],
            0,
            f32_bytes(&vec![0.1; c.n_ctx * e]),
        )
        .tensor(
            "output_norm.weight",
            &[e as u64],
            0,
            f32_bytes(&vec![0.1; e]),
        )
        .tensor("output_norm.bias", &[e as u64], 0, f32_bytes(&vec![0.1; e]))
        .build()
}

use crate::generate::{GenerateError, GenerationConfig, Generator, StopReason};
use crate::gguf::GgufFile;
use crate::gpt2::Gpt2Model;
use crate::gpt2::test::tiny::{TinyConfig, TinyModelData};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::TokenId;

fn tiny_model() -> Gpt2Model {
    let data = TinyModelData::generate(TinyConfig::small(), 0xC0FFEE);
    let file = GgufFile::from_bytes(data.to_gguf(true)).expect("tiny gguf parses");
    Gpt2Model::from_gguf(&file).expect("tiny gguf loads")
}

fn config(max_new_tokens: usize, eos: Option<TokenId>) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        eos,
    }
}

fn sampler(temperature: f32, seed: u64) -> Sampler {
    Sampler::new(SamplerConfig {
        temperature,
        top_k: 0,
        top_p: 1.0,
        min_keep: 1,
        seed,
    })
    .expect("valid sampler config")
}

/// Greedy generation of `n` tokens from `prompt`.
fn greedy_run(model: &Gpt2Model, prompt: &[TokenId], n: usize) -> Vec<TokenId> {
    let mut generator =
        Generator::new(model, Sampler::greedy(), config(n, None), prompt).expect("generator");
    generator.generate_all().expect("generation")
}

#[test]
fn generates_exactly_max_new_tokens() {
    let model = tiny_model();
    for n in [0, 1, 5] {
        let mut generator =
            Generator::new(&model, Sampler::greedy(), config(n, None), &[1, 2, 3]).unwrap();
        let tokens = generator.generate_all().unwrap();
        assert_eq!(tokens.len(), n);
        assert_eq!(generator.stop_reason(), Some(StopReason::MaxTokens));
    }
}

#[test]
fn generated_tokens_are_valid_vocab_ids() {
    let model = tiny_model();
    let n_vocab = model.weights().vocabulary_size() as TokenId;
    for token in greedy_run(&model, &[0], 12) {
        assert!(token < n_vocab, "token {token} outside the vocab");
    }
}

#[test]
fn greedy_generation_is_deterministic() {
    let model = tiny_model();
    assert_eq!(
        greedy_run(&model, &[1, 2], 8),
        greedy_run(&model, &[1, 2], 8)
    );
}

#[test]
fn the_first_greedy_token_is_the_argmax_of_the_prompt_logits() {
    let model = tiny_model();
    let prompt = [3, 1, 4];

    // what the loop produces...
    let first = greedy_run(&model, &prompt, 1)[0];

    // ...against a plain forward pass of the same prompt
    let mut state = model.new_state();
    let last = model.forward(&mut state, &prompt).unwrap();
    let mut expected = 0;
    for (i, &logit) in last.iter().enumerate() {
        if logit > last[expected] {
            expected = i;
        }
    }
    assert_eq!(first, expected as TokenId);
}

#[test]
fn decoding_is_incremental_over_the_cache() {
    let model = tiny_model();
    let prompt = [2, 5, 1];
    let mut generator =
        Generator::new(&model, Sampler::greedy(), config(4, None), &prompt).unwrap();

    generator.next_token().unwrap();
    // the prefill put the whole prompt in the cache
    assert_eq!(generator.state().n_past(), prompt.len());

    for expected in 1..=3 {
        generator.next_token().unwrap();
        // each later step adds exactly one position
        assert_eq!(generator.state().n_past(), prompt.len() + expected);
    }
}

#[test]
fn sampling_the_eos_token_stops_generation() {
    let model = tiny_model();
    // whatever greedy picks first becomes the stop token, so the run must end
    // immediately and emit nothing
    let first = greedy_run(&model, &[1], 1)[0];

    let mut generator =
        Generator::new(&model, Sampler::greedy(), config(16, Some(first)), &[1]).unwrap();
    let tokens = generator.generate_all().unwrap();

    assert!(tokens.is_empty(), "the eos token must not be emitted");
    assert_eq!(generator.stop_reason(), Some(StopReason::Eos));
    assert_eq!(generator.generated(), 1, "eos still counts as sampled");
}

#[test]
fn generation_stops_when_the_context_fills_up() {
    let model = tiny_model();
    let n_ctx = model.hparams().n_ctx;
    let prompt: Vec<TokenId> = (0..n_ctx as TokenId - 2).map(|t| t % 11).collect();

    let mut generator =
        Generator::new(&model, Sampler::greedy(), config(100, None), &prompt).unwrap();
    let tokens = generator.generate_all().unwrap();

    // every forward predicts a token, including the one that fills the last
    // free slot; only the token after that has nowhere to go
    assert_eq!(tokens.len(), n_ctx - prompt.len() + 1);
    assert_eq!(generator.stop_reason(), Some(StopReason::ContextFull));
    assert_eq!(generator.state().n_past(), n_ctx, "the window is full");
}

#[test]
fn seeded_sampling_is_reproducible_and_seed_dependent() {
    let model = tiny_model();
    let run = |seed| {
        let mut generator =
            Generator::new(&model, sampler(1.0, seed), config(12, None), &[1, 2]).unwrap();
        generator.generate_all().unwrap()
    };
    assert_eq!(run(7), run(7), "the same seed must replay exactly");
    assert_ne!(run(7), run(8), "different seeds must diverge");
}

#[test]
fn a_prompt_that_does_not_fit_is_rejected() {
    let model = tiny_model();
    let n_ctx = model.hparams().n_ctx;
    let prompt: Vec<TokenId> = vec![1; n_ctx + 1];

    let result = Generator::new(&model, Sampler::greedy(), config(1, None), &prompt);
    assert!(matches!(result, Err(GenerateError::PromptTooLong { .. })));
}

#[test]
fn an_empty_prompt_is_rejected() {
    let model = tiny_model();
    let result = Generator::new(&model, Sampler::greedy(), config(1, None), &[]);
    assert!(matches!(result, Err(GenerateError::EmptyPrompt)));
}

#[test]
fn an_out_of_range_prompt_token_is_rejected() {
    let model = tiny_model();
    let n_vocab = model.weights().vocabulary_size() as TokenId;
    let mut generator =
        Generator::new(&model, Sampler::greedy(), config(1, None), &[n_vocab]).unwrap();
    assert!(generator.next_token().is_err());
}

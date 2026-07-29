use crate::sampling::{Sampler, SamplerConfig, SamplingError};
use crate::tokenizer::TokenId;

/// The index of the largest logit, ties going to the lowest index.
fn argmax(logits: &[f32]) -> TokenId {
    let mut best = 0;
    for (i, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = i;
        }
    }
    best as TokenId
}

fn config(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_k,
        top_p,
        min_keep: 1,
        seed,
    }
}

/// Logits that are log-probabilities, so an untruncated softmax reproduces them.
fn log_probs(probs: &[f32]) -> Vec<f32> {
    probs.iter().map(|p| p.ln()).collect()
}

#[test]
fn greedy_always_takes_the_argmax() {
    let cases: &[&[f32]] = &[
        &[0.0, 1.0, 2.0],
        &[5.0, -1.0, 0.0],
        &[-3.0, -2.0, -9.0],
        &[1.0],
        &[0.1, 0.2, 0.15, 0.05, 0.3, 0.2],
    ];
    let mut sampler = Sampler::greedy();
    for logits in cases {
        // repeated so a stray RNG draw could not shift the answer
        for _ in 0..8 {
            assert_eq!(
                sampler.sample(logits).unwrap(),
                argmax(logits),
                "{logits:?}"
            );
        }
    }
}

#[test]
fn greedy_breaks_ties_toward_the_lowest_id() {
    let mut sampler = Sampler::greedy();
    for _ in 0..16 {
        assert_eq!(sampler.sample(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
    }
}

#[test]
fn temperature_approaching_zero_converges_to_greedy() {
    let logits = [0.5, 2.0, 1.0, 1.9];
    let mut sampler = Sampler::new(config(1e-6, 0, 1.0, 99)).unwrap();
    for _ in 0..64 {
        assert_eq!(sampler.sample(&logits).unwrap(), argmax(&logits));
    }
}

#[test]
fn the_same_seed_reproduces_the_same_sequence() {
    let logits = log_probs(&[0.2, 0.3, 0.1, 0.25, 0.15]);
    let draw = |seed| {
        let mut sampler = Sampler::new(config(1.0, 0, 1.0, seed)).unwrap();
        (0..64)
            .map(|_| sampler.sample(&logits).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(draw(1234), draw(1234));
}

#[test]
fn different_seeds_produce_different_sequences() {
    // 64 draws over a near-uniform 32-way distribution: an accidental match
    // is not a realistic flake
    let logits = vec![0.0f32; 32];
    let draw = |seed| {
        let mut sampler = Sampler::new(config(1.0, 0, 1.0, seed)).unwrap();
        (0..64)
            .map(|_| sampler.sample(&logits).unwrap())
            .collect::<Vec<_>>()
    };
    assert_ne!(draw(1), draw(2));
}

#[test]
fn draws_follow_the_probabilities() {
    let probs = [0.6f32, 0.3, 0.1];
    let logits = log_probs(&probs);
    let mut sampler = Sampler::new(config(1.0, 0, 1.0, 2024)).unwrap();

    const N: usize = 60_000;
    let mut counts = [0usize; 3];
    for _ in 0..N {
        counts[sampler.sample(&logits).unwrap() as usize] += 1;
    }
    for (i, &expected) in probs.iter().enumerate() {
        let observed = counts[i] as f32 / N as f32;
        assert!(
            (observed - expected).abs() < 0.01,
            "token {i}: sampled {observed}, expected {expected}"
        );
    }
}

#[test]
fn top_k_confines_the_draw_to_the_k_best() {
    // uniform logits, so without top-k every token would appear
    let logits = vec![0.0f32; 10];
    let mut sampler = Sampler::new(config(1.0, 3, 1.0, 5)).unwrap();
    for _ in 0..500 {
        let id = sampler.sample(&logits).unwrap();
        assert!(id < 3, "sampled {id} outside the top 3");
    }
}

#[test]
fn top_p_confines_the_draw_to_the_nucleus() {
    let logits = log_probs(&[0.5, 0.25, 0.15, 0.06, 0.04]);
    // cumulative 0.5 then 0.75: only ids 0 and 1 survive
    let mut sampler = Sampler::new(config(1.0, 0, 0.7, 11)).unwrap();
    for _ in 0..500 {
        let id = sampler.sample(&logits).unwrap();
        assert!(id < 2, "sampled {id} outside the nucleus");
    }
}

#[test]
fn the_chain_applies_temperature_before_truncation() {
    // Temperature rescales but never reorders, so top-k sees the same ranking
    // whatever the temperature: the surviving ids must not depend on it.
    let logits = [0.5f32, 3.0, 1.0, 2.5, 0.1];
    let survivors = |temp| {
        let mut sampler = Sampler::new(config(temp, 2, 1.0, 3)).unwrap();
        sampler.sample(&logits).unwrap();
        let mut ids: Vec<TokenId> = sampler
            .candidates()
            .entries()
            .iter()
            .map(|e| e.id)
            .collect();
        ids.sort_unstable();
        ids
    };
    assert_eq!(survivors(0.5), vec![1, 3]);
    assert_eq!(survivors(2.0), vec![1, 3]);
}

#[test]
fn a_filtered_set_is_renormalized_before_the_draw() {
    let logits = log_probs(&[0.5, 0.25, 0.15, 0.1]);
    let mut sampler = Sampler::new(config(1.0, 2, 1.0, 8)).unwrap();
    sampler.sample(&logits).unwrap();

    let total: f32 = sampler.candidates().entries().iter().map(|e| e.p).sum();
    assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}");
    assert_eq!(sampler.candidates().len(), 2);
}

#[test]
fn invalid_logits_are_rejected() {
    let mut sampler = Sampler::greedy();
    assert_eq!(sampler.sample(&[]).unwrap_err(), SamplingError::EmptyLogits);
    assert_eq!(
        sampler.sample(&[1.0, f32::NAN]).unwrap_err(),
        SamplingError::NaNLogit { index: 1 }
    );
}

#[test]
fn invalid_configs_are_rejected() {
    assert!(matches!(
        Sampler::new(config(f32::NAN, 0, 1.0, 0)).unwrap_err(),
        SamplingError::InvalidTemperature(_)
    ));
    for bad in [-0.1, 1.5, f32::NAN] {
        assert!(
            matches!(
                Sampler::new(config(1.0, 0, bad, 0)).unwrap_err(),
                SamplingError::InvalidTopP(_)
            ),
            "top_p {bad} should be rejected"
        );
    }
}

#[test]
fn a_single_candidate_vocab_always_samples_it() {
    let mut sampler = Sampler::new(config(1.0, 0, 0.5, 4)).unwrap();
    for _ in 0..16 {
        assert_eq!(sampler.sample(&[-2.5]).unwrap(), 0);
    }
}

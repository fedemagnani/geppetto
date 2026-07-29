use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::bench::alloc::AllocStats;
use crate::bench::{CORPUS, Samples, Scenario};
use crate::generate::{GenerateError, GenerationConfig, Generator};
use crate::gguf::{GgufError, GgufFile};
use crate::gpt2::{Gpt2Model, ModelError};
use crate::sampling::Sampler;
use crate::tokenizer::{TokenId, Tokenizer, TokenizerError};

/// Loads the model once and runs scenarios against it.
pub struct Harness {
    model_path: PathBuf,
    model: Gpt2Model,
    tokenizer: Tokenizer,
    /// Token ids of the corpus, cycled to build prompts of any length.
    corpus_tokens: Vec<TokenId>,
}

impl Harness {
    /// Runs one scenario: `warmup` discarded runs, then `samples` measured
    /// ones. Warmup exists because the first run pays for cold pages, a cold
    /// branch predictor and an unboosted clock.
    pub fn run(
        &self,
        scenario: Scenario,
        samples: usize,
        warmup: usize,
    ) -> Result<ScenarioResult, BenchError> {
        let n_ctx = self.model.hparams().n_ctx;
        if scenario.max_context() > n_ctx {
            return Err(BenchError::ContextTooSmall {
                scenario: scenario.to_string(),
                needed: scenario.max_context(),
                n_ctx,
            });
        }

        for _ in 0..warmup {
            self.run_once(scenario)?;
        }
        let mut measurements = Vec::with_capacity(samples);
        for _ in 0..samples {
            let measurement = self.run_once(scenario)?;
            measurements.push(measurement);
        }
        Ok(ScenarioResult {
            scenario,
            measurements,
            peak_rss_bytes: crate::bench::peak_rss_bytes(),
        })
    }

    fn run_once(&self, scenario: Scenario) -> Result<Measurement, BenchError> {
        match scenario {
            Scenario::Load => self.measure_load(),
            Scenario::Decode {
                prompt,
                new_tokens: n,
            } => self.measure_decode(prompt, n),
            Scenario::Prefill { prompt } => self.measure_prefill(prompt),
        }
    }

    /// Times opening the file and building the model. Reported as a duration;
    /// there are no tokens to charge it against.
    fn measure_load(&self) -> Result<Measurement, BenchError> {
        let before = AllocStats::now();
        let started = Instant::now();
        let file = GgufFile::open(&self.model_path)?;
        let model = Gpt2Model::from_gguf(&file)?;
        let elapsed = started.elapsed();
        // keep the model alive until after the clock stops, so the timing does
        // not accidentally include or exclude its drop
        std::hint::black_box(&model);

        Ok(Measurement {
            elapsed,
            tokens: 0,
            forward_calls: 0,
            allocs: AllocStats::now().since(before),
        })
    }

    /// Times `new_tokens` single-token steps on top of a prefilled prompt.
    /// The prefill itself is deliberately outside the clock: this is the
    /// steady-state KPI, not a mixed one.
    fn measure_decode(
        &self,
        prompt_len: usize,
        new_tokens: usize,
    ) -> Result<Measurement, BenchError> {
        let prompt = self.prompt(prompt_len);
        let config = GenerationConfig {
            max_new_tokens: new_tokens + 1,
            // never stop early: a short run would silently become a different
            // measurement from the one requested
            eos: None,
        };
        let mut generator = Generator::new(&self.model, Sampler::greedy(), config, &prompt)?;

        // untimed: this call carries the whole prompt forward
        generator.next_token()?;

        let before = AllocStats::now();
        let started = Instant::now();
        let mut produced = 0;
        for _ in 0..new_tokens {
            if generator.next_token()?.is_none() {
                break;
            }
            produced += 1;
        }
        let elapsed = started.elapsed();
        let allocs = AllocStats::now().since(before);

        if produced == 0 {
            return Err(BenchError::NoTokens(
                Scenario::Decode {
                    prompt: prompt_len,
                    new_tokens,
                }
                .to_string(),
            ));
        }
        Ok(Measurement {
            elapsed,
            tokens: produced,
            // decoding runs one forward pass per token
            forward_calls: produced,
            allocs,
        })
    }

    /// Times the single forward pass that consumes the whole prompt, which is
    /// also the time to first token.
    fn measure_prefill(&self, prompt_len: usize) -> Result<Measurement, BenchError> {
        let prompt = self.prompt(prompt_len);
        let config = GenerationConfig {
            max_new_tokens: 1,
            eos: None,
        };
        let mut generator = Generator::new(&self.model, Sampler::greedy(), config, &prompt)?;

        let before = AllocStats::now();
        let started = Instant::now();
        let token = generator.next_token()?;
        let elapsed = started.elapsed();
        let allocs = AllocStats::now().since(before);

        if token.is_none() {
            return Err(BenchError::NoTokens(
                Scenario::Prefill { prompt: prompt_len }.to_string(),
            ));
        }
        Ok(Measurement {
            elapsed,
            // the prompt is the work that was done, so it is what gets charged
            tokens: prompt_len,
            // however long the prompt, a prefill is a single forward pass
            forward_calls: 1,
            allocs,
        })
    }

    pub fn load(model_path: &Path) -> Result<Harness, BenchError> {
        let file = GgufFile::open(model_path)?;
        let model = Gpt2Model::from_gguf(&file)?;
        let tokenizer = Tokenizer::from_gguf(&file)?;
        let corpus_tokens = tokenizer.encode(CORPUS)?;
        assert!(!corpus_tokens.is_empty(), "corpus tokenized to nothing");

        Ok(Harness {
            model_path: model_path.to_path_buf(),
            model,
            tokenizer,
            corpus_tokens,
        })
    }

    pub fn model(&self) -> &Gpt2Model {
        &self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// A prompt of exactly `n` tokens, by cycling the corpus.
    fn prompt(&self, n: usize) -> Vec<TokenId> {
        self.corpus_tokens.iter().copied().cycle().take(n).collect()
    }
}

/// One timed run of one scenario.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub elapsed: Duration,
    pub tokens: usize,
    /// Forward passes inside the timed region: one per token when decoding,
    /// exactly one for a prefill however long the prompt.
    ///
    /// Allocations are reported against this rather than against `tokens`,
    /// because the forward pass is what allocates. Dividing a prefill's
    /// allocations by its prompt length would report the same work as ~8 per
    /// token at a 128-token prompt and ~2 at 512, which says nothing about
    /// the code.
    pub forward_calls: usize,
    pub allocs: AllocStats,
}

impl Measurement {
    pub fn tokens_per_second(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.tokens as f64 / seconds
    }

    pub fn allocs_per_forward(&self) -> f64 {
        if self.forward_calls == 0 {
            return 0.0;
        }
        self.allocs.count as f64 / self.forward_calls as f64
    }

    pub fn bytes_per_forward(&self) -> f64 {
        if self.forward_calls == 0 {
            return 0.0;
        }
        self.allocs.bytes as f64 / self.forward_calls as f64
    }
}

/// Every sample of one scenario, plus the statistics the report prints.
#[derive(Debug)]
pub struct ScenarioResult {
    pub scenario: Scenario,
    pub measurements: Vec<Measurement>,
    peak_rss_bytes: u64,
}

impl ScenarioResult {
    pub fn samples(&self) -> usize {
        self.measurements.len()
    }

    pub fn tokens_per_second(&self) -> Samples {
        Samples::new(
            self.measurements
                .iter()
                .map(Measurement::tokens_per_second)
                .collect(),
        )
    }

    pub fn seconds(&self) -> Samples {
        Samples::new(
            self.measurements
                .iter()
                .map(|m| m.elapsed.as_secs_f64())
                .collect(),
        )
    }

    /// Milliseconds per timed token; for prefill this is the per-prompt-token
    /// cost, for decode the inter-token latency.
    pub fn millis_per_token(&self) -> Samples {
        Samples::new(
            self.measurements
                .iter()
                .map(|m| {
                    if m.tokens == 0 {
                        0.0
                    } else {
                        m.elapsed.as_secs_f64() * 1000.0 / m.tokens as f64
                    }
                })
                .collect(),
        )
    }

    /// Allocations per forward pass. Comparable across decode and prefill,
    /// unlike a per-token figure.
    pub fn allocs_per_forward(&self) -> Samples {
        Samples::new(
            self.measurements
                .iter()
                .map(Measurement::allocs_per_forward)
                .collect(),
        )
    }

    pub fn bytes_per_forward(&self) -> Samples {
        Samples::new(
            self.measurements
                .iter()
                .map(Measurement::bytes_per_forward)
                .collect(),
        )
    }

    /// Process peak RSS once this scenario finished. Monotonic, so the jump
    /// from the previous scenario is what this one cost.
    pub const fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss_bytes
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Generate(#[from] GenerateError),
    #[error("scenario {scenario} needs {needed} context positions, model has {n_ctx}")]
    ContextTooSmall {
        scenario: String,
        needed: usize,
        n_ctx: usize,
    },
    #[error("scenario {0} produced no tokens, so there is nothing to time")]
    NoTokens(String),
}

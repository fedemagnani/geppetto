use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use geppetto::generate::{GenerationConfig, Generator, StopReason, Utf8Stream};
use geppetto::gguf::GgufFile;
use geppetto::gpt2::Gpt2Model;
use geppetto::sampling::{Sampler, SamplerConfig};
use geppetto::tokenizer::Tokenizer;

/// Generate text with a GPT-2 GGUF model.
#[derive(Parser)]
struct Cli {
    /// Path to the GGUF model.
    model: PathBuf,
    /// Prompt to continue. Empty starts from the end-of-text token.
    #[arg(short, long, default_value = "")]
    prompt: String,
    /// Maximum number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    max_tokens: usize,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.8)]
    temp: f32,
    /// Keep only the k most likely tokens; 0 disables.
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Keep the most likely tokens up to a cumulative probability of p.
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Seed for the sampler.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Keep generating past the end-of-text token.
    #[arg(long)]
    ignore_eos: bool,
}

impl Cli {
    fn sampler(&self) -> Result<Sampler, Box<dyn std::error::Error>> {
        let config = SamplerConfig {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_keep: 1,
            seed: self.seed,
        };
        let out = Sampler::new(config)?;
        Ok(out)
    }
}

#[hotpath::main]
fn main() -> ExitCode {
    let args = Cli::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("generate: {}: {err}", args.model.display());
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::open(&args.model)?;
    let model = Gpt2Model::from_gguf(&file)?;
    let tokenizer = Tokenizer::from_gguf(&file)?;
    let eos = tokenizer.vocab().eos();

    // GPT-2 has no dedicated BOS: unconditional generation is seeded with the
    // end-of-text token, the same way the prompt-less C++ path does it
    let mut prompt_tokens = tokenizer.encode(&args.prompt)?;
    if prompt_tokens.is_empty() {
        prompt_tokens.push(eos);
    }

    let sampler = args.sampler()?;

    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        eos: (!args.ignore_eos).then_some(eos),
    };
    let mut generator = Generator::new(&model, sampler, config, &prompt_tokens)?;

    let hp = model.hparams();
    eprintln!(
        "model: {} layers, n_embd {}, n_ctx {}, vocab {}",
        hp.n_layer,
        hp.n_embd,
        hp.n_ctx,
        model.weights().vocabulary_size()
    );
    eprintln!("prompt: {} tokens\n", prompt_tokens.len());

    let mut stdout = std::io::stdout();
    write!(stdout, "{}", args.prompt)?;
    stdout.flush()?;

    let mut stream = Utf8Stream::new();
    let mut text = String::new();
    let mut bytes = Vec::new();
    let mut n_generated = 0usize;
    let started = Instant::now();

    while let Some(token) = generator.next_token()? {
        // decode into reused buffers, then emit only the complete characters:
        // a multi-byte character split across tokens must not print as garbage
        bytes.clear();
        tokenizer.decode_into(&[token], &mut bytes)?;
        text.clear();
        stream.push(&bytes, &mut text);

        write!(stdout, "{text}")?;
        stdout.flush()?;
        n_generated += 1;
    }
    text.clear();
    stream.finish(&mut text);
    write!(stdout, "{text}")?;
    writeln!(stdout)?;
    stdout.flush()?;

    let elapsed = started.elapsed();
    let reason = match generator.stop_reason() {
        Some(StopReason::Eos) => "end of text",
        Some(StopReason::MaxTokens) => "token limit",
        Some(StopReason::ContextFull) => "context full",
        None => "unfinished",
    };
    let per_second = n_generated as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
    eprintln!(
        "\n{n_generated} tokens in {:.2}s ({per_second:.1} tok/s), stopped: {reason}",
        elapsed.as_secs_f64()
    );
    Ok(())
}

//! Benchmark harness for the KPIs in
//! `docs/prompts/onboarding/OPTIMIZATIONS.md`.
//!
//! Plain run reports KPI-grade numbers:
//!
//! ```text
//! cargo run --release --bin bench -- <model.gguf>
//! ```
//!
//! With profiling, which adds hotpath's per-function timing and allocation
//! tables at the cost of perturbing the timings:
//!
//! ```text
//! cargo run --release --features hotpath,hotpath-alloc --bin bench -- <model.gguf>
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use geppetto::bench::{BenchError, Environment, Harness, Report, Scenario};

// hotpath-alloc installs its own global allocator, so ours stands down to
// avoid two `#[global_allocator]` definitions.
#[cfg(not(feature = "hotpath-alloc"))]
#[global_allocator]
static ALLOCATOR: geppetto::bench::alloc::CountingAllocator =
    geppetto::bench::alloc::CountingAllocator;

/// Measure geppetto's inference KPIs.
#[derive(Parser)]
struct Cli {
    /// Path to a GPT-2 GGUF model with f32/f16 weights.
    model: PathBuf,
    /// Measured samples per scenario. Defaults are per-scenario and scaled to
    /// how long one sample takes.
    #[arg(long)]
    samples: Option<usize>,
    /// Discarded runs before measuring.
    #[arg(long)]
    warmup: Option<usize>,
    /// Emit machine-readable JSON instead of a table.
    #[arg(long)]
    json: bool,
    /// One quick decode scenario, for checking the harness works.
    #[arg(long)]
    quick: bool,
}

fn main() -> ExitCode {
    let args = Cli::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("bench: {}: {err}", args.model.display());
            ExitCode::FAILURE
        }
    }
}

#[hotpath::main]
fn run(args: &Cli) -> Result<(), BenchError> {
    let environment = Environment::detect();
    let model_bytes = std::fs::metadata(&args.model).map_or(0, |m| m.len());

    let scenarios = if args.quick {
        vec![Scenario::Decode {
            prompt: 1,
            new_tokens: 8,
        }]
    } else {
        Scenario::defaults()
    };

    if !args.json {
        eprintln!("loading {} ...", args.model.display());
    }
    let harness = Harness::load(&args.model)?;

    let mut results = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        let samples = args.samples.unwrap_or_else(|| scenario.default_samples());
        let warmup = args.warmup.unwrap_or_else(|| scenario.default_warmup());
        if !args.json {
            eprintln!("  {scenario}: {warmup} warmup + {samples} samples ...");
        }
        results.push(harness.run(scenario, samples, warmup)?);
    }

    let report = Report::new(
        environment,
        args.model.display().to_string(),
        model_bytes,
        results,
    );
    if args.json {
        print!("{}", report.to_json());
    } else {
        print!("{}", report.to_text());
    }
    Ok(())
}

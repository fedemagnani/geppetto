//! Benchmark harness for the KPIs defined in
//! `docs/prompts/onboarding/OPTIMIZATIONS.md`.
//!
//! Every figure in that document's baseline and journal comes from here. The
//! harness owns three things the KPIs depend on: repeated sampling with warmup
//! discarded, order statistics that survive a noisy laptop, and the
//! environment block that makes a number reproducible.

pub mod alloc;
mod calibrate;
mod env;
mod report;
mod runner;
mod scenario;
mod stats;
mod timed;

#[cfg(test)]
mod test;

pub use calibrate::SampleBudget;
pub use env::{Environment, peak_rss_bytes};
pub use report::Report;
pub use runner::{BenchError, Harness, Measurement, ScenarioResult};
pub use scenario::Scenario;
pub use stats::Samples;
pub use timed::TimedCall;

/// Text the harness prompts with. Real English so the tokenizer does
/// representative work; prompts longer than this are built by repeating it,
/// which keeps every run deterministic.
pub const CORPUS: &str = "The quick brown fox jumps over the lazy dog. \
Machine learning models process sequences of tokens, and each token attends \
to every token before it. In a transformer, the cost of that attention grows \
with the length of the context, while the cost of the feed-forward layers \
stays the same per position. Understanding which of the two dominates a given \
workload is the whole point of measuring instead of guessing. ";

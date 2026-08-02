//! Result collection and the CLI table.

use geppetto::bench::{SampleBudget, Samples};
use geppetto::tensor::{MatmulNtKernel, Shape};
use indicatif::{ProgressBar, ProgressStyle};

use crate::workload::Sampler;

/// Accumulates one [`SummaryStats`] row per kernel x shape, then renders
/// them as a table. Kernels are named by their type, so adding one to the
/// run is a turbofish, not a string.
pub struct Collector {
    budget: SampleBudget,
    rows: Vec<SummaryStats>,
}

impl Collector {
    pub fn new(budget: SampleBudget) -> Collector {
        Collector {
            budget,
            rows: Vec::new(),
        }
    }

    /// Benchmarks `K` over every shape pair, labelled with the last
    /// segment of the type's path.
    ///
    /// The progress bar (stderr, cleared on finish) advances only between
    /// cells; the steady tick that animates the spinner redraws from its
    /// own thread every 200ms, a ~microsecond stderr write the median/MAD
    /// statistics absorb without a trace.
    pub fn run_bench<K: MatmulNtKernel + Default>(&mut self, shapes: &[(Shape, Shape)]) {
        let kernel = K::default();
        let name = short_type_name(std::any::type_name::<K>());
        let bar = progress_bar(&name, shapes.len());
        for &(a, b) in shapes {
            bar.set_message(format!("{a} @ {b}"));
            let samples = Sampler::new(&kernel, a, b).compute_samples(self.budget);
            self.rows
                .push(SummaryStats::new(name.clone(), a, b, &samples));
            bar.inc(1);
        }
        bar.finish_and_clear();
    }

    pub fn display(&self) {
        println!(
            "{:<43} {:>12} {:>14} {:>12} {:>12} {:>8} {:>9}",
            "kernel", "a", "b", "p50", "p95", "spread", "GFLOP/s"
        );
        for row in &self.rows {
            println!(
                "{:<43} {:>12} {:>14} {:>12} {:>12} {:>7.1}% {:>9.2}",
                row.kernel,
                row.a.to_string(),
                row.b.to_string(),
                format_seconds(row.p50),
                format_seconds(row.p95),
                row.spread * 100.0,
                row.gflops,
            );
        }
    }
}

/// Drops every module path from a `type_name`, including inside generic
/// arguments: `a::b::AutoVecDotProduct<a::b::Unfused, 8>` becomes
/// `AutoVecDotProduct<Unfused, 8>` -- what the source spells.
fn short_type_name(full: &str) -> String {
    let mut out = String::new();
    let mut segment = String::new();
    for c in full.chars() {
        if c.is_alphanumeric() || c == '_' {
            segment.push(c);
        } else if c == ':' {
            segment.clear();
        } else {
            out.push_str(&segment);
            segment.clear();
            out.push(c);
        }
    }
    out.push_str(&segment);
    out
}

/// A per-kernel bar: spinner, kernel name as bold prefix, elapsed time,
/// the in-flight shape pair as the dimmed message, one tick per cell.
fn progress_bar(kernel: &str, cells: usize) -> ProgressBar {
    let template = "{spinner:.green} {prefix:43.bold.cyan} [{bar:30.cyan/blue}] {pos}/{len} {elapsed} {msg:.dim}";
    let style = ProgressStyle::with_template(template)
        .expect("static template is valid")
        .progress_chars("=> ");
    let bar = ProgressBar::new(cells as u64)
        .with_style(style)
        .with_prefix(kernel.to_string());
    bar.enable_steady_tick(std::time::Duration::from_millis(200));
    bar
}

/// One kernel x shape cell condensed from its [`Samples`], plus the
/// identity of what was measured.
struct SummaryStats {
    kernel: String,
    a: Shape,
    b: Shape,
    p50: f64,
    p95: f64,
    spread: f64,
    gflops: f64,
}

impl SummaryStats {
    fn new(kernel: String, a: Shape, b: Shape, samples: &Samples) -> SummaryStats {
        let p50 = samples.median();
        let flops = (2 * a.rows() * a.cols() * b.cols()) as f64;
        SummaryStats {
            kernel,
            a,
            b,
            p50,
            p95: samples.percentile(0.95),
            spread: samples.relative_spread(),
            gflops: flops / p50 / 1e9,
        }
    }
}

fn format_seconds(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.2} s")
    } else if seconds >= 1e-3 {
        format!("{:.2} ms", seconds * 1e3)
    } else {
        format!("{:.2} µs", seconds * 1e6)
    }
}

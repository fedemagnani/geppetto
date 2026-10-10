//! The kernel-agnostic half of a microkernel bench target: the result
//! collector with its CLI table, the per-kernel progress bar and the
//! type-name shortener. Each bench includes this via `#[path]` and keeps
//! for itself what a cell measures -- its workload and the driver loop
//! that knows how to construct one.

use geppetto::bench::{SampleBudget, Samples};
use indicatif::{ProgressBar, ProgressStyle};

/// Accumulates one [`SummaryStats`] row per kernel x workload cell, then
/// renders them as a table. It never runs anything: the owning bench
/// samples a cell and hands the result to [`record`](Collector::record).
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

    /// The budget every cell samples under, so a bench's driver loop reads
    /// it from here instead of threading a second copy around.
    pub fn budget(&self) -> SampleBudget {
        self.budget
    }

    /// Adds one cell: `workload` labels the row, `flops` is the floating
    /// point work of one timed call at that cell.
    pub fn record(&mut self, kernel: &str, workload: String, flops: f64, samples: &Samples) {
        let row = SummaryStats::new(kernel.to_string(), workload, flops, samples);
        self.rows.push(row);
    }

    pub fn display(&self) {
        println!(
            "{:<43} {:>27} {:>12} {:>12} {:>8} {:>9}",
            "kernel", "workload", "p50", "p95", "spread", "GFLOP/s"
        );
        for row in &self.rows {
            println!(
                "{:<43} {:>27} {:>12} {:>12} {:>7.1}% {:>9.2}",
                row.kernel,
                row.workload,
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
pub fn short_type_name(full: &str) -> String {
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
/// the in-flight workload as the dimmed message, one tick per cell.
///
/// The bar lives on stderr and clears on finish; the steady tick that
/// animates the spinner redraws from its own thread every 200ms, a
/// ~microsecond stderr write the median/MAD statistics absorb without a
/// trace.
pub fn progress_bar(kernel: &str, cells: usize) -> ProgressBar {
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

/// One kernel x workload cell condensed from its [`Samples`], plus the
/// identity of what was measured.
struct SummaryStats {
    kernel: String,
    workload: String,
    p50: f64,
    p95: f64,
    spread: f64,
    gflops: f64,
}

impl SummaryStats {
    fn new(kernel: String, workload: String, flops: f64, samples: &Samples) -> SummaryStats {
        let p50 = samples.median();
        SummaryStats {
            kernel,
            workload,
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

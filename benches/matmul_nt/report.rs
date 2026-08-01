//! Result collection and the CLI table.

use geppetto::bench::{SampleBudget, Samples};
use geppetto::tensor::{MatmulNtKernel, Shape};

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
    pub fn run_bench<K: MatmulNtKernel + Default>(&mut self, shapes: &[(Shape, Shape)]) {
        let kernel = K::default();
        let full_name = std::any::type_name::<K>();
        let name = full_name.rsplit("::").next().unwrap_or(full_name);
        for &(a, b) in shapes {
            let samples = Sampler::new(&kernel, a, b).compute_samples(self.budget);
            self.rows.push(SummaryStats::new(name, a, b, &samples));
        }
    }

    pub fn display(&self) {
        println!(
            "{:<20} {:>12} {:>14} {:>12} {:>12} {:>8} {:>9}",
            "kernel", "a", "b", "p50", "p95", "spread", "GFLOP/s"
        );
        for row in &self.rows {
            println!(
                "{:<20} {:>12} {:>14} {:>12} {:>12} {:>7.1}% {:>9.2}",
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

/// One kernel x shape cell condensed from its [`Samples`], plus the
/// identity of what was measured.
struct SummaryStats {
    kernel: &'static str,
    a: Shape,
    b: Shape,
    p50: f64,
    p95: f64,
    spread: f64,
    gflops: f64,
}

impl SummaryStats {
    fn new(kernel: &'static str, a: Shape, b: Shape, samples: &Samples) -> SummaryStats {
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

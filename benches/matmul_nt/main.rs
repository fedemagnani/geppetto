//! Microkernel benchmark for the `a @ b^T` weight matmul: every
//! [`MatmulNtKernel`] over the shapes GPT-2 large actually calls, timed in
//! isolation on synthetic data.
//!
//! Complements `src/bench` rather than competing with it: this file answers
//! "which kernel is faster at shape X" in seconds per experiment, the
//! end-to-end harness answers "does the win survive in decode tok/s". A new
//! kernel earns a `run_bench::<K>` line here first, a model run second.
//!
//! Run with `cargo bench --bench matmul_nt`. Like every laptop benchmark,
//! medians are the headline and a relative spread above a few percent means
//! the machine was not quiet.

mod report;
mod workload;

use std::time::Duration;

use geppetto::bench::SampleBudget;
use geppetto::tensor::{AutoVecMatMulNt, FmaMatMulNt, NaiveMatMulNt, Shape};

use crate::report::Collector;

/// The matmul shapes one GPT-2 large layer stack issues, as logical
/// `a [m, k] @ b [k, n]` pairs (the weight is stored transposed `[n, k]`
/// per the nt convention). Decode rows (`m = 1`): qkv, attn-proj, mlp-up,
/// mlp-down, unembed; then qkv and mlp-up again at a prefill tile
/// (`m = 128`).
const SHAPES: [(Shape, Shape); 7] = [
    (Shape::new(1, 1280), Shape::new(1280, 3840)),
    (Shape::new(1, 1280), Shape::new(1280, 1280)),
    (Shape::new(1, 1280), Shape::new(1280, 5120)),
    (Shape::new(1, 5120), Shape::new(5120, 1280)),
    (Shape::new(1, 1280), Shape::new(1280, 50257)),
    (Shape::new(128, 1280), Shape::new(1280, 3840)),
    (Shape::new(128, 1280), Shape::new(1280, 5120)),
];

fn main() {
    // enough samples for a stable median, capped so the vocab-sized shapes
    // keep the whole run in tens of seconds
    let budget = SampleBudget::new(Duration::from_millis(250), 5, 200);
    let mut collector = Collector::new(budget);
    collector.run_bench::<NaiveMatMulNt>(&SHAPES);
    collector.run_bench::<AutoVecMatMulNt<8>>(&SHAPES);
    collector.run_bench::<AutoVecMatMulNt<16>>(&SHAPES);
    collector.run_bench::<AutoVecMatMulNt<32>>(&SHAPES);
    collector.run_bench::<AutoVecMatMulNt<64>>(&SHAPES);
    collector.run_bench::<FmaMatMulNt<8>>(&SHAPES);
    collector.run_bench::<FmaMatMulNt<16>>(&SHAPES);
    collector.run_bench::<FmaMatMulNt<32>>(&SHAPES);
    collector.run_bench::<FmaMatMulNt<64>>(&SHAPES);
    collector.display();
}

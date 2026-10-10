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

#[path = "../common/mod.rs"]
mod common;
mod workload;

use std::time::Duration;

use geppetto::bench::SampleBudget;
use geppetto::tensor::{
    MatMulNtAutoVecFma, MatMulNtAutoVecUnfused, MatMulNtNaive, MatMulNtNeonTiled,
    MatMulNtPackedNeon, MatMulNtTiledFma, MatMulNtTiledUnfused, MatmulNtKernel, Shape,
};

use crate::common::{Collector, progress_bar, short_type_name};
use crate::workload::Sampler;

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
    run_bench::<MatMulNtNaive>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecUnfused<8>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecUnfused<16>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecUnfused<32>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecUnfused<64>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecFma<8>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecFma<16>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecFma<32>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtAutoVecFma<64>>(&mut collector, &SHAPES);

    run_bench::<MatMulNtTiledUnfused<2, 16>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledUnfused<4, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledUnfused<4, 8>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<2, 16>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<4, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<4, 8>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<4, 16>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<8, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtTiledFma<8, 8>>(&mut collector, &SHAPES);

    run_bench::<MatMulNtNeonTiled<2, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtNeonTiled<4, 2>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtNeonTiled<4, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtNeonTiled<6, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtNeonTiled<8, 2>>(&mut collector, &SHAPES);

    run_bench::<MatMulNtPackedNeon<2, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtPackedNeon<4, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtPackedNeon<6, 4>>(&mut collector, &SHAPES);
    run_bench::<MatMulNtPackedNeon<8, 2>>(&mut collector, &SHAPES);
    collector.display();
}

/// Benchmarks `K` over every shape pair, labelled with the last segment of
/// the type's path. Cells are built one at a time inside the loop -- the
/// vocab-sized weights are too large to hold all at once.
fn run_bench<K: MatmulNtKernel + Default>(collector: &mut Collector, shapes: &[(Shape, Shape)]) {
    let kernel = K::default();
    let name = short_type_name(std::any::type_name::<K>());
    let bar = progress_bar(&name, shapes.len());
    for &(a, b) in shapes {
        let label = format!("{a} @ {b}");
        bar.set_message(label.clone());
        let sampler = Sampler::new(&kernel, a, b);
        let samples = sampler.compute_samples(collector.budget());
        let flops = (2 * a.rows() * a.cols() * b.cols()) as f64;
        collector.record(&name, label, flops, &samples);
        bar.inc(1);
    }
    bar.finish_and_clear();
}

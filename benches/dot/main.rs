//! Microkernel benchmark for the [`DotProduct`] family: every dot the
//! `matmul_nt` dot kernels are built from, timed on two contiguous rows
//! without the matmul loop nest around it.
//!
//! Sits one level below `benches/matmul_nt`: this file answers "how fast
//! is the dot itself at length k", the matmul bench answers "does that
//! speed survive the loop nest and the weight traffic". A new dot earns a
//! `run_bench::<D>` line here first, a matmul cell second.
//!
//! Run with `cargo bench --bench dot`. Like every laptop benchmark,
//! medians are the headline and a relative spread above a few percent
//! means the machine was not quiet.

#[path = "../common/mod.rs"]
mod common;
mod workload;

use std::time::Duration;

use geppetto::bench::SampleBudget;
use geppetto::tensor::{AutoVecDot, DotProduct, FmaDot, NaiveDotProduct};

use crate::common::{Collector, progress_bar, short_type_name};
use crate::workload::{DOTS_PER_CALL, Sampler};

/// The contracted lengths GPT-2 large dots over (1280 into the layer
/// stack, 5120 down from the MLP), the attention head width 64, and 1283
/// -- coprime to every lane count, so the tail path is never idle.
const LENGTHS: [usize; 4] = [64, 1280, 1283, 5120];

fn main() {
    // enough samples for a stable median, capped so the run stays in
    // seconds even with every lane width enrolled
    let budget = SampleBudget::new(Duration::from_millis(250), 5, 200);
    let mut collector = Collector::new(budget);
    run_bench::<NaiveDotProduct>(&mut collector, &LENGTHS);
    run_bench::<AutoVecDot<8>>(&mut collector, &LENGTHS);
    run_bench::<AutoVecDot<16>>(&mut collector, &LENGTHS);
    run_bench::<AutoVecDot<32>>(&mut collector, &LENGTHS);
    run_bench::<AutoVecDot<64>>(&mut collector, &LENGTHS);
    run_bench::<FmaDot<8>>(&mut collector, &LENGTHS);
    run_bench::<FmaDot<16>>(&mut collector, &LENGTHS);
    run_bench::<FmaDot<32>>(&mut collector, &LENGTHS);
    run_bench::<FmaDot<64>>(&mut collector, &LENGTHS);
    collector.display();
}

/// Benchmarks `D` over every length, labelled with the last segment of
/// the type's path.
fn run_bench<D: DotProduct>(collector: &mut Collector, lengths: &[usize]) {
    let name = short_type_name(std::any::type_name::<D>());
    let bar = progress_bar(&name, lengths.len());
    for &len in lengths {
        let label = format!("k = {len}");
        bar.set_message(label.clone());
        let sampler = Sampler::<D>::new(len);
        let samples = sampler.compute_samples(collector.budget());
        let flops = (2 * len * DOTS_PER_CALL) as f64;
        collector.record(&name, label, flops, &samples);
        bar.inc(1);
    }
    bar.finish_and_clear();
}

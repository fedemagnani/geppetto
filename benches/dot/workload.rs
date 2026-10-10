//! The measured unit: one dot x length cell on synthetic data.

use std::hint::black_box;
use std::marker::PhantomData;

use geppetto::bench::{SampleBudget, Samples, TimedCall};
use geppetto::tensor::DotProduct;
use geppetto::tensor::test::Rng;

/// Dots one timed call performs. A single dot at GPT-2 lengths is tens of
/// nanoseconds, too close to timer resolution to sample alone, so a call
/// measures a batch and the FLOP count scales by the same factor. The
/// operands stay cache-hot across the batch on purpose: this bench asks
/// for the dot's compute ceiling; how it fares against streaming weights
/// is the matmul bench's question.
pub const DOTS_PER_CALL: usize = 512;

/// One dot x length cell: two synthetic operand rows built once, dotted
/// [`DOTS_PER_CALL`] times per timed call. Each cell seeds its own RNG,
/// so a cell's data never depends on which cells ran before it.
pub struct Sampler<D> {
    a: Vec<f32>,
    b: Vec<f32>,
    _dot: PhantomData<D>,
}

impl<D: DotProduct> Sampler<D> {
    pub fn new(len: usize) -> Sampler<D> {
        let mut rng = Rng::new(0xD07);
        Sampler {
            a: rng.vec(len, 1.0),
            b: rng.vec(len, 1.0),
            _dot: PhantomData,
        }
    }

    /// One batch of dots, black-boxed on both sides: the slices re-enter
    /// through `black_box` every iteration so the compiler can neither
    /// hoist the dot out of the loop nor fold the batch into one call.
    fn call(&self) {
        let mut sum = 0.0f32;
        for _ in 0..DOTS_PER_CALL {
            let a = black_box(self.a.as_slice());
            let b = black_box(self.b.as_slice());
            sum += D::dot(a, b);
        }
        black_box(sum);
    }

    /// The budget's calibration call doubles as the warmup, then the
    /// sampling loop.
    pub fn compute_samples(self, budget: SampleBudget) -> Samples {
        let mut timed = TimedCall::new(|| self.call());
        let iters = timed.num_iterations(budget);
        let times = (0..iters).map(|_| timed.measure().as_secs_f64()).collect();
        Samples::new(times)
    }
}

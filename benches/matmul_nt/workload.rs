//! The measured unit: one kernel x shape cell on synthetic data.

use std::hint::black_box;

use bytes::Bytes;
use geppetto::bench::{SampleBudget, Samples, TimedCall};
use geppetto::tensor::test::Rng;
use geppetto::tensor::{DType, MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// One kernel x shape cell: synthetic operands built once, buffers reused
/// across every timed call. The RNG is a construction detail -- each cell
/// seeds its own, so a cell's data never depends on which cells ran before
/// it.
pub struct Sampler<'k, K: MatmulNtKernel> {
    kernel: &'k K,
    a: Shape,
    out: Shape,
    input: Vec<f32>,
    packed: K::Weights,
    out_buf: Vec<f32>,
    scratch: Vec<f32>,
}

impl<'k, K: MatmulNtKernel> Sampler<'k, K> {
    /// `a` and `b` are the logical `[m, k] @ [k, n]` operands; the weight
    /// tensor is built `[n, k]`, transposed at rest per the nt convention.
    pub fn new(kernel: &'k K, a: Shape, b: Shape) -> Sampler<'k, K> {
        assert_eq!(a.cols(), b.rows(), "contracted dim mismatch: {a} vs {b}",);
        let (m, k, n) = (a.rows(), a.cols(), b.cols());
        let mut rng = Rng::new(0xBE7C4);
        let input = rng.vec(m * k, 1.0);
        let weight = rng.vec(n * k, 1.0);
        let packed = kernel.pack(weight_tensor(&weight, n, k));
        let scratch = vec![0.0f32; kernel.scratch_len(m, k, n)];
        let out_buf = vec![0.0f32; m * n];
        Sampler {
            kernel,
            a,
            out: Shape::new(m, n),
            input,
            packed,
            out_buf,
            scratch,
        }
    }

    /// One matmul call, black-boxed: the unit both warmup and sampling
    /// measure. View mounting rides inside the measured region; it is two
    /// stack constructions, noise against a microseconds-scale kernel.
    fn call(&mut self) {
        let a = TensorView::contiguous(black_box(&self.input), self.a);
        let out = TensorViewMut::contiguous(&mut self.out_buf, self.out);
        self.kernel
            .matmul_nt(a, &self.packed, out, &mut self.scratch);
        black_box(&self.out_buf);
    }

    /// The budget's calibration call doubles as the warmup, then the
    /// sampling loop. Consumes the workload: its buffers are cold after
    /// this.
    pub fn compute_samples(mut self, budget: SampleBudget) -> Samples {
        let mut timed = TimedCall::new(|| self.call());
        let iters = timed.num_iterations(budget);
        let times = (0..iters).map(|_| timed.measure().as_secs_f64()).collect();
        Samples::new(times)
    }
}

/// An `[n, k]` weight from raw values, the way the model hands one to
/// [`MatmulNtKernel::pack`].
fn weight_tensor(values: &[f32], n: usize, k: usize) -> WeightTensor {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = Bytes::from(raw);
    WeightTensor::from_gguf_bytes(DType::F32, Shape::new(n, k), bytes).unwrap()
}

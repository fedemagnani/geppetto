//! Matrix multiplication: one free-function driver per file, the
//! [`MatmulKernel`] seam here, and its [`Naive`] baseline in `naive`.

mod naive;
// a `use mul::matmul` downstream resolves the name in both namespaces
// (module and function), so these modules must be as visible as the
// functions they re-export
pub mod matmul;
pub mod matmul_nn;
pub mod matmul_nn_causal;

#[cfg(test)]
mod tests;

pub use matmul::matmul;
pub use matmul_nn::matmul_nn;
pub use matmul_nn_causal::matmul_nn_causal;
pub use naive::NaiveMatMul;

use crate::tensor::{TensorView, TensorViewMut, WeightTensor};

/// A matmul implementation in the weight convention of [`matmul`]: `a` is
/// `[m, k]` activations, `b` is a `[n, k]` weight (`n` output features, each
/// a row of `k` input weights), and `out[i, j] = dot(a row i, b row j)`
/// fully overwrites `out` as `[m, n]`.
///
/// One stable interface the model calls, everything a kernel wants to
/// research hidden behind it -- the shape CPU BLAS-likes converge on (ggml's
/// repacked "extra buffer types", oneDNN's reorder + scratchpad queries,
/// BLIS's swappable microkernel under a fixed loop nest): a pack hook run
/// once at load, a scratch-size query answered before the arena is carved,
/// and a compute call that allocates nothing. Microkernel choice, blocking,
/// threading and GEMV/GEMM dispatch are all internals of an implementation
/// -- the trait never changes for them.
///
/// A kernel is constructed with everything it needs to know (tile sizes,
/// thread count, ...); the trait deliberately carries no such knobs. A
/// threaded kernel owns its pool internally and answers [`scratch_len`] for
/// all of its threads at once, which is why the sizing methods take `&self`.
///
/// [`scratch_len`]: MatmulKernel::scratch_len
pub trait MatmulKernel {
    /// The kernel's own at-rest format for a weight matrix: packed panels
    /// for a blocked GEMM, interleaved columns for a SIMD GEMV, quantized
    /// blocks later. Produced once per weight by [`pack`] at model load, so
    /// no per-call reformatting is ever paid. Implementations that own
    /// storage should back it with an [`Arena`](crate::arena::Arena) to
    /// inherit the one-allocation, 64-byte-aligned regime.
    ///
    /// [`pack`]: MatmulKernel::pack
    type Weights;

    /// Converts a raw `[n, k]` weight into this kernel's format. Runs once
    /// per weight tensor at model load, never on the hot path. The identity
    /// pack (keep the [`WeightTensor`]) is a refcount move, zero bytes.
    fn pack(&self, b: WeightTensor) -> Self::Weights;

    /// Floats of scratch one `[m, k] @ [n, k]^T` call needs. Answered before
    /// generation starts: the model takes the worst case over its call
    /// shapes and carves one arena region that big, which [`matmul`] then
    /// receives. Kernels needing no scratch return 0.
    ///
    /// [`matmul`]: MatmulKernel::matmul
    fn scratch_len(&self, m: usize, k: usize, n: usize) -> usize;

    /// Computes `out = a @ b^T` with `b` already in the kernel's format.
    /// `scratch` is at least [`scratch_len`] floats, unspecified on entry
    /// (NaN-poisoned in debug builds) and garbage after return: write before
    /// reading, never allocate.
    ///
    /// [`scratch_len`]: MatmulKernel::scratch_len
    fn matmul(&self, a: TensorView, b: &Self::Weights, out: TensorViewMut, scratch: &mut [f32]);
}

//! Kernels of the dot-product family: one output element, one dot of two
//! contiguous rows.
//!
//! [`DotMatMulNt`] owns what the whole family shares -- the loop nest, the
//! shape asserts and the [`MatmulNtKernel`] boilerplate (raw weights, zero
//! scratch). An implementation contributes only a [`DotProduct`]: `naive`
//! the serial baseline, `autovec` the chunked-accumulator family. Kernels
//! that break the one-dot-per-output shape (register tiling, threading)
//! implement [`MatmulNtKernel`] directly; this harness is a family
//! convenience, not a second seam.

mod autovec;
mod naive;

pub use autovec::{AutoVecDotProduct, AutoVecMatMulNt, Fma, FmaMatMulNt, MulAdd, Unfused};
pub use naive::{NaiveDotProduct, NaiveMatMulNt, matmul_nt};

use std::marker::PhantomData;

use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// An arbitrary dot product over two equal-length contiguous rows: the
/// entire degree of freedom a dot-family kernel has.
pub trait DotProduct {
    fn dot(a: &[f32], b: &[f32]) -> f32;
}

/// The family harness: `out[i, j] = D::dot(a row i, b row j)` in the
/// weight convention of [`matmul_nt`], with raw weights (identity pack)
/// and zero scratch. Swapping the dot is swapping the kernel.
#[derive(Debug, Default, Clone, Copy)]
pub struct DotMatMulNt<D> {
    _dot: PhantomData<D>,
}

impl<D: DotProduct> MatmulNtKernel for DotMatMulNt<D> {
    type Weights = WeightTensor;

    fn pack(&self, b: WeightTensor) -> WeightTensor {
        b
    }

    fn scratch_len(&self, _m: usize, _k: usize, _n: usize) -> usize {
        0
    }

    fn matmul_nt(&self, a: TensorView, b: &WeightTensor, out: TensorViewMut, _scratch: &mut [f32]) {
        matmul_nt_dot::<D>(a, b.view(), out);
    }
}

/// The shared driver: same convention, same asserts, same loop nest for
/// every dot. One hotpath label serves the family -- a run only ever
/// drives one dot at a time.
#[hotpath::measure]
fn matmul_nt_dot<D: DotProduct>(a: TensorView, b: TensorView, mut out: TensorViewMut) {
    let (m, k, n) = (a.rows(), a.cols(), b.rows());
    assert_eq!(
        k,
        b.cols(),
        "matmul_nt: contracted dim mismatch, {} vs {}",
        a.shape(),
        b.shape(),
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nt: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    for i in 0..m {
        let a_row = a.row(i);
        let out_row = out.row_mut(i);
        for (j, slot) in out_row.iter_mut().enumerate() {
            *slot = D::dot(a_row, b.row(j));
        }
    }
}

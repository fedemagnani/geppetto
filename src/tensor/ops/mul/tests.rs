use crate::tensor::test::{Rng, assert_close, matmul_vec};
use crate::tensor::{Shape, TensorView, TensorViewMut, matmul_nn, matmul_nn_causal};

#[test]
fn matmul_nn_hand_computed() {
    // A [2, 2] @ B [2, 3], textbook layout
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [1.0, 0.0, 2.0, 0.0, 1.0, 3.0];
    let mut out = vec![0.0f32; 2 * 3];
    matmul_nn(
        TensorView::contiguous(&a, Shape::new(2, 2)),
        TensorView::contiguous(&b, Shape::new(2, 3)),
        TensorViewMut::contiguous(&mut out, Shape::new(2, 3)),
    );
    assert_eq!(out, &[1.0, 2.0, 8.0, 3.0, 4.0, 18.0]);
}

#[test]
fn matmul_nn_agrees_with_the_transpose_trick() {
    // out = probs @ v equals the `a @ b^T` convention applied to v^T
    let (m, k, n) = (3, 4, 2);
    let mut rng = Rng::new(0xD07);
    let probs = rng.vec(m * k, 1.0);
    let v = rng.vec(k * n, 2.0);

    let mut v_t = vec![0.0f32; n * k];
    for r in 0..k {
        for c in 0..n {
            v_t[c * k + r] = v[r * n + c];
        }
    }
    let expected = matmul_vec(&probs, &v_t, m, k, n);

    let mut out = vec![0.0f32; m * n];
    matmul_nn(
        TensorView::contiguous(&probs, Shape::new(m, k)),
        TensorView::contiguous(&v, Shape::new(k, n)),
        TensorViewMut::contiguous(&mut out, Shape::new(m, n)),
    );
    assert_close(&out, &expected, 1e-5);
}

#[test]
fn causal_bound_never_reads_past_the_prefix() {
    // row i may only touch a[i, ..n_past + i + 1]; columns beyond hold
    // NaN, so any out-of-bound read would poison the output
    let nan = f32::NAN;
    let a = [2.0, nan, nan, 1.0, 3.0, nan, 0.5, -1.0, 2.0];
    let b = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
    let mut out = vec![0.0f32; 3 * 2];
    matmul_nn_causal(
        TensorView::contiguous(&a, Shape::new(3, 3)),
        TensorView::contiguous(&b, Shape::new(3, 2)),
        TensorViewMut::contiguous(&mut out, Shape::new(3, 2)),
        0,
    );
    let expected = [
        2.0, 4.0, // 2 * b0
        31.0, 62.0, // 1 * b0 + 3 * b1
        190.5, 381.0, // 0.5 * b0 - 1 * b1 + 2 * b2
    ];
    assert_eq!(out, &expected);
}

#[test]
fn causal_bound_overwrites_stale_output() {
    // the driver's contract: out is fully overwritten, never accumulated
    // into across calls
    let a = [3.0];
    let b = [1.0, 2.0];
    let mut out = vec![f32::NAN, 7.0];
    matmul_nn_causal(
        TensorView::contiguous(&a, Shape::new(1, 1)),
        TensorView::contiguous(&b, Shape::new(1, 2)),
        TensorViewMut::contiguous(&mut out, Shape::new(1, 2)),
        0,
    );
    assert_eq!(out, &[3.0, 6.0]);
}

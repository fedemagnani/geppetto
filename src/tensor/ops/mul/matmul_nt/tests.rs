use bytes::Bytes;

use crate::tensor::test::{Rng, assert_close, matmul_vec, naive_matmul};
use crate::tensor::{
    DType, MatMulNtAutoVecFma, MatMulNtAutoVecUnfused, MatMulNtNeonTiled, MatMulNtPackedNeon,
    MatMulNtTiledFma, MatMulNtTiledUnfused, MatmulNtKernel, Shape, TensorView, TensorViewMut,
    WeightTensor, matmul_nt,
};

#[test]
fn hand_computed_pins_the_convention() {
    // input: 2 tokens x 3 features
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // weight: 2 outputs, each 3 input weights
    let weight = [1.0, 0.0, -1.0, 1.0, 1.0, 1.0];
    // out[t, o] = dot(input[t], weight[o])
    assert_eq!(
        matmul_vec(&input, &weight, 2, 3, 2),
        &[-2.0, 6.0, -2.0, 15.0]
    );
}

#[test]
fn agrees_with_naive_reference_on_random_shapes() {
    let mut rng = Rng::new(0xA11CE);
    for &(m, k, n) in &[(1, 1, 1), (3, 4, 2), (5, 5, 5), (2, 7, 3), (8, 1, 4)] {
        let input = rng.vec(m * k, 3.0);
        let weight = rng.vec(n * k, 3.0);
        let got = matmul_vec(&input, &weight, m, k, n);
        assert_close(&got, &naive_matmul(&input, &weight, m, k, n), 1e-4);
    }
}

#[test]
fn strided_operand_matches_a_copied_column_block() {
    // a [2, 6] fused activation; its width-2 block at column 2 used as
    // the matmul_nt input, once via a strided view, once via a copy
    let mut rng = Rng::new(0xFACE);
    let fused = rng.vec(2 * 6, 2.0);
    let weight = rng.vec(3 * 2, 2.0);

    let copied: Vec<f32> = fused.chunks(6).flat_map(|row| row[2..4].to_vec()).collect();
    let expected = matmul_vec(&copied, &weight, 2, 2, 3);

    let a = TensorView::strided(&fused[2..10], Shape::new(2, 2), 6);
    let mut out = vec![0.0f32; 2 * 3];
    matmul_nt(
        a,
        TensorView::contiguous(&weight, Shape::new(3, 2)),
        TensorViewMut::contiguous(&mut out, Shape::new(2, 3)),
    );
    assert_close(&out, &expected, 1e-6);
}

/// An `[n, k]` weight from raw values, the way the model hands one to
/// [`MatmulNtKernel::pack`].
fn weight_tensor(values: &[f32], n: usize, k: usize) -> WeightTensor {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = Bytes::from(raw);
    WeightTensor::from_gguf_bytes(DType::F32, Shape::new(n, k), bytes).unwrap()
}

/// Agreement check every research kernel goes through, over shapes
/// straddling the LANES=16 boundary: tail-only k (1, 7, 15), exact
/// multiples (16, 1280), mixed (17, 33); (1, 1280, 8) is a decode GEMV
/// shape. Scratch is sized as the kernel asks and NaN-poisoned, per the
/// unspecified-on-entry contract.
fn assert_kernel_agrees_with_naive(kernel: &impl MatmulNtKernel) {
    let mut rng = Rng::new(0x5EED);
    let shapes = [
        (1, 1, 1),
        (2, 7, 3),
        (3, 15, 4),
        (3, 16, 2),
        (1, 17, 5),
        (4, 33, 3),
        (1, 1280, 8),
    ];
    for &(m, k, n) in &shapes {
        let input = rng.vec(m * k, 3.0);
        let weight = rng.vec(n * k, 3.0);
        let packed = kernel.pack(weight_tensor(&weight, n, k));
        let mut scratch = vec![f32::NAN; kernel.scratch_len(m, k, n)];

        let mut out = vec![f32::NAN; m * n];
        kernel.matmul_nt(
            TensorView::contiguous(&input, Shape::new(m, k)),
            &packed,
            TensorViewMut::contiguous(&mut out, Shape::new(m, n)),
            &mut scratch,
        );
        // reassociated (and, for fma, fused) accumulation drifts from the
        // serial reference by rounding; at k=1280 dots reach ~1e2, putting
        // that drift above the 1e-4 the other tests use
        assert_close(&out, &naive_matmul(&input, &weight, m, k, n), 1e-3);
    }
}

#[test]
fn unrolled_kernel_agrees_with_naive_on_random_shapes() {
    // a type alias is not a value constructor, and the bare underlying
    // name would not apply the LANES default; the annotated default() does
    // both
    let kernel: MatMulNtAutoVecUnfused = MatMulNtAutoVecUnfused::default();
    assert_kernel_agrees_with_naive(&kernel);
}

#[test]
fn fma_kernel_agrees_with_naive_on_random_shapes() {
    let kernel: MatMulNtAutoVecFma = MatMulNtAutoVecFma::default();
    assert_kernel_agrees_with_naive(&kernel);
}

#[test]
fn tiled_kernel_agrees_with_naive_on_random_shapes() {
    // the agreement shapes' n values (1..8) exercise every tile-tail
    // width for ROWS = 4 and the all-tail case for ROWS = 8
    assert_kernel_agrees_with_naive(&MatMulNtTiledUnfused::<4, 4>::default());
    assert_kernel_agrees_with_naive(&MatMulNtTiledFma::<4, 4>::default());
    assert_kernel_agrees_with_naive(&MatMulNtTiledFma::<2, 16>::default());
    assert_kernel_agrees_with_naive(&MatMulNtTiledFma::<4, 8>::default());
    assert_kernel_agrees_with_naive(&MatMulNtTiledFma::<8, 8>::default());
}

#[test]
fn neon_kernel_agrees_with_naive_on_random_shapes() {
    assert_kernel_agrees_with_naive(&MatMulNtNeonTiled::<2, 4>);
    assert_kernel_agrees_with_naive(&MatMulNtNeonTiled::<4, 2>);
    assert_kernel_agrees_with_naive(&MatMulNtNeonTiled::<4, 4>);
    assert_kernel_agrees_with_naive(&MatMulNtNeonTiled::<8, 2>);
    assert_kernel_agrees_with_naive(&MatMulNtNeonTiled::<1, 1>);
}

#[test]
fn packed_kernel_agrees_with_naive_on_random_shapes() {
    // the shapes' n (1..8) and k (1..33) exercise partial last panels,
    // all-padding chunks and the padded-activation copy
    assert_kernel_agrees_with_naive(&MatMulNtPackedNeon::<2, 4>);
    assert_kernel_agrees_with_naive(&MatMulNtPackedNeon::<4, 2>);
    assert_kernel_agrees_with_naive(&MatMulNtPackedNeon::<4, 4>);
    assert_kernel_agrees_with_naive(&MatMulNtPackedNeon::<8, 2>);
    assert_kernel_agrees_with_naive(&MatMulNtPackedNeon::<1, 1>);
}

#[test]
fn unrolled_pack_is_the_identity() {
    let kernel: MatMulNtAutoVecUnfused = MatMulNtAutoVecUnfused::default();
    let values: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let raw = weight_tensor(&values, 2, 4);
    let raw_ptr = raw.data().as_ptr();
    let packed = kernel.pack(raw);
    assert_eq!(packed.data().as_ptr(), raw_ptr);
}

#[test]
#[should_panic(expected = "contracted dim mismatch")]
fn mismatched_inner_dim_panics() {
    let input = [1.0, 2.0, 3.0];
    let weight = [1.0, 2.0];
    let mut out = vec![0.0f32; 1];
    matmul_nt(
        TensorView::contiguous(&input, Shape::new(1, 3)),
        TensorView::contiguous(&weight, Shape::new(1, 2)),
        TensorViewMut::contiguous(&mut out, Shape::new(1, 1)),
    );
}

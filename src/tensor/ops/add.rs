use crate::tensor::{TensorView, TensorViewMut};

/// `out += rhs`, elementwise and in place. `rhs` is either the same shape as
/// `out` or a single row (`[1, out.cols()]`) broadcast across every row --
/// the bias-add the FFN and attention projections need.
#[hotpath::measure]
pub fn add(mut out: TensorViewMut, rhs: TensorView) {
    let broadcast = rhs.rows() == 1 && out.rows() != 1;
    assert_eq!(
        out.cols(),
        rhs.cols(),
        "add: column mismatch, {} vs {}",
        out.shape(),
        rhs.shape(),
    );
    assert!(
        broadcast || out.rows() == rhs.rows(),
        "add: row mismatch, {} vs {}",
        out.shape(),
        rhs.shape(),
    );

    for r in 0..out.rows() {
        let rhs_row = if broadcast { rhs.row(0) } else { rhs.row(r) };
        let row = out.row_mut(r);
        for (slot, &bias) in row.iter_mut().zip(rhs_row) {
            *slot += bias;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, TensorView, TensorViewMut, add};

    /// `a += b` over fresh views, returning the mutated buffer.
    fn add_vec(a: &[f32], a_shape: Shape, b: &[f32], b_shape: Shape) -> Vec<f32> {
        let mut out = a.to_vec();
        add(
            TensorViewMut::contiguous(&mut out, a_shape),
            TensorView::contiguous(b, b_shape),
        );
        out
    }

    #[test]
    fn elementwise_same_shape() {
        let shape = Shape::new(2, 2);
        let out = add_vec(
            &[1.0, 2.0, 3.0, 4.0],
            shape,
            &[10.0, 20.0, 30.0, 40.0],
            shape,
        );
        assert_eq!(out, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn broadcasts_a_single_row_bias() {
        let out = add_vec(
            &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            Shape::new(3, 2),
            &[0.5, -0.5],
            Shape::new(1, 2),
        );
        assert_eq!(out, &[1.5, 0.5, 2.5, 1.5, 3.5, 2.5]);
    }

    #[test]
    fn is_commutative_for_matching_shapes() {
        let shape = Shape::new(1, 3);
        let a = [1.0, -2.0, 3.0];
        let b = [4.0, 5.0, -6.0];
        assert_eq!(add_vec(&a, shape, &b, shape), add_vec(&b, shape, &a, shape));
    }

    #[test]
    #[should_panic(expected = "column mismatch")]
    fn column_mismatch_panics() {
        let _ = add_vec(
            &[1.0, 2.0, 3.0],
            Shape::new(1, 3),
            &[1.0, 2.0],
            Shape::new(1, 2),
        );
    }
}

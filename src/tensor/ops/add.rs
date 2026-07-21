use std::ops::{Add, AddAssign};

use crate::tensor::Tensor;

/// `self += rhs`, elementwise. `rhs` is either the same shape as `self` or a
/// single row (`[1, self.cols()]`) broadcast across every row -- the bias-add
/// the FFN and attention projections need. In place: `self`'s buffer is
/// mutated, its shape unchanged.
impl AddAssign<&Tensor> for Tensor {
    #[hotpath::measure]
    fn add_assign(&mut self, rhs: &Tensor) {
        let broadcast = rhs.rows() == 1 && self.rows() != 1;
        assert_eq!(
            self.cols(),
            rhs.cols(),
            "add: column mismatch, {} vs {}",
            self.shape(),
            rhs.shape(),
        );
        assert!(
            broadcast || self.rows() == rhs.rows(),
            "add: row mismatch, {} vs {}",
            self.shape(),
            rhs.shape(),
        );

        for (r, row) in self.rows_mut().enumerate() {
            let rhs_row = if broadcast { rhs.row(0) } else { rhs.row(r) };
            for (slot, &bias) in row.iter_mut().zip(rhs_row) {
                *slot += bias;
            }
        }
    }
}

/// `self + rhs`, delegating to [`AddAssign`]. The owned `self` variant reuses
/// its buffer; the `&Tensor` variant clones it first.
impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    fn add(mut self, rhs: &Tensor) -> Tensor {
        self += rhs;
        self
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        self.clone() + rhs
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn elementwise_same_shape() {
        let a = Tensor::new(Shape::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(Shape::new(2, 2), vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!((&a + &b).data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn broadcasts_a_single_row_bias() {
        let a = Tensor::new(Shape::new(3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let bias = Tensor::new(Shape::new(1, 2), vec![0.5, -0.5]);
        assert_eq!((&a + &bias).data(), &[1.5, 0.5, 2.5, 1.5, 3.5, 2.5]);
    }

    #[test]
    fn owned_lhs_reuses_its_buffer() {
        let a = Tensor::new(Shape::new(1, 3), vec![1.0, -2.0, 3.0]);
        let b = Tensor::new(Shape::new(1, 3), vec![4.0, 5.0, -6.0]);
        assert_eq!((a + &b).data(), &[5.0, 3.0, -3.0]);
    }

    #[test]
    fn add_assign_mutates_in_place() {
        let mut a = Tensor::new(Shape::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let bias = Tensor::new(Shape::new(1, 2), vec![10.0, 20.0]);
        a += &bias;
        assert_eq!(a.data(), &[11.0, 22.0, 13.0, 24.0]);
        assert_eq!(a.shape(), Shape::new(2, 2));
    }

    #[test]
    fn is_commutative_for_matching_shapes() {
        let a = Tensor::new(Shape::new(1, 3), vec![1.0, -2.0, 3.0]);
        let b = Tensor::new(Shape::new(1, 3), vec![4.0, 5.0, -6.0]);
        assert_eq!((&a + &b).data(), (&b + &a).data());
    }

    #[test]
    #[should_panic(expected = "column mismatch")]
    fn column_mismatch_panics() {
        let a = Tensor::new(Shape::new(1, 3), vec![1.0, 2.0, 3.0]);
        let b = Tensor::new(Shape::new(1, 2), vec![1.0, 2.0]);
        let _ = &a + &b;
    }
}

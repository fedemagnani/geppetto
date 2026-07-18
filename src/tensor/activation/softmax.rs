use crate::tensor::Tensor;

impl Tensor {
    /// Per-row softmax, numerically stabilized by subtracting each row's max
    /// before exponentiating. When `mask` is given (same shape as `self`) it
    /// is added first -- the additive attention mask, whose `-inf` entries
    /// zero out disallowed positions.
    pub fn softmax(&self, mask: Option<&Tensor>) -> Tensor {
        if let Some(mask) = mask {
            assert_eq!(mask.shape(), self.shape(), "softmax: mask shape must match");
        }

        let mut out = self.clone();
        for (r, row) in out.rows_mut().enumerate() {
            if let Some(mask) = mask {
                for (slot, &m) in row.iter_mut().zip(mask.row(r)) {
                    *slot += m;
                }
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // an all -inf row (fully masked) would divide by zero; guard it
            if max == f32::NEG_INFINITY {
                row.fill(0.0);
                continue;
            }
            let mut sum = 0.0f32;
            for slot in row.iter_mut() {
                *slot = (*slot - max).exp();
                sum += *slot;
            }
            for slot in row.iter_mut() {
                *slot /= sum;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn hand_computed_distribution() {
        let x = Tensor::new(Shape::new(1, 3), vec![1.0, 2.0, 3.0]);
        let out = x.softmax(None);
        let expected = [0.090_030_57, 0.244_728_47, 0.665_240_96];
        for (o, e) in out.data().iter().zip(expected) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    #[test]
    fn each_row_sums_to_one() {
        let x = Tensor::new(
            Shape::new(2, 4),
            vec![0.1, 0.2, 0.3, 0.4, -5.0, 5.0, 0.0, 2.0],
        );
        for row in x.softmax(None).data().chunks(4) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn is_shift_invariant() {
        let x = Tensor::new(Shape::new(1, 3), vec![1.0, 2.0, 3.0]);
        let shifted = Tensor::new(Shape::new(1, 3), vec![101.0, 102.0, 103.0]);
        for (a, b) in x
            .softmax(None)
            .data()
            .iter()
            .zip(shifted.softmax(None).data())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn additive_mask_removes_positions() {
        let x = Tensor::new(Shape::new(1, 3), vec![1.0, 2.0, 3.0]);
        let mask = Tensor::new(Shape::new(1, 3), vec![0.0, 0.0, f32::NEG_INFINITY]);
        let out = x.softmax(Some(&mask));
        assert_eq!(out.data()[2], 0.0);
        assert!((out.data().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // remaining two follow softmax([1, 2])
        assert!((out.data()[0] - 0.268_941_42).abs() < 1e-6);
    }
}

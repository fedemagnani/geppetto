#[cfg(test)]
use crate::tensor::Tensor;
use crate::tensor::TensorViewMut;

/// Per-row softmax over the full row width, in place. Numerically stabilized
/// by subtracting each row's max before exponentiating.
pub fn softmax(x: TensorViewMut) {
    let cols = x.cols();
    // n_past = cols saturates every row's bound to the full width
    softmax_causal(x, cols);
}

/// Per-row softmax over the causal prefix, in place: row `i` is normalized
/// over its first `n_past + i + 1` columns (clamped to the row width); the
/// columns beyond the bound are never read and never written. The bound is
/// the causal mask -- no additive `-inf` mask tensor is needed.
#[hotpath::measure]
pub fn softmax_causal(mut x: TensorViewMut, n_past: usize) {
    let cols = x.cols();
    for i in 0..x.rows() {
        let bound = (n_past + i + 1).min(cols);
        let full_row = x.row_mut(i);
        let row = &mut full_row[..bound];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // an all -inf row would divide by zero; guard it
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
}

#[cfg(test)]
impl Tensor {
    /// Per-row softmax into a fresh tensor. When `mask` is given (same shape
    /// as `self`) it is added first -- the additive attention mask, whose
    /// `-inf` entries zero out disallowed positions. The maskless case
    /// delegates to the in-place [`softmax`] over a view.
    pub fn softmax(&self, mask: Option<&Tensor>) -> Tensor {
        let mut out = self.clone();
        let Some(mask) = mask else {
            softmax(out.as_view_mut());
            return out;
        };

        assert_eq!(mask.shape(), self.shape(), "softmax: mask shape must match");
        for (r, row) in out.rows_mut().enumerate() {
            for (slot, &m) in row.iter_mut().zip(mask.row(r)) {
                *slot += m;
            }
        }
        softmax(out.as_view_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor, softmax_causal};

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

    #[test]
    fn causal_bound_matches_the_additive_mask_on_the_prefix() {
        // scores [n_q=2, n_kv=3] with n_past=1: row i attends to 0..=n_past+i
        let n_past = 1;
        let scores = vec![0.3, -1.2, 9.9, 0.7, 0.1, -0.4];
        let x = Tensor::new(Shape::new(2, 3), scores.clone());

        let mask = Tensor::new(
            Shape::new(2, 3),
            vec![0.0, 0.0, f32::NEG_INFINITY, 0.0, 0.0, 0.0],
        );
        let expected = x.softmax(Some(&mask));

        let mut got = Tensor::new(Shape::new(2, 3), scores);
        softmax_causal(got.as_view_mut(), n_past);
        for i in 0..2 {
            let bound = n_past + i + 1;
            for j in 0..bound {
                let e = expected.row(i)[j];
                let g = got.row(i)[j];
                assert!((e - g).abs() < 1e-6, "[{i}, {j}]: {e} vs {g}");
            }
        }
    }

    #[test]
    fn causal_bound_leaves_the_tail_untouched() {
        let sentinel = 123.0;
        let mut x = Tensor::new(
            Shape::new(2, 3),
            vec![1.0, sentinel, sentinel, 1.0, 1.0, sentinel],
        );
        softmax_causal(x.as_view_mut(), 0);
        assert_eq!(x.row(0)[1], sentinel);
        assert_eq!(x.row(0)[2], sentinel);
        assert_eq!(x.row(1)[2], sentinel);
        // and the prefixes are normalized
        assert_eq!(x.row(0)[0], 1.0);
        assert!((x.row(1)[0] - 0.5).abs() < 1e-6);
    }
}

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
mod tests {
    use crate::tensor::{Shape, TensorViewMut, softmax, softmax_causal};

    /// Full-width softmax of one row, through the driver.
    fn softmax_row(row: &[f32]) -> Vec<f32> {
        let mut out = row.to_vec();
        let shape = Shape::new(1, row.len());
        softmax(TensorViewMut::contiguous(&mut out, shape));
        out
    }

    #[test]
    fn hand_computed_distribution() {
        let out = softmax_row(&[1.0, 2.0, 3.0]);
        let expected = [0.090_030_57, 0.244_728_47, 0.665_240_96];
        for (o, e) in out.iter().zip(expected) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    #[test]
    fn each_row_sums_to_one() {
        let mut x = vec![0.1, 0.2, 0.3, 0.4, -5.0, 5.0, 0.0, 2.0];
        softmax(TensorViewMut::contiguous(&mut x, Shape::new(2, 4)));
        for row in x.chunks(4) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn is_shift_invariant() {
        let a = softmax_row(&[1.0, 2.0, 3.0]);
        let b = softmax_row(&[101.0, 102.0, 103.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn causal_bound_matches_full_softmax_on_the_prefix() {
        // scores [n_q=2, n_kv=3] with n_past=1: row i attends to 0..=n_past+i,
        // and within that prefix must equal a plain softmax over it
        let n_past = 1;
        let scores = [0.3, -1.2, 9.9, 0.7, 0.1, -0.4];

        let mut got = scores.to_vec();
        softmax_causal(
            TensorViewMut::contiguous(&mut got, Shape::new(2, 3)),
            n_past,
        );

        for i in 0..2 {
            let bound = n_past + i + 1;
            let expected = softmax_row(&scores[i * 3..i * 3 + bound]);
            for (j, e) in expected.iter().enumerate() {
                let g = got[i * 3 + j];
                assert!((e - g).abs() < 1e-6, "[{i}, {j}]: {e} vs {g}");
            }
        }
    }

    #[test]
    fn causal_bound_leaves_the_tail_untouched() {
        let sentinel = 123.0;
        let mut x = vec![1.0, sentinel, sentinel, 1.0, 1.0, sentinel];
        softmax_causal(TensorViewMut::contiguous(&mut x, Shape::new(2, 3)), 0);
        assert_eq!(x[1], sentinel);
        assert_eq!(x[2], sentinel);
        assert_eq!(x[5], sentinel);
        // and the prefixes are normalized
        assert_eq!(x[0], 1.0);
        assert!((x[3] - 0.5).abs() < 1e-6);
    }
}

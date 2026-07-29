use crate::tensor::{TensorView, TensorViewMut};

/// Row-normalizes `input` into `out`.
///
/// For each row `x`: `(x - mean) / sqrt(var + eps)`, then elementwise
/// `* weight + bias`. `var` is the biased (population) variance. Never in
/// place: callers need `input` (the residual stream) to survive.
#[hotpath::measure]
pub fn norm(
    input: TensorView,
    weight: TensorView,
    bias: TensorView,
    eps: f32,
    mut out: TensorViewMut,
) {
    let cols = input.cols();
    assert_eq!(weight.rows(), 1, "layernorm: weight must be one row");
    assert_eq!(bias.rows(), 1, "layernorm: bias must be one row");
    assert_eq!(weight.cols(), cols, "layernorm: weight width mismatch");
    assert_eq!(bias.cols(), cols, "layernorm: bias width mismatch");
    assert_eq!(out.shape(), input.shape(), "layernorm: out shape mismatch");

    let (w, b) = (weight.row(0), bias.row(0));
    for r in 0..input.rows() {
        let src = input.row(r);
        let dst = out.row_mut(r);
        let mean = src.iter().sum::<f32>() / cols as f32;
        let var = src.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (((slot, &x), &wi), &bi) in dst.iter_mut().zip(src).zip(w).zip(b) {
            *slot = (x - mean) * inv_std * wi + bi;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::layer::norm;
    use crate::tensor::{Shape, TensorView, TensorViewMut};

    /// Normalizes `input [rows, cols]` with `weight`/`bias` single rows.
    fn norm_vec(input: &[f32], rows: usize, cols: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        let row_shape = Shape::new(1, cols);
        norm(
            TensorView::contiguous(input, Shape::new(rows, cols)),
            TensorView::contiguous(w, row_shape),
            TensorView::contiguous(b, row_shape),
            0.0,
            TensorViewMut::contiguous(&mut out, Shape::new(rows, cols)),
        );
        out
    }

    #[test]
    fn hand_computed_normalization() {
        let out = norm_vec(&[1.0, 2.0, 3.0, 4.0], 1, 4, &[1.0; 4], &[0.0; 4]);
        // mean 2.5, population var 1.25, std ~1.118034
        let expected = [-1.341_641, -0.447_214, 0.447_214, 1.341_641];
        for (o, e) in out.iter().zip(expected) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn each_row_has_zero_mean_unit_variance() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, -2.0, 0.0, 7.0, -3.0, 8.0];
        let out = norm_vec(&x, 2, 5, &[1.0; 5], &[0.0; 5]);
        for row in out.chunks(5) {
            let mean = row.iter().sum::<f32>() / 5.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 5.0;
            assert!(mean.abs() < 1e-4, "mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "var {var}");
        }
    }

    #[test]
    fn weight_and_bias_are_applied() {
        let out = norm_vec(&[0.0, 4.0], 1, 2, &[2.0, 3.0], &[10.0, -10.0]);
        // normalized [-1, 1] -> *[2,3] + [10,-10] = [8, -7]
        assert!((out[0] - 8.0).abs() < 1e-5);
        assert!((out[1] - -7.0).abs() < 1e-5);
    }
}

use crate::tensor::Tensor;

impl Tensor {
    /// Per-row layer normalization, matching ggml's `ggml_norm` followed by
    /// the weight-scale and bias-shift the GPT-2 graph applies.
    ///
    /// For each row `x`: `(x - mean) / sqrt(var + eps)`, then elementwise
    /// `* weight + bias`. `var` is the biased (population) variance ggml uses.
    /// `weight` and `bias` are single rows of length `self.cols()`.
    pub fn layernorm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let cols = self.cols();
        assert_eq!(weight.rows(), 1, "layernorm: weight must be one row");
        assert_eq!(bias.rows(), 1, "layernorm: bias must be one row");
        assert_eq!(weight.cols(), cols, "layernorm: weight width mismatch");
        assert_eq!(bias.cols(), cols, "layernorm: bias width mismatch");

        let (w, b) = (weight.row(0), bias.row(0));
        let mut out = self.clone();
        for row in out.rows_mut() {
            let mean = row.iter().sum::<f32>() / cols as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for ((slot, &wi), &bi) in row.iter_mut().zip(w).zip(b) {
                *slot = (*slot - mean) * inv_std * wi + bi;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor};

    fn ones(cols: usize) -> Tensor {
        Tensor::new(Shape::new(1, cols), vec![1.0; cols])
    }

    fn zeros(cols: usize) -> Tensor {
        Tensor::new(Shape::new(1, cols), vec![0.0; cols])
    }

    #[test]
    fn hand_computed_normalization() {
        let x = Tensor::new(Shape::new(1, 4), vec![1.0, 2.0, 3.0, 4.0]);
        let out = x.layernorm(&ones(4), &zeros(4), 0.0);
        // mean 2.5, population var 1.25, std ~1.118034
        let expected = [-1.341_641, -0.447_214, 0.447_214, 1.341_641];
        for (o, e) in out.data().iter().zip(expected) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn each_row_has_zero_mean_unit_variance() {
        let x = Tensor::new(
            Shape::new(2, 5),
            vec![3.0, 1.0, 4.0, 1.0, 5.0, -2.0, 0.0, 7.0, -3.0, 8.0],
        );
        let out = x.layernorm(&ones(5), &zeros(5), 1e-9);
        for row in out.data().chunks(5) {
            let mean = row.iter().sum::<f32>() / 5.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 5.0;
            assert!(mean.abs() < 1e-4, "mean {mean}");
            assert!((var - 1.0).abs() < 1e-3, "var {var}");
        }
    }

    #[test]
    fn weight_and_bias_are_applied() {
        let x = Tensor::new(Shape::new(1, 2), vec![0.0, 4.0]);
        let weight = Tensor::new(Shape::new(1, 2), vec![2.0, 3.0]);
        let bias = Tensor::new(Shape::new(1, 2), vec![10.0, -10.0]);
        let out = x.layernorm(&weight, &bias, 0.0);
        // normalized [-1, 1] -> *[2,3] + [10,-10] = [8, -7]
        assert!((out.data()[0] - 8.0).abs() < 1e-5);
        assert!((out.data()[1] - -7.0).abs() < 1e-5);
    }
}

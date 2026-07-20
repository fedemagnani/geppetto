use crate::tensor::Tensor;

pub struct LayerNorm<'a> {
    weight: &'a Tensor,
    bias: &'a Tensor,
    eps: f32,
}

impl<'a> LayerNorm<'a> {
    pub fn new(weight: &'a Tensor, bias: &'a Tensor, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Normalizes each row of `input`
    ///
    /// For each row `x`: `(x - mean) / sqrt(var + eps)`, then elementwise
    /// `* weight + bias`. `var` is the biased (population) variance.
    pub fn eval(&self, input: &Tensor) -> Tensor {
        let cols = input.cols();
        assert_eq!(self.weight.rows(), 1, "layernorm: weight must be one row");
        assert_eq!(self.bias.rows(), 1, "layernorm: bias must be one row");
        assert_eq!(self.weight.cols(), cols, "layernorm: weight width mismatch");
        assert_eq!(self.bias.cols(), cols, "layernorm: bias width mismatch");

        let (w, b) = (self.weight.row(0), self.bias.row(0));
        let mut out = input.clone();
        for row in out.rows_mut() {
            let mean = row.iter().sum::<f32>() / cols as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / cols as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for ((slot, &wi), &bi) in row.iter_mut().zip(w).zip(b) {
                *slot = (*slot - mean) * inv_std * wi + bi;
            }
        }
        out
    }
}
#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor, layer::layernorm::LayerNorm};

    fn ones(cols: usize) -> Tensor {
        Tensor::new(Shape::new(1, cols), vec![1.0; cols])
    }

    fn zeros(cols: usize) -> Tensor {
        Tensor::new(Shape::new(1, cols), vec![0.0; cols])
    }

    #[test]
    fn hand_computed_normalization() {
        let x = Tensor::new(Shape::new(1, 4), vec![1.0, 2.0, 3.0, 4.0]);
        let w = ones(4);
        let b = zeros(4);
        let eps = 0.;
        let norm = LayerNorm::new(&w, &b, eps);
        let out = norm.eval(&x);
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

        let w = ones(5);
        let b = zeros(5);
        let eps = 1e-9;
        let norm = LayerNorm::new(&w, &b, eps);
        let out = norm.eval(&x);

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
        let w = Tensor::new(Shape::new(1, 2), vec![2.0, 3.0]);
        let b = Tensor::new(Shape::new(1, 2), vec![10.0, -10.0]);
        let eps = 0.;
        let norm = LayerNorm::new(&w, &b, eps);
        let out = norm.eval(&x);
        // normalized [-1, 1] -> *[2,3] + [10,-10] = [8, -7]
        assert!((out.data()[0] - 8.0).abs() < 1e-5);
        assert!((out.data()[1] - -7.0).abs() < 1e-5);
    }
}

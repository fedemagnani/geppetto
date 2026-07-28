#[cfg(test)]
use crate::tensor::Tensor;
use crate::tensor::TensorViewMut;

/// sqrt(2/pi), matching ggml's constant to f32 precision.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// The cubic coefficient in the tanh GELU approximation, as in ggml.
const GELU_COEF_A: f32 = 0.044_715;

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Elementwise GELU in place, using the tanh approximation ggml applies to
/// GPT-2's FFN.
#[hotpath::measure]
pub fn gelu(mut x: TensorViewMut) {
    for r in 0..x.rows() {
        for slot in x.row_mut(r) {
            *slot = gelu_scalar(*slot);
        }
    }
}

#[cfg(test)]
impl Tensor {
    /// Elementwise GELU into a fresh tensor, delegating to the in-place
    /// [`gelu`] over a view.
    pub fn gelu(&self) -> Tensor {
        let mut out = self.clone();
        gelu(out.as_view_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::gelu_scalar;
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn hand_computed_reference_points() {
        assert_eq!(gelu_scalar(0.0), 0.0);
        // 0.5 * (1 + tanh(0.79788456 * 1.044715)) ~= 0.8412
        assert!((gelu_scalar(1.0) - 0.8412).abs() < 1e-3);
        assert!((gelu_scalar(-1.0) - -0.1588).abs() < 1e-3);
        // saturates towards identity for large positive, towards 0 for large negative
        assert!((gelu_scalar(6.0) - 6.0).abs() < 1e-3);
        assert!(gelu_scalar(-6.0).abs() < 1e-3);
    }

    #[test]
    fn is_monotone_increasing_for_nonnegative_inputs() {
        // GELU is not monotone overall -- it dips to a minimum near x=-0.75 --
        // but it is strictly increasing on [0, inf).
        let mut prev = f32::NEG_INFINITY;
        let mut x = 0.0;
        while x <= 5.0 {
            let y = gelu_scalar(x);
            assert!(y > prev, "not increasing at x={x}: {y} <= {prev}");
            prev = y;
            x += 0.05;
        }
    }

    #[test]
    fn positive_inputs_stay_between_zero_and_identity() {
        // 0 < gelu(x) <= x for x > 0; the upper bound is reached only in the
        // f32-saturated tail, where the correction term underflows.
        let mut x = 0.01;
        while x <= 6.0 {
            let y = gelu_scalar(x);
            assert!(y > 0.0 && y <= x, "gelu({x})={y} not in (0, {x}]");
            x += 0.05;
        }
    }

    #[test]
    fn maps_every_element() {
        let t = Tensor::new(Shape::new(2, 2), vec![0.0, 1.0, -1.0, 2.0]);
        let out = t.gelu();
        assert_eq!(out.shape(), t.shape());
        for (o, i) in out.data().iter().zip(t.data()) {
            assert_eq!(*o, gelu_scalar(*i));
        }
    }
}

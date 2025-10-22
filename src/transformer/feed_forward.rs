use super::*;

/// An implementation of the GELU activation function
///
/// A unit struct in order to implement `candle_core::Module` trait
#[derive(Clone, Debug)]
pub struct GELU;

impl ModuleT for GELU {
    fn forward_t(&self, xs: &Tensor, _: bool) -> candle_core::Result<Tensor> {
        (0.5_f64 * xs)?.mul(
            &((2_f64 / f64::consts::PI).sqrt() * (xs + (xs.mul(xs)?.mul(xs)? * 0.044715f64)?)?)?
                .tanh()?
                .broadcast_add(&Tensor::ones((1,), candle_core::DType::F32, xs.device())?)?,
        )
    }
}

pub struct FeedForward {
    linear_1: Linear,
    gelu: GELU,
    linear_2: Linear,
}
impl FeedForward {
    pub fn new(vb: &VarBuilder, emb_size: usize, bias: bool) -> candle_core::Result<Self> {
        let linear_1 = linear_b(emb_size, 4 * emb_size, bias, vb.pp("c_fc"))?;

        let linear_2 = linear_b(4 * emb_size, emb_size, bias, vb.pp("c_proj"))?;

        let out = Self {
            linear_1,
            linear_2,
            gelu: GELU,
        };
        Ok(out)
    }
}

impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let xs = self.linear_1.forward_t(xs, train)?;
        let xs = self.gelu.forward_t(&xs, train)?;
        let xs = self.linear_2.forward_t(&xs, train)?;
        Ok(xs)
    }
}

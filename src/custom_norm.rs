use super::*;

// Standardizes value computing its zeta scores
// scale and shift are like mu and signa when mapping a generic standard gaussian into a gaussian
// These values are learnt during training, to find the optimal mu and sigma fitting data
pub struct CustomLayerNorm {
    pub scale: Tensor,
    pub shift: Tensor,
}

impl CustomLayerNorm {
    pub fn new(vb: &VarBuilder, emb_dim: usize) -> candle_core::Result<Self> {
        //we start with parameters
        let scale = vb.get_with_hints(emb_dim, "weight", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "weight", candle_nn::Init::Const(0.))?;
        let out = Self { scale, shift };
        Ok(out)
    }
}

impl ModuleT for CustomLayerNorm {
    fn forward_t(&self, xs: &Tensor, _: bool) -> candle_core::Result<Tensor> {
        // mean and var iterating over cols
        let mean = xs.mean_keepdim(D::Minus1)?;
        let var = xs.var_keepdim(D::Minus1)?;
        let num = xs.broadcast_sub(&mean)?;
        let den = var.broadcast_add(&Tensor::new(&[f32::EPSILON], xs.device())?)?;
        let den = den.sqrt()?;
        //(x-mu)/sigma
        let std_gaus = num.broadcast_div(&den)?;
        self.shift
            .broadcast_add(&self.scale.broadcast_mul(&std_gaus)?)
    }
}

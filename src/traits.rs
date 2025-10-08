use super::*;

/// Trait for returning top-k elements of a Tensor
pub trait TopK {
    /// Returns a `Tensor`'s top-k elements and its positions along dim 0
    fn topk_last_dim0(&self, top_k: usize) -> candle_core::Result<(Tensor, Tensor)>;

    /// Returns a `Tensor`'s top-k elements and its positions along dim 1
    fn topk_last_dim1(&self, top_k: usize) -> candle_core::Result<(Tensor, Tensor)>;
}

impl TopK for Tensor {
    fn topk_last_dim0(&self, top_k: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let top_pos = self.arg_sort_last_dim(false)?;
        let top_pos = top_pos.i(..top_k)?;
        let top_els = self.i(top_pos.to_vec1::<u32>()?)?;
        Ok((top_els, top_pos))
    }

    fn topk_last_dim1(&self, top_k: usize) -> candle_core::Result<(Tensor, Tensor)> {
        // get CUDA error sometimes when using `.arg_sort_last_dim`
        // moving to CPU to carry out the op
        let top_pos = self.to_device(&Device::Cpu)?.arg_sort_last_dim(false)?;
        let top_pos = top_pos.to_device(self.device())?;
        let (batch_size, vocab_size) = top_pos.dims2()?;
        let top_pos = top_pos.i((.., ..top_k))?.flatten_all()?;

        // get appropriate sum starting index
        let aux = Tensor::arange(0u32, batch_size as u32, self.device())?;
        let aux = (vocab_size as f64 * aux.broadcast_left(top_k)?.t()?.flatten_all()?)?;
        let top_pos = (top_pos + &aux)?;
        let top_els = self.flatten_all()?.i(top_pos.to_vec1::<u32>()?)?;

        // reshape
        let top_els = top_els.reshape((batch_size, top_k))?;
        let top_pos = (top_pos - &aux)?;
        let top_pos = top_pos.reshape((batch_size, top_k))?;
        Ok((top_els, top_pos))
    }
}

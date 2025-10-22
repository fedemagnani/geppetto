use super::*;

pub struct TextGenerator {
    pub model: GPTModel,
    pub max_new_tokens: usize,
    pub context_length: usize,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub eos_id: Option<Tensor>,
}

impl TextGenerator {
    pub fn generate(&self, rng: &mut ThreadRng, idx: Tensor, train: bool) -> eyre::Result<Tensor> {
        let mut idx = idx.clone();
        for _ in 0..self.max_new_tokens {
            // Limit the context window
            let (b, seq_len) = idx.dims2()?;
            let start = seq_len.saturating_sub(self.context_length);
            let idx_cond = idx.i((.., start..seq_len))?;

            // forward pass
            let logits = self.model.forward_t(&idx_cond, train)?;

            let (_b, c, _vocab_size) = logits.dims3()?;
            let logits = logits.i((.., c - 1, ..))?;

            // Apply top-k filter if present
            let logits = self.prune_logits(logits)?;

            let idx_next = self.compute_probas(rng, &logits, b)?;

            if let Some(ref eos) = self.eos_id {
                // not sure if this is the right thing to do
                // eos_id can appear in any of the batch inputs
                let num_eos = idx_next.broadcast_eq(eos)?.sum_all()?.to_scalar::<u8>()?;
                if num_eos as usize == b {
                    break;
                }
            }

            idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
        }
        Ok(idx)
    }

    fn prune_logits(&self, logits: Tensor) -> eyre::Result<Tensor> {
        let Some(top_k) = self.top_k else {
            return Ok(logits);
        };

        let (top_logits, _top_pos) = logits.contiguous()?.topk_last_dim1(top_k)?;

        let mask = logits.broadcast_lt(&top_logits.min_keepdim(D::Minus1)?)?;
        let on_true = logits
            .ones_like()?
            .broadcast_mul(&Tensor::new(f32::NEG_INFINITY, logits.device())?)?;
        let out = mask.where_cond(&on_true, &logits)?;
        Ok(out)
    }

    fn compute_probas(
        &self,
        rng: &mut ThreadRng,
        logits: &Tensor,
        batch_size: usize,
    ) -> eyre::Result<Tensor> {
        let Some(temp) = self.temperature else {
            // Greedy sampling if no temperature is set
            let probas = softmax(logits, 1)?;
            let out = probas.argmax_keepdim(D::Minus1)?;
            return Ok(out);
        };
        // Temperature scaling
        let logits = (logits / temp)?;
        let probas = softmax(&logits, D::Minus1)?;

        // Multinomial sampling
        let mut idx_next = vec![];
        for bx in 0..batch_size {
            let this_probas = probas.i((bx, ..))?.to_vec1::<f32>()?;
            let next_token_id = Self::sample_multinomial(rng, &this_probas)?;
            idx_next.push(next_token_id);
        }

        let out = Tensor::from_vec(idx_next, (batch_size, 1_usize), logits.device())?;
        Ok(out)
    }

    fn sample_multinomial(rng: &mut ThreadRng, probas: &[f32]) -> eyre::Result<u32> {
        let dist = WeightedIndex::new(probas).map_err(candle_core::Error::wrap)?;
        let sample = dist.sample(rng) as u32;
        Ok(sample)
    }
}

use super::*;

pub struct Tokenizer(CoreBPE);

impl From<CoreBPE> for Tokenizer {
    fn from(value: CoreBPE) -> Self {
        Self(value)
    }
}

impl AsRef<CoreBPE> for Tokenizer {
    fn as_ref(&self) -> &CoreBPE {
        &self.0
    }
}

impl Tokenizer {
    pub fn text_to_token_ids(&self, text: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let encoded = self.0.encode_with_special_tokens(text);
        let num_tokens = encoded.len();
        // encoded tensor
        Tensor::from_vec(encoded, (1_usize, num_tokens), dev)
    }

    pub fn token_ids_to_text(&self, token_ids: &Tensor) -> eyre::Result<String> {
        let flat = token_ids.squeeze(0)?;
        let out = self
            .0
            .decode(flat.to_vec1::<u32>()?)
            .map_err(|e| eyre!(e))?;
        Ok(out)
    }
}

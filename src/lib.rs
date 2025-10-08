use std::{fs, path::Path};

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::ops::softmax;
use tiktoken_rs::CoreBPE;
struct SelfAttentionV1 {
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    scaling: f64,
}

/// The `forward` method of this struct is mapping vector embeddings into context vectors
impl SelfAttentionV1 {
    fn new(vb: &VarBuilder, emb_vec_size: usize, d_k: usize) -> eyre::Result<Self> {
        let w_k = vb.get_with_hints((emb_vec_size, d_k), "W_k", DEFAULT_KAIMING_NORMAL)?;
        let w_q = vb.get_with_hints((emb_vec_size, d_k), "W_q", DEFAULT_KAIMING_NORMAL)?;
        let w_v = vb.get_with_hints((emb_vec_size, d_k), "W_v", DEFAULT_KAIMING_NORMAL)?;
        let out = Self::from_weight_matrices(w_q, w_k, w_v);
        Ok(out)
    }

    fn context_vectors(&self, embeddings: &Tensor) -> eyre::Result<Tensor> {
        let queries = embeddings.matmul(&self.w_q)?;
        let keys = embeddings.matmul(&self.w_k)?;
        let att_scores = queries.matmul(&keys.t()?)?;
        let att_weights = softmax(&(att_scores * self.scaling)?, 1)?;
        let values = embeddings.matmul(&self.w_v)?;
        let out = att_weights.matmul(&values)?;
        Ok(out)
    }

    fn from_weight_matrices(w_q: Tensor, w_k: Tensor, w_v: Tensor) -> Self {
        let d_k = w_q.dims()[1];
        let scaling = 1. / (d_k as f64).sqrt();
        Self {
            w_q,
            w_k,
            w_v,
            scaling,
        }
    }
}

struct Dataset {
    inputs: Vec<Vec<u32>>,
    targets: Vec<Vec<u32>>,
}

impl Dataset {
    fn new(inputs: Vec<Vec<u32>>, targets: Vec<Vec<u32>>) -> Self {
        Self { inputs, targets }
    }

    fn from_path<P: AsRef<Path>>(path: P, tokenizer: CoreBPE) -> eyre::Result<Dataset> {
        let values = fs::read_to_string(path)?;

        //map the whole text in tokenids
        let token_ids = tokenizer.encode_with_special_tokens(&values);

        let max_length = 4;
        let stride = 2;

        // create the input-target matrices
        let mut inputs = vec![];
        let mut targets = vec![];

        for window in token_ids.windows(max_length + 1).step_by(stride) {
            let input = window[..max_length].to_vec();
            let target = window[1..].to_vec();

            inputs.push(input);
            targets.push(target);
        }

        Ok(Self::new(inputs, targets))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap, embedding, init::DEFAULT_KAIMING_NORMAL, ops::softmax};
    use eyre::eyre;
    use tiktoken_rs::{CoreBPE, get_bpe_from_model};

    fn gpt2_tokenizer() -> eyre::Result<CoreBPE> {
        let out = get_bpe_from_model("gpt2").map_err(|e| eyre!("{e}"))?;
        Ok(out)
    }

    #[test]
    fn print_tokens() -> eyre::Result<()> {
        let tokenizer = gpt2_tokenizer()?;

        // "Famous" subwords have single token via gpt2 BPE tokenizer
        let word = "dog";
        let token_ids = tokenizer.encode_with_special_tokens(word);
        assert_eq!(token_ids[0], 9703);
        assert_eq!(token_ids.len(), 1);
        let decoded = tokenizer.decode(token_ids).map_err(|e| eyre!("{e}"))?;
        assert_eq!(decoded, word);

        // Notice that the tokenizer used by gpt2 has a vocabulary of size  50,257: the last tokens are big frequent words (with space) and the last token is the system token "<|endoftext|>"
        let word: &'static str = " gazed";
        let token_ids = tokenizer.encode_with_special_tokens(word);
        assert_eq!(token_ids[0], 50255);
        assert_eq!(token_ids.len(), 1);
        let decoded = tokenizer.decode(token_ids).map_err(|e| eyre!("{e}"))?;
        assert_eq!(decoded, word);

        let word = "<|endoftext|>";
        let token_ids = tokenizer.encode_with_special_tokens(word);
        assert_eq!(token_ids[0], 50256);
        assert_eq!(token_ids.len(), 1);
        let decoded = tokenizer.decode(token_ids).map_err(|e| eyre!("{e}"))?;
        assert_eq!(decoded, word);

        Ok(())
    }

    #[test]
    fn test_input_target_tokenization() -> eyre::Result<()> {
        let tokenizer = gpt2_tokenizer()?;
        let dataset = Dataset::from_path("the-verdict.txt", tokenizer)?;
        let inputs = dataset.inputs;
        let targets = dataset.targets;

        assert_eq!(targets[0][0], inputs[0][1]);
        assert_eq!(targets[0][1], inputs[0][2]);
        assert_eq!(targets[0][2], inputs[0][3]);

        Ok(())
    }

    #[test]
    fn test_embeddings() -> eyre::Result<()> {
        let tokenizer = gpt2_tokenizer()?; //gpt2 BPE tokenizer has 50257 tokens
        let vocab_size = 50_257_usize;
        let embedding_size = 256_usize; //each tokenid will be associated with a 256-dimenisonal vector

        let dataset = Dataset::from_path("the-verdict.txt", tokenizer)?;
        let inputs = dataset.inputs;

        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let emb_layer = embedding(vocab_size, embedding_size, vs.push_prefix("emb_lay"))?;

        let first_input = Tensor::new(inputs[0].clone(), &device)?;
        let max_length = first_input.dims()[0];
        assert_eq!(first_input.dims().first(), Some(&4));
        assert_eq!(first_input.dims().get(1), None);

        // token embeddings: each of the 4 words is now mapped to a 256 vector
        let embeddings = emb_layer.embeddings().index_select(&first_input, 0)?;
        assert_eq!(embeddings.dims().first(), Some(&4));
        assert_eq!(embeddings.dims().get(1), Some(&256));

        //positional embedding: they add positional information to the token embedding
        let pos_layer = embedding(max_length, embedding_size, vs.push_prefix("pos_lay"))?;
        let indices = (0..max_length as u32).collect::<Vec<_>>();
        let pos_ids = Tensor::new(indices, &device)?;
        let pos_embeddings = pos_layer.embeddings().index_select(&pos_ids, 0)?;
        assert_eq!(pos_embeddings.dims().first(), Some(&4));
        assert_eq!(pos_embeddings.dims().get(1), Some(&256));

        // embedings that are position aware:
        let final_embeddings = embeddings.broadcast_add(&pos_embeddings)?;
        assert_eq!(final_embeddings.dims().first(), Some(&4));
        assert_eq!(final_embeddings.dims().get(1), Some(&256));
        Ok(())
    }

    fn mock_embeddings() -> eyre::Result<Tensor> {
        let dev = Device::cuda_if_available(0)?;
        let out = Tensor::new(
            &[
                [0.43_f32, 0.15, 0.89], // Your
                [0.55, 0.87, 0.66],     // journey
                [0.57, 0.85, 0.64],     // starts
                [0.22, 0.58, 0.33],     // with
                [0.77, 0.25, 0.10],     // one
                [0.05, 0.80, 0.55],     // step
            ],
            &dev,
        )?;
        Ok(out)
    }

    #[test]
    fn simple_self_attention() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;

        debug_assert_eq!(6, embeddings.dims()[0]);
        debug_assert_eq!(3, embeddings.dims()[1]);

        let simple_scores = embeddings.matmul(&embeddings.t()?)?;

        let scores_vec = simple_scores.to_vec2::<f32>()?;
        debug_assert_eq!(0.9995, scores_vec[0][0]);
        debug_assert_eq!(0.34739998, scores_vec[4][3]);

        let simple_weights = softmax(&simple_scores, 1)?;
        let weights_vec = simple_weights.to_vec2::<f32>()?;
        debug_assert_eq!(0.20983477, weights_vec[0][0]);
        debug_assert_eq!(0.13668667, weights_vec[4][3]);

        let simple_context = simple_weights.matmul(&embeddings)?;
        let ctx_vec = simple_context.to_vec2::<f32>()?;
        debug_assert_eq!(0.44205943, ctx_vec[0][0]);
        debug_assert_eq!(0.52659655, ctx_vec[4][2]);

        debug_assert_eq!(simple_context.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(simple_context.dims()[1], embeddings.dims()[1]);

        Ok(())
    }

    #[test]
    fn self_attention_w() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let dev = embeddings.device();
        let vs = VarBuilder::from_varmap(&VarMap::new(), DType::F32, dev);

        let d_in = embeddings.dims()[1];
        let d_out = 2;

        let w_query = vs.get_with_hints((d_in, d_out), "W_q", DEFAULT_KAIMING_NORMAL)?;
        let w_key = vs.get_with_hints((d_in, d_out), "W_k", DEFAULT_KAIMING_NORMAL)?;
        let w_value = vs.get_with_hints((d_in, d_out), "W_v", DEFAULT_KAIMING_NORMAL)?;

        let query = embeddings.matmul(&w_query)?;
        let key = embeddings.matmul(&w_key)?;
        let value = embeddings.matmul(&w_value)?;

        debug_assert_eq!(query.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(key.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(value.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(query.dims()[0], 6);
        debug_assert_eq!(query.dims()[1], w_query.dims()[1]);
        debug_assert_eq!(key.dims()[1], w_query.dims()[1]);
        debug_assert_eq!(value.dims()[1], w_query.dims()[1]);
        debug_assert_eq!(query.dims()[1], 2);

        let att_scores = query.matmul(&key.t()?)?;

        let att_scores_alt = embeddings
            .matmul(&w_query)?
            .matmul(&w_key.t()?)?
            .matmul(&embeddings.t()?)?;

        let scores_vec = att_scores.to_vec2::<f32>()?;
        let scores_alt_vec = att_scores_alt.to_vec2::<f32>()?;
        for (a, b) in scores_vec
            .iter()
            .flatten()
            .zip(scores_alt_vec.iter().flatten())
        {
            debug_assert!((*a - *b).abs() < 1e-6);
        }

        debug_assert_eq!(att_scores.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(att_scores.dims()[0], 6);
        debug_assert_eq!(att_scores.dims()[1], embeddings.dims()[0]);
        debug_assert_eq!(att_scores.dims()[1], 6);

        let dk = key.dims()[1] as f32;
        debug_assert_eq!(dk, 2.);

        let att_scores = att_scores.broadcast_div(&Tensor::new(&[dk.sqrt()], dev)?)?;
        let att_weights = softmax(&att_scores, 1)?;

        let context = att_weights.matmul(&value)?;

        debug_assert_eq!(context.dims()[0], embeddings.dims()[0]);
        debug_assert_eq!(context.dims()[0], 6);
        debug_assert_eq!(context.dims()[1], value.dims()[1]);
        debug_assert_eq!(context.dims()[1], 2);

        let layer = SelfAttentionV1::from_weight_matrices(w_query, w_key, w_value);
        let context_alt = layer.context_vectors(&embeddings)?;
        let ctx_vec = context.to_vec2::<f32>()?;
        let ctx_alt_vec = context_alt.to_vec2::<f32>()?;
        for (a, b) in ctx_vec.iter().flatten().zip(ctx_alt_vec.iter().flatten()) {
            debug_assert!((*a - *b).abs() < 1e-6);
        }

        Ok(())
    }
}

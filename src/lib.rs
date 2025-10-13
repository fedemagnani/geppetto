use std::{fs, path::Path};

use candle_core::{D, Device, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::ops::softmax;
use candle_nn::{Dropout, Linear, Module, VarBuilder, linear_b};
use tiktoken_rs::CoreBPE;

pub struct SelfAttention<T> {
    w_q: T,
    w_k: T,
    w_v: T,
    scaling: f64,
}

pub trait AttentionLayer: Module {}

type SelfAttentionV1 = SelfAttention<Tensor>;
impl AttentionLayer for SelfAttentionV1 {}
type SelfAttentionV2 = SelfAttention<Linear>;
impl AttentionLayer for SelfAttentionV2 {}
pub struct CausalAttention<T> {
    attention: SelfAttention<T>,
    dropout: Dropout,
}

type CausalAttentionV2 = CausalAttention<Linear>;
impl AttentionLayer for CausalAttentionV2 {}

pub struct MultiHeadAttention<L: AttentionLayer>(Vec<L>);
impl<L: AttentionLayer> From<Vec<L>> for MultiHeadAttention<L> {
    fn from(value: Vec<L>) -> Self {
        Self(value)
    }
}
impl<L: AttentionLayer> FromIterator<L> for MultiHeadAttention<L> {
    fn from_iter<T: IntoIterator<Item = L>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<L: AttentionLayer> Module for MultiHeadAttention<L> {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let ctx_matrices = self
            .0
            .iter()
            .map(|l| l.forward(xs))
            .collect::<Result<Vec<_>, _>>()?;

        // Last dimenion = column = D::Minus1
        let out = Tensor::cat(&ctx_matrices, D::Minus1)?;
        Ok(out)
    }
}

/// The `forward` method of this struct is mapping vector embeddings into context vectors
impl SelfAttentionV1 {
    pub fn new(vb: &VarBuilder, emb_vec_size: usize, d_k: usize) -> candle_core::Result<Self> {
        let w_k = vb.get_with_hints((emb_vec_size, d_k), "W_k", DEFAULT_KAIMING_NORMAL)?;
        let w_q = vb.get_with_hints((emb_vec_size, d_k), "W_q", DEFAULT_KAIMING_NORMAL)?;
        let w_v = vb.get_with_hints((emb_vec_size, d_k), "W_v", DEFAULT_KAIMING_NORMAL)?;
        let out: SelfAttention<Tensor> = Self::from_weight_matrices(w_q, w_k, w_v);
        Ok(out)
    }

    pub fn attention_scores(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        // E @ (Wq @ Wk.T) @ E.T
        let queries = embeddings.matmul(&self.w_q)?;
        let keys = embeddings.matmul(&self.w_k)?;
        // Recall that
        //  - D::Minus1: last dimenion (columns)
        //  - D::Minus2: second-to-last dimension (rows)
        // let att_scores = queries.broadcast_matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;
        let att_scores = queries.matmul(&keys.t()?)?;

        Ok(att_scores)
    }

    pub fn attention_weights(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let att_scores = self.attention_scores(embeddings)?;
        let att_weights = softmax(&(att_scores * self.scaling)?, 1)?;
        Ok(att_weights)
    }

    pub fn context_vectors(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let att_weights = self.attention_weights(embeddings)?;
        let values = embeddings.matmul(&self.w_v)?;
        let out = att_weights.matmul(&values)?;
        Ok(out)
    }

    pub fn from_weight_matrices(w_q: Tensor, w_k: Tensor, w_v: Tensor) -> Self {
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

impl Module for SelfAttentionV1 {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.context_vectors(xs)
    }
}

impl SelfAttentionV2 {
    pub fn new(
        vb: &VarBuilder,
        emb_vec_size: usize,
        d_k: usize,
        bias: bool,
    ) -> candle_core::Result<Self> {
        let w_k = linear_b(emb_vec_size, d_k, bias, vb.pp("W_k"))?;
        let w_q = linear_b(emb_vec_size, d_k, bias, vb.pp("W_q"))?;
        let w_v = linear_b(emb_vec_size, d_k, bias, vb.pp("W_v"))?;
        let out = Self::from_weight_matrices(w_q, w_k, w_v);
        Ok(out)
    }

    pub fn attention_scores(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let queries = self.w_q.forward(embeddings)?;
        let keys = self.w_k.forward(embeddings)?;
        let att_scores = queries.matmul(&keys.t()?)?;
        Ok(att_scores)
    }

    pub fn attention_weights(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let att_scores = self.attention_scores(embeddings)?;
        let att_weights = softmax(&(att_scores * self.scaling)?, 1)?;
        Ok(att_weights)
    }

    pub fn context_vectors(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let att_weights = self.attention_weights(embeddings)?;
        let values = self.w_v.forward(embeddings)?;
        let out = att_weights.matmul(&values)?;
        Ok(out)
    }

    pub fn from_weight_matrices(w_q: Linear, w_k: Linear, w_v: Linear) -> Self {
        let d_k = w_q.weight().dims()[1];
        let scaling = 1. / (d_k as f64).sqrt();
        Self {
            w_q,
            w_k,
            w_v,
            scaling,
        }
    }
}

impl Module for SelfAttentionV2 {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.context_vectors(xs)
    }
}

impl<T> CausalAttention<T> {
    pub fn get_mask(size: usize, device: &Device) -> candle_core::Result<Tensor> {
        let mask: Vec<_> = (0..size)
            .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
            .collect();
        let out = Tensor::from_slice(&mask, (size, size), device)?;
        Ok(out)
    }

    pub fn masked_fill(
        on_false: &Tensor,
        mask: &Tensor,
        on_true: f32,
    ) -> candle_core::Result<Tensor> {
        let shape = mask.shape();
        let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
        let m = mask.where_cond(&on_true, on_false)?;
        Ok(m)
    }
}

impl CausalAttentionV2 {
    pub fn new(
        vb: &VarBuilder,
        emb_vec_size: usize,
        d_k: usize,
        bias: bool,
        p_drop: f32,
    ) -> candle_core::Result<Self> {
        let attention = SelfAttentionV2::new(vb, emb_vec_size, d_k, bias)?;
        let dropout = Dropout::new(p_drop);
        let out = Self { attention, dropout };
        Ok(out)
    }

    pub fn from_weight_matrices(w_q: Linear, w_k: Linear, w_v: Linear, p_drop: f32) -> Self {
        let attention = SelfAttentionV2::from_weight_matrices(w_q, w_k, w_v);
        let dropout = Dropout::new(p_drop);
        Self { attention, dropout }
    }

    pub fn attention_weights(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let (batches, num_tokens, _) = embeddings.dims3()?;
        let att_scores = self.attention.attention_scores(embeddings)?;

        let mask = Self::get_mask(num_tokens, embeddings.device())?;
        let masked = Self::masked_fill(
            &att_scores,
            &mask.broadcast_left(batches).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut att_weights = softmax(&(masked * self.attention.scaling)?, D::Minus1)?;
        // dropout
        att_weights = self.dropout.forward(&att_weights, true).unwrap();

        Ok(att_weights)
    }

    pub fn context_vectors(&self, embeddings: &Tensor) -> candle_core::Result<Tensor> {
        let att_weights = self.attention_weights(embeddings)?;
        let values = self.attention.w_v.forward(embeddings)?;
        let out = att_weights.matmul(&values)?;
        Ok(out)
    }
}

impl Module for CausalAttentionV2 {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.context_vectors(xs)
    }
}
pub struct Dataset {
    inputs: Vec<Vec<u32>>,
    targets: Vec<Vec<u32>>,
}

impl Dataset {
    pub fn new(inputs: Vec<Vec<u32>>, targets: Vec<Vec<u32>>) -> Self {
        Self { inputs, targets }
    }

    pub fn from_path<P: AsRef<Path>>(path: P, tokenizer: CoreBPE) -> eyre::Result<Dataset> {
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

    pub fn inputs(&self) -> &[Vec<u32>] {
        &self.inputs
    }

    pub fn targets(&self) -> &[Vec<u32>] {
        &self.targets
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{
        Dropout, VarBuilder, VarMap, embedding,
        init::DEFAULT_KAIMING_NORMAL,
        ops::{dropout, softmax},
    };
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

    #[test]
    fn attention_v1_v2() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_lay_2 = SelfAttentionV2::new(&vb, embeddings.dims()[1], 2, false)?;
        let mut att_lay_1 = SelfAttentionV1::from_weight_matrices(
            att_lay_2.w_q.weight().t()?,
            att_lay_2.w_k.weight().t()?,
            att_lay_2.w_v.weight().t()?,
        );
        att_lay_1.scaling = att_lay_2.scaling;

        let ctx2 = att_lay_2.forward(&embeddings)?;
        let ctx1 = att_lay_1.forward(&embeddings)?;

        let ctx2_vec = ctx2.to_vec2::<f32>()?;
        let ctx1_vec = ctx1.to_vec2::<f32>()?;

        for (a, b) in ctx2_vec.iter().flatten().zip(ctx1_vec.iter().flatten()) {
            debug_assert!((*a - *b).abs() < 1e-9);
        }

        Ok(())
    }

    #[test]
    fn causal_attention_via_masking_and_renormalization_mean() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let emb_row = embeddings.dims()[0];
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_layer = SelfAttentionV1::new(&vb, embeddings.dims()[1], 2)?;

        let att_weights = att_layer.attention_weights(&embeddings)?;
        let att_weights = att_weights.to_vec2::<f32>()?;

        let n = att_weights.len();
        let m = att_weights[0].len();
        debug_assert_eq!(n, emb_row);
        debug_assert_eq!(m, emb_row);
        let mut masked = Vec::with_capacity(n * m);

        for (i, row) in att_weights.into_iter().enumerate() {
            let mut sum: f32 = row[..=i].iter().sum();
            if sum == 0.0 {
                sum = 1.0;
            }
            let inv_sum = 1.0 / sum;

            // normalized active part
            for &v in &row[..=i] {
                masked.push(v * inv_sum);
            }
            // masked zeros
            masked.resize(masked.len() + (m - i - 1), 0.0);
        }

        for row in masked.chunks(m) {
            let s: f32 = row.iter().sum();
            debug_assert!((s - 1.).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn causal_attention_via_masking_and_renormalization_softmax() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let emb_row = embeddings.dims()[0];
        let d_k = 2;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_layer = SelfAttentionV1::new(&vb, embeddings.dims()[1], d_k)?;

        let att_weights = att_layer.attention_weights(&embeddings)?;
        let att_weights = att_weights.to_vec2::<f32>()?;

        // Preallocate once
        let n = att_weights.len();
        let m = att_weights[0].len();
        debug_assert_eq!(n, emb_row);
        debug_assert_eq!(m, emb_row);
        let mut masked = Vec::with_capacity(n * m);

        for (i, row) in att_weights.into_iter().enumerate() {
            // Only take up to i+1 actual weights
            masked.extend_from_slice(&row[..=i]);
            // Then pad with zeros
            masked.resize(masked.len() + (m - i - 1), 0.0);
        }

        let att_weights = Tensor::from_vec(masked, (emb_row, emb_row), embeddings.device())?;

        //renormalization
        let att_weights = softmax(&(att_weights * (1. / d_k as f64).sqrt())?, 1)?;

        let att_weights = att_weights.to_vec2::<f32>()?;
        for row in att_weights.into_iter() {
            let s: f32 = row.into_iter().sum();
            debug_assert!((s - 1.).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn causal_attention_dropout_1() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let emb_row = embeddings.dims()[0];
        let d_k = 2;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_layer = SelfAttentionV1::new(&vb, embeddings.dims()[1], d_k)?;

        let att_weights = att_layer.attention_weights(&embeddings)?;

        // dropping out 50% of the elements
        let att_weights = dropout(&att_weights, 0.5)?;
        debug_assert_eq!(att_weights.dims()[0], emb_row);
        debug_assert_eq!(att_weights.dims()[1], emb_row);

        Ok(())
    }

    #[test]
    fn causal_attention_dropout_2() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        let emb_row = embeddings.dims()[0];
        let d_k = 2;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_layer = SelfAttentionV1::new(&vb, embeddings.dims()[1], d_k)?;

        let att_weights = att_layer.attention_weights(&embeddings)?;

        // dropping out 50% of the elements
        let dropout_layer = Dropout::new(0.5);
        let att_weights = dropout_layer.forward(&att_weights, true)?;
        debug_assert_eq!(att_weights.dims()[0], emb_row);
        debug_assert_eq!(att_weights.dims()[1], emb_row);

        Ok(())
    }

    #[test]
    fn causal_attention_2() -> eyre::Result<()> {
        let e = mock_embeddings()?;
        let embeddings = Tensor::stack(&[&e, &e], 0)?;
        let num_batches = embeddings.dims()[0];
        let emb_row = embeddings.dims()[1];
        let emb_col = embeddings.dims()[2];

        let d_k = 2;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let att_layer = CausalAttentionV2::new(&vb, emb_col, d_k, true, 0.5)?;

        let ctx_mat = att_layer.forward(&embeddings)?;

        debug_assert_eq!(ctx_mat.dims()[0], num_batches);
        debug_assert_eq!(ctx_mat.dims()[1], emb_row);
        debug_assert_eq!(ctx_mat.dims()[2], d_k);

        Ok(())
    }

    #[test]
    fn multi_head_causal_attention() -> eyre::Result<()> {
        let embeddings = mock_embeddings()?;
        // We append another dimension to specify number of batches
        let embeddings = embeddings.unsqueeze(0)?;
        let num_batches = embeddings.dims()[0];
        let emb_rows = embeddings.dims()[1];
        let emb_cols = embeddings.dims()[2];

        let d_out = 2;
        let h = 4;

        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, embeddings.device());
        let attentions = (0..h).map(|_| SelfAttentionV2::new(&vb, emb_cols, d_out, true));
        let multi_head = attentions.collect::<Result<MultiHeadAttention<_>, _>>()?;

        let cont_matrices = multi_head.forward(&embeddings)?;

        debug_assert_eq!(cont_matrices.dims()[0], num_batches);
        debug_assert_eq!(cont_matrices.dims()[1], emb_rows);
        debug_assert_eq!(cont_matrices.dims()[2], h * d_out);
        Ok(())
    }
}

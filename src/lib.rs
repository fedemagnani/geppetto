use core::f32;
use std::collections::HashSet;
use std::f64;
use std::{fs, path::Path};

use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::ops::softmax;
use candle_nn::{Dropout, Embedding, Linear, ModuleT, VarBuilder, embedding, linear_b};
use eyre::eyre;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use tiktoken_rs::CoreBPE;

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
        let top_pos = top_pos.to_device(&Device::cuda_if_available(0)?)?;
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

pub struct Tokenizer(CoreBPE);
impl From<CoreBPE> for Tokenizer {
    fn from(value: CoreBPE) -> Self {
        Self(value)
    }
}

impl Tokenizer {
    pub fn text_to_token_ids(&self, text: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let allowed_special = HashSet::from(["<|endoftext|>"]);
        let encoded = self.0.encode(text, allowed_special);
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

pub struct SelfAttention<T> {
    w_q: T,
    w_k: T,
    w_v: T,
    scaling: f64,
}

pub trait AttentionLayer: ModuleT {}

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

pub struct MultiHeadAttentionWrapper<L: AttentionLayer>(Vec<L>);
impl<L: AttentionLayer> From<Vec<L>> for MultiHeadAttentionWrapper<L> {
    fn from(value: Vec<L>) -> Self {
        Self(value)
    }
}
impl<L: AttentionLayer> FromIterator<L> for MultiHeadAttentionWrapper<L> {
    fn from_iter<T: IntoIterator<Item = L>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<L: AttentionLayer> ModuleT for MultiHeadAttentionWrapper<L> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let ctx_matrices = self
            .0
            .iter()
            .map(|l| l.forward_t(xs, train))
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

impl ModuleT for SelfAttentionV1 {
    fn forward_t(&self, xs: &Tensor, _: bool) -> candle_core::Result<Tensor> {
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

    pub fn attention_scores(
        &self,
        embeddings: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let queries = self.w_q.forward_t(embeddings, train)?;
        let keys = self.w_k.forward_t(embeddings, train)?;
        let att_scores = queries.matmul(&keys.t()?)?;
        Ok(att_scores)
    }

    pub fn attention_weights(
        &self,
        embeddings: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let att_scores = self.attention_scores(embeddings, train)?;
        let att_weights = softmax(&(att_scores * self.scaling)?, 1)?;
        Ok(att_weights)
    }

    pub fn context_vectors(&self, embeddings: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let att_weights = self.attention_weights(embeddings, train)?;
        let values = self.w_v.forward_t(embeddings, train)?;
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

impl ModuleT for SelfAttentionV2 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        self.context_vectors(xs, train)
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

    pub fn attention_weights(
        &self,
        embeddings: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let (batches, num_tokens, _) = embeddings.dims3()?;
        let att_scores = self.attention.attention_scores(embeddings, train)?;

        let mask = Self::get_mask(num_tokens, embeddings.device())?;
        let masked = Self::masked_fill(
            &att_scores,
            &mask.broadcast_left(batches).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut att_weights = softmax(&(masked * self.attention.scaling)?, D::Minus1)?;
        // dropout
        att_weights = self.dropout.forward_t(&att_weights, train).unwrap();

        Ok(att_weights)
    }

    pub fn context_vectors(&self, embeddings: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let att_weights = self.attention_weights(embeddings, train)?;
        let values = self.attention.w_v.forward_t(embeddings, train)?;
        let out = att_weights.matmul(&values)?;
        Ok(out)
    }
}

impl ModuleT for CausalAttentionV2 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        self.context_vectors(xs, train)
    }
}
pub struct Dataset {
    inputs: Tensor,
    targets: Tensor,
}

impl Dataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Self {
        Self { inputs, targets }
    }

    pub fn from_vec(
        inputs: Vec<Vec<u32>>,
        targets: Vec<Vec<u32>>,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let n = inputs.len();
        let emb_dim = inputs[0].len();
        let inputs = inputs.into_iter().flatten();
        let targets = targets.into_iter().flatten();
        let inputs = Tensor::from_iter(inputs, device)?.reshape((n, emb_dim))?;
        let targets = Tensor::from_iter(targets, device)?.reshape((n, emb_dim))?;

        Ok(Self { inputs, targets })
    }

    pub fn from_path<P: AsRef<Path>>(
        path: P,
        tokenizer: CoreBPE,
        device: &Device,
    ) -> eyre::Result<Dataset> {
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

        Ok(Self::from_vec(inputs, targets, device)?)
    }

    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    pub fn batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = candle_core::Result<(Tensor, Tensor)>> {
        let n = self.inputs.dims()[0];
        (0..n).step_by(batch_size).map(move |start| {
            let end = (start + batch_size).min(n);
            Ok((
                self.inputs.narrow(0, start, end - start)?,
                self.targets.narrow(0, start, end - start)?,
            ))
        })
    }
}

pub struct MultiHeadAttention {
    num_heads: usize, // number of heads in the multi head attention systems
    d_out: usize,     // columns of the (big) weight matrix
    head_dim: usize,  // columns of the weight matrix for each hed
    w_q: Linear,      // (big) query weight
    w_k: Linear,      // (big) key weight
    w_v: Linear,      // (big) value weight
    out_proj: Linear,
    scaling: f64,     //computed based on the number of columns of each head
    dropout: Dropout, // involved in causal attention
}

impl MultiHeadAttention {
    pub fn new(
        vb: &VarBuilder,
        num_heads: usize,
        emb_vec_size: usize,
        head_dim: usize,
        bias: bool,
        drop_p: f32,
    ) -> candle_core::Result<Self> {
        let d_k = num_heads * head_dim; // total columns of weight matrix
        let w_q = linear_b(emb_vec_size, d_k, bias, vb.pp("W_q"))?;
        let w_k = linear_b(emb_vec_size, d_k, bias, vb.pp("W_k"))?;
        let w_v = linear_b(emb_vec_size, d_k, bias, vb.pp("W_v"))?;

        let scaling = 1. / (head_dim as f64).sqrt();

        let dropout = Dropout::new(drop_p);

        let out_proj = linear_b(d_k, d_k, true, vb.pp("O_p"))?;

        let out = Self {
            num_heads,
            d_out: d_k,
            dropout,
            head_dim,
            out_proj,
            w_k,
            w_q,
            scaling,
            w_v,
        };

        Ok(out)
    }

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

impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (num_batches, emb_vec_size, _) = xs.dims3()?;
        let queries = self.w_q.forward_t(xs, train)?;
        let keys = self.w_k.forward_t(xs, train)?;
        let values = self.w_v.forward_t(xs, train)?;

        // Now we unpack the matrix to create the attention heads
        let queries =
            queries.reshape((num_batches, emb_vec_size, self.num_heads, self.head_dim))?;
        let keys = keys.reshape((num_batches, emb_vec_size, self.num_heads, self.head_dim))?;
        let values = values.reshape((num_batches, emb_vec_size, self.num_heads, self.head_dim))?;

        //we swap dimensions, to complete transformation into multi-heads, so that dimension is now (num_batches, num_heads, emb_vec_size, head_dim) (each had will have dimension (emb_vec_size, head_dim))
        let queries = queries.transpose(1, 2)?.contiguous()?;
        let keys = keys.transpose(1, 2)?.contiguous()?;
        let values = values.transpose(1, 2)?.contiguous()?;

        //we apply the self-attention mechanism
        let att_scores = queries.matmul(&keys.transpose(2, 3)?)?;

        let mask = Self::get_mask(emb_vec_size, xs.device())?;
        let masked = Self::masked_fill(
            &att_scores,
            &mask.broadcast_left((num_batches, self.num_heads))?,
            f32::NEG_INFINITY,
        )?;

        let att_weights = softmax(&(masked * self.scaling)?, 3)?;

        let att_weights = self.dropout.forward_t(&att_weights, train)?;

        let ctx_vec = att_weights.matmul(&values)?;
        let ctx_vec = ctx_vec.transpose(1, 2)?;

        // we re-arrange the ctx vector as a single big matrix
        let ctx_vec = ctx_vec
            .reshape((num_batches, emb_vec_size, self.d_out))?
            .contiguous()?;

        //finally, we project the final ctx vector
        self.out_proj.forward_t(&ctx_vec, train)
    }
}

pub struct GptConfig {
    pub vocab_size: usize, // input of the embedding layer (number of rows of the embedding matrix) = number of token ids (as each token id maps to a row of the matrix)
    pub emb_dim: usize, // columns of the embedding matrix: each token id is mapped to a vector having this dimensions
    pub context_length: usize, //the max number of tokens processed together
    pub drop_p: f32,    //dropout probability
    pub num_trf: usize, //number of transformers invovled in the model
    pub num_heads: usize, //number of heads of multi-head attention module in transformer block
    pub bias: bool,     // whether to add bias or not in the output module
}

pub struct GPTModel {
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub drop_emb: Dropout,
    pub trf_blocks: Vec<TransformerBlock>,
    pub final_norm: LayerNorm,
    pub out_head: Linear,
}

impl GPTModel {
    pub fn new(vb: &VarBuilder, c: &GptConfig) -> candle_core::Result<Self> {
        let tok_emb = embedding(c.vocab_size, c.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(c.context_length, c.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(c.drop_p);
        let mut trf_blocks = Vec::with_capacity(c.num_trf);
        for _ in 0..c.num_trf {
            trf_blocks.push(TransformerBlock::from_config(vb, c)?)
        }
        let final_norm = LayerNorm::new(vb, c.emb_dim)?;
        let out_head = linear_b(c.emb_dim, c.vocab_size, c.bias, vb.pp("out_head"))?;
        let out = Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        };
        Ok(out)
    }

    pub fn generate_text_simple(
        &self,
        mut idx: Tensor,
        max_new_tokens: usize,
        context_size: usize,
    ) -> candle_core::Result<Tensor> {
        for _ in 0..max_new_tokens {
            // Limit the context window
            let (_b, seq_len) = idx.dims2()?;
            let start = seq_len.saturating_sub(context_size);
            let idx_cond = idx.i((.., start..seq_len))?;

            // Forward pass
            let logits = self.forward_t(&idx_cond, false)?;

            // Get last logits
            let (_b, c, _vocab_size) = logits.dims3()?;
            let logits = logits.i((.., c - 1, ..))?;

            // Greedy sampling
            let probas = softmax(&logits, 1)?;
            let idx_next = probas.argmax_keepdim(D::Minus1)?;

            // Append new token
            idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
        }
        Ok(idx)
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (_, seq_len) = xs.dims2()?;
        let tok_emb = self.tok_emb.forward_t(xs, train)?;

        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_emb = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let x = tok_emb.broadcast_add(&pos_emb)?;
        let mut x = self.drop_emb.forward_t(&x, train)?;

        for t in &self.trf_blocks {
            x = t.forward_t(&x, train)?;
        }

        let x = self.final_norm.forward_t(&x, train)?;
        let logits = self.out_head.forward_t(&x, train)?;
        Ok(logits)
    }
}

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

// Standardizes value computing its zeta scores
// scale and shift are like mu and signa when mapping a generic standard gaussian into a gaussian
// These values are learnt during training, to find the optimal mu and sigma fitting data
pub struct LayerNorm {
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn new(vb: &VarBuilder, emb_dim: usize) -> candle_core::Result<Self> {
        //we start with parameters
        let scale = vb.get_with_hints(emb_dim, "layer_scale", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "layer_shift", candle_nn::Init::Const(0.))?;
        let out = Self { scale, shift };
        Ok(out)
    }
}

impl ModuleT for LayerNorm {
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

struct FeedForward {
    linear_1: Linear,
    gelu: GELU,
    linear_2: Linear,
}
impl FeedForward {
    pub fn new(vb: &VarBuilder, emb_size: usize, bias: bool) -> candle_core::Result<Self> {
        let linear_1 = linear_b(emb_size, 4 * emb_size, bias, vb.pp("ff_linear_1"))?;
        let linear_2 = linear_b(4 * emb_size, emb_size, bias, vb.pp("ff_linear_2"))?;

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

pub struct TransformerBlock {
    norm_1: LayerNorm,
    multi_head: MultiHeadAttention,
    norm_2: LayerNorm,
    feed_forward: FeedForward,

    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(
        vb: &VarBuilder,
        num_heads: usize,
        emb_dim: usize,
        bias: bool,
        drop_p: f32,
    ) -> candle_core::Result<Self> {
        let norm_1 = LayerNorm::new(vb, emb_dim)?;
        let multi_head =
            MultiHeadAttention::new(vb, num_heads, emb_dim, emb_dim / num_heads, bias, drop_p)?;
        let norm_2 = LayerNorm::new(vb, emb_dim)?;
        let feed_forward = FeedForward::new(vb, emb_dim, bias)?;
        let dropout = Dropout::new(drop_p);
        let out = Self {
            norm_1,
            multi_head,
            norm_2,
            feed_forward,
            dropout,
        };
        Ok(out)
    }

    pub fn from_config(vb: &VarBuilder, c: &GptConfig) -> candle_core::Result<Self> {
        Self::new(vb, c.num_heads, c.emb_dim, c.bias, c.drop_p)
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let shortcut_1 = xs;
        let x = self.norm_1.forward_t(xs, train)?;
        let x = self.multi_head.forward_t(&x, train)?;
        let x = self.dropout.forward_t(&x, train)?;

        let x = (x + shortcut_1)?; // shortcut connection

        let shortcut_2 = &x;
        let x = self.norm_2.forward_t(&x, train)?;
        let x = self.feed_forward.forward_t(&x, train)?;
        let x = self.dropout.forward_t(&x, train)?;

        x + shortcut_2 // shortcut connection
    }
}

pub struct CrossEntropy {
    pub device: Device,
    pub train: bool,
    pub ignore_index: Option<i64>,
    pub num_batches: Option<usize>,
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            train: false,
            ignore_index: None,
            num_batches: None,
        }
    }
}

impl CrossEntropy {
    pub fn compute(&self, model: &GPTModel, data: Dataset, batch_size: usize) -> eyre::Result<f32> {
        let mut total_loss = 0.;
        let mut count = 0;
        for batch in data.batches(batch_size) {
            let (input_batch, target_batch) = batch?;

            let loss = self.compute_single(model, &input_batch, &target_batch)?;
            total_loss += loss;
            count += 1;
            if let Some(n) = self.num_batches
                && count >= n
            {
                break;
            }
        }
        let out = total_loss / count as f32;
        Ok(out)
    }

    pub fn compute_single(
        &self,
        model: &GPTModel,
        input_batch: &Tensor,
        target_batch: &Tensor,
    ) -> eyre::Result<f32> {
        let input_batch = input_batch.to_device(&self.device)?;
        let target_batch = target_batch.to_device(&self.device)?;

        // Forward pass
        let logits = model.forward_t(&input_batch, self.train)?;

        // flatten
        let logits_flat = logits.flatten(0, 1)?;
        let targets_flat = target_batch.flatten_all()?;

        // Optionally filter out ignored indices (e.g., -100)
        let (logits_flat, targets_flat) = self.filter_indices(logits_flat, targets_flat)?;

        // Forward pass
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
        let loss = loss.to_scalar::<f32>()?;
        Ok(loss)
    }

    fn filter_indices(
        &self,
        logits_flat: Tensor,
        targets_flat: Tensor,
    ) -> eyre::Result<(Tensor, Tensor)> {
        let Some(ignore_val) = self.ignore_index else {
            return Ok((logits_flat, targets_flat));
        };

        // get indices to keep
        let keep = targets_flat
            .to_vec1::<i64>()? // has to be i64 to include ignore_index of -100
            .iter()
            .enumerate()
            .filter(|(_, v)| v != &&ignore_val)
            .map(|(ix, _)| ix as u32)
            .collect::<Vec<_>>();
        let keep = Tensor::new(&keep[..], &self.device)?;

        let logits_flat = logits_flat.index_select(&keep, 0)?;
        let targets_flat = targets_flat.index_select(&keep, 0)?;

        Ok((logits_flat, targets_flat))
    }
}

pub struct TextGenerator {
    pub model: GPTModel,
    pub max_new_tokens: usize,
    pub context_size: usize,
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub eos_id: Option<Tensor>,
}

impl TextGenerator {
    pub fn generate(&self, rng: &mut ThreadRng, idx: Tensor) -> eyre::Result<Tensor> {
        let mut idx = idx.clone();
        for _ in 0..self.max_new_tokens {
            // Limit the context window
            let (b, seq_len) = idx.dims2()?;
            let start = seq_len.saturating_sub(self.context_size);
            let idx_cond = idx.i((.., start..seq_len))?;

            // forward pass
            let logits = self.model.forward_t(&idx_cond, false)?;

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

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{
        Activation, Dropout, VarBuilder, VarMap, embedding,
        init::DEFAULT_KAIMING_NORMAL,
        ops::{dropout, softmax},
    };
    use eyre::eyre;
    use tiktoken_rs::{CoreBPE, get_bpe_from_model};

    fn gpt2_tokenizer() -> eyre::Result<CoreBPE> {
        let out = get_bpe_from_model("gpt2").map_err(|e| eyre!("{e}"))?;
        Ok(out)
    }

    fn vb<'a>() -> VarBuilder<'a> {
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu)
    }

    fn gpt_config() -> GptConfig {
        GptConfig {
            vocab_size: 50257,
            emb_dim: 12,
            context_length: 256,
            drop_p: 0.1,
            num_trf: 2,
            num_heads: 3,
            bias: false,
        }
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
        let dataset = Dataset::from_path("the-verdict.txt", tokenizer, &Device::Cpu)?;
        let inputs = dataset.inputs.to_vec2::<u32>()?;
        let targets = dataset.targets.to_vec2::<u32>()?;

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

        let dataset = Dataset::from_path("the-verdict.txt", tokenizer, &Device::Cpu)?;
        let inputs = dataset.inputs.to_vec2::<u32>()?;

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
    fn test_self_attention_v2_init() -> eyre::Result<()> {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v2_layer = SelfAttentionV2::new(&vb(), d_in, d_out, false)?;

        assert_eq!(attn_v2_layer.w_q.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_k.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_v.weight().dims(), &[d_out, d_in]);
        Ok(())
    }

    #[test]
    fn test_self_attention_v2_forward() -> eyre::Result<()> {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v2_layer = SelfAttentionV2::new(&vb(), d_in, d_out, false)?;

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &Device::Cpu)?;
        let context_vectors = attn_v2_layer.forward_t(&xs, false)?;

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
        Ok(())
    }

    #[test]
    fn test_causal_attention_init() -> eyre::Result<()> {
        let (d_in, d_out) = (3_usize, 5_usize);
        let causal = CausalAttentionV2::new(&vb(), d_in, d_out, false, 0.5)?;

        assert_eq!(causal.attention.w_q.weight().dims(), &[d_out, d_in]);
        assert_eq!(causal.attention.w_k.weight().dims(), &[d_out, d_in]);
        assert_eq!(causal.attention.w_v.weight().dims(), &[d_out, d_in]);
        Ok(())
    }

    #[test]
    fn test_causal_attention_forward() -> eyre::Result<()> {
        let (d_in, d_out) = (3_usize, 5_usize);
        let causal = CausalAttentionV2::new(&vb(), d_in, d_out, false, 0.5)?;

        // create batch
        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &Device::Cpu)?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;
        let context_vectors = causal.forward_t(&batch, false)?;

        assert_eq!(context_vectors.dims(), &[2_usize, input_length, d_out]);
        Ok(())
    }

    #[test]
    fn test_dummy_gpt_model_init() -> eyre::Result<()> {
        let cfg = gpt_config();
        let model = GPTModel::new(&vb(), &cfg)?;

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.num_trf);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
        Ok(())
    }

    #[test]
    fn test_dummy_gpt_model_forward() -> eyre::Result<()> {
        let cfg = gpt_config();

        let batch_size = 2_usize;
        let token_ids = (0..(batch_size * cfg.emb_dim) as u32).collect::<Vec<_>>();
        let batch_token_ids =
            Tensor::from_slice(&token_ids, (batch_size, cfg.emb_dim), &Device::Cpu)?;

        let (batch_size, seq_len) = batch_token_ids.dims2()?;

        let model = GPTModel::new(&vb(), &cfg)?;

        let logits = model.forward_t(&batch_token_ids, false)?;

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_init() -> eyre::Result<()> {
        let cfg = gpt_config();
        let layer_norm = LayerNorm::new(&vb(), cfg.emb_dim)?;
        assert_eq!(layer_norm.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.shift.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.scale.i(..=1)?.to_vec1::<f32>()?, &[1., 1.]);
        assert_eq!(layer_norm.shift.i(..=1)?.to_vec1::<f32>()?, &[0., 0.]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_forward() -> eyre::Result<()> {
        let cfg = gpt_config();
        let batch_size = 2_usize;
        let batch_example = Tensor::rand(0f32, 1f32, (batch_size, cfg.emb_dim), &Device::Cpu)?;
        let layer_norm = LayerNorm::new(&vb(), cfg.emb_dim)?;

        let out_norm = layer_norm.forward_t(&batch_example, false)?;
        let mean = out_norm.mean_keepdim(D::Minus1)?;
        let var = out_norm.var_keepdim(D::Minus1)?;

        let mean_minus_zero = mean.broadcast_sub(&mean.zeros_like()?)?.abs()?;
        let mut mean_minus_zero = mean_minus_zero
            .to_vec2()?
            .into_iter()
            .flatten()
            .map(|x: f32| x.floor());

        let var_minus_one = var.broadcast_sub(&var.ones_like()?)?.abs()?;

        let mut var_minus_one = var_minus_one
            .to_vec2()?
            .into_iter()
            .flatten()
            .map(|x: f32| x.floor());

        assert_eq!(out_norm.dims(), &[batch_size, cfg.emb_dim]);

        assert!(var_minus_one.all(|x| x == 0.));
        assert!(mean_minus_zero.all(|x| x == 0.));

        Ok(())
    }

    #[test]
    fn test_gelu_impl() -> eyre::Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 3_usize), &dev)?;

        // testing manual impl
        let gelu = GELU;
        let out = gelu.forward_t(&batch_example, false)?;

        // reference impl
        let candle_gelu = Activation::Gelu;
        let candle_out = candle_gelu.forward_t(&batch_example, false)?;

        // assert equality
        let tol: f64 = 1e-3;
        let abs_diff = (out - candle_out)?.abs()?;
        assert_eq!(
            abs_diff.lt(tol)?.sum_all()?.to_scalar::<u8>()?,
            (2_usize * 3_usize) as u8
        );
        Ok(())
    }

    #[test]
    fn test_feedforward_forward() -> eyre::Result<()> {
        let cfg = gpt_config();
        let ff = FeedForward::new(&vb(), cfg.emb_dim, false)?;

        // create test batch
        let (batch_size, seq_len) = (2_usize, 3_usize);
        let batch_example =
            Tensor::rand(0f32, 1f32, (batch_size, seq_len, cfg.emb_dim), &Device::Cpu)?;
        let out = ff.forward_t(&batch_example, false)?;

        assert_eq!(out.dims(), &[batch_size, seq_len, cfg.emb_dim]);
        Ok(())
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

        let ctx2 = att_lay_2.forward_t(&embeddings, true)?;
        let ctx1 = att_lay_1.forward_t(&embeddings, true)?;

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

        let ctx_mat = att_layer.forward_t(&embeddings, false)?;

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
        let multi_head = attentions.collect::<Result<MultiHeadAttentionWrapper<_>, _>>()?;

        let cont_matrices = multi_head.forward_t(&embeddings, false)?;

        debug_assert_eq!(cont_matrices.dims()[0], num_batches);
        debug_assert_eq!(cont_matrices.dims()[1], emb_rows);
        debug_assert_eq!(cont_matrices.dims()[2], h * d_out);
        Ok(())
    }

    #[test]
    fn mha() -> eyre::Result<()> {
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 2_usize);
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let mha = MultiHeadAttention::new(&vb, num_heads, d_in, d_out / num_heads, false, 0.5)?;

        assert_eq!(mha.w_q.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_k.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_v.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.out_proj.weight().dims(), &[d_out, d_out]);
        assert_eq!(mha.head_dim, d_out / num_heads);
        Ok(())
    }

    #[test]
    fn mha_forward() -> eyre::Result<()> {
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 3_usize);
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let mha = MultiHeadAttention::new(&vb, num_heads, d_in, d_out / num_heads, false, 0.5)?;

        // create batch
        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;
        let context_vectors = mha.forward_t(&batch, false)?;

        assert_eq!(context_vectors.dims(), &[2_usize, input_length, d_out]);
        Ok(())
    }

    #[test]
    fn transformer_block() -> eyre::Result<()> {
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let num_heads = 4;
        let emb_dim = 768;
        let bias = false;
        let drop_p = 0.2;
        let trans = TransformerBlock::new(&vb, num_heads, emb_dim, bias, drop_p)?;

        let num_tokens_per_group = 4;

        let groups = 1;

        let xs = vb.get_with_hints(
            (groups, num_tokens_per_group, emb_dim),
            "embeddings",
            DEFAULT_KAIMING_NORMAL,
        )?;

        let out = trans.forward_t(&xs, false)?;

        debug_assert_eq!(out.dims()[0], groups);
        debug_assert_eq!(out.dims()[1], num_tokens_per_group);
        debug_assert_eq!(out.dims()[2], emb_dim);

        Ok(())
    }

    #[test]
    fn test_transformer_block_init() -> eyre::Result<()> {
        let vb = vb();
        let cfg = gpt_config();
        let transformer_block = TransformerBlock::from_config(&vb, &cfg)?;

        assert_eq!(transformer_block.multi_head.num_heads, cfg.num_heads);
        assert_eq!(
            transformer_block.multi_head.w_k.weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.multi_head.w_q.weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.multi_head.w_v.weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.multi_head.head_dim,
            cfg.emb_dim / cfg.num_heads
        );
        assert_eq!(transformer_block.norm_1.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(transformer_block.norm_2.shift.dims(), &[cfg.emb_dim]);
        Ok(())
    }

    #[test]
    fn test_transformer_block() -> eyre::Result<()> {
        let vb = vb();
        let cfg = gpt_config();
        let transformer_block = TransformerBlock::from_config(&vb, &cfg)?;

        let batch_size = 2_usize;
        let num_tokens = 4_usize;
        let batch_example = Tensor::rand(
            0f32,
            1f32,
            (batch_size, num_tokens, cfg.emb_dim),
            vb.device(),
        )?;

        let out = transformer_block.forward_t(&batch_example, false)?;
        assert_eq!(out.dims(), batch_example.dims());
        Ok(())
    }

    #[test]
    fn test_gpt_model_init() -> eyre::Result<()> {
        let cfg = gpt_config();
        let model = GPTModel::new(&vb(), &cfg)?;

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.num_trf);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
        Ok(())
    }

    #[test]
    fn test_gpt_model_forward() -> eyre::Result<()> {
        let cfg = gpt_config();

        let batch_size = 2_usize;
        let token_ids = (0..(batch_size * cfg.emb_dim) as u32).collect::<Vec<_>>();
        let batch_token_ids =
            Tensor::from_slice(&token_ids, (batch_size, cfg.emb_dim), &Device::Cpu)?;

        let (batch_size, seq_len) = batch_token_ids.dims2()?;

        let model = GPTModel::new(&vb(), &cfg)?;

        let logits = model.forward_t(&batch_token_ids, false)?;

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
        Ok(())
    }

    #[test]
    fn test_gpt() -> eyre::Result<()> {
        let cfg = gpt_config();

        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);

        let vocab_size = cfg.vocab_size;
        let gpt = GPTModel::new(&vb, &cfg)?;

        let batch_size = 2;
        let seq_len = 4; // Must be <= context_length (10)

        let xs = Tensor::rand(0f32, vocab_size as f32, (batch_size, seq_len), &Device::Cpu)?
            .to_dtype(DType::U32)?;

        let _ = gpt.forward_t(&xs, false)?;

        Ok(())
    }

    #[test]
    fn test_generate_text_simple() -> eyre::Result<()> {
        let cfg = gpt_config();

        let batch_size = 2_usize;
        let token_ids = (0..(batch_size * cfg.emb_dim) as u32).collect::<Vec<_>>();
        let batch_token_ids =
            Tensor::from_slice(&token_ids, (batch_size, cfg.emb_dim), &Device::Cpu)?;

        let model = GPTModel::new(&vb(), &cfg)?;

        // create sample idx
        let (batch_size, seq_len) = batch_token_ids.dims2()?;
        let (context_size, max_new_tokens) = (2_usize, 3_usize);
        let idx = model.generate_text_simple(batch_token_ids, max_new_tokens, context_size)?;

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
        Ok(())
    }

    #[test]
    fn tokenizer_test() -> eyre::Result<()> {
        let tokenizer = gpt2_tokenizer()?;
        let mut string = "Every effort moves you".to_string();

        let tokens = tokenizer.encode_with_special_tokens(&string);
        let tokens = Tensor::from_iter(tokens.into_iter(), &Device::Cpu)?;
        let tokens = tokens.unsqueeze(0)?;

        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let model = GPTModel::new(&vb, &gpt_config())?;

        let logits = model.forward_t(&tokens, true)?;

        let (_, c, _) = logits.dims3()?;
        // Pick last column and return the position of the highest logit
        let logits = logits.i((.., c - 1, ..))?;
        let token_id = logits.argmax(D::Minus1)?;
        let out = tokenizer
            .decode(token_id.to_vec1()?)
            .map_err(|e| eyre!(e))?;

        string.push_str(&out);

        println!("{}", string);

        Ok(())
    }

    #[test]
    fn cross_entropy() -> eyre::Result<()> {
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let model = GPTModel::new(&vb, &gpt_config())?;

        let inputs = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device())?;
        let targets = Tensor::new(&[[1_u32, 2, 3], [4, 5, 9]], vb.device())?;

        let c_entropy = CrossEntropy::default();
        let loss = c_entropy.compute_single(&model, &inputs, &targets)?;
        debug_assert!(loss > 0.);

        let data = Dataset::new(inputs, targets);
        let loss2 = c_entropy.compute(&model, data, 1)?;
        debug_assert!((loss - loss2).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_calc_loss_batch_with_ignore_index() -> eyre::Result<()> {
        // create model
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let model = GPTModel::new(&vb, &gpt_config())?;

        // create sample inputs
        let inputs = Tensor::new(&[[100_u32, 20, 300]], vb.device())?;
        let targets = Tensor::new(&[[1_u32, 2, 3]], vb.device())?;
        let loss = CrossEntropy::default().compute_single(&model, &inputs, &targets)?;

        let inputs_2 = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device())?;
        let targets_2 = Tensor::new(&[[1_i64, 2, 3], [-100, -100, -100]], vb.device())?;
        let c_entropy = CrossEntropy {
            ignore_index: Some(-100),
            train: false,
            ..Default::default()
        };
        let loss_2 = c_entropy.compute_single(&model, &inputs_2, &targets_2)?;

        assert_eq!(loss, loss_2);

        Ok(())
    }

    #[test]
    fn loss_metrics() -> eyre::Result<()> {
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu);
        let model = GPTModel::new(&vb, &gpt_config())?;
        let tokenizer = gpt2_tokenizer()?;

        // inputs and target tensors
        let inputs = Tensor::new(&[[16833_u32, 3626, 6100], [40, 1107, 588]], vb.device())?;
        let targets = Tensor::new(&[[3626_u32, 6100, 345], [1107, 588, 11311]], vb.device())?;

        let logits = model.forward_t(&inputs, false)?;
        let probas = softmax(&logits, D::Minus1)?;

        let predicted = probas.argmax_keepdim(D::Minus1)?; //token ids

        let target_text = tokenizer
            .decode(targets.i(0)?.to_vec1::<u32>()?)
            .map_err(|e| eyre!(e))?;

        let predicted_text = tokenizer
            .decode(predicted.i(0)?.flatten_all()?.to_vec1::<u32>()?)
            .map_err(|e| eyre!(e))?;

        println!("{target_text}");
        println!("{predicted_text}");
        Ok(())
    }

    #[test]
    fn encdec() -> eyre::Result<()> {
        let t = gpt2_tokenizer().map(Tokenizer::from)?;
        let txt = "In the heart of the city";
        let enc = t.text_to_token_ids(txt, &Device::Cpu)?;
        let dec = t.token_ids_to_text(&enc)?;
        debug_assert_eq!(txt, dec);
        Ok(())
    }

    #[test]
    fn test_generate() -> eyre::Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let batch_token_ids = Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev)?;

        let model = GPTModel::new(&vb(), &gpt_config())?;

        // create sample idx
        let (batch_size, seq_len) = batch_token_ids.dims2()?;
        let (context_size, max_new_tokens) = (2_usize, 3_usize);
        let rng = &mut rand::rng();

        let txt_gen = TextGenerator {
            model,
            max_new_tokens,
            context_size,
            temperature: Some(1.),
            top_k: Some(3),
            eos_id: None,
        };

        let idx = txt_gen.generate(rng, batch_token_ids)?;

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
        Ok(())
    }
}

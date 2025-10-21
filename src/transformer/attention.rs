use super::*;

pub struct MultiHeadAttention {
    pub num_heads: usize, // number of heads in the multi head attention systems
    pub d_out: usize,     // columns of the (big) weight matrix
    pub head_dim: usize,  // columns of the weight matrix for each hed
    pub w_q: Linear,      // (big) query weight
    pub w_k: Linear,      // (big) key weight
    pub w_v: Linear,      // (big) value weight
    pub out_proj: Linear,
    pub scaling: f64,     //computed based on the number of columns of each head
    pub dropout: Dropout, // involved in causal attention
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
        let w_q = linear_b(emb_vec_size, d_k, bias, vb.pp("query"))?; // TODO: Need to decompose from c_attn
        let w_k = linear_b(emb_vec_size, d_k, bias, vb.pp("key"))?; // TODO: Need to decompose from c_attn
        let w_v = linear_b(emb_vec_size, d_k, bias, vb.pp("value"))?; // TODO: Need to decompose from c_attn

        let scaling = 1. / (head_dim as f64).sqrt();

        let dropout = Dropout::new(drop_p);

        let out_proj = linear_b(d_k, d_k, bias, vb.pp("c_proj"))?;

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

pub struct SelfAttention<T> {
    pub w_q: T,
    pub w_k: T,
    pub w_v: T,
    pub scaling: f64,
}

pub trait AttentionLayer: ModuleT {}

pub type SelfAttentionV1 = SelfAttention<Tensor>;
impl AttentionLayer for SelfAttentionV1 {}
pub type SelfAttentionV2 = SelfAttention<Linear>;
impl AttentionLayer for SelfAttentionV2 {}
pub struct CausalAttention<T> {
    pub attention: SelfAttention<T>,
    pub dropout: Dropout,
}

pub type CausalAttentionV2 = CausalAttention<Linear>;
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

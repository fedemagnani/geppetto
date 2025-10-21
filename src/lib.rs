use candle_core::DType;
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::LayerNorm;
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::ops::softmax;
use candle_nn::{
    AdamW, Dropout, Embedding, Linear, ModuleT, Optimizer, VarBuilder, embedding, linear_b,
};
use candle_nn::{LayerNormConfig, layer_norm};

use core::f32;
use eyre::eyre;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::f64;
use std::{fs, path::Path};
use tiktoken_rs::CoreBPE;

mod tokenizer;
pub use tokenizer::*;

mod traits;
pub use traits::*;

mod transformer;
pub use transformer::*;

mod data;
pub use data::*;

mod gpt;
pub use gpt::*;

mod custom_norm;
pub use custom_norm::*;

#[cfg(test)]
mod tests {

    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{
        Activation, Dropout, ParamsAdamW, VarBuilder, VarMap, embedding,
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
            emb_dim: 768,
            context_length: 256,
            drop_p: 0.1,
            num_trf: 12,
            num_heads: 12,
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
        let tokenizer: Tokenizer = gpt2_tokenizer()?.into();

        let dataset = Dataset::from_path("the-verdict.txt", &tokenizer, &Device::Cpu, 1.)?;
        let inputs = dataset.train.inputs.to_vec2::<u32>()?;
        let targets = dataset.train.targets.to_vec2::<u32>()?;

        assert_eq!(targets[0][0], inputs[0][1]);
        assert_eq!(targets[0][1], inputs[0][2]);
        assert_eq!(targets[0][2], inputs[0][3]);

        Ok(())
    }

    #[test]
    fn test_embeddings() -> eyre::Result<()> {
        let tokenizer: Tokenizer = gpt2_tokenizer()?.into(); //gpt2 BPE tokenizer has 50257 tokens
        let vocab_size = 50_257_usize;
        let embedding_size = 256_usize; //each tokenid will be associated with a 256-dimenisonal vector

        let dataset = Dataset::from_path("the-verdict.txt", &tokenizer, &Device::Cpu, 1.)?;
        let inputs = dataset.train.inputs.to_vec2::<u32>()?;

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
        let layer_norm = CustomLayerNorm::new(&vb(), cfg.emb_dim)?;
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
        let layer_norm = CustomLayerNorm::new(&vb(), cfg.emb_dim)?;

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
        let idx =
            model.generate_text_simple(batch_token_ids, max_new_tokens, context_size, false)?;

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
        let loss = c_entropy.loss_scalar(&model, &inputs, &targets)?;
        debug_assert!(loss > 0.);

        let data = TensorDataset::new(inputs, targets);
        let loss2 = c_entropy.compute(&model, data, 1, false)?;
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
        let loss = CrossEntropy::default().loss_scalar(&model, &inputs, &targets)?;

        let inputs_2 = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device())?;
        let targets_2 = Tensor::new(&[[1_i64, 2, 3], [-100, -100, -100]], vb.device())?;
        let c_entropy = CrossEntropy {
            ignore_index: Some(-100),
            train: false,
            ..Default::default()
        };
        let loss_2 = c_entropy.loss_scalar(&model, &inputs_2, &targets_2)?;

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

        let idx = txt_gen.generate(rng, batch_token_ids, false)?;

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
        Ok(())
    }

    #[test]
    fn test_train() -> eyre::Result<()> {
        println!("Starting");
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &Device::new_metal(0)?);
        let cfg = gpt_config();
        let model = GPTModel::new(&vb, &cfg)?;
        let tokenizer: Tokenizer = gpt2_tokenizer()?.into();
        println!("Reading the novel");
        let data = Dataset::from_path("the-verdict.txt", &tokenizer, vb.device(), 0.8)?;
        println!("starting training");
        let entropy = CrossEntropy {
            train: true,
            device: vb.device().clone(),
            ..Default::default()
        };
        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: 0.0004,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let mut trainer = Trainer::new(entropy, optimizer, true);
        trainer.train(&model, data, 10, 2)?;

        let ids = tokenizer.text_to_token_ids("In the hearth of the city", vb.device())?;
        let out = model.generate_text_simple(ids, 20, cfg.context_length, false)?;
        let text = tokenizer.token_ids_to_text(&out)?;
        println!("{text}");

        Ok(())
    }
}

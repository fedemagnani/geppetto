use candle_core::{DType, Device, IndexOp};
use candle_nn::VarBuilder;
use eyre::eyre;
use geppetto::{GPTModel, GptConfig, TextGenerator, Tokenizer};
use hf_hub::api::sync::Api;
use rand::rng;
use tiktoken_rs::{CoreBPE, get_bpe_from_model};
fn gpt2_tokenizer() -> eyre::Result<CoreBPE> {
    let out = get_bpe_from_model("gpt2").map_err(|e| eyre!("{e}"))?;
    Ok(out)
}

fn gpt2_small_config() -> GptConfig {
    GptConfig {
        vocab_size: 50257,
        context_length: 1024,
        emb_dim: 768,
        num_trf: 12,
        num_heads: 12,
        drop_p: 0.1,
        bias: false,
    }
}

fn main() -> eyre::Result<()> {
    let dev = Device::new_metal(0)?;

    // get weights from HF Hub
    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let weights = repo.get("model.safetensors")?;
    let mut weights = candle_core::safetensors::load(weights, &dev)?;

    let mut i = 0;

    // query-key-value dimensions must be unwrapped, and all the weights associated with linear layers must be transposed due to bad convention
    while let (Some(w), Some(b)) = (
        weights.remove(&format!("h.{i}.attn.c_attn.weight")),
        weights.remove(&format!("h.{i}.attn.c_attn.bias")),
    ) {
        let dim = b.dims()[0] / 3_usize;
        let (q_w, q_b) = (w.i((.., ..dim))?, b.i(..dim)?);
        let (k_w, k_b) = (w.i((.., dim..2 * dim))?, b.i(dim..2 * dim)?);
        let (v_w, v_b) = (w.i((.., 2 * dim..))?, b.i(2 * dim..)?);

        weights.insert(format!("h.{i}.attn.query.weight"), q_w.t()?.contiguous()?);
        weights.insert(format!("h.{i}.attn.key.weight"), k_w.t()?.contiguous()?);
        weights.insert(format!("h.{i}.attn.value.weight"), v_w.t()?.contiguous()?);
        weights.insert(format!("h.{i}.attn.query.bias"), q_b);
        weights.insert(format!("h.{i}.attn.key.bias"), k_b);
        weights.insert(format!("h.{i}.attn.value.bias"), v_b);

        i += 1
    }

    for (k, v) in weights.iter_mut() {
        if !k.contains("c_proj.weight") && !k.contains("c_fc.weight") {
            continue;
        }
        *v = v.t()?;
    }

    for (k, _) in weights.iter() {
        println!("{k}");
    }

    println!("\nNumber of weights: {}", weights.len());

    let vb = VarBuilder::from_tensors(weights, DType::F32, &dev);
    let cfg = gpt2_small_config();
    let model = GPTModel::new(&vb, &cfg)?;

    let tokenizer: Tokenizer = gpt2_tokenizer()?.into();

    let ids = tokenizer.text_to_token_ids("Every effort moves you", vb.device())?;

    let tt = TextGenerator {
        model,
        max_new_tokens: 20,
        context_length: cfg.context_length,
        eos_id: None,
        temperature: Some(0.1_f64),
        top_k: Some(50_usize),
    };

    // let out = model.generate_text_simple(ids, 20, cfg.context_length, false)?;
    let out = tt.generate(&mut rng(), ids, false)?;
    let text = tokenizer.token_ids_to_text(&out)?;

    println!("{text}");

    Ok(())
}

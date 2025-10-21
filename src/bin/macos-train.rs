use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use eyre::eyre;
use geppetto::{CrossEntropy, Dataset, GPTModel, GptConfig, Tokenizer, Trainer};
use tiktoken_rs::{CoreBPE, get_bpe_from_model};

fn gpt2_tokenizer() -> eyre::Result<CoreBPE> {
    let out = get_bpe_from_model("gpt2").map_err(|e| eyre!("{e}"))?;
    Ok(out)
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

fn main() -> eyre::Result<()> {
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

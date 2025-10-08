use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use geppetto::{GPTModel, GptConfig, TextGenerator, Tokenizer};
use hf_hub::api::sync::Api;
use rand::rng;

fn main() -> eyre::Result<()> {
    let dev = Device::Cpu;

    // get weights from HF Hub
    let api = Api::new()?;
    let repo = api.model("openai-community/gpt2".to_string());
    let weights = repo.get("model.safetensors")?;
    let mut weights = candle_core::safetensors::load(weights, &dev)?;

    GptConfig::adapt_gpt2_weights(&mut weights)?;

    for (k, _) in weights.iter() {
        println!("{k}");
    }

    println!("\nNumber of weights: {}", weights.len());

    let vb = VarBuilder::from_tensors(weights, DType::F32, &dev);
    let cfg = GptConfig::gpt2_small();
    let model = GPTModel::new(&vb, &cfg)?;

    let tokenizer = Tokenizer::gpt2()?;

    let ids = tokenizer.text_to_token_ids("Every effort moves you", vb.device())?;

    let tt = TextGenerator {
        model,
        max_new_tokens: 25,
        context_length: cfg.context_length,
        eos_id: None,
        temperature: Some(0.1_f64),
        top_k: Some(50_usize),
    };

    let out = tt.generate(&mut rng(), ids, false)?;

    let text = tokenizer.token_ids_to_text(&out)?;

    println!("{text}");

    Ok(())
}

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use geppetto::gguf::GgufFile;
use geppetto::tokenizer::Tokenizer;

/// Encode text with a GGUF vocab and print the token ids.
#[derive(Parser)]
struct Cli {
    /// Path to a GGUF file with a gpt2 tokenizer.
    model: PathBuf,
    /// Text to encode.
    text: String,
}

fn main() -> ExitCode {
    let args = Cli::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("tokenize: {}: {err}", args.model.display());
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let file = GgufFile::open(&args.model)?;
    let tokenizer = Tokenizer::from_gguf(&file)?;

    let ids = tokenizer.encode(&args.text)?;
    println!("{} tokens", ids.len());
    for &id in &ids {
        println!("  {id:>6} -> {:?}", tokenizer.decode_lossy(&[id])?);
    }
    println!("round trip: {:?}", tokenizer.decode_lossy(&ids)?);
    Ok(())
}

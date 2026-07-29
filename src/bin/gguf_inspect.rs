use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use geppetto::gguf::{ArrayValue, GgufFile, Value};

/// Print the metadata and tensor table of a GGUF file.
#[derive(Parser)]
struct Cli {
    /// Path to the GGUF file.
    path: PathBuf,
}

fn main() -> ExitCode {
    let args = Cli::parse();
    let path = args.path.display();
    let file = match GgufFile::open(&args.path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("gguf-inspect: {path}: {err}");
            return ExitCode::FAILURE;
        }
    };

    println!("{path}");
    println!("  version:   {}", file.version());
    println!("  alignment: {}", file.alignment());
    println!("  kv pairs:  {}", file.metadata().len());
    println!("  tensors:   {}", file.tensors().len());

    println!("\nmetadata:");
    for (key, value) in file.metadata().iter() {
        println!("  {key}: {}", fmt_value(value));
    }

    println!("\ntensor table:");
    for info in file.tensors().values() {
        let dims = info
            .dims
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(" x ");
        let dtype = info
            .dtype()
            .map_or_else(|| format!("type#{}", info.type_id), |d| d.to_string());
        let bytes = info.nbytes().map_or_else(|| "?".into(), |b| b.to_string());
        println!(
            "  {:<48} {:>6} [{dims}] offset={} bytes={bytes}",
            info.name, dtype, info.offset
        );
    }
    ExitCode::SUCCESS
}

fn fmt_value(value: &Value) -> String {
    match value {
        Value::U8(v) => format!("[u8] {v}"),
        Value::I8(v) => format!("[i8] {v}"),
        Value::U16(v) => format!("[u16] {v}"),
        Value::I16(v) => format!("[i16] {v}"),
        Value::U32(v) => format!("[u32] {v}"),
        Value::I32(v) => format!("[i32] {v}"),
        Value::F32(v) => format!("[f32] {v}"),
        Value::Bool(v) => format!("[bool] {v}"),
        Value::String(v) => format!("[string] {}", fmt_str(v)),
        Value::U64(v) => format!("[u64] {v}"),
        Value::I64(v) => format!("[i64] {v}"),
        Value::F64(v) => format!("[f64] {v}"),
        Value::Array(arr) => {
            let head = fmt_array_head(arr, 4);
            let tail = if arr.len() > 4 { ", ..." } else { "" };
            format!("[{} x {}] [{head}{tail}]", arr.elem_type(), arr.len())
        }
    }
}

fn fmt_array_head(arr: &ArrayValue, n: usize) -> String {
    fn join<T, F: Fn(&T) -> String>(v: &[T], n: usize, f: F) -> String {
        v.iter().take(n).map(f).collect::<Vec<_>>().join(", ")
    }
    match arr {
        ArrayValue::U8(v) => join(v, n, u8::to_string),
        ArrayValue::I8(v) => join(v, n, i8::to_string),
        ArrayValue::U16(v) => join(v, n, u16::to_string),
        ArrayValue::I16(v) => join(v, n, i16::to_string),
        ArrayValue::U32(v) => join(v, n, u32::to_string),
        ArrayValue::I32(v) => join(v, n, i32::to_string),
        ArrayValue::F32(v) => join(v, n, f32::to_string),
        ArrayValue::Bool(v) => join(v, n, bool::to_string),
        ArrayValue::String(v) => join(v, n, |s| fmt_str(s)),
        ArrayValue::U64(v) => join(v, n, u64::to_string),
        ArrayValue::I64(v) => join(v, n, i64::to_string),
        ArrayValue::F64(v) => join(v, n, f64::to_string),
    }
}

fn fmt_str(s: &str) -> String {
    let mut out: String = s.chars().take(60).collect();
    if s.chars().count() > 60 {
        out.push_str("...");
    }
    format!("{out:?}")
}

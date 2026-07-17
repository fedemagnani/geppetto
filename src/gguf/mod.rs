//! GGUF container format reader (versions 2 and 3, little-endian).
//!
//! Reference: `ggml/src/gguf.cpp` in llama.cpp. The reader is format-complete
//! for metadata and tensor infos; tensor *data* access requires a type that
//! maps to [`crate::tensor::DType`] (quantized types arrive in a later epoch).

mod error;
mod reader;
mod tensor_info;
mod value;

#[cfg(test)]
mod test;

pub use error::GgufError;
pub use reader::{GgufFile, MetadataTable, TensorTable};
pub use tensor_info::{MAX_DIMS, MAX_NAME_LEN, TensorInfo};
pub use value::{ArrayValue, Value, ValueType};

pub const MAGIC: [u8; 4] = *b"GGUF";

/// Hard cap on array lengths, mirroring `GGUF_MAX_ARRAY_ELEMENTS`.
pub const MAX_ARRAY_ELEMENTS: u64 = 1 << 30;

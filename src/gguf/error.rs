use crate::gguf::ValueType;
use crate::tensor::TensorError;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("unexpected end of file: need {wanted} bytes at offset {offset}")]
    UnexpectedEof { offset: usize, wanted: usize },
    #[error("invalid magic {0:?}, expected \"GGUF\"")]
    BadMagic([u8; 4]),
    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error("string at offset {offset} is not valid UTF-8")]
    InvalidUtf8 { offset: usize },
    #[error("invalid metadata value type id {0}")]
    InvalidValueType(u32),
    #[error("key '{0}': arrays of arrays are not supported")]
    NestedArray(String),
    #[error("key '{key}': array length {len} exceeds the maximum")]
    ArrayTooLong { key: String, len: u64 },
    #[error("invalid bool byte {0:#x}, expected 0 or 1")]
    InvalidBool(u8),
    #[error("duplicate metadata key '{0}'")]
    DuplicateKey(String),
    #[error("alignment {0} is not a positive power of 2")]
    InvalidAlignment(u64),
    #[error("duplicate tensor name '{0}'")]
    DuplicateTensor(String),
    #[error("tensor name of {0} bytes is too long")]
    NameTooLong(usize),
    #[error("tensor '{name}' has {n_dims} dimensions, more than the maximum")]
    TooManyDims { name: String, n_dims: u32 },
    #[error("tensor '{name}' has a zero dimension")]
    ZeroDim { name: String },
    #[error("tensor '{name}' shape does not fit in a signed 64-bit element count")]
    ElementsOverflow { name: String },
    #[error("tensor '{name}' size in bytes overflows")]
    SizeOverflow { name: String },
    #[error("tensor '{name}' has offset {offset}, expected {expected}")]
    BadTensorOffset {
        name: String,
        offset: u64,
        expected: u64,
    },
    #[error("tensor '{name}' data is out of the file's bounds")]
    TensorOutOfBounds { name: String },
    #[error("tensor '{name}' has type id {type_id}, not supported for data access yet")]
    UnsupportedTensorType { name: String, type_id: u32 },
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),
    #[error("metadata key '{0}' not found")]
    KeyNotFound(String),
    #[error("metadata key '{key}' has type {found}, expected {expected}")]
    TypeMismatch {
        key: String,
        expected: &'static str,
        found: ValueType,
    },
    #[error(
        "tensor '{name}' has shape [{got_rows}, {got_cols}], expected [{want_rows}, {want_cols}]"
    )]
    ShapeMismatch {
        name: String,
        got_rows: usize,
        got_cols: usize,
        want_rows: usize,
        want_cols: usize,
    },
}

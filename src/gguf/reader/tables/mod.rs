//! The two lookup tables of a GGUF file, both backed by the generic
//! insertion-ordered [`table::Table`].

mod metadata_table;
mod table;
mod tensor_table;

pub use metadata_table::MetadataTable;
pub use tensor_table::TensorTable;

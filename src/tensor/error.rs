use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    #[error(
        "row of {n_elements} {dtype} elements is not a whole number of blocks (block size {block_size})"
    )]
    PartialBlock {
        dtype: DType,
        n_elements: usize,
        block_size: usize,
    },
    #[error("{dtype} tensor of {n_elements} elements needs {expected} bytes but got {got}")]
    ByteCountMismatch {
        dtype: DType,
        n_elements: usize,
        expected: usize,
        got: usize,
    },
}

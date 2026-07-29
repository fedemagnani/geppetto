//! CPU tensors and the ops the GPT-2 forward pass is built from.
//!
//! Everything works over borrowed views: ops are free functions consuming
//! [`TensorView`]/[`TensorViewMut`] mounted over the arena, and model weights
//! are [`WeightTensor`]s backed by the mmap'd GGUF file. Nothing here owns
//! activation memory -- the arena does.

mod activation;
mod dtype;
mod error;
pub mod layer;
mod ops;
mod shape;
mod view;
mod weight;

#[cfg(test)]
pub(crate) mod test_support;

pub use activation::{gelu, softmax, softmax_causal};
pub use dtype::DType;
pub use error::TensorError;
pub use ops::{add, matmul, matmul_nn, matmul_nn_causal};
pub use shape::Shape;
pub use view::{TensorView, TensorViewMut};
pub use weight::WeightTensor;

/// Size in bytes of a contiguous row of `n_elements` stored as `dtype`.
///
/// Rows must contain a whole number of blocks.
///
/// A free function for now: row layout is not a property of the data type,
/// and its natural owner (the tensor/layout machinery) does not exist yet.
pub const fn row_size(dtype: DType, n_elements: usize) -> Result<usize, TensorError> {
    let block_size = dtype.block_size();
    if !n_elements.is_multiple_of(block_size) {
        return Err(TensorError::PartialBlock {
            dtype,
            n_elements,
            block_size,
        });
    }
    Ok(n_elements / block_size * dtype.type_size())
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn row_size_scales_with_element_count() {
        assert_eq!(row_size(DType::F32, 0), Ok(0));
        assert_eq!(row_size(DType::F32, 3), Ok(12));
        assert_eq!(row_size(DType::F16, 768), Ok(1536));
    }

    #[test]
    fn whole_rows_for_every_dtype() {
        for dtype in DType::iter() {
            let n = dtype.block_size() * 5;
            assert_eq!(row_size(dtype, n), Ok(5 * dtype.type_size()));
        }
    }
}

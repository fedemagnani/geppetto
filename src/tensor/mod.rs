//! CPU tensor types. Epoch 0 provides the data-type vocabulary; the tensor
//! struct and ops arrive in epoch 3.

mod dtype;
mod error;

pub use dtype::DType;
pub use error::TensorError;

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

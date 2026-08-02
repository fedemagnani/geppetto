use crate::tensor::TensorError;

/// Element type of tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::Display, strum::EnumIter)]
#[strum(serialize_all = "lowercase")]
pub enum DType {
    F32,
    F16,
}

impl DType {
    /// Number of elements stored in one block.
    pub const fn block_size(self) -> usize {
        match self {
            DType::F32 | DType::F16 => 1,
        }
    }

    /// Size in bytes of one block.
    pub const fn type_size(self) -> usize {
        match self {
            DType::F32 => size_of::<f32>(),
            // no f16 primitive on stable; u16 is the storage type
            DType::F16 => size_of::<u16>(),
        }
    }

    /// Size in bytes of a contiguous row of `n_elements` stored as `dtype`.
    ///
    /// Rows must contain a whole number of blocks.
    pub const fn row_byte_size(self, n_elements: usize) -> Result<usize, TensorError> {
        let block_size = self.block_size();
        if !n_elements.is_multiple_of(block_size) {
            return Err(TensorError::PartialBlock {
                dtype: self,
                n_elements,
                block_size,
            });
        }
        Ok(n_elements / block_size * self.type_size())
    }
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn row_size_scales_with_element_count() {
        assert_eq!(DType::F32.row_byte_size(0), Ok(0));
        assert_eq!(DType::F32.row_byte_size(3), Ok(4 * 3));
        assert_eq!(DType::F16.row_byte_size(768), Ok(2 * 768));
    }

    #[test]
    fn whole_rows_for_every_dtype() {
        for dtype in DType::iter() {
            let n = dtype.block_size() * 5;
            assert_eq!(dtype.row_byte_size(n), Ok(5 * dtype.type_size()));
        }
    }
}

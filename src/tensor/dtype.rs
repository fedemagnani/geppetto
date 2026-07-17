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
}

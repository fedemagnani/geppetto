use crate::gguf::GgufError;
use crate::tensor::{DType, row_size};

/// Mirrors `GGML_MAX_DIMS`.
pub const MAX_DIMS: usize = 4;
/// Mirrors `GGML_MAX_NAME`; names must be strictly shorter.
pub const MAX_NAME_LEN: usize = 64;

/// One entry of the tensor table. Fields are validated by the reader; the
/// type id is kept raw so files containing not-yet-supported (e.g. quantized)
/// types still parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions as written in the file; `dims[0]` is the row length.
    pub dims: Vec<u64>,
    /// Raw ggml type id.
    pub type_id: u32,
    /// Offset from the start of the data section, a multiple of the alignment.
    pub offset: u64,
}

impl TensorInfo {
    /// The element type, when the ggml type id maps to a supported [`DType`].
    pub const fn dtype(&self) -> Option<DType> {
        match self.type_id {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            _ => None,
        }
    }

    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Checks this tensor starts at `expected` (the layout rule the C++
    /// reader enforces: tensors are packed sequentially in file order) and
    /// returns where the next tensor's data must start, padded to
    /// `alignment`. `Ok(None)` when this tensor's size is uncomputable, which
    /// leaves the rest of the layout unknown.
    pub fn end_offset(&self, expected: u64, alignment: u64) -> Result<Option<u64>, GgufError> {
        if self.offset != expected {
            return Err(GgufError::BadTensorOffset {
                name: self.name.clone(),
                offset: self.offset,
                expected,
            });
        }
        let Some(nbytes) = self.nbytes() else {
            return Ok(None);
        };
        nbytes
            .checked_next_multiple_of(alignment)
            .and_then(|padded| expected.checked_add(padded))
            .map(Some)
            .ok_or_else(|| GgufError::SizeOverflow {
                name: self.name.clone(),
            })
    }

    /// Size of the tensor data in bytes. `None` when the type is unsupported,
    /// the row length is not a whole number of blocks, or the size overflows;
    /// the reader rejects the latter two at parse time.
    pub fn nbytes(&self) -> Option<u64> {
        let dtype = self.dtype()?;
        let n_rows = self.dims.iter().skip(1).product::<u64>();
        let row = row_size(
            dtype,
            usize::try_from(self.dims.first().copied().unwrap_or(1)).ok()?,
        )
        .ok()?;
        u64::try_from(row).ok()?.checked_mul(n_rows)
    }
}

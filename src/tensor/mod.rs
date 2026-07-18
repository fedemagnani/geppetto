//! CPU tensors and the ops the GPT-2 forward pass is built from.
//!
//! A [`Tensor`] is a 2D, row-major, owned `Vec<f32>` -- activations and
//! weights alike. Weights arriving as raw GGUF bytes are converted to f32 at
//! construction: F32 is copied, F16 is widened. That is correctness-first; the
//! zero-copy F32 borrow and per-tile F16 dequant the plan mentions are an
//! epoch-6 optimization. Quantized dtypes plug in at [`Tensor::from_bytes`].

mod activation;
mod dtype;
mod error;
mod layer;
mod ops;
mod shape;

#[cfg(test)]
mod test_support;

pub use dtype::DType;
pub use error::TensorError;
pub use shape::Shape;

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

/// A 2D, row-major tensor of f32. See the module docs for why everything is
/// owned f32 at this epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
}

impl Tensor {
    /// Wraps `data` in `shape`; the two must agree in length.
    pub fn new(shape: Shape, data: Vec<f32>) -> Tensor {
        assert_eq!(
            data.len(),
            shape.len(),
            "tensor data length {} does not match shape {shape}",
            data.len(),
        );
        Tensor { shape, data }
    }

    pub fn zeros(shape: Shape) -> Tensor {
        Tensor {
            data: vec![0.0; shape.len()],
            shape,
        }
    }

    /// Builds a tensor from a GGUF tensor's raw bytes, dispatching on `dtype`.
    /// The byte count must match `shape` exactly for `dtype`.
    pub fn from_bytes(dtype: DType, shape: Shape, bytes: &[u8]) -> Result<Tensor, TensorError> {
        let data = match dtype {
            DType::F32 => decode_f32(shape, bytes)?,
            DType::F16 => decode_f16(shape, bytes)?,
        };
        Ok(Tensor::new(shape, data))
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Row `r` as a contiguous slice of `cols` elements.
    pub fn row(&self, r: usize) -> &[f32] {
        let cols = self.cols();
        &self.data[r * cols..r * cols + cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let cols = self.cols();
        &mut self.data[r * cols..r * cols + cols]
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        let cols = self.cols().max(1);
        self.data.chunks_mut(cols)
    }
}

fn byte_count_error(dtype: DType, shape: Shape, got: usize) -> TensorError {
    TensorError::ByteCountMismatch {
        dtype,
        n_elements: shape.len(),
        expected: shape.len() * dtype.type_size(),
        got,
    }
}

fn decode_f32(shape: Shape, bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    if bytes.len() != shape.len() * DType::F32.type_size() {
        return Err(byte_count_error(DType::F32, shape, bytes.len()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn decode_f16(shape: Shape, bytes: &[u8]) -> Result<Vec<f32>, TensorError> {
    if bytes.len() != shape.len() * DType::F16.type_size() {
        return Err(byte_count_error(DType::F16, shape, bytes.len()));
    }
    // half -> f32 widening is exact; the `half` dependency is removed for a
    // hand-rolled conversion in epoch 6
    Ok(bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
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

    #[test]
    fn rows_are_contiguous_slices() {
        let t = Tensor::new(Shape::new(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_f32_bytes_round_trips() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t = Tensor::from_bytes(DType::F32, Shape::new(2, 2), &bytes).unwrap();
        assert_eq!(t.data(), &values);
    }

    #[test]
    fn from_f16_bytes_widens_exactly_on_representable_values() {
        // 1.0, 2.0, -2.0, 0.5 as f16 bit patterns
        let bits: [u16; 4] = [0x3C00, 0x4000, 0xC000, 0x3800];
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let t = Tensor::from_bytes(DType::F16, Shape::new(1, 4), &bytes).unwrap();
        assert_eq!(t.data(), &[1.0, 2.0, -2.0, 0.5]);
    }

    #[test]
    fn from_f16_bytes_is_monotone_over_a_sweep() {
        let bits: Vec<u16> = (0x0001u16..0x7C00).step_by(7).collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let shape = Shape::new(1, bits.len());
        let t = Tensor::from_bytes(DType::F16, shape, &bytes).unwrap();
        for pair in t.data().windows(2) {
            assert!(
                pair[1] > pair[0],
                "not increasing: {} then {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn wrong_byte_count_is_rejected() {
        let err = Tensor::from_bytes(DType::F32, Shape::new(2, 2), &[0u8; 8]).unwrap_err();
        assert!(matches!(
            err,
            TensorError::ByteCountMismatch {
                expected: 16,
                got: 8,
                ..
            }
        ));
    }
}

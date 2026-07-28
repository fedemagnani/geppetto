//! Model weights backed by the mmap'd GGUF file wherever possible.
//!
//! A [`WeightTensor`] is the seam between the file's storage dtype and the
//! f32 compute of this epoch: F32 tensor data is borrowed zero-copy from the
//! file mapping (a refcounted `Bytes` slice -- page-cache backed, lazily
//! faulted in, evictable under memory pressure, shared across processes);
//! F16 is widened once into an owned buffer at load. Future quantized dtypes
//! plug in here without touching any driver signature: drivers only ever see
//! [`TensorView`]s.
//!
//! Documented assumption, inherited from `GgufFile::open`: the model file is
//! not modified while loaded.

use bytes::Bytes;

use crate::tensor::{DType, Shape, TensorError, TensorView};

// mapped tensor bytes are reinterpreted as f32 in place, which is only valid
// because GGUF stores them little-endian
#[cfg(target_endian = "big")]
compile_error!("GGUF tensor data is little-endian; big-endian targets would need a byte swap");

/// An owned f32 buffer presentable as bytes, so widened weights ride the
/// same refcounted [`Bytes`] machinery as mapped ones. A `Vec<f32>` is
/// always f32-aligned, so the round trip through `[u8]` is valid by
/// construction.
struct F32Buffer(Vec<f32>);

impl F32Buffer {
    /// Copies little-endian f32 bytes (the fallback for an unaligned source).
    fn copy_f32(bytes: &[u8]) -> Bytes {
        let floats = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Bytes::from_owner(F32Buffer(floats))
    }

    /// Widens little-endian f16 bytes; half -> f32 is exact.
    fn widen_f16(bytes: &[u8]) -> Bytes {
        let floats = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        Bytes::from_owner(F32Buffer(floats))
    }
}

impl AsRef<[u8]> for F32Buffer {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

/// A weight matrix, `[rows, cols]` row-major f32 however the file stores it:
/// one invariant, refcounted bytes viewed as f32. F32 tensor data borrows
/// the file mapping zero-copy; F16 is widened once at bind into an owned
/// buffer behind the same `Bytes`. Cheap to clone either way (the tied
/// unembedding clones `token_embd` without copying it).
#[derive(Debug, Clone)]
pub struct WeightTensor {
    data: Bytes,
    shape: Shape,
}

impl WeightTensor {
    /// Binds a GGUF tensor's raw data. F32 borrows the bytes zero-copy after
    /// one alignment check here at bind time -- never trusted; an unaligned
    /// source (possible for byte buffers that are not the 32-byte-aligned
    /// file mapping) falls back to a one-time copy. F16 is widened into an
    /// owned buffer.
    pub fn from_gguf_bytes(
        dtype: DType,
        shape: Shape,
        bytes: Bytes,
    ) -> Result<WeightTensor, TensorError> {
        let expected = shape.len() * dtype.type_size();
        if bytes.len() != expected {
            return Err(TensorError::ByteCountMismatch {
                dtype,
                n_elements: shape.len(),
                expected,
                got: bytes.len(),
            });
        }
        let data = match dtype {
            DType::F32 if bytemuck::try_cast_slice::<u8, f32>(bytes.as_ref()).is_ok() => bytes,
            DType::F32 => F32Buffer::copy_f32(bytes.as_ref()),
            DType::F16 => F32Buffer::widen_f16(bytes.as_ref()),
        };
        Ok(WeightTensor { data, shape })
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

    /// The full weight as a contiguous f32 slice.
    pub fn data(&self) -> &[f32] {
        bytemuck::cast_slice(self.data.as_ref())
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView::contiguous(self.data(), self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Bytes {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Bytes::from(raw)
    }

    #[test]
    fn f32_round_trips() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let w = WeightTensor::from_gguf_bytes(DType::F32, Shape::new(2, 2), f32_bytes(&values))
            .unwrap();
        assert_eq!(w.data(), &values);
        assert_eq!(w.view().row(1), &[3.25, 0.0]);
    }

    #[test]
    fn f16_widens_exactly_on_representable_values() {
        // 1.0, 2.0, -2.0, 0.5 as f16 bit patterns
        let bits: [u16; 4] = [0x3C00, 0x4000, 0xC000, 0x3800];
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let w =
            WeightTensor::from_gguf_bytes(DType::F16, Shape::new(1, 4), Bytes::from(raw)).unwrap();
        assert_eq!(w.data(), &[1.0, 2.0, -2.0, 0.5]);
    }

    #[test]
    fn f16_widening_is_monotone_over_a_sweep() {
        let bits: Vec<u16> = (0x0001u16..0x7C00).step_by(7).collect();
        let raw: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let shape = Shape::new(1, bits.len());
        let w = WeightTensor::from_gguf_bytes(DType::F16, shape, Bytes::from(raw)).unwrap();
        for pair in w.data().windows(2) {
            assert!(
                pair[1] > pair[0],
                "not increasing: {} then {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn unaligned_f32_bytes_fall_back_to_a_copy() {
        // slicing at 1 byte guarantees a misaligned f32 start
        let mut raw = vec![0u8];
        raw.extend(1.5f32.to_le_bytes());
        raw.extend(2.5f32.to_le_bytes());
        let bytes = Bytes::from(raw).slice(1..);
        let w = WeightTensor::from_gguf_bytes(DType::F32, Shape::new(1, 2), bytes).unwrap();
        assert_eq!(w.data(), &[1.5, 2.5]);
    }

    #[test]
    fn clones_share_the_backing_storage() {
        let values: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let w = WeightTensor::from_gguf_bytes(DType::F32, Shape::new(2, 4), f32_bytes(&values))
            .unwrap();
        let tied = w.clone();
        assert_eq!(w.data().as_ptr(), tied.data().as_ptr());
    }

    #[test]
    fn wrong_byte_count_is_rejected() {
        let err = WeightTensor::from_gguf_bytes(DType::F32, Shape::new(2, 2), f32_bytes(&[1.0]))
            .unwrap_err();
        assert!(matches!(err, TensorError::ByteCountMismatch { .. }));
    }
}

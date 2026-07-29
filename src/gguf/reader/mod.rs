mod tables;

pub use tables::{MetadataTable, TensorTable};

use std::path::Path;

use bytes::Bytes;

use crate::{
    gguf::{
        ArrayValue, GgufError, MAGIC, MAX_ARRAY_ELEMENTS, MAX_DIMS, MAX_NAME_LEN, TensorInfo,
        Value, ValueType,
    },
    tensor::WeightTensor,
};

/// A parsed GGUF file. Metadata and tensor infos are read eagerly; tensor
/// data stays in the shared `Bytes` buffer and is handed out zero-copy.
pub struct GgufFile {
    data: Bytes,
    version: u32,
    alignment: u64,
    metadata: MetadataTable,
    tensors: TensorTable,
    /// Absolute offset of the data section.
    data_offset: usize,
}

impl GgufFile {
    #[hotpath::measure]
    pub fn open(path: impl AsRef<Path>) -> Result<GgufFile, GgufError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the map is read-only and geppetto never mutates model files;
        // truncation of the file by another process while mapped is UB we
        // accept, like the C++ mmap loader does.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let raw = Bytes::from_owner(mmap);
        Self::parse(raw)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<GgufFile, GgufError> {
        let raw = Bytes::from(bytes);
        Self::parse(raw)
    }

    fn parse(data: Bytes) -> Result<GgufFile, GgufError> {
        let mut cur = Cursor {
            data: data.as_ref(),
            pos: 0,
        };

        // Check the magic tag
        let magic: [u8; 4] = cur.read_bytes(4)?.try_into().expect("length checked");
        if magic != MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        // Assert the version is correct
        let version = cur.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors = cur.read_u64()?;
        let n_kv = cur.read_u64()?;

        // Populates the metadata
        let capacity = usize::try_from(n_kv).unwrap_or(0).min(1 << 10);
        // n_kv is untrusted: cap the pre-allocation so a hostile header
        // cannot force a huge reservation (real models have < 100 keys)
        let mut metadata = MetadataTable::with_capacity(capacity);
        for _ in 0..n_kv {
            let key = cur.read_string()?;
            let value = cur.read_value(&key)?;
            metadata.insert(key, value)?;
        }

        // Check the alignment from the populated metadata
        let alignment = metadata.alignment()?;

        // Populates the tensor table, checking the layout as it goes
        let capacity = usize::try_from(n_tensors).unwrap_or(0).min(1 << 13);
        // n_tensors is untrusted like n_kv; large models stay in the thousands
        let mut tensors = TensorTable::with_capacity(capacity);
        // running end of the data layout; None once a size is uncomputable
        let mut end: Option<u64> = Some(0);
        for _ in 0..n_tensors {
            let tensor_info = cur.read_tensor_info()?;
            if let Some(expected) = end {
                end = tensor_info.end_offset(expected, alignment)?;
            }
            tensors.insert(tensor_info)?;
        }

        let data_offset = if n_tensors > 0 {
            ((cur.pos as u64).div_ceil(alignment) * alignment) as usize
        } else {
            cur.pos
        };
        if data_offset > data.len() {
            return Err(GgufError::UnexpectedEof {
                offset: cur.pos,
                wanted: data_offset - cur.pos,
            });
        }

        // the one layout rule the loop cannot check: the padded data section
        // must actually be present in the file
        let data_len = (data.len() - data_offset) as u64;
        if let Some(size) = end
            && size > data_len
        {
            return Err(GgufError::UnexpectedEof {
                offset: data.len(),
                wanted: usize::try_from(size - data_len).unwrap_or(usize::MAX),
            });
        }

        Ok(GgufFile {
            version,
            alignment,
            metadata,
            tensors,
            data_offset,
            data,
        })
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn metadata(&self) -> &MetadataTable {
        &self.metadata
    }

    pub fn tensors(&self) -> &TensorTable {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// A tensor's data as an owned, refcount-shared slice into the file,
    /// bounds-checked against it. Zero-copy: the returned `Bytes` shares the
    /// backing buffer and can outlive this `GgufFile`.
    pub fn tensor_data(&self, name: &str) -> Result<Bytes, GgufError> {
        let info = self
            .tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.into()))?;
        let nbytes = info
            .nbytes()
            .ok_or_else(|| GgufError::UnsupportedTensorType {
                name: name.into(),
                type_id: info.type_id,
            })?;
        let start = self
            .data_offset
            .checked_add(usize::try_from(info.offset).ok().ok_or_else(|| oob(name))?)
            .ok_or_else(|| oob(name))?;
        let end = start
            .checked_add(usize::try_from(nbytes).ok().ok_or_else(|| oob(name))?)
            .ok_or_else(|| oob(name))?;
        if end > self.data.len() {
            return Err(oob(name));
        }
        Ok(self.data.slice(start..end))
    }

    /// Binds a tensor as a [`WeightTensor`]: F32 data is borrowed zero-copy
    /// from the file mapping, F16 is widened once. The mapping stays alive as
    /// long as any bound weight does (the `Bytes` refcount owns it).
    pub fn tensor_from(&self, name: &str) -> Result<WeightTensor, GgufError> {
        let info = self
            .tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.into()))?;
        let dtype = info
            .dtype()
            .ok_or_else(|| GgufError::UnsupportedTensorType {
                name: name.into(),
                type_id: info.type_id,
            })?;
        let shape = info.dims_to_shape();
        let bytes = self.tensor_data(name)?;
        let tensor = WeightTensor::from_gguf_bytes(dtype, shape, bytes)?;

        Ok(tensor)
    }

    /// Loads a tensor and checks it is exactly `[rows, cols]`.
    pub fn load(
        &self,
        name: &str,
        want_rows: usize,
        want_cols: usize,
    ) -> Result<WeightTensor, GgufError> {
        let tensor = self.tensor_from(name)?;
        let got_rows = tensor.rows();
        let got_cols = tensor.cols();
        if got_rows != want_rows || got_cols != want_cols {
            return Err(GgufError::ShapeMismatch {
                name: name.into(),
                got_rows,
                got_cols,
                want_rows,
                want_cols,
            });
        }
        Ok(tensor)
    }

    /// Loads a tensor checking only its row width, returning it with whatever row
    /// count the file declares (used for `token_embd`, which fixes `n_vocab`).
    pub fn load_cols(&self, name: &str, want_cols: usize) -> Result<WeightTensor, GgufError> {
        let tensor = self.tensor_from(name)?;
        let got_rows = tensor.rows();
        let got_cols = tensor.cols();
        if got_cols != want_cols {
            return Err(GgufError::ShapeMismatch {
                name: name.into(),
                got_rows,
                got_cols,
                want_rows: got_rows,
                want_cols,
            });
        }
        Ok(tensor)
    }
}

fn oob(name: &str) -> GgufError {
    GgufError::TensorOutOfBounds { name: name.into() }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let bytes = self
            .data
            .get(
                self.pos..self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof {
                    offset: self.pos,
                    wanted: n,
                })?,
            )
            .ok_or(GgufError::UnexpectedEof {
                offset: self.pos,
                wanted: n,
            })?;
        self.pos += n;
        Ok(bytes)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        Ok(self.read_bytes(N)?.try_into().expect("length checked"))
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(GgufError::InvalidBool(b)),
        }
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        let len = usize::try_from(len).map_err(|_| GgufError::UnexpectedEof {
            offset: self.pos,
            wanted: usize::MAX,
        })?;
        let offset = self.pos;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8 { offset })
    }

    fn read_value(&mut self, key: &str) -> Result<Value, GgufError> {
        let type_id = self.read_u32()?;
        let vtype = ValueType::from_id(type_id).ok_or(GgufError::InvalidValueType(type_id))?;
        if vtype != ValueType::Array {
            return self.read_scalar(vtype);
        }

        let elem_id = self.read_u32()?;
        let elem = ValueType::from_id(elem_id).ok_or(GgufError::InvalidValueType(elem_id))?;
        if elem == ValueType::Array {
            return Err(GgufError::NestedArray(key.into()));
        }
        let n = self.read_u64()?;
        if n > MAX_ARRAY_ELEMENTS {
            return Err(GgufError::ArrayTooLong {
                key: key.into(),
                len: n,
            });
        }
        let n = n as usize;

        macro_rules! read_vec {
            ($variant:ident, $read:ident) => {{
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.$read()?);
                }
                ArrayValue::$variant(v)
            }};
        }
        let arr = match elem {
            ValueType::U8 => read_vec!(U8, read_u8),
            ValueType::I8 => read_vec!(I8, read_i8),
            ValueType::U16 => read_vec!(U16, read_u16),
            ValueType::I16 => read_vec!(I16, read_i16),
            ValueType::U32 => read_vec!(U32, read_u32),
            ValueType::I32 => read_vec!(I32, read_i32),
            ValueType::F32 => read_vec!(F32, read_f32),
            ValueType::Bool => read_vec!(Bool, read_bool),
            ValueType::String => read_vec!(String, read_string),
            ValueType::U64 => read_vec!(U64, read_u64),
            ValueType::I64 => read_vec!(I64, read_i64),
            ValueType::F64 => read_vec!(F64, read_f64),
            ValueType::Array => unreachable!("rejected above"),
        };
        Ok(Value::Array(arr))
    }

    fn read_scalar(&mut self, vtype: ValueType) -> Result<Value, GgufError> {
        Ok(match vtype {
            ValueType::U8 => Value::U8(self.read_u8()?),
            ValueType::I8 => Value::I8(self.read_i8()?),
            ValueType::U16 => Value::U16(self.read_u16()?),
            ValueType::I16 => Value::I16(self.read_i16()?),
            ValueType::U32 => Value::U32(self.read_u32()?),
            ValueType::I32 => Value::I32(self.read_i32()?),
            ValueType::F32 => Value::F32(self.read_f32()?),
            ValueType::Bool => Value::Bool(self.read_bool()?),
            ValueType::String => Value::String(self.read_string()?),
            ValueType::U64 => Value::U64(self.read_u64()?),
            ValueType::I64 => Value::I64(self.read_i64()?),
            ValueType::F64 => Value::F64(self.read_f64()?),
            ValueType::Array => unreachable!("handled by read_value"),
        })
    }

    fn read_tensor_info(&mut self) -> Result<TensorInfo, GgufError> {
        let name = self.read_string()?;
        if name.len() >= MAX_NAME_LEN {
            return Err(GgufError::NameTooLong(name.len()));
        }
        let n_dims = self.read_u32()?;
        if n_dims as usize > MAX_DIMS {
            return Err(GgufError::TooManyDims { name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut n_elements: u64 = 1;
        for _ in 0..n_dims {
            let d = self.read_u64()?;
            n_elements = match n_elements.checked_mul(d) {
                Some(n) if n <= i64::MAX as u64 && d <= i64::MAX as u64 => n,
                _ => return Err(GgufError::ElementsOverflow { name }),
            };
            dims.push(d);
        }
        let type_id = self.read_u32()?;
        let offset = self.read_u64()?;
        let info = TensorInfo {
            name,
            dims,
            type_id,
            offset,
        };

        if let Some(dtype) = info.dtype() {
            // surface a PartialBlock error now; nbytes() folds it into None
            crate::tensor::row_size(dtype, info.dims.first().copied().unwrap_or(1) as usize)?;
            if info.nbytes().is_none() {
                return Err(GgufError::SizeOverflow { name: info.name });
            }
        }
        Ok(info)
    }
}

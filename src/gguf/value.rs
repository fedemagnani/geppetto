/// Metadata value type identifiers, as encoded in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub const fn from_id(id: u32) -> Option<ValueType> {
        Some(match id {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            _ => return None,
        })
    }

    pub const fn id(self) -> u32 {
        match self {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }
}

/// A single metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(ArrayValue),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// A homogeneous metadata array. Arrays of arrays are not supported, matching
/// the C++ reader.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValue {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl Value {
    pub const fn value_type(&self) -> ValueType {
        match self {
            Value::U8(_) => ValueType::U8,
            Value::I8(_) => ValueType::I8,
            Value::U16(_) => ValueType::U16,
            Value::I16(_) => ValueType::I16,
            Value::U32(_) => ValueType::U32,
            Value::I32(_) => ValueType::I32,
            Value::F32(_) => ValueType::F32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_) => ValueType::Array,
            Value::U64(_) => ValueType::U64,
            Value::I64(_) => ValueType::I64,
            Value::F64(_) => ValueType::F64,
        }
    }

    /// Lossless widening: any unsigned type that fits in u32.
    pub const fn as_u32(&self) -> Option<u32> {
        match *self {
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            _ => None,
        }
    }

    /// Lossless widening: any unsigned type.
    pub const fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            _ => None,
        }
    }

    /// Lossless widening: any signed type that fits in i32.
    pub const fn as_i32(&self) -> Option<i32> {
        match *self {
            Value::I8(v) => Some(v as i32),
            Value::I16(v) => Some(v as i32),
            Value::I32(v) => Some(v),
            _ => None,
        }
    }

    pub const fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            _ => None,
        }
    }

    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(v) => Some(v),
            _ => None,
        }
    }

    pub const fn as_array(&self) -> Option<&ArrayValue> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }
}

impl ArrayValue {
    pub const fn elem_type(&self) -> ValueType {
        match self {
            ArrayValue::U8(_) => ValueType::U8,
            ArrayValue::I8(_) => ValueType::I8,
            ArrayValue::U16(_) => ValueType::U16,
            ArrayValue::I16(_) => ValueType::I16,
            ArrayValue::U32(_) => ValueType::U32,
            ArrayValue::I32(_) => ValueType::I32,
            ArrayValue::F32(_) => ValueType::F32,
            ArrayValue::Bool(_) => ValueType::Bool,
            ArrayValue::String(_) => ValueType::String,
            ArrayValue::U64(_) => ValueType::U64,
            ArrayValue::I64(_) => ValueType::I64,
            ArrayValue::F64(_) => ValueType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ArrayValue::U8(v) => v.len(),
            ArrayValue::I8(v) => v.len(),
            ArrayValue::U16(v) => v.len(),
            ArrayValue::I16(v) => v.len(),
            ArrayValue::U32(v) => v.len(),
            ArrayValue::I32(v) => v.len(),
            ArrayValue::F32(v) => v.len(),
            ArrayValue::Bool(v) => v.len(),
            ArrayValue::String(v) => v.len(),
            ArrayValue::U64(v) => v.len(),
            ArrayValue::I64(v) => v.len(),
            ArrayValue::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_str_slice(&self) -> Option<&[String]> {
        match self {
            ArrayValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        match self {
            ArrayValue::U32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        match self {
            ArrayValue::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            ArrayValue::F32(v) => Some(v),
            _ => None,
        }
    }
}

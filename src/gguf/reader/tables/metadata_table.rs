use super::table::{DuplicateKey, Table};
use crate::gguf::{ArrayValue, GgufError, Value};

/// The metadata key-value store of a GGUF file. Wraps [`Table`], mapping
/// duplicate keys to [`GgufError::DuplicateKey`].
#[derive(Debug, Default)]
pub struct MetadataTable(Table<Value>);

impl MetadataTable {
    /// Key that overrides the data-section alignment.
    pub const ALIGNMENT_KEY: &'static str = "general.alignment";
    pub const DEFAULT_ALIGNMENT: u64 = 32;

    pub fn new() -> MetadataTable {
        MetadataTable::default()
    }

    pub fn with_capacity(n: usize) -> MetadataTable {
        MetadataTable(Table::with_capacity(n))
    }

    pub fn insert(&mut self, key: String, value: Value) -> Result<(), GgufError> {
        self.0
            .insert(key, value)
            .map_err(|DuplicateKey(key)| GgufError::DuplicateKey(key))
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// Entries in file order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The data-section alignment: `general.alignment` when present (must be
    /// a u32 power of two), [`Metadata::DEFAULT_ALIGNMENT`] otherwise.
    pub fn alignment(&self) -> Result<u64, GgufError> {
        let Some(value) = self.get(Self::ALIGNMENT_KEY) else {
            return Ok(Self::DEFAULT_ALIGNMENT);
        };
        let Some(alignment) = value.as_u32() else {
            return Err(GgufError::TypeMismatch {
                key: Self::ALIGNMENT_KEY.into(),
                expected: "u32",
                found: value.value_type(),
            });
        };
        let alignment = u64::from(alignment);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }
        Ok(alignment)
    }

    // typed getters, mirroring `llama_model_loader::get_key`; unsigned
    // integer getters widen losslessly
    pub fn get_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.get_typed(key, "u32", Value::as_u32)
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, GgufError> {
        self.get_typed(key, "u64", Value::as_u64)
    }

    pub fn get_i32(&self, key: &str) -> Result<i32, GgufError> {
        self.get_typed(key, "i32", Value::as_i32)
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.get_typed(key, "f32", Value::as_f32)
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, GgufError> {
        self.get_typed(key, "bool", Value::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Result<&str, GgufError> {
        self.get_typed(key, "string", Value::as_str)
    }

    pub fn get_arr_str(&self, key: &str) -> Result<&[String], GgufError> {
        self.get_arr_typed(key, "string array", ArrayValue::as_str_slice)
    }

    pub fn get_arr_u32(&self, key: &str) -> Result<&[u32], GgufError> {
        self.get_arr_typed(key, "u32 array", ArrayValue::as_u32_slice)
    }

    pub fn get_arr_i32(&self, key: &str) -> Result<&[i32], GgufError> {
        self.get_arr_typed(key, "i32 array", ArrayValue::as_i32_slice)
    }

    pub fn get_arr_f32(&self, key: &str) -> Result<&[f32], GgufError> {
        self.get_arr_typed(key, "f32 array", ArrayValue::as_f32_slice)
    }

    /// Shared body of the typed getters: key lookup, then conversion, with a
    /// typed error for each failure.
    fn get_typed<'a, T>(
        &'a self,
        key: &str,
        expected: &'static str,
        as_fn: impl Fn(&'a Value) -> Option<T>,
    ) -> Result<T, GgufError> {
        let value = self
            .get(key)
            .ok_or_else(|| GgufError::KeyNotFound(key.into()))?;
        as_fn(value).ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected,
            found: value.value_type(),
        })
    }

    /// Like [`Self::get_typed`], for elements of array values; mismatches
    /// report the array's element type.
    fn get_arr_typed<'a, T>(
        &'a self,
        key: &str,
        expected: &'static str,
        as_fn: impl Fn(&'a ArrayValue) -> Option<T>,
    ) -> Result<T, GgufError> {
        let arr = self.get_typed(key, "array", Value::as_array)?;
        as_fn(arr).ok_or_else(|| GgufError::TypeMismatch {
            key: key.into(),
            expected,
            found: arr.elem_type(),
        })
    }
}

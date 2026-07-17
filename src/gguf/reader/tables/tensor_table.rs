use super::table::{DuplicateKey, Table};
use crate::gguf::{GgufError, TensorInfo};

/// The tensor table of a GGUF file, keyed by tensor name. Wraps [`Table`],
/// mapping duplicate names to [`GgufError::DuplicateTensor`]. Layout
/// validation is the reader's concern, not the table's: see
/// [`TensorInfo::end_offset`].
#[derive(Debug, Default)]
pub struct TensorTable(Table<TensorInfo>);

impl TensorTable {
    pub fn with_capacity(n: usize) -> TensorTable {
        TensorTable(Table::with_capacity(n))
    }

    pub fn insert(&mut self, info: TensorInfo) -> Result<(), GgufError> {
        self.0
            .insert(info.name.clone(), info)
            .map_err(|DuplicateKey(name)| GgufError::DuplicateTensor(name))
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.0.get(name)
    }

    /// Tensor infos in file order.
    pub fn values(&self) -> impl Iterator<Item = &TensorInfo> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

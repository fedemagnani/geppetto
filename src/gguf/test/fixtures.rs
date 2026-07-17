//! Test-only GGUF writer: builds fixture files in memory. Kept minimal; it
//! graduates to a real writer module when the HF import epoch needs one.

use crate::gguf::{ArrayValue, MAGIC, MetadataTable, Value};

pub struct FixtureTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub data: Vec<u8>,
    /// Overrides the computed sequential offset, to craft malformed layouts.
    pub offset_override: Option<u64>,
}

pub struct FixtureBuilder {
    version: u32,
    alignment: Option<u32>,
    kvs: Vec<(String, Value)>,
    /// Serialized verbatim after `kvs`, counted in `n_kv`; lets tests craft
    /// entries the typed API refuses to build (e.g. nested arrays).
    raw_kvs: Vec<Vec<u8>>,
    tensors: Vec<FixtureTensor>,
}

impl FixtureBuilder {
    pub fn new() -> FixtureBuilder {
        FixtureBuilder {
            version: 3,
            alignment: None,
            kvs: Vec::new(),
            raw_kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Emits a `general.alignment` key; also used to compute tensor offsets.
    pub fn alignment(mut self, alignment: u32) -> Self {
        self.alignment = Some(alignment);
        self
    }

    pub fn kv(mut self, key: &str, value: Value) -> Self {
        self.kvs.push((key.into(), value));
        self
    }

    pub fn raw_kv(mut self, bytes: Vec<u8>) -> Self {
        self.raw_kvs.push(bytes);
        self
    }

    pub fn tensor(self, name: &str, dims: &[u64], type_id: u32, data: Vec<u8>) -> Self {
        self.push_tensor(name, dims, type_id, data, None)
    }

    pub fn tensor_at(
        self,
        name: &str,
        dims: &[u64],
        type_id: u32,
        data: Vec<u8>,
        offset: u64,
    ) -> Self {
        self.push_tensor(name, dims, type_id, data, Some(offset))
    }

    fn push_tensor(
        mut self,
        name: &str,
        dims: &[u64],
        type_id: u32,
        data: Vec<u8>,
        offset_override: Option<u64>,
    ) -> Self {
        self.tensors.push(FixtureTensor {
            name: name.into(),
            dims: dims.to_vec(),
            type_id,
            data,
            offset_override,
        });
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let alignment = u64::from(
            self.alignment
                .unwrap_or(MetadataTable::DEFAULT_ALIGNMENT as u32),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        w_u32(&mut out, self.version);
        w_u64(&mut out, self.tensors.len() as u64);
        let n_kv = self.kvs.len() + self.raw_kvs.len() + usize::from(self.alignment.is_some());
        w_u64(&mut out, n_kv as u64);

        if let Some(a) = self.alignment {
            w_string(&mut out, MetadataTable::ALIGNMENT_KEY);
            w_u32(&mut out, 4); // u32 type id
            w_u32(&mut out, a);
        }
        for (key, value) in &self.kvs {
            w_string(&mut out, key);
            w_value(&mut out, value);
        }
        for raw in &self.raw_kvs {
            out.extend_from_slice(raw);
        }

        // sequential offsets, each tensor padded to the alignment
        let mut offset: u64 = 0;
        for t in &self.tensors {
            w_string(&mut out, &t.name);
            w_u32(&mut out, t.dims.len() as u32);
            for &d in &t.dims {
                w_u64(&mut out, d);
            }
            w_u32(&mut out, t.type_id);
            w_u64(&mut out, t.offset_override.unwrap_or(offset));
            offset += (t.data.len() as u64).div_ceil(alignment) * alignment;
        }

        if !self.tensors.is_empty() {
            let data_start = (out.len() as u64).div_ceil(alignment) * alignment;
            out.resize(data_start as usize, 0);
            for t in &self.tensors {
                out.extend_from_slice(&t.data);
                let padded = (out.len() as u64).div_ceil(alignment) * alignment;
                out.resize(padded as usize, 0);
            }
        }
        out
    }
}

pub fn w_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn w_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn w_string(out: &mut Vec<u8>, s: &str) {
    w_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

fn w_value(out: &mut Vec<u8>, value: &Value) {
    w_u32(out, value.value_type().id());
    match value {
        Value::U8(v) => out.push(*v),
        Value::I8(v) => out.push(*v as u8),
        Value::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::Bool(v) => out.push(u8::from(*v)),
        Value::String(v) => w_string(out, v),
        Value::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        Value::Array(arr) => {
            w_u32(out, arr.elem_type().id());
            w_u64(out, arr.len() as u64);
            match arr {
                ArrayValue::U8(v) => out.extend_from_slice(v),
                ArrayValue::I8(v) => v.iter().for_each(|x| out.push(*x as u8)),
                ArrayValue::U16(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::I16(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::U32(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::I32(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::F32(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::Bool(v) => v.iter().for_each(|x| out.push(u8::from(*x))),
                ArrayValue::String(v) => v.iter().for_each(|x| w_string(out, x)),
                ArrayValue::U64(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::I64(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
                ArrayValue::F64(v) => v
                    .iter()
                    .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            }
        }
    }
}

use std::collections::HashMap;

/// Insertion-ordered collection indexed by string key: preserves file order
/// for display, offers O(1) lookup, and rejects duplicate keys on insert.
#[derive(Debug)]
pub struct Table<V> {
    entries: Vec<(String, V)>,
    index: HashMap<String, usize>,
}

/// Rejected insert: the key is already present. Carries the key so callers
/// can build their own error.
#[derive(Debug, PartialEq, Eq)]
pub struct DuplicateKey(pub String);

impl<V> Default for Table<V> {
    fn default() -> Table<V> {
        Table {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }
}

impl<V> Table<V> {
    pub fn with_capacity(n: usize) -> Table<V> {
        Table {
            entries: Vec::with_capacity(n),
            index: HashMap::with_capacity(n),
        }
    }

    pub fn insert(&mut self, key: String, value: V) -> Result<(), DuplicateKey> {
        if self.index.contains_key(&key) {
            return Err(DuplicateKey(key));
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.index.get(key).map(|&i| &self.entries[i].1)
    }

    /// Entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &V)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Values in insertion order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

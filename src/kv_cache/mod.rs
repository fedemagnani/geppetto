//! Single-sequence key/value cache: one contiguous K and V buffer per layer,
//! each preallocated to `n_ctx` positions. No paging, no multi-sequence cells,
//! no defrag -- that is epoch 9 territory.

use crate::tensor::Tensor;

pub struct KvCache {
    n_embd: usize,
    n_ctx: usize,
    /// Per layer, `[stored, n_embd]` row-major keys and values.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layer: usize, n_embd: usize, n_ctx: usize) -> KvCache {
        let buffers = || {
            (0..n_layer)
                .map(|_| Vec::with_capacity(n_ctx * n_embd))
                .collect()
        };
        KvCache {
            n_embd,
            n_ctx,
            keys: buffers(),
            values: buffers(),
        }
    }

    /// Positions currently stored. Every layer holds the same count, so the
    /// forward pass reads this once as `n_past` before appending. Layer 0 is
    /// the reference; empty when there are no layers.
    pub fn len(&self) -> usize {
        self.keys.first().map_or(0, |buf| buf.len() / self.n_embd)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends one step's keys and values for `layer`, both `[n_new, n_embd]`.
    pub fn append(&mut self, layer: usize, keys: &Tensor, values: &Tensor) {
        assert_eq!(keys.cols(), self.n_embd, "kv append: key width");
        assert_eq!(values.cols(), self.n_embd, "kv append: value width");
        assert_eq!(keys.rows(), values.rows(), "kv append: key/value row count");
        assert!(
            self.stored(layer) + keys.rows() <= self.n_ctx,
            "kv cache overflow: {} + {} > {}",
            self.stored(layer),
            keys.rows(),
            self.n_ctx,
        );

        self.keys[layer].extend_from_slice(keys.data());
        self.values[layer].extend_from_slice(values.data());
    }

    /// Resembles the number of rows of the stored keys associated with position `layer`
    fn stored(&self, layer: usize) -> usize {
        self.keys[layer].len() / self.n_embd
    }

    /// The layer's stored keys as `[stored, n_embd]`, row-major.
    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    /// The layer's stored values as `[stored, n_embd]`, row-major.
    pub fn values(&self, layer: usize) -> &[f32] {
        &self.values[layer]
    }

    /// Rolls every layer back to `n` stored positions, reusing the allocation.
    pub fn truncate(&mut self, n: usize) {
        for buf in self.keys.iter_mut().chain(self.values.iter_mut()) {
            buf.truncate(n * self.n_embd);
        }
    }

    /// Clears every layer, so that `keys` and `values` become vectors of empty vectors.
    pub fn clear(&mut self) {
        self.truncate(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    fn rows(n: usize, embd: usize, fill: f32) -> Tensor {
        Tensor::new(Shape::new(n, embd), vec![fill; n * embd])
    }

    #[test]
    fn appends_grow_the_stored_length() {
        let mut cache = KvCache::new(2, 4, 16);
        assert_eq!(cache.len(), 0);
        cache.append(0, &rows(3, 4, 1.0), &rows(3, 4, 2.0));
        cache.append(1, &rows(3, 4, 3.0), &rows(3, 4, 4.0));
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.keys(0).len(), 12);
        assert_eq!(cache.keys(0)[0], 1.0);
        assert_eq!(cache.values(1)[0], 4.0);
    }

    #[test]
    fn incremental_appends_concatenate() {
        let mut cache = KvCache::new(1, 2, 16);
        cache.append(0, &rows(1, 2, 1.0), &rows(1, 2, 1.0));
        cache.append(0, &rows(1, 2, 2.0), &rows(1, 2, 2.0));
        assert_eq!(cache.keys(0), &[1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn truncate_and_clear_roll_back() {
        let mut cache = KvCache::new(1, 2, 16);
        cache.append(0, &rows(3, 2, 1.0), &rows(3, 2, 1.0));
        cache.truncate(1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.keys(0), &[1.0, 1.0]);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn appending_past_capacity_panics() {
        let mut cache = KvCache::new(1, 2, 2);
        cache.append(0, &rows(3, 2, 1.0), &rows(3, 2, 1.0));
    }
}

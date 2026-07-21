use crate::tensor::{Shape, Tensor};

impl Tensor {
    /// Gathers rows `ids` from `self` (an `[n_rows, cols]` matrix), producing
    /// `[ids.len(), cols]`. This is `ggml_get_rows`: the embedding lookup that
    /// turns token ids into rows of the embedding table, and also how the graph
    /// selects the output positions to keep.
    #[hotpath::measure]
    pub fn get_rows(&self, ids: &[u32]) -> Tensor {
        let cols = self.cols();
        let n_rows = self.rows();
        let mut out = vec![0.0f32; ids.len() * cols];
        for (dst, &id) in out.chunks_mut(cols).zip(ids) {
            let id = id as usize;
            assert!(
                id < n_rows,
                "get_rows: id {id} out of range for {n_rows} rows"
            );
            dst.copy_from_slice(self.row(id));
        }
        Tensor::new(Shape::new(ids.len(), cols), out)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn selects_and_reorders_rows() {
        let table = Tensor::new(Shape::new(3, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = table.get_rows(&[2, 0, 2]);
        assert_eq!(out.shape(), Shape::new(3, 2));
        assert_eq!(out.data(), &[4.0, 5.0, 0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn empty_id_list_yields_no_rows() {
        let table = Tensor::new(Shape::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let out = table.get_rows(&[]);
        assert_eq!(out.shape(), Shape::new(0, 2));
        assert!(out.data().is_empty());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_id_panics() {
        let table = Tensor::new(Shape::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let _ = table.get_rows(&[2]);
    }
}

#[cfg(test)]
use crate::tensor::Tensor;
use crate::tensor::{Shape, TensorView, TensorViewMut};

/// Gathers rows `ids` from `table` (an `[n_rows, cols]` matrix) into `out`
/// (`[ids.len(), cols]`, fully overwritten). This is `ggml_get_rows`: the
/// embedding lookup that turns token ids into rows of the embedding table,
/// and also how the graph selects the output positions to keep.
#[hotpath::measure]
pub fn get_rows(table: TensorView, ids: &[u32], mut out: TensorViewMut) {
    let n_rows = table.rows();
    let expected = Shape::new(ids.len(), table.cols());
    assert_eq!(out.shape(), expected, "get_rows: out shape mismatch");
    for (r, &id) in ids.iter().enumerate() {
        let id = id as usize;
        assert!(
            id < n_rows,
            "get_rows: id {id} out of range for {n_rows} rows"
        );
        out.row_mut(r).copy_from_slice(table.row(id));
    }
}

#[cfg(test)]
impl Tensor {
    /// Gathers rows `ids` from `self` into a fresh tensor, delegating to
    /// [`get_rows`].
    pub fn get_rows(&self, ids: &[u32]) -> Tensor {
        let mut out = Tensor::zeros(Shape::new(ids.len(), self.cols()));
        get_rows(self.as_view(), ids, out.as_view_mut());
        out
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

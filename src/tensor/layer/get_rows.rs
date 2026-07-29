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
mod tests {
    use crate::tensor::layer::get_rows;
    use crate::tensor::{Shape, TensorView, TensorViewMut};

    fn gather(table: &[f32], rows: usize, cols: usize, ids: &[u32]) -> Vec<f32> {
        let mut out = vec![0.0f32; ids.len() * cols];
        get_rows(
            TensorView::contiguous(table, Shape::new(rows, cols)),
            ids,
            TensorViewMut::contiguous(&mut out, Shape::new(ids.len(), cols)),
        );
        out
    }

    #[test]
    fn selects_and_reorders_rows() {
        let table = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = gather(&table, 3, 2, &[2, 0, 2]);
        assert_eq!(out, &[4.0, 5.0, 0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn empty_id_list_yields_no_rows() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let out = gather(&table, 2, 2, &[]);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn out_of_range_id_panics() {
        let table = [1.0, 2.0, 3.0, 4.0];
        let _ = gather(&table, 2, 2, &[2]);
    }
}

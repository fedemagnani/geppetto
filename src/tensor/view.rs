//! Borrowed 2D views over f32 data: the operand and output types the op
//! drivers consume. A view is `(data, shape, row_stride)` -- `rows` rows of
//! `cols` elements whose starts sit `row_stride` apart. `row_stride == cols`
//! is the contiguous common case; a larger stride mounts a column block of a
//! wider buffer (one attention head of a fused QKV activation) without
//! copying. Views never own or allocate: the data lives in the arena or the
//! mmap'd weights, and a view is mounted just in time for each call.

use crate::tensor::Shape;

/// Shared borrowed view; `Copy`, passed by value.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape,
    row_stride: usize,
}

/// Exclusive borrowed view, for op outputs and in-place ops. Passed by value
/// and consumed; use [`TensorViewMut::reborrow`] to keep the original usable.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    data: &'a mut [f32],
    shape: Shape,
    row_stride: usize,
}

/// The strided-view invariant, checked once at mount time: every row is
/// `cols` long, row starts are `row_stride` apart, and the last row has no
/// trailing gap (so `data` is exactly spanned).
fn check(len: usize, shape: Shape, row_stride: usize) {
    assert!(
        row_stride >= shape.cols(),
        "view: row_stride {row_stride} shorter than a row of {shape}"
    );
    assert_eq!(
        len,
        shape.strided_len(row_stride),
        "view: data length {len} does not fit {shape} with row_stride {row_stride}"
    );
}

impl<'a> TensorView<'a> {
    /// Contiguous view: rows packed back to back.
    pub fn contiguous(data: &'a [f32], shape: Shape) -> TensorView<'a> {
        TensorView::strided(data, shape, shape.cols())
    }

    /// Strided view over a column block of a wider buffer.
    pub fn strided(data: &'a [f32], shape: Shape, row_stride: usize) -> TensorView<'a> {
        check(data.len(), shape, row_stride);
        TensorView {
            data,
            shape,
            row_stride,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    pub fn is_contiguous(&self) -> bool {
        self.row_stride == self.shape.cols()
    }

    /// Row `r` as a contiguous slice of `cols` elements.
    pub fn row(&self, r: usize) -> &'a [f32] {
        let start = r * self.row_stride;
        &self.data[start..start + self.shape.cols()]
    }
}

impl<'a> TensorViewMut<'a> {
    /// Contiguous view: rows packed back to back.
    pub fn contiguous(data: &'a mut [f32], shape: Shape) -> TensorViewMut<'a> {
        TensorViewMut::strided(data, shape, shape.cols())
    }

    /// Strided view over a column block of a wider buffer.
    pub fn strided(data: &'a mut [f32], shape: Shape, row_stride: usize) -> TensorViewMut<'a> {
        check(data.len(), shape, row_stride);
        TensorViewMut {
            data,
            shape,
            row_stride,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape.rows()
    }

    pub fn cols(&self) -> usize {
        self.shape.cols()
    }

    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    pub fn is_contiguous(&self) -> bool {
        self.row_stride == self.shape.cols()
    }

    /// Row `r` as a contiguous slice of `cols` elements.
    pub fn row(&self, r: usize) -> &[f32] {
        let start = r * self.row_stride;
        &self.data[start..start + self.shape.cols()]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.row_stride;
        &mut self.data[start..start + self.shape.cols()]
    }

    /// A shared view of the same data, for ops that read their output operand.
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data,
            shape: self.shape,
            row_stride: self.row_stride,
        }
    }

    /// A shorter-lived exclusive view of the same data, so the original
    /// survives being passed by value to an op.
    pub fn reborrow(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            data: self.data,
            shape: self.shape,
            row_stride: self.row_stride,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_rows_round_trip() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = TensorView::contiguous(&data, Shape::new(2, 3));
        assert!(v.is_contiguous());
        assert_eq!(v.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(v.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn strided_view_mounts_a_column_block() {
        // a [2, 6] fused buffer; mount the width-2 block at column 2
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let block = &data[2..10];
        let v = TensorView::strided(block, Shape::new(2, 2), 6);
        assert!(!v.is_contiguous());
        assert_eq!(v.row(0), &[2.0, 3.0]);
        assert_eq!(v.row(1), &[8.0, 9.0]);
    }

    #[test]
    fn row_mut_writes_through_the_stride() {
        let mut data: Vec<f32> = vec![0.0; 12];
        let block = &mut data[2..10];
        let mut v = TensorViewMut::strided(block, Shape::new(2, 2), 6);
        v.row_mut(0).fill(1.0);
        v.row_mut(1).fill(2.0);
        let expected = [
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 2.0, 0.0, 0.0,
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn as_view_and_reborrow_preserve_the_layout() {
        let mut data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let slice = &mut data[1..7];
        let mut v = TensorViewMut::strided(slice, Shape::new(2, 2), 4);
        let r = v.reborrow();
        assert_eq!(r.row_stride(), 4);
        let shared = v.as_view();
        assert_eq!(shared.row(1), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn length_mismatch_panics() {
        let data = [0.0; 5];
        let _ = TensorView::contiguous(&data, Shape::new(2, 3));
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn trailing_gap_after_last_row_panics() {
        // 2 rows of 2 at stride 4 span 6 elements, not 8
        let data = [0.0; 8];
        let _ = TensorView::strided(&data, Shape::new(2, 2), 4);
    }

    #[test]
    #[should_panic(expected = "shorter than a row")]
    fn stride_shorter_than_a_row_panics() {
        let data = [0.0; 6];
        let _ = TensorView::strided(&data, Shape::new(2, 3), 2);
    }
}

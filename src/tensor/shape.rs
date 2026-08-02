use std::fmt;

/// The shape of a 2D, row-major tensor: `rows` vectors each of `cols`
/// elements. Both are non-zero by construction, so every `Shape` is a real
/// rectangle and no consumer handles a degenerate one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

// `len` can never be 0, so there is no `is_empty` to pair it with
#[allow(clippy::len_without_is_empty)]
impl Shape {
    pub const fn new(rows: usize, cols: usize) -> Shape {
        assert!(
            rows > 0 && cols > 0,
            "shape: rows and cols must be non-zero"
        );
        Shape { rows, cols }
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Total number of elements; the length the backing `Vec<f32>` must have.
    pub const fn len(self) -> usize {
        self.rows * self.cols
    }

    /// Length of the shortest subslice holding the matrix with `self.rows` rows
    /// and `self.cols` columns such that moving from one row to the next one of
    /// requires jumping `row_stride` elements of the subslice
    pub const fn strided_len(self, row_stride: usize) -> usize {
        (self.rows - 1) * row_stride + self.cols
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_len_equals_len_when_contiguous() {
        let s = Shape::new(3, 4);
        assert_eq!(s.strided_len(4), s.len());
    }

    #[test]
    fn strided_len_spans_to_the_last_rows_end() {
        // 2 rows of 2 at stride 6: row 1 starts at 6, ends at 8
        assert_eq!(Shape::new(2, 2).strided_len(6), 8);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_rows_are_rejected() {
        let _ = Shape::new(0, 5);
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn zero_cols_are_rejected() {
        let _ = Shape::new(3, 0);
    }
}

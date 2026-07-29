use std::fmt;

/// The shape of a 2D, row-major tensor: `rows` vectors each of `cols`
/// elements. Everything GPT-2 touches at this epoch is 2D -- a linear weight
/// is `[n_out, n_in]`, an activation is `[n_tokens, n_embd]` -- so a fixed 2D
/// shape keeps indexing simple and total. Higher-rank views (splitting the
/// embedding into attention heads) belong to the forward pass, not here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl Shape {
    pub const fn new(rows: usize, cols: usize) -> Shape {
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

    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.rows, self.cols)
    }
}

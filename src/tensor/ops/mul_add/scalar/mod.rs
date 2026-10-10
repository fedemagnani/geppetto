//! The scalar [`MulAdd`](super::MulAdd) strategies, one per file: the
//! entire experiment is whether the multiply and the add fuse.

mod fma;
mod unfused;

pub use fma::Fma;
pub use unfused::Unfused;

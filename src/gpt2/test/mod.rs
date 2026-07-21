//! Test support: an in-memory tiny GPT-2 (seeded random weights) serialized
//! through the epoch-1 GGUF writer, and the behavioral test suite.

mod tests;
/// Visible crate-wide: the generation tests drive this same tiny model.
pub(crate) mod tiny;

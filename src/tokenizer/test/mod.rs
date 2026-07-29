//! Behavioral tests. Pure unit tests live next to their modules; everything
//! here drives the real GPT-2 vocab, a fixture checked into the parent
//! llama.cpp repo (`models/ggml-vocab-gpt-2.gguf` and its `.inp`/`.out`
//! token vectors, shared with the C++ `test-tokenizer-0`).

mod tests;

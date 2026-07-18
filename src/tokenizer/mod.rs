//! GPT-2 byte-level BPE tokenizer.
//!
//! Reference: the gpt2 paths of `src/llama-vocab.cpp` in llama.cpp. The
//! pipeline is: pre-tokenize the raw text with the GPT-2 split regex, map
//! each pre-token's UTF-8 bytes straight to the vocab ids of their
//! single-character tokens, then run BPE merges over those ids. Decoding
//! goes token string -> byte map inverse -> raw bytes, so encode/decode is
//! lossless at the byte level.

mod error;
mod pretokenizer;
mod vocab;

#[cfg(test)]
mod test;

pub use error::TokenizerError;
pub use pretokenizer::{GPT2_SPLIT_PATTERN, Pretokenizer};
pub use vocab::{ByteCharBmap, TextTokenIdBmap, Vocab};

use crate::gguf::GgufFile;

/// A vocab token id: the token's position in the vocab list.
pub type TokenId = u32;

pub struct Tokenizer {
    vocab: Vocab,
    pretokenizer: Pretokenizer,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Result<Tokenizer, TokenizerError> {
        Ok(Tokenizer {
            vocab,
            pretokenizer: Pretokenizer::gpt2()?,
        })
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Tokenizer, TokenizerError> {
        Self::new(Vocab::from_gguf(file)?)
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// [`Self::encode`] appending to a caller-owned buffer, which is not
    /// cleared. Allocation-free beyond the buffer's own growth: each
    /// pre-token's byte ids are appended and BPE-merged in place on the tail.
    pub fn encode_into(&self, text: &str, out: &mut Vec<TokenId>) -> Result<(), TokenizerError> {
        for piece in self.pretokenizer.split_iter(text) {
            let piece = piece?;
            // this will define the starting index from which the new tail will be appended
            let from = out.len();
            // cast each byte of the substringo into a token id
            let piece_ids = piece.as_bytes().iter().map(|&b| self.vocab.byte_to_id(b));
            // the tail of this buffer now has one token per byte
            out.extend(piece_ids);
            self.vocab.apply_bpe_merges(out, from);
        }
        Ok(())
    }

    /// [`Self::decode`] appending to a caller-owned buffer, which is not
    /// cleared.
    pub fn decode_into(&self, ids: &[TokenId], out: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &id in ids {
            let bytes = self.vocab.id_to_bytes(id);
            let bytes = bytes.ok_or(TokenizerError::IdOutOfRange {
                id,
                n_tokens: self.vocab.n_tokens(),
            })?;
            out.extend_from_slice(bytes);
        }
        Ok(())
    }

    /// Token ids of `text`. No BOS/EOS is added and no special-token text is
    /// recognized, matching llama.cpp's gpt2 defaults.
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let mut ids = Vec::new();
        self.encode_into(text, &mut ids)?;
        Ok(ids)
    }

    /// The bytes `ids` stand for. Byte-level BPE can split multi-byte
    /// characters across tokens, so the byte buffer is the honest output;
    /// [`Self::decode_lossy`] is the string-typed convenience.
    pub fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes = Vec::new();
        self.decode_into(ids, &mut bytes)?;
        Ok(bytes)
    }

    /// Like [`Self::decode`], with invalid UTF-8 replaced by U+FFFD.
    pub fn decode_lossy(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        let decoded = self.decode(ids)?;
        let out = String::from_utf8_lossy(&decoded);
        Ok(out.into_owned())
    }
}

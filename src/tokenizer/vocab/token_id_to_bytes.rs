use super::byte_char_bmap::ByteCharBmap;
use super::text_token_id_bmap::TextTokenIdBmap;
use crate::tokenizer::{TokenId, TokenizerError};

/// The decode-side exit table, derived from the two sources of truth: the raw
/// bytes each token id stands for, its mapped-space string translated back
/// through the byte map (`id -> text` via [`TextTokenIdBmap`], `char -> byte`
/// via [`ByteCharBmap`]). Stored as one flat arena plus per-id offsets.
/// Construction fails if any token contains an unmapped char, which is what
/// makes [`Self::get`] infallible for in-range ids.
pub struct TokenIdToBytes {
    bytes: Box<[u8]>,
    /// `n_tokens + 1` offsets into `bytes`; token `id` occupies
    /// `bytes[offsets[id]..offsets[id + 1]]`.
    offsets: Box<[usize]>,
}

impl TokenIdToBytes {
    pub fn new(
        token_table: &TextTokenIdBmap,
        byte_map: &ByteCharBmap,
    ) -> Result<TokenIdToBytes, TokenizerError> {
        let n = token_table.len();
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for id in 0..n as TokenId {
            let token = token_table.token(id).expect("id < len");
            for ch in token.chars() {
                let byte = byte_map
                    .char_to_byte(ch)
                    .ok_or(TokenizerError::UnmappableChar { id, ch })?;
                bytes.push(byte);
            }
            offsets.push(bytes.len());
        }
        Ok(TokenIdToBytes {
            bytes: bytes.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        })
    }

    /// The raw bytes of `id`, or `None` if `id` is out of range.
    pub fn get(&self, id: TokenId) -> Option<&[u8]> {
        let start = *self.offsets.get(id as usize)?;
        let end = self.offsets[id as usize + 1];
        Some(&self.bytes[start..end])
    }
}

use super::byte_char_bmap::ByteCharBmap;
use super::text_token_id_bmap::TextTokenIdBmap;
use crate::tokenizer::{TokenId, TokenizerError};

/// The encode-side entry table, derived from the two sources of truth: for
/// each raw byte, the vocab id of its single-character mapped token
/// (`byte -> char` via [`ByteCharMap`], `char -> id` via [`TokenTable`]).
/// Construction fails if any byte lacks its token, which is what makes
/// [`Self::get`] infallible.
pub struct ByteToTokenId([TokenId; 256]);

impl ByteToTokenId {
    pub fn new(
        token_table: &TextTokenIdBmap,
        byte_map: &ByteCharBmap,
    ) -> Result<ByteToTokenId, TokenizerError> {
        let mut table = [0; 256];
        let mut buf = [0u8; 4];
        for (byte, slot) in table.iter_mut().enumerate() {
            let mapped = byte_map.byte_to_char(byte as u8);
            let token = mapped.encode_utf8(&mut buf);
            let Some(id) = token_table.id(token) else {
                return Err(TokenizerError::ByteNotInVocab {
                    byte: byte as u8,
                    token: token.into(),
                });
            };
            *slot = id;
        }
        Ok(ByteToTokenId(table))
    }

    /// The id of `byte`'s single-character token.
    pub const fn get(&self, byte: u8) -> TokenId {
        self.0[byte as usize]
    }
}

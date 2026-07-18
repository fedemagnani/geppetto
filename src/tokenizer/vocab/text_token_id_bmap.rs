use std::collections::HashMap;

use crate::tokenizer::{TokenId, TokenizerError};

/// The bidirectional token string <-> id mapping of a vocab: an id is the
/// token's position in the vocab list, string lookup is O(1). Construction
/// rejects duplicate tokens, so the two directions always agree.
pub struct TextTokenIdBmap {
    text_tokens: Vec<String>,
    token_ids: HashMap<String, TokenId>,
}

impl TextTokenIdBmap {
    pub fn new(tokens: Vec<String>) -> Result<TextTokenIdBmap, TokenizerError> {
        let mut ids = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            if ids.insert(token.clone(), id as TokenId).is_some() {
                return Err(TokenizerError::DuplicateToken(token.clone()));
            }
        }
        Ok(TextTokenIdBmap {
            text_tokens: tokens,
            token_ids: ids,
        })
    }

    /// The token string of `id`.
    pub fn token(&self, id: TokenId) -> Option<&str> {
        self.text_tokens.get(id as usize).map(String::as_str)
    }

    /// The id of a token string.
    pub fn id(&self, text: &str) -> Option<TokenId> {
        self.token_ids.get(text).copied()
    }

    pub fn len(&self) -> usize {
        self.text_tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text_tokens.is_empty()
    }
}

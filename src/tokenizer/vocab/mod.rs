mod bpe;
mod byte_char_bmap;
mod byte_to_token_id;
mod text_token_id_bmap;
mod token_id_to_bytes;

pub use byte_char_bmap::ByteCharBmap;
pub use text_token_id_bmap::TextTokenIdBmap;

use self::bpe::BPEMergeTable;
use self::byte_to_token_id::ByteToTokenId;
use self::token_id_to_bytes::TokenIdToBytes;
use crate::gguf::GgufFile;
use crate::tokenizer::{TokenId, TokenizerError};

/// The GPT-2 vocab. Sources of truth: the token string <-> id table
/// ([`TextTokenIdBmap`], strings in mapped space, e.g. " hello" as "Ġhello")
/// and the byte <-> char alphabet ([`ByteCharBmap`]). The three derived tables
/// are built from them at construction, which owns their validation: the BPE
/// merge table, the encode-side byte -> id table, and the decode-side
/// id -> bytes table, so both hot paths can never fail on the alphabet.
pub struct Vocab {
    bpe_merge_table: BPEMergeTable,
    /// text <--> token_id
    text_tokenid_bmap: TextTokenIdBmap,
    /// byte --> token_id
    byte_to_tokenid: ByteToTokenId,
    /// token_id --> bytes
    tokenid_to_bytes: TokenIdToBytes,
    bos: TokenId,
    eos: TokenId,
}

impl Vocab {
    pub const MODEL_KEY: &'static str = "tokenizer.ggml.model";
    pub const MODEL_VALUE: &'static str = "gpt2";
    pub const MODEL_VALUE_PRE: &'static str = "gpt-2";

    pub const PRE_KEY: &'static str = "tokenizer.ggml.pre";
    pub const TOKENS_KEY: &'static str = "tokenizer.ggml.tokens";
    pub const MERGES_KEY: &'static str = "tokenizer.ggml.merges";
    pub const BOS_KEY: &'static str = "tokenizer.ggml.bos_token_id";
    pub const EOS_KEY: &'static str = "tokenizer.ggml.eos_token_id";

    /// Pure assembly: each derived table is built from the sources of truth
    /// by its own constructor, which owns the validation.
    pub fn new(
        tokens: Vec<String>,
        merges: &[String],
        bos: TokenId,
        eos: TokenId,
    ) -> Result<Vocab, TokenizerError> {
        let text_tokenid_bmap = TextTokenIdBmap::new(tokens)?;
        // sources of truth for the derived tables; not stored past construction
        let byte_char_bmap = ByteCharBmap::new();
        let byte_to_tokenid = ByteToTokenId::new(&text_tokenid_bmap, &byte_char_bmap)?;
        let tokenid_to_bytes = TokenIdToBytes::new(&text_tokenid_bmap, &byte_char_bmap)?;
        let bpe_merge_table = BPEMergeTable::from_merges(merges, &text_tokenid_bmap)?;

        // sanitize beginning and end of sequence
        for id in [bos, eos] {
            if id as usize >= text_tokenid_bmap.len() {
                return Err(TokenizerError::IdOutOfRange {
                    id,
                    n_tokens: text_tokenid_bmap.len(),
                });
            }
        }

        Ok(Vocab {
            text_tokenid_bmap,
            bpe_merge_table,
            byte_to_tokenid,
            tokenid_to_bytes,
            bos,
            eos,
        })
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Vocab, TokenizerError> {
        let metadata = file.metadata();
        let model = metadata.get_str(Self::MODEL_KEY)?;
        if model != Self::MODEL_VALUE {
            return Err(TokenizerError::UnsupportedModel(model.into()));
        }
        // absent `pre` is accepted (older files); a different one is not
        if let Some(pre) = metadata.get(Self::PRE_KEY).and_then(|v| v.as_str())
            && pre != Self::MODEL_VALUE_PRE
        {
            return Err(TokenizerError::UnsupportedPretokenizer(pre.into()));
        }
        let tokens = metadata.get_arr_str(Self::TOKENS_KEY)?.to_vec();
        let merges = metadata.get_arr_str(Self::MERGES_KEY)?;
        let bos = metadata.get_u32(Self::BOS_KEY)?;
        let eos = metadata.get_u32(Self::EOS_KEY)?;
        Self::new(tokens, merges, bos, eos)
    }

    /// The mapped-space string of `id`.
    pub fn id_to_text(&self, id: TokenId) -> Option<&str> {
        self.text_tokenid_bmap.token(id)
    }

    /// The id of a mapped-space token string.
    pub fn text_to_id(&self, text: &str) -> Option<TokenId> {
        self.text_tokenid_bmap.id(text)
    }

    /// The id of a byte's single-character token. Infallible: coverage of
    /// all 256 bytes is checked at construction.
    pub const fn byte_to_id(&self, byte: u8) -> TokenId {
        self.byte_to_tokenid.get(byte)
    }

    /// The raw bytes `id` stands for, or `None` if `id` is out of range.
    /// The byte-map translation was validated at construction, so any
    /// in-range id resolves.
    pub fn id_to_bytes(&self, id: TokenId) -> Option<&[u8]> {
        self.tokenid_to_bytes.get(id)
    }

    /// Runs BPE in place over the tail of `symbols` starting at `from`:
    /// append one pre-token's byte-level ids, then merge just that region.
    pub fn apply_bpe_merges(&self, symbols: &mut Vec<TokenId>, from: usize) {
        self.bpe_merge_table.apply_bpe(symbols, from);
    }

    pub fn n_tokens(&self) -> usize {
        self.text_tokenid_bmap.len()
    }

    pub const fn bos(&self) -> TokenId {
        self.bos
    }

    pub const fn eos(&self) -> TokenId {
        self.eos
    }
}

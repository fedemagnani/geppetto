use fancy_regex::Regex;

use crate::tokenizer::TokenizerError;

/// The original OpenAI GPT-2 split pattern. llama.cpp's copy
/// (`src/llama-vocab.cpp`, `LLAMA_VOCAB_PRE_TYPE_GPT2`) drops the final
/// `\s+` branch because its splitter keeps unmatched gaps as pieces; with
/// match-based iteration the full pattern is the equivalent form.
pub const GPT2_SPLIT_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Pretokenizer {
    regex: Regex,
}

impl Pretokenizer {
    pub fn gpt2() -> Result<Pretokenizer, TokenizerError> {
        Ok(Pretokenizer {
            regex: Regex::new(GPT2_SPLIT_PATTERN)?,
        })
    }

    /// Pre-tokens of `text`, in order, covering the whole string. The
    /// pattern covers every char class, so gaps between matches should not
    /// happen; they are kept as their own pieces rather than silently
    /// dropped.
    pub fn split_iter<'r, 't>(&'r self, text: &'t str) -> SplitIter<'r, 't> {
        SplitIter {
            text,
            matches: self.regex.find_iter(text),
            last: 0,
            pending: None,
            done: false,
        }
    }
}

pub struct SplitIter<'r, 't> {
    text: &'t str,
    matches: fancy_regex::Matches<'r, 't>,
    /// End of the last piece handed out.
    last: usize,
    /// A match held back while the gap before it is handed out.
    pending: Option<(usize, usize)>,
    done: bool,
}

impl<'t> Iterator for SplitIter<'_, 't> {
    type Item = Result<&'t str, TokenizerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if let Some((start, end)) = self.pending.take() {
            self.last = end;
            return Some(Ok(&self.text[start..end]));
        }
        match self.matches.next() {
            Some(Err(err)) => {
                self.done = true;
                Some(Err(err.into()))
            }
            Some(Ok(m)) if m.start() > self.last => {
                let gap = &self.text[self.last..m.start()];
                self.pending = Some((m.start(), m.end()));
                Some(Ok(gap))
            }
            Some(Ok(m)) => {
                self.last = m.end();
                Some(Ok(m.as_str()))
            }
            None => {
                self.done = true;
                if self.last < self.text.len() {
                    Some(Ok(&self.text[self.last..]))
                } else {
                    None
                }
            }
        }
    }
}

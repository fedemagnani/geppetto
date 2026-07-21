//! Incremental UTF-8 assembly for streaming output.
//!
//! Byte-level BPE splits multi-byte characters across tokens, so the bytes of
//! a single token are often not valid UTF-8 on their own. This buffers the
//! incomplete tail until the following token completes it.

/// Accumulates decoded bytes and hands back only the text that is complete.
///
/// The output matches `String::from_utf8_lossy` over the concatenated input:
/// genuinely invalid sequences become U+FFFD, while a merely unfinished tail
/// waits for the next push.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    /// Bytes seen but not yet emitted: always a proper prefix of some
    /// multi-byte character.
    buffer: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream::default()
    }

    /// Appends `bytes` and pushes every character that is now complete onto
    /// `out`.
    pub fn push(&mut self, bytes: &[u8], out: &mut String) {
        self.buffer.extend_from_slice(bytes);
        loop {
            let error = match std::str::from_utf8(&self.buffer) {
                Ok(text) => {
                    out.push_str(text);
                    self.buffer.clear();
                    return;
                }
                Err(error) => error,
            };

            let valid_up_to = error.valid_up_to();
            // `valid_up_to` is by definition the length of a valid prefix
            let valid =
                std::str::from_utf8(&self.buffer[..valid_up_to]).expect("prefix is valid utf-8");
            out.push_str(valid);

            let Some(invalid_len) = error.error_len() else {
                // an unfinished character: keep it and wait for more bytes
                self.buffer.drain(..valid_up_to);
                return;
            };
            // a genuinely malformed sequence: replace it and keep scanning
            out.push(char::REPLACEMENT_CHARACTER);
            self.buffer.drain(..valid_up_to + invalid_len);
        }
    }

    /// Flushes whatever is left. Anything still buffered is an unfinished
    /// character that will never be completed, so it becomes one U+FFFD.
    pub fn finish(&mut self, out: &mut String) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer.clear();
        out.push(char::REPLACEMENT_CHARACTER);
    }

    /// Whether any bytes are held back waiting for completion.
    pub fn is_pending(&self) -> bool {
        !self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `text` one chunk at a time and returns everything emitted.
    fn stream_in_chunks(bytes: &[u8], chunk: usize) -> String {
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for piece in bytes.chunks(chunk.max(1)) {
            stream.push(piece, &mut out);
        }
        stream.finish(&mut out);
        out
    }

    #[test]
    fn byte_at_a_time_matches_the_whole_string() {
        let text = "hello 🦙 wörld — 你好 🏳️‍🌈 done";
        for chunk in 1..=8 {
            assert_eq!(
                stream_in_chunks(text.as_bytes(), chunk),
                text,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn a_split_character_is_held_until_complete() {
        let llama = "🦙".as_bytes();
        let mut stream = Utf8Stream::new();
        let mut out = String::new();

        stream.push(&llama[..2], &mut out);
        assert!(out.is_empty(), "emitted an incomplete character");
        assert!(stream.is_pending());

        stream.push(&llama[2..], &mut out);
        assert_eq!(out, "🦙");
        assert!(!stream.is_pending());
    }

    #[test]
    fn output_always_matches_from_utf8_lossy() {
        let cases: &[&[u8]] = &[
            b"plain ascii",
            &[0xF0, 0x9F, 0xA6, 0x99],       // a llama
            &[0xFF, 0xFE],                   // never valid
            &[b'a', 0xE4, 0xBD, 0xA0, b'b'], // valid, surrounded
            &[b'a', 0x80, b'b'],             // a stray continuation byte
            &[0xE4, 0xBD],                   // truncated at the end
        ];
        for bytes in cases {
            let expected = String::from_utf8_lossy(bytes);
            for chunk in 1..=4 {
                assert_eq!(
                    stream_in_chunks(bytes, chunk),
                    expected,
                    "{bytes:?} in chunks of {chunk}"
                );
            }
        }
    }

    #[test]
    fn output_grows_as_a_prefix_of_the_final_text() {
        // arbitrary bytes, including sequences that are never valid
        let bytes: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let expected = String::from_utf8_lossy(&bytes);

        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for piece in bytes.chunks(3) {
            stream.push(piece, &mut out);
            // nothing is ever emitted that the full decode would not produce,
            // so a streaming reader never sees text it has to take back
            assert!(expected.starts_with(&out), "emitted text diverged");
        }
        stream.finish(&mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn finishing_an_empty_stream_emits_nothing() {
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        stream.push(b"ok", &mut out);
        stream.finish(&mut out);
        assert_eq!(out, "ok");
    }
}

//! GPT-2's bijection between raw bytes and printable unicode codepoints.

use std::collections::HashMap;

/// The byte <-> codepoint tables. BPE vocab entries must be printable
/// strings, so GPT-2 remaps every byte to a codepoint: bytes that are
/// already printable Latin-1 map to themselves, the rest (whitespace,
/// control bytes, 0x7F..0xA0, 0xAD) take consecutive codepoints from U+0100
/// up (e.g. space becomes 'Ġ' = U+0120).
pub struct ByteCharBmap {
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl ByteCharBmap {
    pub fn new() -> ByteCharBmap {
        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = HashMap::with_capacity(256);
        let mut next = 0x100u32;

        let is_printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);

        for (index, slot) in byte_to_char.iter_mut().enumerate() {
            let index = index as u8;
            let c = if is_printable(index) {
                index as char
            } else {
                let c = char::from_u32(next).expect("below surrogate range");
                next += 1;
                c
            };
            *slot = c;
            char_to_byte.insert(c, index);
        }

        ByteCharBmap {
            byte_to_char,
            char_to_byte,
        }
    }

    pub fn byte_to_char(&self, b: u8) -> char {
        self.byte_to_char[b as usize]
    }

    /// `None` for chars outside the 256-codepoint mapped alphabet.
    pub fn char_to_byte(&self, c: char) -> Option<u8> {
        self.char_to_byte.get(&c).copied()
    }
}

impl Default for ByteCharBmap {
    fn default() -> ByteCharBmap {
        ByteCharBmap::new()
    }
}

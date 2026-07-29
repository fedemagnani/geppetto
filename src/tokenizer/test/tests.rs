use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::gguf::GgufFile;
use crate::tokenizer::{ByteCharBmap, Pretokenizer, Tokenizer};

fn models_path(file: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models");
    assert!(
        path.is_dir(),
        "these tests need the llama.cpp models directory next to geppetto \
         (fixtures: ggml-vocab-gpt-2.gguf and its .inp/.out files)"
    );
    path.join(file)
}

static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    let file = GgufFile::open(models_path("ggml-vocab-gpt-2.gguf")).expect("vocab fixture parses");
    Tokenizer::from_gguf(&file).expect("vocab fixture is a gpt2 vocab")
});

/// Cases transcribed from the `.inp`/`.out` fixtures, frozen here so the
/// core suite stands alone if geppetto moves out of the llama.cpp tree.
const KNOWN_CASES: &[(&str, &[u32])] = &[
    ("", &[]),
    (" ", &[220]),
    ("  ", &[220, 220]),
    ("   ", &[220, 220, 220]),
    ("\t", &[197]),
    ("\n", &[198]),
    ("\n\n", &[628]),
    ("\n\n\n", &[628, 198]),
    ("\t\n", &[197, 198]),
    ("Hello world", &[15496, 995]),
    (" Hello world", &[18435, 995]),
    ("Hello World", &[15496, 2159]),
    (" Hello World", &[18435, 2159]),
    (" Hello World!", &[18435, 2159, 0]),
    ("Hello, world!", &[15496, 11, 995, 0]),
    (" Hello, world!", &[18435, 11, 995, 0]),
    (" this is 🦙.cpp", &[428, 318, 12520, 99, 247, 13, 20322]),
    (
        "w048 7tuijk dsdfhu",
        &[86, 47202, 767, 28047, 45961, 288, 82, 7568, 13415],
    ),
    ("ied 4 ½ months", &[798, 604, 25208, 1933]),
    ("Äpfel", &[127, 226, 79, 69, 417]),
    (
        "нещо на Български",
        &[
            22177, 16843, 141, 231, 15166, 12466, 121, 16142, 12466, 239, 141, 232, 30143, 140,
            111, 16142, 21169, 21727, 31583, 18849,
        ],
    ),
    ("Hello", &[15496]),
    (" Hello", &[18435]),
    ("  Hello", &[220, 18435]),
    ("   Hello", &[220, 220, 18435]),
    ("    Hello", &[220, 220, 220, 18435]),
    (
        "    Hello\n    Hello",
        &[220, 220, 220, 18435, 198, 220, 220, 220, 18435],
    ),
    (" (", &[357]),
    ("\n =", &[198, 796]),
    ("' era", &[6, 6980]),
    ("!!!!!!", &[13896, 3228]),
    ("3", &[18]),
    ("33", &[2091]),
    ("333", &[20370]),
    ("3333", &[24840]),
    ("33333", &[2091, 20370]),
    ("333333", &[24840, 2091]),
    ("3333333", &[24840, 20370]),
    ("33333333", &[24840, 24840]),
    ("333333333", &[24840, 2091, 20370]),
    (
        "Cửa Việt",
        &[34, 157, 119, 255, 64, 16049, 157, 119, 229, 83],
    ),
    (" discards", &[1221, 1371]),
];

#[test]
fn known_answers_encode_exactly() {
    for (text, ids) in KNOWN_CASES {
        assert_eq!(
            &TOKENIZER.encode(text).unwrap(),
            ids,
            "encode mismatch for {text:?}"
        );
    }
}

#[test]
fn full_reference_suite_matches_the_cpp_tokenizer() {
    let inp = std::fs::read_to_string(models_path("ggml-vocab-gpt-2.gguf.inp")).unwrap();
    let out = std::fs::read_to_string(models_path("ggml-vocab-gpt-2.gguf.out")).unwrap();

    let cases: Vec<&str> = {
        let mut cases: Vec<&str> = inp.split("\n__ggml_vocab_test__\n").collect();
        assert_eq!(cases.pop(), Some(""), "trailing marker expected");
        cases
    };
    let expected: Vec<Vec<u32>> = out
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|id| id.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(cases.len(), expected.len(), "fixture files disagree");
    assert!(cases.len() >= 40, "suspiciously few reference cases");

    for (text, ids) in cases.iter().zip(&expected) {
        assert_eq!(
            &TOKENIZER.encode(text).unwrap(),
            ids,
            "encode mismatch for {text:?}"
        );
    }
}

#[test]
fn encode_decode_round_trips_arbitrary_text() {
    let inp = std::fs::read_to_string(models_path("ggml-vocab-gpt-2.gguf.inp")).unwrap();
    let extra = [
        "no special handling of <|endoftext|> either",
        "mixed \u{0000} control \u{0007} bytes",
        "combining a\u{0301}ccents and ﬁ ligatures",
        "🇮🇹 flags and 🏳️‍🌈 zwj sequences",
    ];
    for text in inp
        .split("\n__ggml_vocab_test__\n")
        .chain(extra.iter().copied())
    {
        let ids = TOKENIZER.encode(text).unwrap();
        assert_eq!(
            TOKENIZER.decode(&ids).unwrap(),
            text.as_bytes(),
            "round trip failed for {text:?}"
        );
    }
}

#[test]
fn into_variants_append_without_clearing() {
    let mut ids = vec![42];
    TOKENIZER.encode_into("Hello world", &mut ids).unwrap();
    assert_eq!(ids, [42, 15496, 995]);

    let mut bytes = b"x".to_vec();
    TOKENIZER.decode_into(&[15496, 995], &mut bytes).unwrap();
    assert_eq!(bytes, b"xHello world");
}

#[test]
fn every_vocab_token_decodes_to_bytes() {
    let n = TOKENIZER.vocab().n_tokens() as u32;
    assert_eq!(n, 50257);
    for id in 0..n {
        TOKENIZER
            .decode(&[id])
            .unwrap_or_else(|e| panic!("token {id} does not decode: {e}"));
    }
}

#[test]
fn bos_and_eos_are_endoftext() {
    assert_eq!(TOKENIZER.vocab().bos(), 50256);
    assert_eq!(TOKENIZER.vocab().eos(), 50256);
    assert_eq!(TOKENIZER.vocab().id_to_text(50256), Some("<|endoftext|>"));
}

#[test]
fn byte_map_is_a_bijection() {
    let map = ByteCharBmap::new();
    let mut seen = std::collections::HashSet::new();
    for b in 0..=255u8 {
        let c = map.byte_to_char(b);
        assert!(seen.insert(c), "codepoint {c:?} mapped twice");
        assert_eq!(map.char_to_byte(c), Some(b));
    }
    assert_eq!(map.byte_to_char(b' '), 'Ġ');
    assert_eq!(map.byte_to_char(b'!'), '!');
    assert_eq!(map.byte_to_char(0xA1), '¡');
    assert_eq!(map.char_to_byte('a'), Some(b'a'));
    assert_eq!(map.char_to_byte('本'), None);
}

#[test]
fn pretokenizer_splits_match_gpt2_semantics() {
    let pre = Pretokenizer::gpt2().unwrap();
    let split = |text| pre.split_iter(text).collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(split("Hello world"), vec!["Hello", " world"]);
    assert_eq!(split("don't"), vec!["don", "'t"]);
    assert_eq!(split("I've 'RE"), vec!["I", "'ve", " '", "RE"]);
    assert_eq!(split("  Hello"), vec![" ", " Hello"]);
    assert_eq!(split("ab   "), vec!["ab", "   "]);
    assert_eq!(split("x1y2"), vec!["x", "1", "y", "2"]);
    assert_eq!(split("3.3"), vec!["3", ".", "3"]);
    assert_eq!(split("a\n\nb"), vec!["a", "\n", "\n", "b"]);
    assert_eq!(split(" (normal)"), vec![" (", "normal", ")"]);
    let text: String = split("empty gaps never drop text 🚀\u{200d}ok").concat();
    assert_eq!(text, "empty gaps never drop text 🚀\u{200d}ok");
}

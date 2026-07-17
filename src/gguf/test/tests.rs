use super::fixtures::{FixtureBuilder, w_string, w_u32, w_u64};
use crate::gguf::{ArrayValue, GgufError, GgufFile, Value};

const F32_ID: u32 = 0;
const F16_ID: u32 = 1;
const Q8_0_ID: u32 = 8;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn empty_file_parses() {
    let file = GgufFile::from_bytes(FixtureBuilder::new().build()).unwrap();
    assert_eq!(file.version(), 3);
    assert_eq!(file.alignment(), 32);
    assert!(file.metadata().is_empty());
    assert!(file.tensors().is_empty());
}

#[test]
fn scalar_values_round_trip() {
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .kv("k.u8", Value::U8(7))
            .kv("k.i8", Value::I8(-7))
            .kv("k.u16", Value::U16(300))
            .kv("k.i16", Value::I16(-300))
            .kv("k.u32", Value::U32(70_000))
            .kv("k.i32", Value::I32(-70_000))
            .kv("k.f32", Value::F32(1.5))
            .kv("k.bool", Value::Bool(true))
            .kv("k.str", Value::String("hello".into()))
            .kv("k.u64", Value::U64(1 << 40))
            .kv("k.i64", Value::I64(-(1 << 40)))
            .kv("k.f64", Value::F64(2.5))
            .build(),
    )
    .unwrap();

    assert_eq!(file.metadata().get_u32("k.u32").unwrap(), 70_000);
    assert_eq!(file.metadata().get_i32("k.i32").unwrap(), -70_000);
    assert_eq!(file.metadata().get_f32("k.f32").unwrap(), 1.5);
    assert!(file.metadata().get_bool("k.bool").unwrap());
    assert_eq!(file.metadata().get_str("k.str").unwrap(), "hello");
    assert_eq!(file.metadata().get_u64("k.u64").unwrap(), 1 << 40);
    assert_eq!(file.metadata().get("k.f64"), Some(&Value::F64(2.5)));
    assert_eq!(file.metadata().get("k.i8"), Some(&Value::I8(-7)));
    assert_eq!(file.metadata().get("k.u16"), Some(&Value::U16(300)));
    assert_eq!(file.metadata().get("k.i16"), Some(&Value::I16(-300)));
    assert_eq!(file.metadata().get("k.i64"), Some(&Value::I64(-(1 << 40))));
    // insertion order is preserved
    assert_eq!(file.metadata().iter().next().unwrap().0, "k.u8");
}

#[test]
fn integer_getters_widen_losslessly() {
    let file =
        GgufFile::from_bytes(FixtureBuilder::new().kv("small", Value::U8(12)).build()).unwrap();
    assert_eq!(file.metadata().get_u32("small").unwrap(), 12);
    assert_eq!(file.metadata().get_u64("small").unwrap(), 12);
    assert!(matches!(
        file.metadata().get_i32("small"),
        Err(GgufError::TypeMismatch { .. })
    ));
}

#[test]
fn arrays_round_trip() {
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .kv("a.u32", Value::Array(ArrayValue::U32(vec![1, 2, 3])))
            .kv("a.f32", Value::Array(ArrayValue::F32(vec![0.5, -0.5])))
            .kv("a.i32", Value::Array(ArrayValue::I32(vec![-1, 1])))
            .kv(
                "a.str",
                Value::Array(ArrayValue::String(vec!["ab".into(), "".into(), "c".into()])),
            )
            .kv("a.bool", Value::Array(ArrayValue::Bool(vec![true, false])))
            .kv("a.empty", Value::Array(ArrayValue::I64(vec![])))
            .build(),
    )
    .unwrap();

    assert_eq!(file.metadata().get_arr_u32("a.u32").unwrap(), &[1, 2, 3]);
    assert_eq!(file.metadata().get_arr_f32("a.f32").unwrap(), &[0.5, -0.5]);
    assert_eq!(file.metadata().get_arr_i32("a.i32").unwrap(), &[-1, 1]);
    assert_eq!(
        file.metadata().get_arr_str("a.str").unwrap(),
        &["ab".to_string(), "".to_string(), "c".to_string()]
    );
    assert_eq!(
        file.metadata().get("a.bool").unwrap(),
        &Value::Array(ArrayValue::Bool(vec![true, false]))
    );
    assert!(
        file.metadata()
            .get("a.empty")
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = FixtureBuilder::new().build();
    bytes[..4].copy_from_slice(b"GGML");
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::BadMagic(m)) if &m == b"GGML"
    ));
}

#[test]
fn unsupported_versions_are_rejected() {
    for version in [0, 1, 4] {
        assert!(matches!(
            GgufFile::from_bytes(FixtureBuilder::new().version(version).build()),
            Err(GgufError::UnsupportedVersion(v)) if v == version
        ));
    }
    assert!(GgufFile::from_bytes(FixtureBuilder::new().version(2).build()).is_ok());
}

#[test]
fn truncation_is_rejected_at_every_point() {
    let bytes = FixtureBuilder::new()
        .kv("key", Value::String("value".into()))
        .tensor("t", &[4], F32_ID, f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .build();
    assert!(GgufFile::from_bytes(bytes.clone()).is_ok());
    for len in 0..bytes.len() {
        assert!(
            GgufFile::from_bytes(bytes[..len].to_vec()).is_err(),
            "prefix of {len} bytes should not parse"
        );
    }
}

#[test]
fn duplicate_keys_and_tensor_names_are_rejected() {
    let bytes = FixtureBuilder::new()
        .kv("dup", Value::U8(1))
        .kv("dup", Value::U8(2))
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::DuplicateKey(k)) if k == "dup"
    ));

    let bytes = FixtureBuilder::new()
        .tensor("t", &[1], F32_ID, f32_bytes(&[1.0]))
        .tensor_at("t", &[1], F32_ID, f32_bytes(&[2.0]), 32)
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::DuplicateTensor(n)) if n == "t"
    ));
}

#[test]
fn nested_arrays_are_rejected() {
    let mut raw = Vec::new();
    w_string(&mut raw, "nested");
    w_u32(&mut raw, 9); // array
    w_u32(&mut raw, 9); // element type: array
    w_u64(&mut raw, 0);
    assert!(matches!(
        GgufFile::from_bytes(FixtureBuilder::new().raw_kv(raw).build()),
        Err(GgufError::NestedArray(k)) if k == "nested"
    ));
}

#[test]
fn invalid_value_type_and_bool_are_rejected() {
    let mut raw = Vec::new();
    w_string(&mut raw, "bad");
    w_u32(&mut raw, 13);
    assert!(matches!(
        GgufFile::from_bytes(FixtureBuilder::new().raw_kv(raw).build()),
        Err(GgufError::InvalidValueType(13))
    ));

    let mut raw = Vec::new();
    w_string(&mut raw, "bad-bool");
    w_u32(&mut raw, 7);
    raw.push(2);
    assert!(matches!(
        GgufFile::from_bytes(FixtureBuilder::new().raw_kv(raw).build()),
        Err(GgufError::InvalidBool(2))
    ));
}

#[test]
fn non_power_of_two_alignment_is_rejected() {
    for alignment in [0, 24] {
        assert!(matches!(
            GgufFile::from_bytes(FixtureBuilder::new().alignment(alignment).build()),
            Err(GgufError::InvalidAlignment(a)) if a == u64::from(alignment)
        ));
    }
}

#[test]
fn tensor_data_round_trips() {
    let values = [1.0f32, -2.0, 3.5, 0.25];
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .tensor("weight", &[2, 2], F32_ID, f32_bytes(&values))
            .build(),
    )
    .unwrap();

    let info = file.tensor("weight").unwrap();
    assert_eq!(info.dims, vec![2, 2]);
    assert_eq!(info.n_elements(), 4);
    assert_eq!(info.nbytes(), Some(16));
    assert_eq!(file.tensor_data("weight").unwrap(), f32_bytes(&values));
    assert!(matches!(
        file.tensor_data("missing"),
        Err(GgufError::TensorNotFound(_))
    ));
}

#[test]
fn custom_alignment_places_tensors_correctly() {
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .alignment(64)
            .tensor("a", &[3], F32_ID, f32_bytes(&[1.0, 2.0, 3.0]))
            .tensor("b", &[2], F32_ID, f32_bytes(&[4.0, 5.0]))
            .build(),
    )
    .unwrap();

    assert_eq!(file.alignment(), 64);
    assert_eq!(file.tensor("a").unwrap().offset, 0);
    assert_eq!(file.tensor("b").unwrap().offset, 64);
    assert_eq!(file.tensor_data("a").unwrap(), f32_bytes(&[1.0, 2.0, 3.0]));
    assert_eq!(file.tensor_data("b").unwrap(), f32_bytes(&[4.0, 5.0]));
}

#[test]
fn wrong_tensor_offsets_are_rejected() {
    // second tensor claims offset 0: overlaps the first
    let bytes = FixtureBuilder::new()
        .tensor("a", &[4], F32_ID, f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
        .tensor_at("b", &[4], F32_ID, f32_bytes(&[5.0, 6.0, 7.0, 8.0]), 0)
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::BadTensorOffset { name, offset: 0, expected: 32 }) if name == "b"
    ));

    // gap before the first tensor is equally malformed
    let bytes = FixtureBuilder::new()
        .tensor_at("a", &[1], F32_ID, f32_bytes(&[1.0]), 32)
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::BadTensorOffset { .. })
    ));
}

#[test]
fn f16_tensors_use_two_bytes_per_element() {
    let data: Vec<u8> = (0..8).collect();
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .tensor("h", &[4], F16_ID, data.clone())
            .build(),
    )
    .unwrap();
    assert_eq!(file.tensor("h").unwrap().nbytes(), Some(8));
    assert_eq!(file.tensor_data("h").unwrap(), data);
}

#[test]
fn unknown_tensor_types_parse_but_refuse_data_access() {
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .tensor("q", &[32], Q8_0_ID, vec![0; 34])
            .build(),
    )
    .unwrap();

    let info = file.tensor("q").unwrap();
    assert_eq!(info.type_id, Q8_0_ID);
    assert_eq!(info.dtype(), None);
    assert_eq!(info.nbytes(), None);
    assert!(matches!(
        file.tensor_data("q"),
        Err(GgufError::UnsupportedTensorType {
            type_id: Q8_0_ID,
            ..
        })
    ));
}

#[test]
fn getter_errors_are_typed() {
    let file = GgufFile::from_bytes(
        FixtureBuilder::new()
            .kv("s", Value::String("x".into()))
            .build(),
    )
    .unwrap();
    assert!(matches!(
        file.metadata().get_u32("missing"),
        Err(GgufError::KeyNotFound(k)) if k == "missing"
    ));
    assert!(matches!(
        file.metadata().get_u32("s"),
        Err(GgufError::TypeMismatch {
            expected: "u32",
            ..
        })
    ));
    assert!(matches!(
        file.metadata().get_arr_str("s"),
        Err(GgufError::TypeMismatch {
            expected: "array",
            ..
        })
    ));
}

#[test]
fn oversized_names_and_dims_are_rejected() {
    let bytes = FixtureBuilder::new()
        .tensor(&"x".repeat(64), &[1], F32_ID, f32_bytes(&[1.0]))
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::NameTooLong(64))
    ));

    let bytes = FixtureBuilder::new()
        .tensor("t", &[1, 1, 1, 1, 1], F32_ID, f32_bytes(&[1.0]))
        .build();
    assert!(matches!(
        GgufFile::from_bytes(bytes),
        Err(GgufError::TooManyDims { n_dims: 5, .. })
    ));
}

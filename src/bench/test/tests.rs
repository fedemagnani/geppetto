use crate::bench::{Environment, Report, Scenario};

/// The harness needs a real GGUF path and a gpt2 tokenizer, which the tiny
/// test model deliberately lacks, so `Harness` itself is exercised by the
/// manual `bench` runs. What is tested here is everything that can go wrong
/// without a model: scenario shape, statistics, and report rendering.
fn env() -> Environment {
    Environment::detect()
}

#[test]
fn every_default_scenario_has_samples_and_warmup() {
    for scenario in Scenario::defaults() {
        assert!(
            scenario.default_samples() >= 3,
            "{scenario} takes too few samples to have a spread"
        );
        assert!(
            scenario.default_warmup() >= 1,
            "{scenario} must discard at least one warmup run"
        );
    }
}

#[test]
fn default_scenarios_cover_decode_prefill_and_load() {
    let kinds: Vec<&str> = Scenario::defaults().iter().map(|s| s.key()).collect();
    assert!(kinds.contains(&"decode"), "no decode scenario");
    assert!(kinds.contains(&"prefill"), "no prefill scenario");
    assert!(kinds.contains(&"load"), "no load scenario");
}

#[test]
fn decode_scenarios_span_more_than_one_cache_depth() {
    // a single depth would hide regressions that only appear at long context
    let depths: Vec<usize> = Scenario::defaults()
        .iter()
        .filter(|s| s.key() == "decode")
        .map(|s| s.cache_depth())
        .collect();
    assert!(depths.len() >= 2, "decode measured at only {depths:?}");
    assert!(
        depths.iter().min() != depths.iter().max(),
        "decode depths are all the same: {depths:?}"
    );
}

#[test]
fn an_empty_report_still_renders() {
    let report = Report::new(env(), "none.gguf".into(), 0, Vec::new());
    let text = report.to_text();
    assert!(text.contains("geppetto bench"));
    assert!(text.contains("peak RSS"));
}

#[test]
fn a_debug_report_is_labelled_not_kpi_grade() {
    // the suite runs in debug, so the warning must be present: this is the
    // guard against pasting a debug number into the journal
    let report = Report::new(env(), "none.gguf".into(), 0, Vec::new());
    let text = report.to_text();
    if !report.environment.timings_are_kpi_grade() {
        assert!(
            text.contains("NOT KPI grade"),
            "missing the warning banner:\n{text}"
        );
    }
}

#[test]
fn json_output_is_well_formed_and_carries_provenance() {
    let report = Report::new(env(), "model.gguf".into(), 1234, Vec::new());
    let json = report.to_json();

    assert!(json.starts_with('{'), "json must be an object");
    assert!(json.trim_end().ends_with('}'), "json must be closed");
    assert_eq!(
        json.matches('{').count(),
        json.matches('}').count(),
        "unbalanced braces:\n{json}"
    );
    assert!(json.contains("\"model\": \"model.gguf\""));
    assert!(json.contains("\"model_bytes\": 1234"));
    // provenance is the whole point of the json form
    assert!(json.contains("\"commit\""));
    assert!(json.contains("\"rustc\""));
    assert!(json.contains("\"kpi_grade\""));
    assert!(json.contains("\"peak_rss_bytes\""));
}

#[test]
fn peak_rss_is_reported_in_bytes() {
    let report = Report::new(env(), "m".into(), 0, Vec::new());
    // a live test process is above 1 MB and far below 1 TB; catches the
    // classic ru_maxrss kB-vs-B unit slip
    assert!(report.peak_rss_bytes > 1 << 20);
    assert!(report.peak_rss_bytes < 1 << 40);
}

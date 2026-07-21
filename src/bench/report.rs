use std::fmt::Write as _;

use crate::bench::alloc;
use crate::bench::{Environment, Samples, Scenario, ScenarioResult, peak_rss_bytes};

/// A completed run: every scenario's samples plus the environment they were
/// taken in.
pub struct Report {
    pub environment: Environment,
    pub model: String,
    pub model_bytes: u64,
    pub results: Vec<ScenarioResult>,
    pub peak_rss_bytes: u64,
}

impl Report {
    pub fn new(
        environment: Environment,
        model: String,
        model_bytes: u64,
        results: Vec<ScenarioResult>,
    ) -> Report {
        Report {
            environment,
            model,
            model_bytes,
            results,
            peak_rss_bytes: peak_rss_bytes(),
        }
    }

    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write_header(&mut out);
        self.write_table(&mut out);
        self.write_summary(&mut out);
        out
    }

    fn write_header(&self, out: &mut String) {
        let env = &self.environment;
        let _ = writeln!(out, "geppetto bench");
        let _ = writeln!(out, "  model      {}", self.model);
        let _ = writeln!(out, "  model size {:.1} MB", self.model_bytes as f64 / 1e6);
        let _ = writeln!(out, "  commit     {}", env.commit);
        let _ = writeln!(out, "  rustc      {}", env.rustc);
        let _ = writeln!(out, "  profile    {}", env.profile);
        let _ = writeln!(out, "  cores      {}", env.cores);

        let instrumentation = if env.instrumentation.is_empty() {
            "none".to_string()
        } else {
            env.instrumentation.join(", ")
        };
        let _ = writeln!(out, "  profiling  {instrumentation}");

        if !env.timings_are_kpi_grade() {
            let _ = writeln!(
                out,
                "\n  WARNING: these timings are NOT KPI grade.\n  \
                 {}Record KPIs from a --release build with no profiling features.",
                if env.profile == "debug" {
                    "This is a debug build. "
                } else {
                    "Instrumentation perturbs what it measures. "
                }
            );
        }
        let _ = writeln!(out);
    }

    fn write_table(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "{:<18} {:>7} {:>12} {:>12} {:>12} {:>8}",
            "scenario", "samples", "median", "p10", "p90", "spread"
        );
        let _ = writeln!(out, "{}", "-".repeat(74));

        for result in &self.results {
            let scenario = result.scenario;
            if scenario == Scenario::Load {
                self.write_row(out, &scenario.to_string(), "s", &result.seconds());
                continue;
            }

            let per_second = result.tokens_per_second();
            self.write_row(out, &scenario.to_string(), "tok/s", &per_second);
            self.write_row(out, "  latency", "ms/tok", &result.millis_per_token());

            if alloc::is_installed() {
                self.write_row(out, "  allocs", "/fwd", &result.allocs_per_forward());
                self.write_row(out, "  alloc bytes", "B/fwd", &result.bytes_per_forward());
            }
            let _ = writeln!(
                out,
                "{:<18} {:>7} {:>12}",
                "  peak RSS",
                "",
                format!("{:.0} MB", result.peak_rss_bytes() as f64 / 1e6)
            );
        }
        let _ = writeln!(out);
    }

    fn write_row(&self, out: &mut String, label: &str, unit: &str, samples: &Samples) {
        let _ = writeln!(
            out,
            "{:<18} {:>7} {:>12} {:>12} {:>12} {:>7.1}%",
            label,
            samples.len(),
            format_value(samples.median(), unit),
            format_value(samples.percentile(0.10), unit),
            format_value(samples.percentile(0.90), unit),
            samples.relative_spread() * 100.0,
        );
    }

    fn write_summary(&self, out: &mut String) {
        let _ = writeln!(
            out,
            "peak RSS   {:.0} MB (whole process, all scenarios)",
            self.peak_rss_bytes as f64 / 1e6
        );
        if self.model_bytes > 0 {
            let _ = writeln!(
                out,
                "           {:.2}x the model file",
                self.peak_rss_bytes as f64 / self.model_bytes as f64
            );
        }
        if !alloc::is_installed() {
            let _ = writeln!(
                out,
                "\nallocation counts come from the hotpath table below \
                 (hotpath-alloc owns the global allocator)"
            );
        }
    }

    /// Machine-readable form, so journal entries are pasted rather than
    /// transcribed. Hand-rolled rather than pulling in serde for one struct.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{{");
        let _ = writeln!(out, "  \"model\": {},", quote(&self.model));
        let _ = writeln!(out, "  \"model_bytes\": {},", self.model_bytes);
        let _ = writeln!(out, "  \"commit\": {},", quote(&self.environment.commit));
        let _ = writeln!(out, "  \"rustc\": {},", quote(&self.environment.rustc));
        let _ = writeln!(out, "  \"profile\": {},", quote(self.environment.profile));
        let _ = writeln!(out, "  \"cores\": {},", self.environment.cores);
        let _ = writeln!(
            out,
            "  \"instrumentation\": [{}],",
            self.environment
                .instrumentation
                .iter()
                .map(|s| quote(s))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "  \"kpi_grade\": {},",
            self.environment.timings_are_kpi_grade()
        );
        let _ = writeln!(out, "  \"peak_rss_bytes\": {},", self.peak_rss_bytes);
        let _ = writeln!(out, "  \"scenarios\": [");

        let rows: Vec<String> = self.results.iter().map(scenario_json).collect();
        let _ = writeln!(out, "{}", rows.join(",\n"));
        let _ = writeln!(out, "  ]");
        let _ = writeln!(out, "}}");
        out
    }
}

fn scenario_json(result: &ScenarioResult) -> String {
    let mut out = String::new();
    let seconds = result.seconds();
    let _ = write!(out, "    {{");
    let _ = write!(out, "\"name\": {}, ", quote(&result.scenario.to_string()));
    let _ = write!(out, "\"kind\": {}, ", quote(result.scenario.key()));
    let _ = write!(out, "\"cache_depth\": {}, ", result.scenario.cache_depth());
    let _ = write!(out, "\"samples\": {}, ", result.samples());
    let _ = write!(out, "\"seconds_median\": {:.6}, ", seconds.median());

    if result.scenario != Scenario::Load {
        let per_second = result.tokens_per_second();
        let _ = write!(out, "\"tok_per_s_median\": {:.4}, ", per_second.median());
        let _ = write!(
            out,
            "\"tok_per_s_p10\": {:.4}, ",
            per_second.percentile(0.10)
        );
        let _ = write!(
            out,
            "\"tok_per_s_p90\": {:.4}, ",
            per_second.percentile(0.90)
        );
        let _ = write!(
            out,
            "\"ms_per_token_median\": {:.4}, ",
            result.millis_per_token().median()
        );
        if alloc::is_installed() {
            let _ = write!(
                out,
                "\"allocs_per_forward_median\": {:.1}, ",
                result.allocs_per_forward().median()
            );
            let _ = write!(
                out,
                "\"alloc_bytes_per_forward_median\": {:.0}, ",
                result.bytes_per_forward().median()
            );
        }
        let _ = write!(out, "\"peak_rss_bytes\": {}, ", result.peak_rss_bytes());
    }
    let _ = write!(
        out,
        "\"relative_spread\": {:.4}}}",
        seconds.relative_spread()
    );
    out
}

/// Minimal JSON string escaping: enough for paths, versions and scenario
/// names, which is all this emits.
fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn format_value(value: f64, unit: &str) -> String {
    match unit {
        "tok/s" => format!("{value:.1} tok/s"),
        "ms/tok" => format!("{value:.2} ms"),
        "s" => format!("{value:.3} s"),
        "/fwd" => format!("{value:.0}"),
        "B/fwd" => format!("{:.1} KB", value / 1024.0),
        _ => format!("{value:.3}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_are_escaped() {
        assert_eq!(quote("plain"), "\"plain\"");
        assert_eq!(quote("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(quote("back\\slash"), "\"back\\\\slash\"");
        assert_eq!(quote("line\nbreak"), "\"line\\nbreak\"");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn values_carry_their_units() {
        assert_eq!(format_value(19.34, "tok/s"), "19.3 tok/s");
        assert_eq!(format_value(51.82, "ms/tok"), "51.82 ms");
        assert_eq!(format_value(1013.0, "/fwd"), "1013");
    }
}

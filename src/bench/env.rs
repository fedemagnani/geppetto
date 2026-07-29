//! The environment a result was measured on. A benchmark number without this
//! block is not reproducible.

use std::process::Command;

#[derive(Debug, Clone)]
pub struct Environment {
    pub cores: usize,
    pub rustc: String,
    pub commit: String,
    pub profile: &'static str,
    /// Profiling features compiled in, which perturb the timings.
    pub instrumentation: Vec<&'static str>,
}

impl Environment {
    pub fn detect() -> Environment {
        let mut instrumentation = Vec::new();
        if cfg!(feature = "hotpath") {
            instrumentation.push("hotpath");
        }
        if cfg!(feature = "hotpath-alloc") {
            instrumentation.push("hotpath-alloc");
        }
        if cfg!(feature = "hotpath-cpu") {
            instrumentation.push("hotpath-cpu");
        }

        let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
        let rustc = capture("rustc", &["--version"]).unwrap_or_else(|| "unknown".into());
        let commit =
            capture("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
        let profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };

        Environment {
            cores,
            rustc,
            commit,
            profile,
            instrumentation,
        }
    }

    /// Whether measured timings can be trusted as KPIs. Instrumentation and
    /// debug builds both invalidate them, in opposite directions of severity
    /// but with the same conclusion.
    pub fn timings_are_kpi_grade(&self) -> bool {
        self.instrumentation.is_empty() && self.profile == "release"
    }
}

fn capture(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

/// Peak resident set size of this process, in bytes.
///
/// `ru_maxrss` is bytes on macOS and kilobytes on Linux, which is a classic
/// way to publish a benchmark figure that is wrong by 1024x.
#[cfg(unix)]
pub fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid, correctly sized rusage for the kernel to
    // fill; the call cannot fail for RUSAGE_SELF.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) };
    if rc != 0 {
        return 0;
    }
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

#[cfg(not(unix))]
pub fn peak_rss_bytes() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_fills_in_the_basics() {
        let env = Environment::detect();
        assert!(env.cores >= 1);
        assert!(env.profile == "debug" || env.profile == "release");
    }

    #[test]
    fn tests_run_unprofiled_are_not_kpi_grade_in_debug() {
        let env = Environment::detect();
        // the suite runs under `cargo test` (debug), so this must be false;
        // it is the guard that stops a debug run being recorded as a KPI
        if env.profile == "debug" {
            assert!(!env.timings_are_kpi_grade());
        }
    }

    #[test]
    #[cfg(unix)]
    fn peak_rss_is_plausible() {
        let rss = peak_rss_bytes();
        // any live process has touched at least a megabyte, and a test binary
        // is nowhere near a terabyte: this catches the kB/B unit mistake
        assert!(rss > 1 << 20, "peak rss {rss} implausibly small");
        assert!(rss < 1 << 40, "peak rss {rss} implausibly large");
    }
}

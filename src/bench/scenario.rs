use std::fmt;

/// One measurable workload. Sample counts differ per scenario because a
/// sample here costs seconds, not microseconds: a uniform count would either
/// make the cheap scenarios imprecise or the expensive ones unbearable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// Steady-state decoding: prefill `prompt` tokens untimed, then time
    /// `new_tokens` single-token steps. `prompt` sets the cache depth the
    /// decode runs at, which is what makes attention cost visible.
    Decode { prompt: usize, new_tokens: usize },
    /// One batched forward pass over `prompt` tokens, timed end to end. Also
    /// the time-to-first-token measurement.
    Prefill { prompt: usize },
    /// Opening the GGUF and building the model.
    Load,
}

impl Scenario {
    /// The default set. The headline decode KPI runs first so a broken run
    /// fails fast, and `Load` runs *last* on purpose: it builds a second model
    /// while the harness still holds the first, and peak RSS is monotonic, so
    /// measuring it earlier would inflate every scenario after it.
    pub fn defaults() -> Vec<Scenario> {
        vec![
            Scenario::Decode {
                prompt: 1,
                new_tokens: 32,
            },
            Scenario::Decode {
                prompt: 256,
                new_tokens: 32,
            },
            Scenario::Prefill { prompt: 128 },
            Scenario::Prefill { prompt: 512 },
            Scenario::Load,
        ]
    }

    /// Samples taken when the caller does not override the count. Scaled by
    /// how long one sample takes, so the whole default run stays in minutes.
    pub const fn default_samples(self) -> usize {
        match self {
            Scenario::Load => 3,
            Scenario::Decode { prompt, .. } => {
                if prompt <= 1 {
                    10
                } else {
                    3
                }
            }
            Scenario::Prefill { prompt } => {
                if prompt <= 128 {
                    10
                } else {
                    3
                }
            }
        }
    }

    pub const fn default_warmup(self) -> usize {
        match self {
            Scenario::Load => 1,
            Scenario::Decode { prompt, .. } => {
                if prompt <= 1 {
                    2
                } else {
                    1
                }
            }
            Scenario::Prefill { .. } => 1,
        }
    }

    /// Tokens whose production is being timed. Prefill charges the whole
    /// prompt, since that is the work the forward pass did.
    pub const fn timed_tokens(self) -> usize {
        match self {
            Scenario::Load => 0,
            Scenario::Decode { new_tokens, .. } => new_tokens,
            Scenario::Prefill { prompt } => prompt,
        }
    }

    /// Cache depth the timed work starts from, which the report prints so a
    /// decode figure is never read without its context length.
    pub const fn cache_depth(self) -> usize {
        match self {
            Scenario::Load | Scenario::Prefill { .. } => 0,
            Scenario::Decode { prompt, .. } => prompt,
        }
    }

    /// Longest context the scenario touches, checked against `n_ctx` before
    /// running so a misconfigured sweep fails immediately instead of midway.
    pub const fn max_context(self) -> usize {
        match self {
            Scenario::Load => 0,
            Scenario::Decode { prompt, new_tokens } => prompt + new_tokens + 1,
            Scenario::Prefill { prompt } => prompt + 1,
        }
    }

    pub const fn key(self) -> &'static str {
        match self {
            Scenario::Load => "load",
            Scenario::Decode { .. } => "decode",
            Scenario::Prefill { .. } => "prefill",
        }
    }
}

impl fmt::Display for Scenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scenario::Load => write!(f, "load"),
            Scenario::Decode { prompt, new_tokens } => {
                write!(f, "decode@{prompt} x{new_tokens}")
            }
            Scenario::Prefill { prompt } => write!(f, "prefill@{prompt}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scenarios_fit_a_1024_token_window() {
        for scenario in Scenario::defaults() {
            assert!(
                scenario.max_context() <= 1024,
                "{scenario} needs {} positions",
                scenario.max_context()
            );
        }
    }

    #[test]
    fn expensive_scenarios_take_fewer_samples() {
        let cheap = Scenario::Decode {
            prompt: 1,
            new_tokens: 32,
        };
        let dear = Scenario::Decode {
            prompt: 256,
            new_tokens: 32,
        };
        assert!(cheap.default_samples() > dear.default_samples());
    }

    #[test]
    fn timed_tokens_reflect_what_was_measured() {
        // decode charges only the generated tokens...
        assert_eq!(
            Scenario::Decode {
                prompt: 256,
                new_tokens: 32
            }
            .timed_tokens(),
            32
        );
        // ...prefill charges the whole prompt it processed
        assert_eq!(Scenario::Prefill { prompt: 512 }.timed_tokens(), 512);
        assert_eq!(Scenario::Load.timed_tokens(), 0);
    }

    #[test]
    fn display_is_unambiguous_about_depth() {
        assert_eq!(
            Scenario::Decode {
                prompt: 256,
                new_tokens: 32
            }
            .to_string(),
            "decode@256 x32"
        );
        assert_eq!(Scenario::Prefill { prompt: 128 }.to_string(), "prefill@128");
    }
}

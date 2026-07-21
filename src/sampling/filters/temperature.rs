use crate::sampling::Candidates;
use crate::sampling::filters::Filter;

/// Logit scaling. Higher temperature flattens the distribution, lower sharpens
/// it, and `<= 0` collapses it to the argmax.
#[derive(Debug)]
pub struct Temperature {
    temp: f32,
}

impl Temperature {
    pub const fn new(temp: f32) -> Temperature {
        Temperature { temp }
    }
}

impl Filter for Temperature {
    /// Mirrors `llama_sampler_temp_impl`. Dividing by a positive temperature
    /// is monotonic, so a sorted set stays sorted; the `<= 0` branch keeps the
    /// argmax and sends every other logit to -inf, which leaves all the
    /// probability mass on one token and makes the later draw a formality.
    /// That is why greedy decoding needs no separate code path.
    fn apply(&self, candidates: &mut Candidates) {
        if self.temp > 0.0 {
            for entry in candidates.entries_mut() {
                entry.logit /= self.temp;
            }
            return;
        }

        let Some(max) = candidates.argmax_index() else {
            return;
        };
        for (index, entry) in candidates.entries_mut().iter_mut().enumerate() {
            if index != max {
                entry.logit = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::Candidates;

    fn candidates(logits: &[f32]) -> Candidates {
        let mut c = Candidates::new();
        c.reset_from_logits(logits).expect("valid logits");
        c
    }

    #[test]
    fn positive_temperature_scales_every_logit() {
        let mut c = candidates(&[1.0, 2.0, 4.0]);
        Temperature::new(2.0).apply(&mut c);
        let logits: Vec<f32> = c.entries().iter().map(|e| e.logit).collect();
        assert_eq!(logits, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn scaling_preserves_the_ranking() {
        let mut c = candidates(&[1.0, 3.0, 2.0]);
        Temperature::new(0.25).apply(&mut c);
        assert_eq!(c.argmax_index(), Some(1));
    }

    #[test]
    fn zero_temperature_collapses_onto_the_argmax() {
        let mut c = candidates(&[1.0, 3.0, 2.0]);
        Temperature::new(0.0).apply(&mut c);
        let logits: Vec<f32> = c.entries().iter().map(|e| e.logit).collect();
        assert_eq!(logits, vec![f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY]);

        c.softmax();
        assert_eq!(c.entries()[1].p, 1.0);
    }

    #[test]
    fn collapse_keeps_the_first_of_tied_maxima() {
        let mut c = candidates(&[3.0, 3.0]);
        Temperature::new(-1.0).apply(&mut c);
        assert_eq!(c.entries()[0].logit, 3.0);
        assert_eq!(c.entries()[1].logit, f32::NEG_INFINITY);
    }
}

use crate::sampling::SamplingError;
use crate::tokenizer::TokenId;

/// One token in the running candidate set: its id, its (possibly rescaled)
/// logit, and the probability [`Candidates::softmax`] last wrote.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub id: TokenId,
    pub logit: f32,
    pub p: f32,
}

/// The candidate set a sampler chain narrows down, mirroring llama.cpp's
/// `llama_token_data_array`. Filters mutate it in place; the buffer is reused
/// across tokens so steady-state sampling allocates nothing.
#[derive(Debug, Default)]
pub struct Candidates {
    entries: Vec<Candidate>,
    /// Whether `entries` is in descending logit order. Tracked rather than
    /// recomputed so a chain sorts at most once.
    sorted: bool,
}

impl Candidates {
    pub fn new() -> Candidates {
        Candidates::default()
    }

    /// Refills from one row of logits, one candidate per vocab entry. The
    /// existing allocation is reused.
    pub fn reset_from_logits(&mut self, logits: &[f32]) -> Result<(), SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }
        self.entries.clear();
        self.entries.reserve(logits.len());
        for (index, &logit) in logits.iter().enumerate() {
            // NaN would poison every comparison downstream; a NaN logit means
            // the forward pass is broken, so say so here rather than sample garbage
            if logit.is_nan() {
                return Err(SamplingError::NaNLogit { index });
            }
            self.entries.push(Candidate {
                id: index as TokenId,
                logit,
                p: 0.0,
            });
        }
        self.sorted = false;
        Ok(())
    }

    pub fn entries(&self) -> &[Candidate] {
        &self.entries
    }

    /// Mutable view for filters. Rescaling logits monotonically (or setting
    /// non-maximal ones to -inf) preserves the sort order; any other change
    /// must be followed by [`Self::mark_unsorted`].
    pub fn entries_mut(&mut self) -> &mut [Candidate] {
        &mut self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub const fn is_sorted(&self) -> bool {
        self.sorted
    }

    pub const fn mark_unsorted(&mut self) {
        self.sorted = false;
    }

    /// Sorts by descending logit, breaking ties by ascending id so a set with
    /// equal logits always collapses to the same token.
    pub fn sort_desc(&mut self) {
        if self.sorted {
            return;
        }
        self.entries
            .sort_unstable_by(|a, b| b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id)));
        self.sorted = true;
    }

    /// Keeps the first `n` candidates. Never empties the set.
    pub fn truncate(&mut self, n: usize) {
        debug_assert!(n >= 1, "truncating the candidate set to nothing");
        self.entries.truncate(n.max(1));
    }

    /// Index of the highest logit, first one winning a tie -- the comparison
    /// `llama_sampler_temp_impl` uses.
    pub fn argmax_index(&self) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            let better = best.is_none_or(|(_, logit)| entry.logit > logit);
            if better {
                best = Some((i, entry.logit));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Writes normalized probabilities into every candidate's `p`, shifting by
    /// the max logit for numerical stability.
    pub fn softmax(&mut self) {
        let max = self.max_logit();
        let mut sum = 0.0f64;
        if max.is_finite() {
            for entry in &mut self.entries {
                let p = (entry.logit - max).exp();
                entry.p = p;
                sum += f64::from(p);
            }
        } else {
            // every logit is -inf (or the set carries no finite mass): fall
            // back to a uniform distribution rather than divide by zero
            for entry in &mut self.entries {
                entry.p = 1.0;
                sum += 1.0;
            }
        }
        let inv = 1.0 / sum;
        for entry in &mut self.entries {
            entry.p = (f64::from(entry.p) * inv) as f32;
        }
    }

    fn max_logit(&self) -> f32 {
        if self.sorted {
            return self.entries.first().map_or(f32::NEG_INFINITY, |e| e.logit);
        }
        self.entries
            .iter()
            .fold(f32::NEG_INFINITY, |acc, e| acc.max(e.logit))
    }

    /// Picks a candidate by inverse-CDF over the current `p` values, where
    /// `r` is uniform in `[0, 1)`.
    ///
    /// The comparison is strict so a zero-probability candidate is never
    /// selected at `r == 0`; if rounding leaves the cumulative sum just short
    /// of `r`, the most probable candidate is the fallback.
    pub fn draw(&self, r: f64) -> Option<TokenId> {
        let mut best = *self.entries.first()?;
        let mut cumulative = 0.0f64;
        for entry in &self.entries {
            cumulative += f64::from(entry.p);
            if cumulative > r {
                return Some(entry.id);
            }
            if entry.p > best.p {
                best = *entry;
            }
        }
        Some(best.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(logits: &[f32]) -> Candidates {
        let mut c = Candidates::new();
        c.reset_from_logits(logits).expect("valid logits");
        c
    }

    #[test]
    fn reset_rejects_empty_and_nan() {
        let mut c = Candidates::new();
        assert_eq!(
            c.reset_from_logits(&[]).unwrap_err(),
            SamplingError::EmptyLogits
        );
        assert_eq!(
            c.reset_from_logits(&[1.0, f32::NAN]).unwrap_err(),
            SamplingError::NaNLogit { index: 1 }
        );
    }

    #[test]
    fn ids_follow_the_logit_positions() {
        let c = candidates(&[0.5, -1.0, 2.0]);
        assert_eq!(c.len(), 3);
        assert_eq!(c.entries()[2].id, 2);
        assert_eq!(c.entries()[2].logit, 2.0);
        assert!(!c.is_sorted());
    }

    #[test]
    fn sort_is_descending_with_ties_by_id() {
        let mut c = candidates(&[1.0, 3.0, 3.0, 2.0]);
        c.sort_desc();
        let ids: Vec<u32> = c.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 0]);
        assert!(c.is_sorted());
    }

    #[test]
    fn argmax_breaks_ties_toward_the_first() {
        assert_eq!(candidates(&[1.0, 3.0, 3.0]).argmax_index(), Some(1));
        assert_eq!(candidates(&[5.0]).argmax_index(), Some(0));
    }

    #[test]
    fn softmax_normalizes_and_is_shift_invariant() {
        let mut a = candidates(&[1.0, 2.0, 3.0]);
        let mut b = candidates(&[101.0, 102.0, 103.0]);
        a.softmax();
        b.softmax();

        let sum: f32 = a.entries().iter().map(|e| e.p).sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities sum to {sum}");
        for (x, y) in a.entries().iter().zip(b.entries()) {
            assert!((x.p - y.p).abs() < 1e-6, "{} vs {}", x.p, y.p);
        }
        // and ordered like the logits
        assert!(a.entries()[0].p < a.entries()[2].p);
    }

    #[test]
    fn softmax_treats_an_all_neg_inf_set_as_uniform() {
        let mut c = candidates(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        c.softmax();
        for entry in c.entries() {
            assert!((entry.p - 0.5).abs() < 1e-6, "got {}", entry.p);
        }
    }

    #[test]
    fn draw_walks_the_cumulative_distribution() {
        let mut c = candidates(&[0.0, 0.0]);
        c.softmax(); // p = [0.5, 0.5]
        assert_eq!(c.draw(0.0), Some(0));
        assert_eq!(c.draw(0.49), Some(0));
        assert_eq!(c.draw(0.5), Some(1));
        assert_eq!(c.draw(0.999), Some(1));
    }

    #[test]
    fn draw_never_picks_a_zero_probability_candidate() {
        // a collapsed set: all mass on the last entry
        let mut c = candidates(&[f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]);
        c.softmax();
        for r in [0.0, 0.25, 0.5, 0.75, 0.999] {
            assert_eq!(c.draw(r), Some(2), "r = {r}");
        }
    }

    #[test]
    fn truncate_keeps_the_leading_candidates() {
        let mut c = candidates(&[1.0, 3.0, 2.0]);
        c.sort_desc();
        c.truncate(2);
        let ids: Vec<u32> = c.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }
}

use crate::sampling::Candidates;
use crate::sampling::filters::Filter;

/// Nucleus sampling: keeps the shortest leading run of candidates whose
/// probabilities reach `p`.
#[derive(Debug)]
pub struct TopP {
    p: f32,
    min_keep: usize,
}

impl TopP {
    /// `p >= 1.0` disables the filter. `min_keep` is a floor on the surviving
    /// count, so a single dominant token cannot starve the set.
    pub const fn new(p: f32, min_keep: usize) -> TopP {
        TopP { p, min_keep }
    }
}

impl Filter for TopP {
    /// Mirrors `llama_sampler_top_p_apply`: probabilities are accumulated in
    /// descending order and the cut lands just past the candidate that brings
    /// the running sum to `p`.
    fn apply(&self, candidates: &mut Candidates) {
        if self.p >= 1.0 {
            return;
        }
        candidates.sort_desc();
        candidates.softmax();

        let min_keep = self.min_keep.max(1);
        let mut cumulative = 0.0f32;
        let mut last = candidates.len();
        for (index, entry) in candidates.entries().iter().enumerate() {
            cumulative += entry.p;
            if cumulative >= self.p && index + 1 >= min_keep {
                last = index + 1;
                break;
            }
        }
        candidates.truncate(last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Logits that are log-probabilities, so softmax reproduces `probs`.
    fn candidates_with_probs(probs: &[f32]) -> Candidates {
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let mut c = Candidates::new();
        c.reset_from_logits(&logits).expect("valid logits");
        c
    }

    #[test]
    fn keeps_the_minimal_prefix_reaching_p() {
        // cumulative: 0.5, 0.75, 0.9, 1.0
        let mut c = candidates_with_probs(&[0.5, 0.25, 0.15, 0.1]);
        TopP::new(0.7, 1).apply(&mut c);
        let ids: Vec<u32> = c.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn a_single_candidate_can_satisfy_p() {
        let mut c = candidates_with_probs(&[0.9, 0.05, 0.05]);
        TopP::new(0.8, 1).apply(&mut c);
        assert_eq!(c.len(), 1);
        assert_eq!(c.entries()[0].id, 0);
    }

    #[test]
    fn min_keep_is_a_floor_on_the_survivors() {
        let mut c = candidates_with_probs(&[0.9, 0.05, 0.05]);
        TopP::new(0.8, 3).apply(&mut c);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn p_of_one_disables_the_filter() {
        let mut c = candidates_with_probs(&[0.5, 0.25, 0.25]);
        TopP::new(1.0, 1).apply(&mut c);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn survivors_are_the_most_probable_ones() {
        let mut c = candidates_with_probs(&[0.1, 0.6, 0.3]);
        TopP::new(0.85, 1).apply(&mut c);
        let ids: Vec<u32> = c.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2]);
    }
}

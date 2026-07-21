use crate::sampling::Candidates;
use crate::sampling::filters::Filter;

/// Keeps the `k` highest-logit candidates.
#[derive(Debug)]
pub struct TopK {
    k: usize,
}

impl TopK {
    /// `k == 0` disables the filter, matching llama.cpp's `k <= 0` guard.
    pub const fn new(k: usize) -> TopK {
        TopK { k }
    }
}

impl Filter for TopK {
    fn apply(&self, candidates: &mut Candidates) {
        if self.k == 0 || self.k >= candidates.len() {
            return;
        }
        candidates.sort_desc();
        candidates.truncate(self.k);
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
    fn keeps_exactly_k_highest_candidates() {
        let mut c = candidates(&[0.0, 5.0, 1.0, 4.0, 2.0]);
        TopK::new(3).apply(&mut c);
        let ids: Vec<u32> = c.entries().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 3, 4]);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn zero_disables_the_filter() {
        let mut c = candidates(&[0.0, 5.0, 1.0]);
        TopK::new(0).apply(&mut c);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn a_k_past_the_end_keeps_everything() {
        let mut c = candidates(&[0.0, 5.0, 1.0]);
        TopK::new(99).apply(&mut c);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn k_of_one_leaves_the_argmax() {
        let mut c = candidates(&[0.0, 5.0, 1.0]);
        TopK::new(1).apply(&mut c);
        assert_eq!(c.len(), 1);
        assert_eq!(c.entries()[0].id, 1);
    }
}

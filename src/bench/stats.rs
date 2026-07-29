//! Order statistics over a sample set.
//!
//! The median is the headline everywhere rather than the mean: benchmark noise
//! on a laptop is one-sided (something else steals a core, never gives one
//! back), so the mean chases outliers the median ignores.

/// A sorted sample set. Construction sorts once; every query is then a lookup.
#[derive(Debug, Clone)]
pub struct Samples {
    sorted: Vec<f64>,
}

impl Samples {
    /// Panics if `values` is empty: a statistic over no samples is a bug in
    /// the caller, not a value to propagate.
    pub fn new(mut values: Vec<f64>) -> Samples {
        assert!(!values.is_empty(), "statistics over an empty sample set");
        values.sort_by(f64::total_cmp);
        Samples { sorted: values }
    }

    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    pub fn min(&self) -> f64 {
        self.sorted[0]
    }

    pub fn max(&self) -> f64 {
        self.sorted[self.sorted.len() - 1]
    }

    /// Linearly interpolated percentile, `q` in `[0, 1]`. Interpolating rather
    /// than picking a nearest rank matters at the sample counts we use: with
    /// 10 samples, nearest-rank p10 and p90 would just be the min and max.
    pub fn percentile(&self, q: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&q), "percentile out of range: {q}");
        let n = self.sorted.len();
        if n == 1 {
            return self.sorted[0];
        }
        let pos = q * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            return self.sorted[lo];
        }
        let frac = pos - lo as f64;
        self.sorted[lo] * (1.0 - frac) + self.sorted[hi] * frac
    }

    pub fn median(&self) -> f64 {
        self.percentile(0.5)
    }

    /// Median absolute deviation: the median of the distances from the median.
    /// Reported instead of the standard deviation because it is not inflated
    /// by the single slow run a scheduler hiccup produces.
    pub fn mad(&self) -> f64 {
        let median = self.median();
        let deviations = self.sorted.iter().map(|v| (v - median).abs()).collect();
        Samples::new(deviations).median()
    }

    /// MAD as a fraction of the median, the spread figure worth eyeballing:
    /// anything above a few percent means the machine was not quiet.
    pub fn relative_spread(&self) -> f64 {
        let median = self.median();
        if median == 0.0 {
            return 0.0;
        }
        self.mad() / median
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn median_of_odd_and_even_counts() {
        assert!(close(Samples::new(vec![3.0, 1.0, 2.0]).median(), 2.0));
        assert!(close(Samples::new(vec![4.0, 1.0, 3.0, 2.0]).median(), 2.5));
    }

    #[test]
    fn a_single_sample_is_its_own_every_statistic() {
        let s = Samples::new(vec![7.5]);
        assert!(close(s.median(), 7.5));
        assert!(close(s.percentile(0.1), 7.5));
        assert!(close(s.percentile(0.9), 7.5));
        assert!(close(s.mad(), 0.0));
    }

    #[test]
    fn percentiles_interpolate_between_ranks() {
        // 0..=10, so the q-th percentile is exactly 10q
        let s = Samples::new((0..=10).map(f64::from).collect());
        assert!(close(s.percentile(0.0), 0.0));
        assert!(close(s.percentile(0.1), 1.0));
        assert!(close(s.percentile(0.5), 5.0));
        assert!(close(s.percentile(0.9), 9.0));
        assert!(close(s.percentile(1.0), 10.0));
        // between ranks
        assert!(close(s.percentile(0.25), 2.5));
    }

    #[test]
    fn input_order_does_not_matter() {
        let ascending = Samples::new(vec![1.0, 2.0, 3.0, 4.0]);
        let shuffled = Samples::new(vec![3.0, 1.0, 4.0, 2.0]);
        assert!(close(ascending.median(), shuffled.median()));
        assert!(close(ascending.percentile(0.9), shuffled.percentile(0.9)));
    }

    #[test]
    fn mad_ignores_a_lone_outlier() {
        // one run took 100x as long; the median and MAD barely move
        let clean = Samples::new(vec![10.0, 10.0, 10.0, 11.0, 9.0]);
        let spiked = Samples::new(vec![10.0, 10.0, 10.0, 11.0, 1000.0]);
        assert!(close(clean.median(), spiked.median()));
        assert!(spiked.mad() <= 1.0, "mad was {}", spiked.mad());
    }

    #[test]
    fn relative_spread_is_zero_for_identical_samples() {
        let s = Samples::new(vec![5.0; 8]);
        assert!(close(s.relative_spread(), 0.0));
        assert!(close(s.min(), 5.0));
        assert!(close(s.max(), 5.0));
    }
}

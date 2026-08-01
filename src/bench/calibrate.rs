//! Turns a wall-clock budget into a sample count.

use std::time::Duration;

/// How many samples a time budget affords: divide the budget by one call's
/// duration, clamp to `[min, max]` samples. Pure arithmetic over a
/// measurement someone else took -- [`TimedCall::num_iterations`] is the
/// pairing that takes it.
///
/// One measured call is deliberately all the calibration spends: the
/// functions benchmarked here (matmul-sized, microseconds and up) dwarf
/// timer resolution, and the `min` floor absorbs what noise remains. Calls
/// too fast to time saturate to `max` samples; calls slower than the whole
/// budget still get `min`.
///
/// [`TimedCall::num_iterations`]: crate::bench::TimedCall::num_iterations
#[derive(Debug, Clone, Copy)]
pub struct SampleBudget {
    target: Duration,
    min_samples: usize,
    max_samples: usize,
}

impl SampleBudget {
    /// Panics if the bounds are inverted or `min_samples` is zero: a budget
    /// affording no samples is a bug in the caller, not a value to return.
    pub fn new(target: Duration, min_samples: usize, max_samples: usize) -> SampleBudget {
        assert!(min_samples >= 1, "a budget must afford at least one sample");
        assert!(
            min_samples <= max_samples,
            "inverted sample bounds: {min_samples} > {max_samples}"
        );
        SampleBudget {
            target,
            min_samples,
            max_samples,
        }
    }

    /// How many iterations of a call taking `once` fit
    /// [`target`](SampleBudget::new). A zero duration divides to infinity,
    /// which the float-to-int cast saturates and the clamp caps at
    /// `max_samples`.
    pub fn num_iterations(&self, once: Duration) -> usize {
        let fits = (self.target.as_secs_f64() / once.as_secs_f64()) as usize;
        fits.clamp(self.min_samples, self.max_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_duration_call_is_capped_at_max_samples() {
        let budget = SampleBudget::new(Duration::from_millis(50), 2, 30);
        assert_eq!(budget.num_iterations(Duration::ZERO), 30);
    }

    #[test]
    fn a_call_slower_than_the_budget_still_gets_min_samples() {
        let budget = SampleBudget::new(Duration::from_millis(1), 3, 30);
        assert_eq!(budget.num_iterations(Duration::from_millis(5)), 3);
    }

    #[test]
    fn a_mid_range_call_divides_the_budget_exactly() {
        let budget = SampleBudget::new(Duration::from_millis(50), 1, 1000);
        assert_eq!(budget.num_iterations(Duration::from_millis(5)), 10);
    }

    #[test]
    #[should_panic(expected = "inverted sample bounds")]
    fn inverted_bounds_panic() {
        SampleBudget::new(Duration::from_millis(1), 10, 5);
    }
}

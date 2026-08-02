//! The timing primitive every bench measurement goes through.

use std::time::{Duration, Instant};

use crate::bench::SampleBudget;

/// A closure captured once, measured on demand: [`measure`] runs it and
/// returns how long it took. The one place bench code touches a clock --
/// calibration ([`SampleBudget`]) and sampling loops share an instance, so
/// a bench author never hand-rolls `Instant` pairs or forgets which side of
/// the call to read the clock on.
///
/// Setup that should not be timed (buffer allocation, weight packing)
/// belongs outside the closure, in whatever constructed it.
///
/// [`measure`]: TimedCall::measure
/// [`SampleBudget`]: crate::bench::SampleBudget
pub struct TimedCall<F: FnMut()> {
    call: F,
}

impl<F: FnMut()> TimedCall<F> {
    pub fn new(call: F) -> TimedCall<F> {
        TimedCall { call }
    }

    /// Runs the call once and returns its wall-clock duration.
    pub fn measure(&mut self) -> Duration {
        let start = Instant::now();
        (self.call)();
        start.elapsed()
    }

    /// Measures one call and asks `budget` how many samples it affords.
    /// The measured execution is real, so it doubles as the caller's
    /// warmup -- one discarded run serving both purposes.
    pub fn num_iterations(&mut self, budget: SampleBudget) -> usize {
        budget.num_iterations(self.measure())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_measure_runs_the_call() {
        let mut runs = 0;
        let mut timed = TimedCall::new(|| runs += 1);
        timed.measure();
        timed.measure();
        assert_eq!(runs, 2);
    }

    #[test]
    fn a_sleep_measures_at_least_its_duration() {
        let nap = Duration::from_millis(5);
        let mut timed = TimedCall::new(|| std::thread::sleep(nap));
        assert!(timed.measure() >= nap);
    }

    #[test]
    fn calibration_executes_the_call_and_respects_the_bounds() {
        let budget = SampleBudget::new(Duration::from_millis(1), 2, 10);
        let mut ran = false;
        let mut timed = TimedCall::new(|| ran = true);
        let iters = timed.num_iterations(budget);
        assert!(ran);
        assert!((2..=10).contains(&iters), "iters was {iters}");
    }
}

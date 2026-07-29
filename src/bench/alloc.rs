//! A counting global allocator, so allocations per token can be measured
//! with profiling compiled out.
//!
//! Two relaxed atomic adds per allocation cost a few nanoseconds against the
//! ~1000 allocations of a 50 ms token, which is far below the noise floor of
//! the timings it sits alongside. `hotpath-alloc` gives the same totals broken
//! down per function, but it installs its own global allocator, so this one
//! stands down when that feature is on.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: the caller upholds GlobalAlloc's contract; we only count and
        // delegate to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above, delegated unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // a realloc that has to move is a fresh allocation from the caller's
        // point of view, so it is counted as one
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: as above, delegated unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Allocation totals since process start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocStats {
    pub count: u64,
    pub bytes: u64,
}

impl AllocStats {
    pub fn now() -> AllocStats {
        AllocStats {
            count: ALLOC_COUNT.load(Ordering::Relaxed),
            bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    /// Totals accumulated since `self` was taken.
    pub fn since(self, earlier: AllocStats) -> AllocStats {
        AllocStats {
            count: self.count.saturating_sub(earlier.count),
            bytes: self.bytes.saturating_sub(earlier.bytes),
        }
    }
}

/// Whether the `bench` binary installs [`CountingAllocator`] as the global
/// allocator, and so whether [`AllocStats`] means anything there. False when
/// `hotpath-alloc` owns the global allocator, in which case its per-function
/// table is the source of allocation figures instead.
pub const fn is_installed() -> bool {
    !cfg!(feature = "hotpath-alloc")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// The two tests below read exact deltas of the shared counters while
    /// driving the allocator directly; run concurrently they see each other's
    /// bumps. Serialize them (ignoring poisoning: a failed test must not fail
    /// the other one spuriously).
    static COUNTER_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn deltas_subtract_the_baseline() {
        let a = AllocStats {
            count: 10,
            bytes: 100,
        };
        let b = AllocStats {
            count: 25,
            bytes: 400,
        };
        assert_eq!(
            b.since(a),
            AllocStats {
                count: 15,
                bytes: 300
            }
        );
    }

    #[test]
    fn a_backwards_delta_saturates_instead_of_wrapping() {
        let a = AllocStats {
            count: 10,
            bytes: 100,
        };
        let b = AllocStats {
            count: 5,
            bytes: 50,
        };
        assert_eq!(b.since(a), AllocStats { count: 0, bytes: 0 });
    }

    /// The counters only move when the allocator is installed as *the* global
    /// allocator, which happens in the `bench` binary rather than in this test
    /// harness. So drive the counting logic directly.
    #[test]
    fn allocating_through_the_counter_moves_it() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let layout = Layout::from_size_align(4096, 8).expect("valid layout");
        let before = AllocStats::now();

        // SAFETY: a non-zero-sized layout, freed below with the same layout.
        let ptr = unsafe { CountingAllocator.alloc(layout) };
        assert!(!ptr.is_null(), "allocation failed");
        // SAFETY: `ptr` came from the call above with this exact layout.
        unsafe { CountingAllocator.dealloc(ptr, layout) };

        let delta = AllocStats::now().since(before);
        assert_eq!(delta.count, 1, "expected exactly one counted allocation");
        assert_eq!(delta.bytes, 4096);
    }

    #[test]
    fn reallocating_counts_as_a_further_allocation() {
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let layout = Layout::from_size_align(64, 8).expect("valid layout");
        // SAFETY: non-zero-sized layout; the pointer is grown then freed with
        // the layout each call requires.
        unsafe {
            let ptr = CountingAllocator.alloc(layout);
            assert!(!ptr.is_null());
            let before = AllocStats::now();
            let grown = CountingAllocator.realloc(ptr, layout, 256);
            assert!(!grown.is_null());
            let delta = AllocStats::now().since(before);
            assert_eq!(delta.count, 1);
            assert_eq!(delta.bytes, 256);
            CountingAllocator.dealloc(grown, Layout::from_size_align(256, 8).unwrap());
        }
    }
}

//! The pre-allocated f32 arena backing all forward-pass memory.
//!
//! One allocation at model load, zero allocations on the hot path: a model's
//! `Layout` sizes the arena once, splits it into a scratch partition (dead
//! after every pass) and a persistent tail (the KV cache), and carves the
//! scratch into named worst-case regions. The arena itself is dumb storage --
//! all layout intelligence lives with the model.

/// A fixed span of f32s whose start is 64-byte aligned. `Vec<f32>` only
/// guarantees 4-byte alignment, so the buffer over-allocates by up to one
/// cacheline and exposes the aligned subslice.
pub struct Arena {
    buf: Vec<f32>,
    start: usize,
    len: usize,
}

impl Arena {
    /// The alignment target: one cacheline.
    pub const CACHELINE_BYTES: usize = 64;

    /// How many [`f32`] can be contained by a single cacheline; region sizes
    /// round up to this.
    pub const ALIGN_FLOATS: usize = Self::CACHELINE_BYTES / size_of::<f32>();

    /// Rounds a region size up to a whole number of cachelines: sizes that
    /// are multiples of [`Self::ALIGN_FLOATS`] keep every region start
    /// 64-byte aligned when carved from the arena's aligned base.
    pub const fn round_up(n_floats: usize) -> usize {
        n_floats.next_multiple_of(Self::ALIGN_FLOATS)
    }

    /// Allocates `len` usable floats starting on a 64-byte boundary.
    pub fn new(len: usize) -> Self {
        let buf = vec![0.0f32; len + Self::ALIGN_FLOATS - 1];

        // Return the virtual memory address of the allocated buffer. This corresponds to the
        // virtual memory address of the first byte of the vector.
        //
        // A virtual memory address is always `(page_number|offset)` where:
        //
        // - `page_number`: is the index of the page containing that number
        //
        // - `offset` (low bits): are the offsets within the page where the byte is located.
        //                        notice in fact that a page contains `2^{offset_bits}` bytes.
        //                        The capacity of each page (and so the number of bits reserved
        //                        for the offset) depend on the OS, so they are not knowable
        //                        at runtime. As a result, the `offset` part of the memory
        //                        address is always a value in `0..2^{offset_bits}`
        //
        // Because of bit concatenation, when encoded as an integer, the virtual memory address
        // is equal to `page_number * 2^{offset_bits} + offset` (because the `page_number` is
        // shifted to the left by `offset_bits` bits)
        //
        // Encoded as integer, any virtual memory address is a multiple of the `alignment` of
        // the associated type. Since the virtual memory address of a buffer is equal to the
        // virtual memory address of the first byte, a buffer inherits the same `alignment`
        // of the inner type.
        //
        // A value is "aligned" indeed when the associated virtual memory address is a multiple
        // of the alignment of the associated type, and it is a desirable property both for
        // performance reasons (e.g. be sure of loading the value in a single cache line load)
        // and security reasons (e.g. allowing atomic operations).
        // Indeed, while memory looks like a flat line of bytess, the machine manages it as a
        // nested power of 2 boxes (word ⊂ cacheline ⊂ page) and alignment promises that the
        // values will tile efficiently within this boxes, so that the object never straddles
        // any bigger boundary it could have fitted inside.
        let addr = buf.as_ptr() as usize;

        // The address is guaranteed to be a multiple of `align_of::<T>()`, but not of
        // `CACHELINE_BYTES`, so we sacrifice sacrifice up to a cacheline of leading padding
        // so that the first element actually exposed by the vector has a memory address that
        // is multiple of `CACHELINE_BYTES`
        let misaligned_floats = (addr % Self::CACHELINE_BYTES) / size_of::<f32>();
        let start = if misaligned_floats == 0 {
            0
        } else {
            Self::ALIGN_FLOATS - misaligned_floats
        };
        Self { buf, start, len }
    }

    /// Usable floats, excluding the alignment padding.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The usable, aligned span; all carving starts here.
    pub fn floats_mut(&mut self) -> &mut [f32] {
        &mut self.buf[self.start..self.start + self.len]
    }

    pub fn floats(&self) -> &[f32] {
        &self.buf[self.start..self.start + self.len]
    }
}

/// Debug-build write-before-read tripwire: fills a scratch region with NaN so
/// a read of stale data poisons everything downstream.
///
/// This must be a no-op in release builds
pub fn poison(region: &mut [f32]) {
    if cfg!(debug_assertions) {
        region.fill(f32::NAN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_starts_on_a_cacheline_boundary() {
        for len in [0, 1, 15, 16, 1000] {
            let mut arena = Arena::new(len);
            let span = arena.floats_mut();
            assert_eq!(span.len(), len);
            let addr = span.as_ptr() as usize;
            assert_eq!(
                addr % Arena::CACHELINE_BYTES,
                0,
                "misaligned span for len {len}"
            );
        }
    }

    #[test]
    fn poison_is_a_debug_tripwire() {
        let mut region = vec![1.0f32; 8];
        poison(&mut region);
        if cfg!(debug_assertions) {
            assert!(region.iter().all(|v| v.is_nan()));
        } else {
            assert!(region.iter().all(|&v| v == 1.0));
        }
    }
}

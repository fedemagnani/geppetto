//! The fixed memory plan for the GPT-2 forward pass over an
//! [`Arena`](crate::arena::Arena): which regions exist, how large each is,
//! and in what order they are carved. The region list mirrors the dataflow
//! table in `prompts/3-pre-allocate-arena.md`; edit them together.

use crate::arena::{Arena, poison};
use crate::gpt2::HParams;

/// The scratch partition, obtainable only from [`Gpt2Layout::split`]: holding
/// one is proof the bytes are the head of the arena's aligned span. Dead data
/// after each pass -- poison it, carve it, drop it.
pub struct Gpt2Scratch<'a>(&'a mut [f32]);

impl Gpt2Scratch<'_> {
    /// Debug-build NaN prefill of the whole scratch partition, the
    /// write-before-read tripwire. [`Gpt2Persistent`] deliberately has no
    /// such method: it holds live data.
    pub fn poison(&mut self) {
        poison(self.0);
    }
}

/// The persistent partition (the arena's tail), obtainable only from
/// [`Gpt2Layout::split`]; its data lives across passes (the KV cache).
pub struct Gpt2Persistent<'a>(&'a mut [f32]);

impl<'a> Gpt2Persistent<'a> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn floats_mut(&mut self) -> &mut [f32] {
        self.0
    }

    /// Mounts the KV cache over the partition, consuming it: per layer,
    /// `[n_ctx, n_embd]` keys then `[n_ctx, n_embd]` values.
    pub fn into_kv_cache(self, n_layer: usize, n_ctx: usize, n_embd: usize) -> Gpt2KvCache<'a> {
        let needed = n_layer * 2 * n_ctx * n_embd;
        assert!(
            self.0.len() >= needed,
            "kv: persistent partition holds {} floats, {needed} needed",
            self.0.len(),
        );
        Gpt2KvCache {
            data: self.0,
            n_ctx,
            n_embd,
        }
    }
}

/// The persistent partition interpreted as the single-sequence KV cache. The
/// position bookkeeping (`n_past`) lives with the state that owns the arena;
/// this view only does the address arithmetic.
pub struct Gpt2KvCache<'a> {
    data: &'a mut [f32],
    n_ctx: usize,
    n_embd: usize,
}

impl Gpt2KvCache<'_> {
    fn layer_base(&self, layer: usize) -> usize {
        layer * 2 * self.n_ctx * self.n_embd
    }

    /// Writes one position's key and value rows for `layer` at `pos`.
    pub fn append_row(&mut self, layer: usize, pos: usize, k_row: &[f32], v_row: &[f32]) {
        assert!(pos < self.n_ctx, "kv: position {pos} beyond n_ctx");
        assert_eq!(k_row.len(), self.n_embd, "kv: key width");
        assert_eq!(v_row.len(), self.n_embd, "kv: value width");
        let k_at = self.layer_base(layer) + pos * self.n_embd;
        let v_at = k_at + self.n_ctx * self.n_embd;
        self.data[k_at..k_at + self.n_embd].copy_from_slice(k_row);
        self.data[v_at..v_at + self.n_embd].copy_from_slice(v_row);
    }

    /// The layer's first `n_kv` key rows as `[n_kv, n_embd]`, row-major.
    pub fn keys(&self, layer: usize, n_kv: usize) -> &[f32] {
        let base = self.layer_base(layer);
        &self.data[base..base + n_kv * self.n_embd]
    }

    /// The layer's first `n_kv` value rows as `[n_kv, n_embd]`, row-major.
    pub fn values(&self, layer: usize, n_kv: usize) -> &[f32] {
        let base = self.layer_base(layer) + self.n_ctx * self.n_embd;
        &self.data[base..base + n_kv * self.n_embd]
    }
}

/// The scratch partition carved into named worst-case regions. Raw slices,
/// not tensor views: each pass mounts `TensorView`s over the active prefix of
/// a region ([n tokens, width]), never over the whole carve.
pub struct Gpt2Views<'a> {
    /// Residual stream `[n_ctx, n_embd]`; updated in place only.
    pub x: &'a mut [f32],
    /// `[n_ctx, n_embd]`; remounted for ln1, ln2, ln_f.
    pub norm_out: &'a mut [f32],
    /// `[n_ctx, 3 * n_embd]`; rows are `q | k | v`.
    pub qkv: &'a mut [f32],
    /// `[n_ctx, n_ctx]`; one head at a time (sequential per-head loop).
    pub scores: &'a mut [f32],
    /// `[n_ctx, n_embd]`; heads write strided column blocks.
    pub attn_out: &'a mut [f32],
    /// `[n_ctx, n_ff]`; gelu applied in place.
    pub ffn_up: &'a mut [f32],
    /// `[n_ctx, n_embd]`; remounted for the attention and FFN-down projections.
    pub proj_out: &'a mut [f32],
    /// `[n_vocab]`, last token only; outlives the pass until the next
    /// forward call overwrites it.
    pub logits: &'a mut [f32],
}

/// Region sizes in floats, each pre-rounded, in carve order. Constructed once
/// at model load and stored, never recomputed per call: a field reorder or
/// size tweak in a "conveniently" rebuilt layout would silently shear the
/// persistent KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt2Layout {
    // scratch: dead after each pass, worst-case (n_ctx) sized
    x: usize,
    norm_out: usize,
    qkv: usize,
    scores: usize,
    attn_out: usize,
    ffn_up: usize,
    proj_out: usize,
    logits: usize,
    // persistent tail: lives across passes
    kv_cache: usize,
}

impl Gpt2Layout {
    /// Sizes every region from the hyperparameters. `n_vocab` is passed
    /// separately: it is derived from the token-embedding tensor at weight
    /// load, not from the `gpt2.*` metadata.
    pub fn new(hp: &HParams, n_vocab: usize) -> Gpt2Layout {
        let n_ctx = hp.n_ctx;
        let n_embd = hp.n_embd;
        Gpt2Layout {
            x: Arena::round_up(n_ctx * n_embd),
            norm_out: Arena::round_up(n_ctx * n_embd),
            qkv: Arena::round_up(n_ctx * 3 * n_embd),
            scores: Arena::round_up(n_ctx * n_ctx),
            attn_out: Arena::round_up(n_ctx * n_embd),
            ffn_up: Arena::round_up(n_ctx * hp.n_ff),
            proj_out: Arena::round_up(n_ctx * n_embd),
            logits: Arena::round_up(n_vocab),
            kv_cache: Arena::round_up(hp.n_layer * 2 * n_ctx * n_embd),
        }
    }

    pub fn scratch_total(&self) -> usize {
        self.x
            + self.norm_out
            + self.qkv
            + self.scores
            + self.attn_out
            + self.ffn_up
            + self.proj_out
            + self.logits
    }

    pub fn persistent_total(&self) -> usize {
        self.kv_cache
    }

    /// The arena size this layout needs.
    pub fn total(&self) -> usize {
        self.scratch_total() + self.persistent_total()
    }

    /// Splits the arena's aligned span into `(scratch | persistent)`, done
    /// once per pass before carving so KV and scratch borrows can coexist.
    /// Takes the [`Arena`] itself, not a slice: only its span is guaranteed
    /// to start 64-byte aligned, which every region offset builds on. The
    /// returned wrappers can be built nowhere else, so anything downstream
    /// that holds one inherits the guarantee.
    pub fn split<'a>(&self, arena: &'a mut Arena) -> (Gpt2Scratch<'a>, Gpt2Persistent<'a>) {
        let span = arena.floats_mut();
        assert_eq!(
            span.len(),
            self.total(),
            "layout: arena span does not match the layout total"
        );
        let (scratch, persistent) = span.split_at_mut(self.scratch_total());
        (Gpt2Scratch(scratch), Gpt2Persistent(persistent))
    }

    /// Carves the scratch partition into its named regions, exhausting it
    /// exactly: a region missing here (or a stale size) fails loudly instead
    /// of shearing every later region. Consumes the [`Gpt2Scratch`], so a
    /// carve can only ever see bytes that came out of [`Gpt2Layout::split`].
    pub fn carve_scratch<'a>(&self, scratch: Gpt2Scratch<'a>) -> Gpt2Views<'a> {
        let (x, rest) = scratch.0.split_at_mut(self.x);
        let (norm_out, rest) = rest.split_at_mut(self.norm_out);
        let (qkv, rest) = rest.split_at_mut(self.qkv);
        let (scores, rest) = rest.split_at_mut(self.scores);
        let (attn_out, rest) = rest.split_at_mut(self.attn_out);
        let (ffn_up, rest) = rest.split_at_mut(self.ffn_up);
        let (proj_out, rest) = rest.split_at_mut(self.proj_out);
        let (logits, rest) = rest.split_at_mut(self.logits);
        assert!(rest.is_empty(), "carve must exhaust the scratch partition");
        Gpt2Views {
            x,
            norm_out,
            qkv,
            scores,
            attn_out,
            ffn_up,
            proj_out,
            logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;

    /// GPT-2 small, the shapes the dataflow table is written in.
    fn small() -> HParams {
        HParams {
            n_ctx: 1024,
            n_embd: 768,
            n_ff: 3072,
            n_head: 12,
            n_layer: 12,
            eps: 1e-5,
        }
    }

    const N_VOCAB: usize = 50257;

    #[test]
    fn regions_match_the_dataflow_table() {
        let hp = small();
        let layout = Gpt2Layout::new(&hp, N_VOCAB);
        let mut arena = Arena::new(layout.total());
        let (scratch, persistent) = layout.split(&mut arena);
        assert_eq!(persistent.len(), 12 * 2 * 1024 * 768);

        let views = layout.carve_scratch(scratch);
        assert_eq!(views.x.len(), 1024 * 768);
        assert_eq!(views.norm_out.len(), 1024 * 768);
        assert_eq!(views.qkv.len(), 1024 * 2304);
        assert_eq!(views.scores.len(), 1024 * 1024);
        assert_eq!(views.attn_out.len(), 1024 * 768);
        assert_eq!(views.ffn_up.len(), 1024 * 3072);
        assert_eq!(views.proj_out.len(), 1024 * 768);
        // n_vocab is not a multiple of 16; the region is rounded up
        assert_eq!(views.logits.len(), 50272);
    }

    #[test]
    fn every_region_start_is_cacheline_aligned() {
        // odd hparams force rounding on every region
        let hp = HParams {
            n_ctx: 3,
            n_embd: 5,
            n_ff: 7,
            n_head: 1,
            n_layer: 2,
            eps: 1e-5,
        };
        let layout = Gpt2Layout::new(&hp, 11);
        let mut arena = Arena::new(layout.total());
        let (scratch, mut persistent) = layout.split(&mut arena);
        let views = layout.carve_scratch(scratch);
        let regions: [&[f32]; 9] = [
            views.x,
            views.norm_out,
            views.qkv,
            views.scores,
            views.attn_out,
            views.ffn_up,
            views.proj_out,
            views.logits,
            persistent.floats_mut(),
        ];
        for (i, region) in regions.iter().enumerate() {
            let addr = region.as_ptr() as usize;
            assert_eq!(addr % 64, 0, "region {i} start is misaligned");
            assert!(
                region.len().is_multiple_of(Arena::ALIGN_FLOATS),
                "region {i} size"
            );
        }
    }

    #[test]
    #[should_panic(expected = "does not match the layout total")]
    fn split_rejects_a_mismatched_span() {
        let layout = Gpt2Layout::new(&small(), N_VOCAB);
        let mut arena = Arena::new(layout.total() + Arena::ALIGN_FLOATS);
        let _ = layout.split(&mut arena);
    }

    #[test]
    fn poison_reaches_scratch_and_never_persistent() {
        let hp = small();
        let layout = Gpt2Layout::new(&hp, N_VOCAB);
        let mut arena = Arena::new(layout.total());
        let (mut scratch, mut persistent) = layout.split(&mut arena);
        scratch.poison();
        assert!(persistent.floats_mut().iter().all(|v| v.is_finite()));
        if cfg!(debug_assertions) {
            let views = layout.carve_scratch(scratch);
            assert!(views.x.iter().all(|v| v.is_nan()));
            assert!(views.logits.iter().all(|v| v.is_nan()));
        }
    }

    #[test]
    fn layouts_from_equal_hparams_are_identical() {
        // the stored-once rule leans on this: rebuilding from the same inputs
        // must not move any region
        let a = Gpt2Layout::new(&small(), N_VOCAB);
        let b = Gpt2Layout::new(&small(), N_VOCAB);
        assert_eq!(a, b);
    }
}

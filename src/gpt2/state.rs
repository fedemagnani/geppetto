//! Per-sequence inference state: the arena, its layout, and the KV position
//! count. The model stays immutable during a forward pass -- everything a
//! pass writes lives here, so two sequences over one model are two states.

use crate::arena::Arena;
use crate::gpt2::HParams;
use crate::gpt2::layout::{Gpt2KvCache, Gpt2Layout, Gpt2Views};

pub struct Gpt2State {
    hp: HParams,
    /// Stored once here, never rebuilt per call: a rebuilt layout with a
    /// tweaked size would silently relocate the KV cache under the data.
    layout: Gpt2Layout,
    arena: Arena,
    n_past: usize,
}

impl Gpt2State {
    /// `kernel_scratch` is the worst-case float count the model's matmul
    /// kernel needs per call, already maxed over its call shapes -- a plain
    /// size, so state and layout stay kernel-agnostic.
    pub fn new(hp: HParams, n_vocab: usize, kernel_scratch: usize) -> Gpt2State {
        let layout = Gpt2Layout::new(&hp, n_vocab, kernel_scratch);
        let arena = Arena::new(layout.total());
        Gpt2State {
            hp,
            layout,
            arena,
            n_past: 0,
        }
    }

    /// Positions stored in the KV cache; what the next pass builds on.
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn is_empty(&self) -> bool {
        self.n_past == 0
    }

    /// Forgets all cached positions, reusing the arena as is.
    pub fn clear(&mut self) {
        self.n_past = 0;
    }

    pub(crate) fn advance(&mut self, n_new: usize) {
        self.n_past += n_new;
    }

    /// Splits and carves the arena for one pass: NaN-poisons the scratch
    /// partition (debug builds only), carves it into the named regions, and
    /// mounts the KV cache over the persistent tail.
    pub(crate) fn begin_pass(&mut self) -> (Gpt2Views<'_>, Gpt2KvCache<'_>) {
        let (mut scratch, persistent) = self.layout.split(&mut self.arena);
        scratch.poison();
        let views = self.layout.carve_scratch(scratch);
        let kv = persistent.into_kv_cache(self.hp.n_layer, self.hp.n_ctx, self.hp.n_embd);
        (views, kv)
    }
}

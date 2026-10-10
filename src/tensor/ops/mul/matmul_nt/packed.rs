//! The panel-packed tile: the NEON register tile of `neon` fed from one
//! sequential stream instead of `ROWS` parallel ones.
//!
//! The stream probe measured this machine's prefetcher sustaining full
//! streaming bandwidth (~73 GB/s) for at most 2 concurrent forward
//! streams, halving at 4 -- which is why every wider register tile lost
//! despite asm-perfect codegen. This kernel spends the [`pack`] hook to
//! dissolve the streams: each group of `ROWS` weight rows is interleaved
//! chunk-wise into one contiguous panel at model load, so the hot loop
//! walks one forward stream while keeping the wide tile's
//! `1 + 1/ROWS` loads per FMA.
//!
//! Padding replaces every tail path. Panels are zero-padded to a full
//! `ROWS` rows and to a `k` multiple of the chunk width, and the
//! activation rows are copied once per call into a zero-padded scratch
//! region ([`scratch_len`] is the family's first non-zero answer), so the
//! tile loop is the whole kernel: no remainder dots, no scalar `k` tail.
//! Zero rows cost zero output -- a padded lane only ever multiplies into
//! a sum nobody reads.
//!
//! [`pack`]: MatmulNtKernel::pack
//! [`scratch_len`]: MatmulNtKernel::scratch_len

use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

use crate::arena::Arena;
use crate::tensor::{MatmulNtKernel, Shape, TensorView, TensorViewMut, WeightTensor};

/// f32 lanes in one 128-bit NEON vector.
const FLOATS_IN_VECTOR: usize = 4;

/// The packed kernel: `ROWS` weight rows per panel, `VECS` accumulator
/// vectors per row, the register budget of the unpacked NEON tile
/// (`(ROWS + 1) x VECS <= 32`) with the stream count always 2: the
/// activations and the panel. The default tile is the bench sweep's peak,
/// 8 rows x 2 vectors: with the stream cliff dissolved, wider reuse
/// finally pays, and the deepest tile that still fits the register file
/// wins on every shape.
#[derive(Debug, Default, Clone, Copy)]
pub struct MatMulNtPackedNeon<const ROWS: usize = 8, const VECS: usize = 2>;

/// A weight repacked into panels: groups of `ROWS` rows interleaved
/// chunk-wise -- `LANES` floats of row 0, of row 1, ... of row
/// `ROWS - 1`, then every row's next chunk -- so one panel is read start
/// to end as a single sequential stream. Rows are zero-padded to a full
/// last panel and `k` to `k_pad`, a multiple of the chunk width; the
/// arena backing keeps the whole pack one 64-byte-aligned allocation.
pub struct PackedPanels {
    panels: Arena,
    n: usize,
    k: usize,
    k_pad: usize,
    rows: usize,
    lanes: usize,
}

impl<const ROWS: usize, const VECS: usize> MatmulNtKernel for MatMulNtPackedNeon<ROWS, VECS> {
    type Weights = PackedPanels;

    fn pack(&self, b: WeightTensor) -> PackedPanels {
        const {
            assert!(ROWS > 0 && VECS > 0);
        }
        let view = b.view();
        let (n, k) = (view.rows(), view.cols());
        let lanes = FLOATS_IN_VECTOR * VECS;
        let k_pad = k.next_multiple_of(lanes);
        let n_panels = n.div_ceil(ROWS);

        let mut panels = Arena::new(n_panels * ROWS * k_pad);
        let panel_len = ROWS * k_pad;
        let data = panels.floats_mut();
        // every slot is written exactly once: a source chunk where the
        // weight has one, zeros in the k tail and the missing rows of a
        // partial last panel
        for (dst, slot) in data.chunks_exact_mut(lanes).enumerate() {
            let panel = dst * lanes / panel_len;
            let within = dst * lanes % panel_len;
            let chunk = within / (ROWS * lanes);
            let row = within / lanes % ROWS;
            let j = panel * ROWS + row;
            let src_start = chunk * lanes;
            if j >= n || src_start >= k {
                slot.fill(0.0);
                continue;
            }
            let src_row = view.row(j);
            let src_end = k.min(src_start + lanes);
            let src = &src_row[src_start..src_end];
            slot[..src.len()].copy_from_slice(src);
            slot[src.len()..].fill(0.0);
        }
        PackedPanels {
            panels,
            n,
            k,
            k_pad,
            rows: ROWS,
            lanes,
        }
    }

    /// One zero-padded copy of the activations: `m` rows of `k_pad`.
    fn scratch_len(&self, m: usize, k: usize, _n: usize) -> usize {
        let lanes = FLOATS_IN_VECTOR * VECS;
        m * k.next_multiple_of(lanes)
    }

    fn matmul_nt(&self, a: TensorView, b: &PackedPanels, out: TensorViewMut, scratch: &mut [f32]) {
        matmul_nt_packed::<ROWS, VECS>(a, b, out, scratch);
    }
}

/// The driver. Panels run outer, activation rows inner: at `m > 1` the
/// panel then stays cache-hot across every row instead of the whole
/// weight being re-streamed per row; at `m = 1` the orders coincide.
#[hotpath::measure]
fn matmul_nt_packed<const ROWS: usize, const VECS: usize>(
    a: TensorView,
    b: &PackedPanels,
    mut out: TensorViewMut,
    scratch: &mut [f32],
) {
    let (m, k, n) = (a.rows(), a.cols(), b.n);
    assert_eq!(
        k,
        b.k,
        "matmul_nt: contracted dim mismatch, {} vs packed k {}",
        a.shape(),
        b.k,
    );
    assert_eq!(
        (b.rows, b.lanes),
        (ROWS, FLOATS_IN_VECTOR * VECS),
        "matmul_nt: weights were packed for another tile shape",
    );
    assert_eq!(
        out.shape(),
        Shape::new(m, n),
        "matmul_nt: out must be [{m}, {n}], got {}",
        out.shape(),
    );

    let k_pad = b.k_pad;
    let a_pad = &mut scratch[..m * k_pad];
    for (i, row_pad) in a_pad.chunks_exact_mut(k_pad).enumerate() {
        row_pad[..k].copy_from_slice(a.row(i));
        row_pad[k..].fill(0.0);
    }

    let panel_len = ROWS * k_pad;
    let panels = b.panels.floats();
    for (pi, panel) in panels.chunks_exact(panel_len).enumerate() {
        let j = pi * ROWS;
        let width = ROWS.min(n - j);
        for (i, row_pad) in a_pad.chunks_exact(k_pad).enumerate() {
            let sums = panel_tile::<ROWS, VECS>(row_pad, panel);
            let out_row = out.row_mut(i);
            out_row[j..j + width].copy_from_slice(&sums[..width]);
        }
    }
}

/// One panel against one padded activation row: the NEON register tile
/// with its `ROWS x VECS` `vld1q_f32` loads walking the panel strictly
/// forward, one stream, no tails.
#[inline(always)]
fn panel_tile<const ROWS: usize, const VECS: usize>(a_pad: &[f32], panel: &[f32]) -> [f32; ROWS] {
    let k_pad = a_pad.len();
    let lanes = FLOATS_IN_VECTOR * VECS;
    assert_eq!(panel.len(), ROWS * k_pad);
    assert_eq!(k_pad % lanes, 0);

    // SAFETY: `a` loads read 4 floats at `base + 4v` with
    // `base + lanes <= k_pad`; panel loads advance 4 floats per read,
    // `ROWS x VECS` reads per chunk for `k_pad / lanes` chunks, exactly
    // `panel.len()` floats.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); VECS]; ROWS];
        let a_ptr = a_pad.as_ptr();
        let mut panel_ptr = panel.as_ptr();
        let mut base = 0;
        while base < k_pad {
            let mut a_vecs = [vdupq_n_f32(0.0); VECS];
            for (v, a_vec) in a_vecs.iter_mut().enumerate() {
                *a_vec = vld1q_f32(a_ptr.add(base + FLOATS_IN_VECTOR * v));
            }
            for acc_row in acc.iter_mut() {
                for (v, acc_vec) in acc_row.iter_mut().enumerate() {
                    let b_vec = vld1q_f32(panel_ptr);
                    panel_ptr = panel_ptr.add(FLOATS_IN_VECTOR);
                    *acc_vec = vfmaq_f32(*acc_vec, a_vecs[v], b_vec);
                }
            }
            base += lanes;
        }

        let mut sums = [0.0f32; ROWS];
        for (row, sum) in sums.iter_mut().enumerate() {
            let acc_row = acc[row];
            let mut vec_sum = acc_row[0];
            for acc_vec in &acc_row[1..] {
                vec_sum = vaddq_f32(vec_sum, *acc_vec);
            }
            *sum = vaddvq_f32(vec_sum);
        }
        sums
    }
}

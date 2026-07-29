use crate::arena::poison;
use crate::tensor::{Shape, TensorView, TensorViewMut, matmul, matmul_nn_causal, softmax_causal};

/// Multi-head causal self-attention for one layer, over arena regions --
/// no allocation, no mask tensor, no per-head copies.
///
/// `qkv` is the fused activation region, rows `q | k | v` of width
/// `3 * n_embd`; only the Q columns are read here (K and V were already
/// appended to the cache). `keys` and `values` are the layer's cache prefixes
/// `[n_kv, n_embd]`, row-major. `scores` (`>= n_new * n_kv`) and `attn_out`
/// (`>= n_new * n_embd`) are scratch regions this function fully owns for the
/// call: `attn_out`'s active prefix is the output.
///
/// Heads run strictly sequentially: head `h` touches only its `head_dim`
/// column block, but the *backing slices* of sibling blocks overlap (the
/// stride gaps), so a strided view is mounted inside the loop and dropped
/// before the next iteration -- never collected.
///
/// The causal structure is the per-row prefix bound `n_past + i + 1` shared
/// by [`softmax_causal`] and [`matmul_nn_causal`]; columns beyond it are
/// never read, so the score matmul's upper triangle is dead work only during
/// prefill and no `-inf` mask ever exists.
#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    qkv: &[f32],
    keys: &[f32],
    values: &[f32],
    scores: &mut [f32],
    attn_out: &mut [f32],
    n_new: usize,
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
) {
    let n_embd = n_head * head_dim;
    let qkv_width = 3 * n_embd;
    let n_past = n_kv - n_new;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // one poison for the whole loop: heads write disjoint column blocks, and
    // a per-head poison would wipe the previous heads' output
    poison(attn_out);

    for h in 0..n_head {
        let off = h * head_dim;
        let head_shape = |rows: usize| Shape::new(rows, head_dim);

        let q_shape = head_shape(n_new);
        let q_block = &qkv[off..off + q_shape.strided_len(qkv_width)];
        let q_h = TensorView::strided(q_block, q_shape, qkv_width);
        let kv_shape = head_shape(n_kv);
        let k_block = &keys[off..off + kv_shape.strided_len(n_embd)];
        let k_h = TensorView::strided(k_block, kv_shape, n_embd);

        poison(scores);
        let scores_prefix = &mut scores[..n_new * n_kv];
        let scores_shape = Shape::new(n_new, n_kv);
        matmul(
            q_h,
            k_h,
            TensorViewMut::contiguous(scores_prefix, scores_shape),
        );
        for (i, row) in scores_prefix.chunks_exact_mut(n_kv).enumerate() {
            let bound = (n_past + i + 1).min(n_kv);
            for s in &mut row[..bound] {
                *s *= scale;
            }
        }
        softmax_causal(
            TensorViewMut::contiguous(scores_prefix, scores_shape),
            n_past,
        );

        let v_block = &values[off..off + kv_shape.strided_len(n_embd)];
        let v_h = TensorView::strided(v_block, kv_shape, n_embd);
        let out_block = &mut attn_out[off..off + q_shape.strided_len(n_embd)];
        let out_h = TensorViewMut::strided(out_block, q_shape, n_embd);
        let probs = TensorView::contiguous(&scores[..n_new * n_kv], scores_shape);
        matmul_nn_causal(probs, v_h, out_h, n_past);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs attention with q rows padded into a fused `q | k | v` layout (the
    /// K/V columns hold junk: attention must only read Q) and fresh scratch.
    fn attend(
        q: &[f32],
        keys: &[f32],
        values: &[f32],
        n_new: usize,
        n_kv: usize,
        n_head: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let n_embd = n_head * head_dim;
        let mut qkv = vec![777.0; n_new * 3 * n_embd];
        for (row, q_row) in qkv.chunks_exact_mut(3 * n_embd).zip(q.chunks_exact(n_embd)) {
            row[..n_embd].copy_from_slice(q_row);
        }
        let mut scores = vec![0.0; n_new * n_kv];
        let mut attn_out = vec![0.0; n_new * n_embd];
        multi_head_attention(
            &qkv,
            keys,
            values,
            &mut scores,
            &mut attn_out,
            n_new,
            n_kv,
            n_head,
            head_dim,
        );
        attn_out
    }

    #[test]
    fn single_head_single_token_is_the_value() {
        // one query, one key/value position: softmax over a single score is 1,
        // so the output is exactly the value vector.
        let out = attend(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0], 1, 1, 1, 2);
        assert_eq!(out, &[5.0, 6.0]);
    }

    #[test]
    fn first_query_attends_only_to_itself() {
        // two positions, one head of dim 2. Row 0 (n_past=0) may see only key 0,
        // so its output is value row 0 regardless of key/query values.
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [9.0, 9.0, -9.0, -9.0];
        let v = [1.0, 2.0, 3.0, 4.0];
        let out = attend(&q, &k, &v, 2, 2, 1, 2);
        assert_eq!(&out[0..2], &[1.0, 2.0]);
    }

    #[test]
    fn heads_are_independent_column_blocks() {
        // two heads of dim 1, one position: each head's output is its own value.
        let out = attend(&[1.0, 1.0], &[2.0, 2.0], &[7.0, 8.0], 1, 1, 2, 1);
        assert_eq!(out, &[7.0, 8.0]);
    }

    #[test]
    fn decode_step_on_a_grown_cache_uses_the_full_prefix() {
        // n_new=1 on top of n_past=2, one head of dim 1: with equal keys the
        // probs are uniform, so the output is the mean of the three values.
        let q = [1.0];
        let k = [2.0, 2.0, 2.0];
        let v = [3.0, 6.0, 9.0];
        let out = attend(&q, &k, &v, 1, 3, 1, 1);
        assert!((out[0] - 6.0).abs() < 1e-6);
    }
}

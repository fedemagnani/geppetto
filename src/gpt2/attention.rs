use crate::tensor::{Shape, Tensor};

/// Multi-head causal self-attention for one layer.
///
/// `q` is this step's queries `[n_q, n_embd]`; `k` and `v` are the layer's full
/// cache `[n_kv, n_embd]`, including the `n_q` positions just appended, all
/// row-major slices. `n_past = n_kv - n_q` offsets the causal mask: query row
/// `i` (absolute position `n_past + i`) may attend to key positions
/// `0..=n_past + i`. Within a row the `n_head` heads occupy contiguous
/// `head_dim` column blocks, matching the fused-QKV layout. Returns
/// `[n_q, n_embd]`.
pub fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    n_head: usize,
    head_dim: usize,
) -> Tensor {
    let n_embd = n_head * head_dim;
    let n_past = n_kv - n_q;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask = causal_mask(n_q, n_kv, n_past);

    let mut out = vec![0.0f32; n_q * n_embd];
    for h in 0..n_head {
        let off = h * head_dim;
        let qh = gather_head(q, n_q, n_embd, off, head_dim); // [n_q, head_dim]
        let kh = gather_head(k, n_kv, n_embd, off, head_dim); // [n_kv, head_dim]

        let mut scores = &qh * &kh; // [n_q, n_kv], scores[i, j] = qh_i . kh_j
        for s in scores.data_mut() {
            *s *= scale;
        }
        let probs = scores.softmax(Some(&mask));

        // out_h = probs @ v_h; `*` gives probs @ rhs^T, so transpose v_h first
        let vh_t = gather_head_transposed(v, n_kv, n_embd, off, head_dim); // [head_dim, n_kv]
        let out_h = &probs * &vh_t; // [n_q, head_dim]

        for i in 0..n_q {
            out[i * n_embd + off..i * n_embd + off + head_dim].copy_from_slice(out_h.row(i));
        }
    }
    Tensor::new(Shape::new(n_q, n_embd), out)
}

/// `[n_q, n_kv]` additive mask: `0` where a query may attend, `-inf` otherwise.
fn causal_mask(n_q: usize, n_kv: usize, n_past: usize) -> Tensor {
    let mut mask = vec![0.0f32; n_q * n_kv];
    for i in 0..n_q {
        for j in (n_past + i + 1)..n_kv {
            mask[i * n_kv + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::new(Shape::new(n_q, n_kv), mask)
}

/// Head `[off, off + head_dim)` of every row: `[rows, head_dim]`.
fn gather_head(src: &[f32], rows: usize, n_embd: usize, off: usize, head_dim: usize) -> Tensor {
    let mut out = vec![0.0f32; rows * head_dim];
    for r in 0..rows {
        let from = r * n_embd + off;
        out[r * head_dim..(r + 1) * head_dim].copy_from_slice(&src[from..from + head_dim]);
    }
    Tensor::new(Shape::new(rows, head_dim), out)
}

/// Head `[off, off + head_dim)` transposed to `[head_dim, rows]`, so the V
/// multiply can use the `self @ rhs^T` operator.
fn gather_head_transposed(
    src: &[f32],
    rows: usize,
    n_embd: usize,
    off: usize,
    head_dim: usize,
) -> Tensor {
    let mut out = vec![0.0f32; head_dim * rows];
    for r in 0..rows {
        for d in 0..head_dim {
            out[d * rows + r] = src[r * n_embd + off + d];
        }
    }
    Tensor::new(Shape::new(head_dim, rows), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_head_single_token_is_the_value() {
        // one query, one key/value position: softmax over a single score is 1,
        // so the output is exactly the value vector.
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let out = multi_head_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(out.data(), &[5.0, 6.0]);
    }

    #[test]
    fn first_query_attends_only_to_itself() {
        // two positions, one head of dim 2. Row 0 (n_past=0) may see only key 0,
        // so its output is value row 0 regardless of key/query values.
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![9.0, 9.0, -9.0, -9.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = multi_head_attention(&q, &k, &v, 2, 2, 1, 2);
        assert_eq!(&out.data()[0..2], &[1.0, 2.0]);
    }

    #[test]
    fn heads_are_independent_column_blocks() {
        // two heads of dim 1, one position: each head's output is its own value.
        let q = vec![1.0, 1.0];
        let k = vec![2.0, 2.0];
        let v = vec![7.0, 8.0];
        let out = multi_head_attention(&q, &k, &v, 1, 1, 2, 1);
        assert_eq!(out.data(), &[7.0, 8.0]);
    }
}

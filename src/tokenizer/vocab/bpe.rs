use std::collections::HashMap;

use super::text_token_id_bmap::TextTokenIdBmap;
use crate::tokenizer::{TokenId, TokenizerError};

/// The token `left` immediately followed by the token `right`. Ordered:
/// `(left, right)` and `(right, left)` are different pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenPair {
    pub left: TokenId,
    pub right: TokenId,
}

impl TokenPair {
    pub const fn new(left: TokenId, right: TokenId) -> TokenPair {
        TokenPair { left, right }
    }
}

/// Merge rules keyed by their [`TokenPair`]. Symbols are vocab ids
/// throughout: every single mapped character and every merge result is
/// itself a vocab token, so no string is ever built during encoding.
#[derive(Debug, Default)]
pub struct BPEMergeTable(HashMap<TokenPair, Merge>);

impl BPEMergeTable {
    pub fn with_capacity(n: usize) -> BPEMergeTable {
        BPEMergeTable(HashMap::with_capacity(n))
    }

    /// Compiles the learned merge list against the vocab. Each line is two
    /// space-separated symbols ("Ġ t"); its position is the rank. Symbols
    /// are vocab ids during BPE, so a merge's parts and their concatenation
    /// must all be tokens themselves.
    pub fn from_merges(
        merges: &[String],
        token_table: &TextTokenIdBmap,
    ) -> Result<BPEMergeTable, TokenizerError> {
        let mut table = BPEMergeTable::with_capacity(merges.len());
        for (index, merge) in merges.iter().enumerate() {
            let split = merge.split_once(' ');
            let split = split.filter(|(left, right)| {
                !left.is_empty() && !right.is_empty() && !right.contains(' ')
            });
            let Some((left, right)) = split else {
                return Err(TokenizerError::BadMerge {
                    index,
                    merge: merge.clone(),
                });
            };
            let concat = [left, right].concat();
            let resolved = (
                token_table.id(left),
                token_table.id(right),
                token_table.id(&concat),
            );
            let (Some(left_id), Some(right_id), Some(merged_id)) = resolved else {
                return Err(TokenizerError::MergeNotInVocab {
                    index,
                    merge: merge.clone(),
                });
            };
            table.insert(
                TokenPair::new(left_id, right_id),
                Merge {
                    rank: index as u32,
                    merged: merged_id,
                },
            );
        }
        Ok(table)
    }

    /// First occurrence wins, keeping the lowest rank for a pair.
    pub fn insert(&mut self, pair: TokenPair, merge: Merge) {
        self.0.entry(pair).or_insert(merge);
    }

    /// The rule for `pair`, if it was learned.
    pub fn get(&self, pair: TokenPair) -> Option<Merge> {
        self.0.get(&pair).copied()
    }

    /// Runs BPE in place over the tail of `symbols` starting at `from``:
    ///
    /// repeatedly replaces the adjacent pair with the lowest rank via the
    /// learned [`MergeTable`].
    ///
    /// It exits when:
    /// - EITHER the tail has collapsed to a single symbol (no more pairs to inspect)
    /// - OR no adjacent pair in the new tail appear in the [`MergeTable`]
    pub fn apply_bpe(&self, symbols: &mut Vec<TokenId>, from: usize) {
        while symbols.len() > from + 1 {
            let best = symbols[from..]
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let pair = TokenPair::new(w[0], w[1]);
                    self.get(pair).map(|merge| (merge, i))
                })
                .min();

            let Some((merge, i)) = best else { break };
            // replace the pair with the fused one
            symbols[from + i] = merge.merged;
            symbols.remove(from + i + 1);
        }
    }
}

/// One merge rule: the pair it applies to is the key in [`MergeTable`],
/// `merged` is the vocab id of the concatenation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merge {
    /// Position in the merges list; lower merges first. A priority, not a
    /// token id.
    pub rank: u32,
    pub merged: TokenId,
}

/// Priority order: by rank alone. `merged` takes part only to stay
/// consistent with `Eq`; ranks are unique within a table, so it never
/// decides between two table entries.
impl Ord for Merge {
    fn cmp(&self, other: &Merge) -> std::cmp::Ordering {
        (self.rank, self.merged).cmp(&(other.rank, other.merged))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Merge) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(merges: &[((u32, u32), u32)]) -> BPEMergeTable {
        let mut table = BPEMergeTable::with_capacity(merges.len());
        for (rank, &((left, right), merged)) in merges.iter().enumerate() {
            table.insert(
                TokenPair::new(left, right),
                Merge {
                    rank: rank as u32,
                    merged,
                },
            );
        }
        table
    }

    fn apply(mut symbols: Vec<u32>, merges: &BPEMergeTable) -> Vec<u32> {
        merges.apply_bpe(&mut symbols, 0);
        symbols
    }

    // ids: l=0 o=1 w=2 e=3 r=4 s=9; lo=5 low=6 er=7 lower=8
    const LOWER: &[((u32, u32), u32)] = &[((0, 1), 5), ((5, 2), 6), ((3, 4), 7), ((6, 7), 8)];

    #[test]
    fn merges_follow_rank_order() {
        let merges = table(LOWER);
        assert_eq!(apply(vec![0, 1, 2, 3, 4], &merges), vec![8]);
        assert_eq!(apply(vec![0, 1, 2, 3, 4, 9], &merges), vec![8, 9]);
        assert_eq!(apply(vec![2, 3], &merges), vec![2, 3]);
    }

    #[test]
    fn ties_on_the_same_pair_merge_left_to_right() {
        let merges = table(&[((0, 0), 1)]);
        assert_eq!(apply(vec![0, 0, 0], &merges), vec![1, 0]);
        assert_eq!(apply(vec![0, 0, 0, 0], &merges), vec![1, 1]);
    }

    #[test]
    fn merging_from_an_offset_leaves_the_prefix_alone() {
        let merges = table(&[((0, 1), 2)]);
        let mut symbols = vec![0, 1, 0, 1];
        merges.apply_bpe(&mut symbols, 2);
        assert_eq!(symbols, vec![0, 1, 2]);
    }

    #[test]
    fn a_pair_keeps_its_first_rank() {
        let mut merges = table(&[((0, 1), 2)]);
        merges.insert(TokenPair::new(0, 1), Merge { rank: 9, merged: 7 });
        assert_eq!(
            merges.get(TokenPair::new(0, 1)),
            Some(Merge { rank: 0, merged: 2 })
        );
        assert_eq!(merges.get(TokenPair::new(1, 0)), None, "pairs are ordered");
        assert_eq!(apply(vec![0, 1], &merges), vec![2]);
    }

    #[test]
    fn from_merges_compiles_against_the_vocab() {
        let strings = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let vocab = TextTokenIdBmap::new(strings(&["a", "b", "ab", "abb"])).unwrap();

        let merges = BPEMergeTable::from_merges(&strings(&["a b", "ab b"]), &vocab).unwrap();
        assert_eq!(
            merges.get(TokenPair::new(0, 1)),
            Some(Merge { rank: 0, merged: 2 })
        );
        assert_eq!(
            merges.get(TokenPair::new(2, 1)),
            Some(Merge { rank: 1, merged: 3 })
        );

        use crate::tokenizer::TokenizerError;
        let bad = BPEMergeTable::from_merges(&strings(&["a b c"]), &vocab);
        assert!(matches!(
            bad,
            Err(TokenizerError::BadMerge { index: 0, .. })
        ));
        let unknown = BPEMergeTable::from_merges(&strings(&["b b"]), &vocab);
        assert!(matches!(
            unknown,
            Err(TokenizerError::MergeNotInVocab { index: 0, .. })
        ));
    }

    #[test]
    fn short_and_unknown_pieces_pass_through() {
        let merges = table(&[((0, 1), 2)]);
        assert_eq!(apply(vec![7], &merges), vec![7]);
        assert_eq!(apply(vec![], &merges), Vec::<u32>::new());
        assert_eq!(apply(vec![3, 4, 5], &merges), vec![3, 4, 5]);
    }
}

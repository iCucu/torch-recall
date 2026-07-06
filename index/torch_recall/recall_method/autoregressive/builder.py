"""Vectorized builder for the Trie + GenerativeRecall model.

Construction follows the STATIC approach (Su et al., 2026): sort SIDs,
diff-scan to identify unique prefixes, assign contiguous global state IDs,
then pack edges into CSR format.  Dense lookup tables are synthesised in
parallel for the top ``d_dense`` levels.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from torch_recall.schema import Item, Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.autoregressive.trie import Trie


class GenerativeBuilder:
    """Builds a :class:`GenerativeRecall` from items with ``sid_path``."""

    def __init__(
        self,
        schema: Schema,
        decoder: torch.nn.Module,
        beam_width: int = 10,
        dense_levels: int = 2,
        sid_vocab_size: int = 2048,
        path_length: int = 5,
    ):
        self.schema = schema
        self.decoder = decoder
        self.beam_width = beam_width
        self.dense_levels = dense_levels
        self.sid_vocab_size = sid_vocab_size
        self.path_length = path_length

    # -- public API --------------------------------------------------------

    def build(self, items: list[Item]) -> tuple[torch.nn.Module, dict]:
        from torch_recall.recall_method.autoregressive.recall import GenerativeRecall

        self._validate_items(items)
        N = len(items)
        V = self.sid_vocab_size
        L = self.path_length
        d_dense = self.dense_levels

        targeting_model, targeting_meta = TargetingBuilder(self.schema).build(items)

        sid_paths = np.array([item.sid_path for item in items], dtype=np.int32)
        sort_idx = np.lexsort(sid_paths.T[::-1])
        sorted_sids = sid_paths[sort_idx]

        (
            packed_csr,
            csr_indptr,
            max_branch_factors,
            start_mask,
            dense_mask,
            dense_states,
            level_start,
            level_count,
            num_states,
            state_ids,
        ) = _build_static_index(sorted_sids, V, d_dense)

        leaf_item_ids, leaf_item_valid = _build_leaf_mapping(
            state_ids, sort_idx, level_start[-1], level_count[-1]
        )

        user_dim = getattr(self.decoder, "user_dim", 64)

        trie = Trie(
            start_mask=torch.from_numpy(start_mask),
            dense_mask=torch.from_numpy(dense_mask),
            dense_states=torch.from_numpy(dense_states).long(),
            packed_csr=torch.from_numpy(packed_csr).long(),
            csr_indptr=torch.from_numpy(csr_indptr).long(),
            leaf_item_ids=torch.from_numpy(leaf_item_ids).long(),
            leaf_item_valid=torch.from_numpy(leaf_item_valid),
            level_start=level_start,
            level_count=level_count,
            max_branch_factors=max_branch_factors,
            num_items=N,
            num_states=num_states,
            d_dense=d_dense,
            path_length=L,
            vocab_size=V,
        )

        model = GenerativeRecall(
            targeting=targeting_model,
            trie=trie,
            decoder=self.decoder,
            beam_width=self.beam_width,
            num_items=N,
            num_preds=targeting_meta["num_preds"],
            user_dim=user_dim,
        )

        meta: dict = {
            "num_items": N,
            "num_preds": targeting_meta["num_preds"],
            "beam_width": self.beam_width,
            "sid_vocab_size": V,
            "path_length": L,
            "dense_levels": d_dense,
            "user_dim": user_dim,
            "targeting": targeting_meta,
            "item_ids": [item.id for item in items] if items[0].id else None,
            "num_states": num_states,
            "max_branch_factors": max_branch_factors,
            "nodes_per_level": [1] + level_count,
            "num_leaves": level_count[-1],
            "max_items_per_leaf": int(leaf_item_ids.shape[1]),
        }
        return model, meta

    def save_meta(self, meta: dict, path: str) -> None:
        serializable = {k: v for k, v in meta.items() if k != "targeting"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # -- validation --------------------------------------------------------

    def _validate_items(self, items: list[Item]) -> None:
        for i, item in enumerate(items):
            if item.sid_path is None:
                raise ValueError(f"Item {i}: sid_path is None")
            if len(item.sid_path) != self.path_length:
                raise ValueError(
                    f"Item {i}: sid_path length {len(item.sid_path)} != {self.path_length}"
                )
            for j, sid in enumerate(item.sid_path):
                if not (0 <= sid < self.sid_vocab_size):
                    raise ValueError(
                        f"Item {i}, position {j}: sid {sid} not in [0, {self.sid_vocab_size})"
                    )


# ---------------------------------------------------------------------------
# Vectorised index construction (pure NumPy)
# ---------------------------------------------------------------------------


def _build_static_index(
    sorted_sids: np.ndarray, V: int, d_dense: int
) -> tuple:
    """Build CSR + dense index from *sorted* SID array.

    Returns (packed_csr, csr_indptr, max_branch_factors, start_mask,
             dense_mask, dense_states, level_start, level_count,
             num_states, state_ids).
    """
    N, L = sorted_sids.shape

    # 1. start mask
    start_mask = np.zeros(V, dtype=bool)
    start_mask[np.unique(sorted_sids[:, 0])] = True

    # 2. unique-prefix identification via diff-scan
    is_new = np.zeros((N, L), dtype=bool)
    is_new[0, :] = True
    if N > 1:
        diff = sorted_sids[1:] != sorted_sids[:-1]
        first_diff = np.full(N - 1, L, dtype=np.int8)
        has_diff = diff.any(axis=1)
        first_diff[has_diff] = diff[has_diff].argmax(axis=1)
        for depth in range(L):
            is_new[1:, depth] = first_diff <= depth

    # 3. state-ID assignment (all levels including leaves)
    state_ids = np.zeros((N, L), dtype=np.int32)
    state_ids[:, 0] = sorted_sids[:, 0].astype(np.int32) + 1

    level_start: list[int] = [1]
    level_count: list[int] = [V]
    cur = V + 1
    for depth in range(1, L):
        mask = is_new[:, depth]
        n_new = int(mask.sum())
        level_start.append(cur)
        level_count.append(n_new)
        state_ids[mask, depth] = np.arange(cur, cur + n_new, dtype=np.int32)
        state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])
        cur += n_new

    num_states = cur

    # 4. edge collection (depth 1 → L-1)
    all_par, all_tok, all_ch = [], [], []
    for depth in range(1, L):
        mask = is_new[:, depth]
        all_par.append(state_ids[mask, depth - 1])
        all_tok.append(sorted_sids[mask, depth].astype(np.int32))
        all_ch.append(state_ids[mask, depth])

    parents = np.concatenate(all_par) if all_par else np.array([], dtype=np.int32)
    tokens = np.concatenate(all_tok) if all_tok else np.array([], dtype=np.int32)
    children = np.concatenate(all_ch) if all_ch else np.array([], dtype=np.int32)

    # 5. dense tables (first d_dense codewords)
    dense_shape = tuple([V] * d_dense)
    dense_mask = np.zeros(dense_shape, dtype=bool)
    dense_states_arr = np.zeros(dense_shape, dtype=np.int32)
    indices = tuple(sorted_sids[:, i].astype(np.int32) for i in range(d_dense))
    dense_mask[indices] = True
    dense_states_arr[indices] = state_ids[:, d_dense - 1]

    # 6. CSR packing
    counts = np.bincount(parents, minlength=num_states) if len(parents) else np.zeros(num_states, dtype=int)
    indptr = np.zeros(num_states + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts)

    # OOB-safe padding row (token = V, child = 0)
    raw_tok = np.concatenate([tokens, np.full(V, V, dtype=np.int32)])
    raw_ch = np.concatenate([children, np.zeros(V, dtype=np.int32)])
    indptr = np.append(indptr, indptr[-1] + V)
    packed_csr = np.ascontiguousarray(np.vstack([raw_tok, raw_ch]).T)

    # 7. max branch factors (one per "edge level")
    mbf: list[int] = [int(start_mask.sum())]
    l0_cnt = counts[1 : V + 1]
    mbf.append(int(l0_cnt.max()) if len(l0_cnt) > 0 else 0)
    for lv in range(1, L - 1):
        s, c = level_start[lv], level_count[lv]
        if c > 0 and s < len(counts):
            mbf.append(int(counts[s : s + c].max()))
        else:
            mbf.append(0)
    while len(mbf) < L:
        mbf.append(1)

    return (
        packed_csr,
        indptr,
        mbf,
        start_mask,
        dense_mask,
        dense_states_arr,
        level_start,
        level_count,
        num_states,
        state_ids,
    )


def _build_leaf_mapping(
    state_ids: np.ndarray,
    sort_idx: np.ndarray,
    leaf_start: int,
    num_leaves: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map leaf state IDs → original item indices."""
    N, L = state_ids.shape
    leaf_to_items: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        local = int(state_ids[i, L - 1] - leaf_start)
        leaf_to_items[local].append(int(sort_idx[i]))

    max_items = max((len(v) for v in leaf_to_items.values()), default=1)
    max_items = max(max_items, 1)

    ids = np.zeros((num_leaves, max_items), dtype=np.int64)
    valid = np.zeros((num_leaves, max_items), dtype=bool)
    for leaf_idx, items in leaf_to_items.items():
        for j, item in enumerate(items):
            ids[leaf_idx, j] = item
            valid[leaf_idx, j] = True
    return ids, valid

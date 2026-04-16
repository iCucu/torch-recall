"""Hybrid Dense/CSR trie for constrained beam search.

Inspired by STATIC (Su et al., 2026, arXiv:2602.22647). Uses dense
lookup tables for the top ``d_dense`` levels and a Compressed Sparse Row
(CSR) matrix for the sparse tail.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Trie(nn.Module):
    """Hybrid Dense/CSR trie with online targeting-mask propagation.

    State layout (global IDs)::

        0            — padding / unused
        1 … V       — L0 nodes  (state = first_token + 1)
        V+1 …       — L1, L2, … nodes (contiguous per level)
        last range   — leaf nodes (depth L-1)

    Dense tables (first ``d_dense`` levels) duplicate the CSR for O(1)
    beam-search access on the "hot head".  The CSR covers *all* transitions
    (including dense ones) and is the sole data source for bottom-up
    validity propagation.
    """

    def __init__(
        self,
        *,
        start_mask: torch.Tensor,
        dense_mask: torch.Tensor,
        dense_states: torch.Tensor,
        packed_csr: torch.Tensor,
        csr_indptr: torch.Tensor,
        leaf_item_ids: torch.Tensor,
        leaf_item_valid: torch.Tensor,
        level_start: list[int],
        level_count: list[int],
        max_branch_factors: list[int],
        num_items: int,
        num_states: int,
        d_dense: int,
        path_length: int,
        vocab_size: int,
    ):
        super().__init__()
        self.register_buffer("start_mask", start_mask)            # [V]      bool
        self.register_buffer("dense_mask", dense_mask)            # [V]*d    bool
        self.register_buffer("dense_states", dense_states)        # [V]*d    int64
        self.register_buffer("packed_csr", packed_csr)            # [E+V, 2] int64
        self.register_buffer("csr_indptr", csr_indptr)            # [S+2]    int64
        self.register_buffer("leaf_item_ids", leaf_item_ids)      # [nL, M]  int64
        self.register_buffer("leaf_item_valid", leaf_item_valid)  # [nL, M]  bool

        self._level_start = level_start
        self._level_count = level_count
        self._max_br = max_branch_factors
        self._num_items = num_items
        self._num_states = num_states
        self._d_dense = d_dense
        self._path_length = path_length
        self._vocab_size = vocab_size

    # -- properties --------------------------------------------------------

    @property
    def num_items(self) -> int:
        return self._num_items

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def d_dense(self) -> int:
        return self._d_dense

    @property
    def path_length(self) -> int:
        return self._path_length

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_leaves(self) -> int:
        return self.leaf_item_ids.shape[0]

    # -- bottom-up validity propagation ------------------------------------

    def propagate_validity(self, item_mask: torch.Tensor) -> torch.Tensor:
        """Propagate per-user item mask to per-node validity.

        从叶子层向根方向逐层传播：如果一个节点的子树中存在至少一个
        对当前用户合法的 item，则该节点 valid=True。

        Args:
            item_mask: [B, N] bool — 每个用户(B)对每个 item(N) 是否通过 targeting
        Returns:
            node_valid: [B, num_states] bool — 每个用户视角下每个 trie 节点是否有效
        """
        B = item_mask.shape[0]
        device = item_mask.device
        # 全部 state 初始化为 False，后续自底向上逐层填充
        node_valid = torch.zeros(B, self._num_states, dtype=torch.bool, device=device)

        # ---- 第一步：初始化叶子层 ----
        leaf_s = self._level_start[-1]   # 叶子层第一个 state 的 global ID
        leaf_c = self._level_count[-1]   # 叶子层共多少个 state

        # leaf_item_ids: [nL, M]，每个叶子关联的 item index（M 为 padding 后的最大值）
        # item_mask[:, leaf_item_ids] → [B, nL, M]，用 fancy indexing 一次取出
        # 每个叶子的每个关联 item 对当前用户是否合法
        items_g = item_mask[:, self.leaf_item_ids]

        # leaf_item_valid: [nL, M] bool，区分真实 item 和 padding
        # (items_g & leaf_item_valid) 排除 padding 后，any(dim=2) 沿 M 维归约：
        # 只要叶子关联的 item 中有一个合法且非 padding → 该叶子 valid
        # leaf_v: [B, nL] bool
        leaf_v = (items_g & self.leaf_item_valid).any(dim=2)
        node_valid[:, leaf_s : leaf_s + leaf_c] = leaf_v

        # ---- 第二步：自底向上逐层传播（从倒数第二层到 L0） ----
        max_csr = self.packed_csr.shape[0] - 1  # clamp 上界，防止越界
        for d in range(self._path_length - 2, -1, -1):
            s_start = self._level_start[d]       # 本层第一个 state 的 global ID
            s_count = self._level_count[d]        # 本层 state 数
            max_br = self._max_br[d + 1]          # 下一层的最大分支因子
            if s_count == 0 or max_br == 0:
                continue

            # states: [s_count]，本层所有 state 的 global ID
            states = torch.arange(s_start, s_start + s_count, device=device)

            # 通过 CSR indptr 找到每个 state 的子节点在 packed_csr 中的起止位置
            # starts[i] = csr_indptr[state_i]，是 packed_csr 中的起始行号
            # actual_lens[i] = indptr[state_i+1] - indptr[state_i]，即实际子节点数
            starts = self.csr_indptr[states]                      # [s_count]
            actual_lens = self.csr_indptr[states + 1] - starts    # [s_count]

            # offsets = [0, 1, ..., max_br-1]，用于逐个取子节点
            offsets = torch.arange(max_br, device=device)

            # gather_idx[i, j] = starts[i] + j → packed_csr 中第 i 个 state
            # 的第 j 个子节点所在的行号
            # unsqueeze 是为了 broadcast: [s_count,1] + [1,max_br] → [s_count, max_br]
            # clamp 确保不越界（超出实际子节点数的位置会被 struct_ok 屏蔽）
            gather_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=max_csr)

            # packed_csr[:, 1] 存的是 child_state_id
            # ch_states: [s_count, max_br]，每个 state 的（至多 max_br 个）子节点 ID
            ch_states = self.packed_csr[gather_idx, 1]

            # struct_ok: [s_count, max_br] bool，标记哪些位置是真实子节点、哪些是 padding
            # 例如某 state 只有 1 个子节点但 max_br=2，则 offset=1 处 struct_ok=False
            struct_ok = offsets.unsqueeze(0) < actual_lens.unsqueeze(1)

            # node_valid[:, ch_states]: [B, s_count, max_br]，每个子节点对每个用户的有效性
            # & struct_ok: 排除 padding 位置
            # .any(dim=2): 沿 max_br 维归约——只要有一个子节点 valid，父节点就 valid
            child_v = node_valid[:, ch_states] & struct_ok.unsqueeze(0)
            node_valid[:, s_start : s_start + s_count] = child_v.any(dim=2)

        return node_valid

    # -- leaf → item resolution --------------------------------------------

    def resolve_leaves(
        self, leaf_states: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Map beam leaf-state IDs + scores → per-item score tensor [B, N].

        Args:
            leaf_states: [B, beam] int64
            scores:      [B, beam] float
        Returns:
            [B, N] float  (``-inf`` for items not reached by any beam)
        """
        B = leaf_states.shape[0]
        leaf_s = self._level_start[-1]
        local = (leaf_states - leaf_s).clamp(min=0, max=self.num_leaves - 1)

        items = self.leaf_item_ids[local]
        valid = self.leaf_item_valid[local]

        scores_exp = scores.unsqueeze(-1).expand_as(items)
        flat_items = items.reshape(B, -1)
        flat_scores = scores_exp.masked_fill(~valid, float("-inf")).reshape(B, -1)

        output = torch.full(
            (B, self._num_items),
            float("-inf"),
            device=scores.device,
            dtype=scores.dtype,
        )
        output.scatter_reduce_(1, flat_items, flat_scores, reduce="amax")
        return output

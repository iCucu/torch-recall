"""Generative recall via trie-constrained beam search.

The beam search uses a hybrid strategy: dense lookup tables for the
first ``d_dense`` codewords, then CSR burst-reads for the sparse tail.
Validity masks from online targeting are intersected at every step.
"""

from __future__ import annotations

import torch

from torch_recall.recall_method.base import RecallOp
from torch_recall.recall_method.targeting.recall import TargetingRecall
from torch_recall.recall_method.generative.trie import Trie


def _gather_beams(x: torch.Tensor, beam_idx: torch.Tensor) -> torch.Tensor:
    """Reorder beams: ``x[B, old, ...]`` → ``x[B, new, ...]``."""
    B, new = beam_idx.shape
    shape = [B, new] + [1] * (x.dim() - 2)
    exp = [B, new] + list(x.shape[2:])
    return x.gather(1, beam_idx.view(shape).expand(exp))


class GenerativeRecall(RecallOp):
    """Generative recall via trie-constrained beam search.

    ``forward(pred_satisfied [B, P], query [B, D]) -> [B, N] float``
    """

    def __init__(
        self,
        targeting: TargetingRecall,
        trie: Trie,
        decoder: torch.nn.Module,
        beam_width: int,
        num_items: int,
        num_preds: int,
        user_dim: int,
        query_offset: int = 0,
    ):
        super().__init__()
        self.targeting = targeting
        self.trie = trie
        self.decoder = decoder
        self.beam_width = beam_width
        self.num_items = num_items
        self.num_preds = num_preds
        self.user_dim = user_dim
        self.query_offset = query_offset

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        B = pred_satisfied.shape[0]
        device = pred_satisfied.device
        beam = self.beam_width
        V = self.trie.vocab_size
        L = self.trie.path_length
        d_dense = self.trie.d_dense

        query = query[:, self.query_offset : self.query_offset + self.user_dim]

        # Phase 1 — targeting mask
        targeting_scores = self.targeting(pred_satisfied, query)
        item_mask = targeting_scores > float("-inf")

        # Phase 2 — bottom-up trie validity propagation
        node_valid = self.trie.propagate_validity(item_mask)

        # Phase 3 — beam search
        # -- step 0: choose first token ------------------------------------
        dummy = torch.zeros(B, L, dtype=torch.int64, device=device)
        logits0 = self.decoder(query, dummy)
        lp0 = torch.log_softmax(logits0, dim=-1)

        l0_valid = node_valid[:, 1 : V + 1]
        lp0 = lp0.masked_fill(~(self.trie.start_mask.unsqueeze(0) & l0_valid), float("-inf"))

        top_lp, top_tok = lp0.topk(beam, dim=-1)            # [B, beam]

        beam_states = top_tok + 1                              # L0 state IDs
        beam_scores = top_lp
        beam_paths = torch.zeros(B, beam, L, dtype=torch.int64, device=device)
        beam_paths[:, :, 0] = top_tok

        # batch-index helper: maps flat (B*beam) position → batch
        bb = torch.arange(B, device=device).repeat_interleave(beam)

        # -- steps 1 … L-1 ------------------------------------------------
        for step in range(1, L):
            user_repr = query.unsqueeze(1).expand(B, beam, -1).reshape(B * beam, -1)
            paths_flat = beam_paths.reshape(B * beam, L)
            logits = self.decoder(user_repr, paths_flat)       # [B*beam, V]
            lp = torch.log_softmax(logits, dim=-1)

            flat_st = beam_states.reshape(B * beam)

            if step < d_dense:
                beam_states, beam_scores, beam_paths = self._step_dense(
                    lp, flat_st, node_valid, bb,
                    beam_scores, beam_paths, step, B, beam, V,
                )
            else:
                limit = max(self.trie._max_br[step], 1)
                beam_states, beam_scores, beam_paths = self._step_csr(
                    lp, flat_st, node_valid, bb,
                    beam_scores, beam_paths, step, limit, B, beam, V,
                )

        # Phase 4 — resolve leaf states → item scores
        return self.trie.resolve_leaves(beam_states, beam_scores)

    # -- dense beam-search step --------------------------------------------

    def _step_dense(
        self,
        lp: torch.Tensor,          # [B*beam, V]
        flat_st: torch.Tensor,     # [B*beam]
        node_valid: torch.Tensor,  # [B, S]
        bb: torch.Tensor,          # [B*beam]
        beam_scores: torch.Tensor, # [B, beam]
        beam_paths: torch.Tensor,  # [B, beam, L]
        step: int,
        B: int, beam: int, V: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_tok = (flat_st - 1).long()
        masks = self.trie.dense_mask[parent_tok]               # [B*beam, V]
        st_table = self.trie.dense_states[parent_tok]          # [B*beam, V]

        nv = node_valid[bb.unsqueeze(1), st_table]             # [B*beam, V]
        combined = masks & nv
        lp = lp.masked_fill(~combined, float("-inf"))

        lp3 = lp.view(B, beam, V)
        st3 = st_table.view(B, beam, V)

        cand = beam_scores.unsqueeze(-1) + lp3
        top_sc, top_flat = cand.view(B, -1).topk(beam, dim=-1)
        top_bm = top_flat // V
        top_tk = top_flat % V

        bi = torch.arange(B, device=lp.device).unsqueeze(1)
        new_st = st3[bi, top_bm, top_tk]
        new_paths = _gather_beams(beam_paths, top_bm)
        new_paths[:, :, step] = top_tk
        return new_st, top_sc, new_paths

    # -- CSR beam-search step ----------------------------------------------

    def _step_csr(
        self,
        lp: torch.Tensor,          # [B*beam, V]
        flat_st: torch.Tensor,     # [B*beam]
        node_valid: torch.Tensor,  # [B, S]
        bb: torch.Tensor,          # [B*beam]
        beam_scores: torch.Tensor, # [B, beam]
        beam_paths: torch.Tensor,  # [B, beam, L]
        step: int, limit: int,
        B: int, beam: int, V: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = self.trie.csr_indptr[flat_st.long()]
        a_lens = self.trie.csr_indptr[flat_st.long() + 1] - starts

        offs = torch.arange(limit, device=lp.device)
        gi = (starts.unsqueeze(1) + offs.unsqueeze(0)).clamp(
            max=self.trie.packed_csr.shape[0] - 1
        )
        gathered = self.trie.packed_csr[gi]                    # [B*beam, K, 2]
        c_tok = gathered[..., 0]
        c_st = gathered[..., 1]

        struct_ok = offs.unsqueeze(0) < a_lens.unsqueeze(1)
        c_nv = node_valid[bb.unsqueeze(1), c_st]
        valid = struct_ok & c_nv

        safe_tok = c_tok.clamp(min=0, max=V - 1).long()
        c_lp = lp.gather(1, safe_tok).masked_fill(~valid, float("-inf"))

        c_lp = c_lp.view(B, beam, limit)
        c_tok = c_tok.view(B, beam, limit)
        c_st = c_st.view(B, beam, limit)

        cand = beam_scores.unsqueeze(-1) + c_lp
        top_sc, top_flat = cand.view(B, -1).topk(beam, dim=-1)
        top_bm = top_flat // limit
        top_lc = top_flat % limit

        bi = torch.arange(B, device=lp.device).unsqueeze(1)
        new_st = c_st[bi, top_bm, top_lc]
        new_sid = c_tok[bi, top_bm, top_lc]
        new_paths = _gather_beams(beam_paths, top_bm)
        new_paths[:, :, step] = new_sid
        return new_st, top_sc, new_paths

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.num_preds, dtype=torch.bool, device=device),
            torch.randn(1, self.user_dim, device=device),
        )

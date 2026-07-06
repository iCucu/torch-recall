"""Discrete-diffusion recall with SidPathFilter constraint.

Contrast with :class:`autoregressive.recall.GenerativeRecall`:

    AR:        fixes generation order depth 0 → L-1;
               constraint  = Trie state machine (prefix-based).
    Diffusion: commits highest-confidence position first;
               constraint  = path bitmap that narrows after every commit.

At each of the L decode steps the model sees the *full* partial sequence
(including MASK tokens) via its bidirectional decoder and returns logits
for *all* positions simultaneously.  The position with the globally
highest average confidence (across beams) is selected for beam expansion.

Resolve step: because ``beam_mask[b, k]`` encodes exactly which items
are consistent with beam ``k`` of user ``b``, no path-to-item lookup is
needed — the mask is the item set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_recall.recall_method.base import RecallOp
from torch_recall.recall_method.targeting.recall import TargetingRecall
from torch_recall.recall_method.diffusion.path_filter import SidPathFilter

MASK_ID: int = -1


def _gather_beams(x: torch.Tensor, beam_idx: torch.Tensor) -> torch.Tensor:
    """Reorder beams: ``x[B, old, ...]`` → ``x[B, new, ...]``."""
    B, new = beam_idx.shape
    shape = [B, new] + [1] * (x.dim() - 2)
    exp = [B, new] + list(x.shape[2:])
    return x.gather(1, beam_idx.view(shape).expand(exp))


class DiffusionRecall(RecallOp):
    """Generative recall via SidPathFilter + adaptive-order beam search.

    ``forward(pred_satisfied [B, P], query [B, D]) -> [B, N] float``

    The decode loop runs for exactly L steps.  At each step:

    1. Decoder forward → ``logits [B, beam, L, V]``
    2. Apply per-beam path constraints → ``lp [B, beam, L, V]``
    3. Select the unfilled position with highest mean confidence → ``m_t``
    4. Beam expand at ``m_t``: top-beam from ``beam × V`` candidates
    5. ``filter_beam_paths`` narrows ``beam_mask`` for the new beams

    After L steps every position is committed and ``beam_mask`` encodes
    the surviving candidate items per beam.
    """

    def __init__(
        self,
        targeting: TargetingRecall,
        path_filter: SidPathFilter,
        decoder: torch.nn.Module,
        beam_width: int,
        num_items: int,
        num_preds: int,
        user_dim: int,
        query_offset: int = 0,
    ):
        super().__init__()
        self.targeting = targeting
        self.path_filter = path_filter
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
        L = self.path_filter.path_length
        V = self.path_filter.vocab_size
        N = self.path_filter.num_paths

        query = query[:, self.query_offset : self.query_offset + self.user_dim]

        # Phase 1 — targeting
        targeting_scores = self.targeting(pred_satisfied, query)
        item_mask = targeting_scores > float("-inf")           # [B, N]

        # Phase 2 — initialise per-beam path mask
        # beam_mask[b, k, n] = True ↔ path n is still consistent
        #                              for beam k of user b
        beam_mask = self.path_filter.init_beam_mask(item_mask, beam)  # [B, beam, N]

        # Phase 3 — adaptive-order diffusion decode (L steps, 1 commit per step)
        paths = torch.full((B, beam, L), MASK_ID, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B, beam, device=device)

        for _step in range(L):
            # (a) Decoder: all L positions in parallel
            user_repr = (
                query.unsqueeze(1).expand(B, beam, -1).reshape(B * beam, -1)
            )
            paths_flat = paths.reshape(B * beam, L)
            logits = self.decoder(user_repr, paths_flat)       # [B*beam, L, V]
            lp = F.log_softmax(logits, dim=-1).view(B, beam, L, V)

            # (b) Mask invalid tokens at every unfilled position
            is_unfilled = paths == MASK_ID                     # [B, beam, L]
            for m in range(L):
                if not is_unfilled[:, :, m].any():
                    continue
                valid_tok = self.path_filter.collect_valid_tokens(beam_mask, m)
                lp[:, :, m, :] = lp[:, :, m, :].masked_fill(
                    ~valid_tok, float("-inf")
                )

            # (c) Select best position per batch item
            # conf = max log-prob over vocab; −inf for already-filled
            conf = lp.max(dim=-1).values                       # [B, beam, L]
            conf_masked = conf.masked_fill(~is_unfilled, float("-inf"))
            # Mean confidence across beams to obtain a single position signal
            mean_conf = conf_masked.mean(dim=1)                # [B, L]
            best_pos = mean_conf.argmax(dim=-1)                # [B]

            # (d) Beam expand — process each batch item independently
            #     (positions can differ across batch items)
            new_paths = paths.clone()
            new_scores = beam_scores.clone()
            new_beam_mask = beam_mask.clone()

            for b in range(B):
                m = int(best_pos[b].item())
                lp_bm = lp[b, :, m, :]                        # [beam, V]
                cand = beam_scores[b].unsqueeze(-1) + lp_bm   # [beam, V]
                top_sc, top_flat = cand.reshape(-1).topk(
                    beam, largest=True, sorted=True
                )
                top_bm = top_flat // V                         # old beam idx
                top_tk = top_flat % V                          # chosen token

                # Reorder paths/mask from old beams, then commit token
                new_paths[b] = paths[b, top_bm, :]
                new_paths[b, :, m] = top_tk
                new_scores[b] = top_sc

                # (e) Narrow path mask for newly created beams
                old_mask = beam_mask[b, top_bm, :]            # [beam, N]
                path_tok = self.path_filter.path_vectors[:, m]
                consistent = (
                    path_tok.unsqueeze(0) == top_tk.unsqueeze(-1)
                )                                              # [beam, N]
                new_beam_mask[b] = old_mask & consistent

            paths = new_paths
            beam_scores = new_scores
            beam_mask = new_beam_mask

        # Phase 4 — resolve
        # beam_mask[b, k] is True for exactly the items consistent with
        # beam k; no path matching needed.
        scores_exp = beam_scores.unsqueeze(-1).expand(B, beam, N)
        scores_masked = scores_exp.masked_fill(~beam_mask, float("-inf"))
        return scores_masked.max(dim=1).values                 # [B, N]

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.num_preds, dtype=torch.bool, device=device),
            torch.randn(1, self.user_dim, device=device),
        )

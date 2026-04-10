from __future__ import annotations

import torch

from torch_recall.recall_method.base import RecallOp


class TargetingRecall(RecallOp):
    """Reverse recall: items carry targeting rules, users carry attributes.

    Given a boolean matrix indicating which predicates each user satisfies,
    returns per-item scores: 0.0 for match, -inf for no match.

    The forward pass uses only gather, elementwise ops, and reductions
    — fully compatible with torch.export / AOTInductor.
    """

    def __init__(
        self,
        conj_pred_ids: torch.Tensor,
        conj_pred_negated: torch.Tensor,
        conj_pred_valid: torch.Tensor,
        item_conj_ids: torch.Tensor,
        item_conj_valid: torch.Tensor,
        num_items: int,
        num_preds: int,
    ):
        super().__init__()
        self.register_buffer("conj_pred_ids", conj_pred_ids)          # [C, K] int64
        self.register_buffer("conj_pred_negated", conj_pred_negated)  # [C, K] bool
        self.register_buffer("conj_pred_valid", conj_pred_valid)      # [C, K] bool
        self.register_buffer("item_conj_ids", item_conj_ids)          # [N, J] int64
        self.register_buffer("item_conj_valid", item_conj_valid)      # [N, J] bool
        self.num_items = num_items
        self.num_preds = num_preds

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_satisfied: [B, P] bool — which predicates each user satisfies.
            query: [B, D] float — unused (present for unified interface).
        Returns:
            [B, N] float — 0.0 for matched items, -inf for unmatched.
        """
        sat = pred_satisfied[:, self.conj_pred_ids]   # [B, C, K]
        sat = sat ^ self.conj_pred_negated            # [B, C, K]
        sat = sat | ~self.conj_pred_valid             # [B, C, K]
        conj_ok = sat.all(dim=2)                      # [B, C]

        item_conj_ok = conj_ok[:, self.item_conj_ids] # [B, N, J]
        item_conj_ok = item_conj_ok & self.item_conj_valid
        item_ok = item_conj_ok.any(dim=2)             # [B, N] bool

        scores = torch.where(
            item_ok,
            torch.zeros(1, device=item_ok.device),
            torch.full((1,), float("-inf"), device=item_ok.device),
        )
        return scores  # [B, N]

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.num_preds, dtype=torch.bool, device=device),
            torch.zeros(1, 1, device=device),
        )

    # ------------------------------------------------------------------
    # High-level API (not traced by torch.export)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def query(self, user_attrs: dict, meta: dict) -> torch.Tensor:
        """Encode user attributes and return matched items as [N] bool."""
        from torch_recall.recall_method.targeting.encoder import encode_user

        pred_satisfied = encode_user(user_attrs, meta)      # [P]
        pred_satisfied = pred_satisfied.unsqueeze(0)         # [1, P]
        dummy_query = torch.zeros(1, 1)
        scores = self(pred_satisfied, dummy_query)           # [1, N]
        return scores.squeeze(0) > float("-inf")             # [N] bool

    def save_user_tensors(self, user_attrs: dict, meta: dict, path: str) -> None:
        """Encode user attributes and save for C++ inference."""
        from torch_recall.recall_method.targeting.encoder import encode_user

        pred_satisfied = encode_user(user_attrs, meta)       # [P]
        pred_satisfied = pred_satisfied.unsqueeze(0)          # [1, P]
        dummy_query = torch.zeros(1, 1)
        torch.save([[pred_satisfied, dummy_query]], path)

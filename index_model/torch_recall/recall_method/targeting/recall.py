from __future__ import annotations

import torch
import torch.nn as nn

from torch_recall.schema import MAX_PREDS_PER_CONJ, MAX_CONJ_PER_ITEM


class TargetingRecall(nn.Module):
    """Reverse recall: items carry targeting rules, users carry attributes.

    Given a boolean vector indicating which predicates a user satisfies,
    returns a boolean vector indicating which items match.

    The forward pass uses only gather, elementwise bool ops, and reductions
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

    def forward(self, pred_satisfied: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_satisfied: [P] bool — which predicates the user satisfies.
        Returns:
            [N] bool — which items match.
        """
        # Step 1: conjunction matching
        sat = pred_satisfied[self.conj_pred_ids]      # [C, K]
        sat = sat ^ self.conj_pred_negated            # handle NOT
        sat = sat | ~self.conj_pred_valid             # padding → True
        conj_ok = sat.all(dim=1)                      # [C]

        # Step 2: item matching
        item_conj_ok = conj_ok[self.item_conj_ids]    # [N, J]
        item_conj_ok = item_conj_ok & self.item_conj_valid
        item_ok = item_conj_ok.any(dim=1)             # [N]

        return item_ok

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        return (torch.zeros(self.num_preds, dtype=torch.bool, device=device),)

    # ------------------------------------------------------------------
    # High-level API (not traced by torch.export)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def query(self, user_attrs: dict, meta: dict) -> torch.Tensor:
        """Encode user attributes and run forward in one call."""
        from torch_recall.recall_method.targeting.encoder import encode_user

        pred_satisfied = encode_user(user_attrs, meta)
        return self(pred_satisfied)

    def save_user_tensors(self, user_attrs: dict, meta: dict, path: str) -> None:
        """Encode user attributes and save for C++ inference."""
        from torch_recall.recall_method.targeting.encoder import encode_user

        pred_satisfied = encode_user(user_attrs, meta)
        torch.save([[pred_satisfied]], path)

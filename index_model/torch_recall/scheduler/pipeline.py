from __future__ import annotations

import torch

from torch_recall.recall_method.base import RecallOp


class AndModule(RecallOp):
    """Intersection: ``score = sum(branches)``.

    Because targeting scores are 0 (match) / -inf (no match), summing
    with KNN scores naturally implements hard filtering.
    """

    def __init__(self, branches: list[RecallOp]):
        super().__init__()
        self.branches = torch.nn.ModuleList(branches)

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        result = self.branches[0](pred_satisfied, query)
        for i in range(1, len(self.branches)):
            result = result + self.branches[i](pred_satisfied, query)
        return result


class OrModule(RecallOp):
    """Union: ``score = max(branches)``."""

    def __init__(self, branches: list[RecallOp]):
        super().__init__()
        self.branches = torch.nn.ModuleList(branches)

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        result = self.branches[0](pred_satisfied, query)
        for i in range(1, len(self.branches)):
            result = torch.max(result, self.branches[i](pred_satisfied, query))
        return result


class RecallPipeline(torch.nn.Module):
    """Top-level module: runs the composed score tree, then topk.

    forward(pred_satisfied, query) -> (top_scores [B, K], top_indices [B, K])
    """

    def __init__(
        self,
        root: RecallOp,
        k: int,
        num_items: int,
        num_preds: int,
        total_query_dim: int,
    ):
        super().__init__()
        self.root = root
        self.k = k
        self.num_items = num_items
        self.num_preds = num_preds
        self.total_query_dim = total_query_dim

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.root(pred_satisfied, query)  # [B, N]
        top_vals, top_idx = scores.topk(self.k, dim=1)
        return top_vals, top_idx

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.num_preds, dtype=torch.bool, device=device),
            torch.randn(1, self.total_query_dim, device=device),
        )

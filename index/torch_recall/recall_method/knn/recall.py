from __future__ import annotations

import torch

from torch_recall.recall_method.base import RecallOp


VALID_METRICS = frozenset({"inner_product", "cosine", "l2"})


class KNNRecall(RecallOp):
    """K-Nearest Neighbor recall via brute-force matmul.

    Given a batch of query embeddings, returns per-item similarity scores.

    The forward pass uses only matmul and elementwise arithmetic
    — fully compatible with torch.export / AOTInductor.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        embedding_norms: torch.Tensor,
        k: int,
        metric: str,
        num_items: int,
        embedding_dim: int,
        offset: int = 0,
        weight: float = 1.0,
    ):
        super().__init__()
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}, expected one of {sorted(VALID_METRICS)}"
            )
        self.register_buffer("embeddings", embeddings)            # [N, D]
        self.register_buffer("embedding_norms", embedding_norms)  # [N]
        self.k = k
        self.metric = metric
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.offset = offset
        self.weight = weight

    def forward(
        self, pred_satisfied: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_satisfied: [B, P] bool — unused (present for unified interface).
            query: [B, D_total] float — concatenated query; this op slices
                   its portion via offset/dim.
        Returns:
            [B, N] float — per-item similarity scores (higher = more relevant).
        """
        q = query[:, self.offset : self.offset + self.embedding_dim]  # [B, D]

        if self.metric == "inner_product":
            scores = q @ self.embeddings.T  # [B, N]

        elif self.metric == "cosine":
            q_norm = q.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scores = (q / q_norm) @ self.embeddings.T  # [B, N]

        else:  # l2 — negate so higher = closer
            q_sq = (q * q).sum(dim=1, keepdim=True)            # [B, 1]
            cross = q @ self.embeddings.T                       # [B, N]
            dists = q_sq + self.embedding_norms.unsqueeze(0) - 2 * cross
            scores = -dists                                     # [B, N]

        return scores * self.weight

    def example_inputs(self, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        total_dim = self.offset + self.embedding_dim
        return (
            torch.zeros(1, 1, dtype=torch.bool, device=device),
            torch.randn(1, total_dim, device=device),
        )

    # ------------------------------------------------------------------
    # High-level API (not traced by torch.export)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def query(self, query_vector, meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a query vector and return top-K results.

        Returns:
            (scores [K], indices [K]) — squeezed single-query result.
        """
        from torch_recall.recall_method.knn.encoder import encode_query

        q = encode_query(query_vector, meta)  # [1, D]
        if self.offset > 0:
            q = torch.cat([torch.zeros(1, self.offset), q], dim=1)
        dummy_pred = torch.zeros(1, 1, dtype=torch.bool)
        scores = self(dummy_pred, q)           # [1, N]
        top_vals, top_idx = scores.topk(self.k, dim=1)

        if self.metric == "l2":
            return (-top_vals).squeeze(0), top_idx.squeeze(0)
        return top_vals.squeeze(0), top_idx.squeeze(0)

    def save_query_tensors(self, query_vector, meta: dict, path: str) -> None:
        """Encode a query vector and save for C++ inference."""
        from torch_recall.recall_method.knn.encoder import encode_query

        q = encode_query(query_vector, meta)  # [1, D]
        if self.offset > 0:
            q = torch.cat([torch.zeros(1, self.offset), q], dim=1)
        dummy_pred = torch.zeros(1, 1, dtype=torch.bool)
        torch.save([[dummy_pred, q]], path)

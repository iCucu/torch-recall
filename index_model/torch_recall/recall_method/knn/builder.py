from __future__ import annotations

import json

import torch

from torch_recall.schema import Item
from torch_recall.recall_method.knn.recall import KNNRecall, VALID_METRICS


class KNNBuilder:
    """Build a KNNRecall model from item embeddings.

    Embeddings are registered as frozen buffers; the resulting module
    computes brute-force matmul + top-K in its forward pass.
    """

    def __init__(self, k: int, metric: str = "cosine"):
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}, expected one of {sorted(VALID_METRICS)}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.metric = metric

    def build(self, items: list[Item]) -> tuple[KNNRecall, dict]:
        """Build the KNN recall model from a list of items.

        Args:
            items: list of Item objects, each must have embedding set.
        Returns:
            (model, meta) tuple.
        """
        N = len(items)
        if self.k > N:
            raise ValueError(f"k={self.k} exceeds num_items={N}")

        rows: list[list[float]] = []
        for idx, item in enumerate(items):
            if item.embedding is None:
                raise ValueError(f"Item {idx}: embedding is None")
            rows.append(item.embedding)

        embeddings = torch.tensor(rows, dtype=torch.float32)
        D = embeddings.shape[1]

        raw_ids = [item.id for item in items]
        item_ids = raw_ids if any(i is not None for i in raw_ids) else None

        embeddings = embeddings.float()

        if self.metric == "cosine":
            norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            embeddings = embeddings / norms
            embedding_norms = torch.zeros(N)
        elif self.metric == "l2":
            embedding_norms = (embeddings * embeddings).sum(dim=1)  # [N]
        else:
            embedding_norms = torch.zeros(N)

        meta: dict = {
            "num_items": N,
            "embedding_dim": D,
            "k": self.k,
            "metric": self.metric,
            "item_ids": item_ids,
        }

        model = KNNRecall(
            embeddings=embeddings,
            embedding_norms=embedding_norms,
            k=self.k,
            metric=self.metric,
            num_items=N,
            embedding_dim=D,
        )
        return model, meta

    @staticmethod
    def save_meta(meta: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

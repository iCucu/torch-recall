from __future__ import annotations

import json

import torch

from torch_recall.recall_method.base import RecallOp
from torch_recall.schema import Item
from torch_recall.scheduler.spec import (
    RecallSpec,
    Targeting as TargetingSpec,
    KNN as KNNSpec,
    And as AndSpec,
    Or as OrSpec,
)
from torch_recall.scheduler.pipeline import (
    AndModule,
    OrModule,
    RecallPipeline,
)
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.knn.builder import KNNBuilder


def _collect_leaves(
    spec: RecallSpec,
    targeting: list[TargetingSpec],
    knn: list[KNNSpec],
) -> None:
    """DFS to collect all leaf specs (in tree-walk order)."""
    if isinstance(spec, TargetingSpec):
        targeting.append(spec)
    elif isinstance(spec, KNNSpec):
        knn.append(spec)
    elif isinstance(spec, (AndSpec, OrSpec)):
        for child in spec.children:
            _collect_leaves(child, targeting, knn)
    else:
        raise TypeError(f"Unknown spec node: {type(spec)}")


class PipelineBuilder:
    """Compiles a declarative spec tree + items into a RecallPipeline."""

    def __init__(self, spec: RecallSpec, k: int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.spec = spec
        self.k = k

    def build(self, items: list[Item]) -> tuple[RecallPipeline, dict]:
        """Build the pipeline model and its meta from a list of items.

        Returns:
            (RecallPipeline, meta_dict)
        """
        N = len(items)
        if self.k > N:
            raise ValueError(f"k={self.k} exceeds num_items={N}")

        # --- 1. collect leaves ---
        targeting_leaves: list[TargetingSpec] = []
        knn_leaves: list[KNNSpec] = []
        _collect_leaves(self.spec, targeting_leaves, knn_leaves)

        # --- 2. build shared targeting model (if any) ---
        targeting_model = None
        targeting_meta: dict | None = None
        if targeting_leaves:
            first = targeting_leaves[0]
            for other in targeting_leaves[1:]:
                if other.schema is not first.schema:
                    raise ValueError(
                        "Multiple Targeting specs must share the same Schema instance"
                    )
            builder = TargetingBuilder(
                first.schema,
                max_preds_per_conj=first.max_preds_per_conj,
                max_conj_per_item=first.max_conj_per_item,
            )
            targeting_model, targeting_meta = builder.build(items)

        num_preds = targeting_meta["num_preds"] if targeting_meta else 1

        # --- 3. build KNN models + assign query offsets ---
        knn_models: list[tuple[KNNSpec, RecallOp]] = []
        knn_meta_list: list[dict] = []
        offset = 0

        for spec_node in knn_leaves:
            knn_builder = KNNBuilder(k=self.k, metric=spec_node.metric)
            knn_model, knn_meta = knn_builder.build(items)
            dim = knn_meta["embedding_dim"]

            knn_model.offset = offset
            knn_model.weight = spec_node.weight

            knn_models.append((spec_node, knn_model))
            knn_meta_list.append({
                "offset": offset,
                "dim": dim,
                "metric": spec_node.metric,
                "weight": spec_node.weight,
            })
            offset += dim

        total_query_dim = offset if offset > 0 else 1

        # --- 4. recursively compile the spec tree into RecallOps ---
        knn_iter = iter(knn_models)

        def _compile(node: RecallSpec) -> RecallOp:
            if isinstance(node, TargetingSpec):
                assert targeting_model is not None
                return targeting_model
            if isinstance(node, KNNSpec):
                _, model = next(knn_iter)
                return model
            if isinstance(node, AndSpec):
                return AndModule([_compile(c) for c in node.children])
            if isinstance(node, OrSpec):
                return OrModule([_compile(c) for c in node.children])
            raise TypeError(f"Unknown spec node: {type(node)}")

        root = _compile(self.spec)

        # --- 5. wrap in RecallPipeline ---
        pipeline = RecallPipeline(
            root=root,
            k=self.k,
            num_items=N,
            num_preds=num_preds,
            total_query_dim=total_query_dim,
        )

        # --- 6. build merged meta ---
        raw_ids = [item.id for item in items]
        item_ids = raw_ids if any(i is not None for i in raw_ids) else None

        meta: dict = {
            "k": self.k,
            "num_items": N,
            "num_preds": num_preds,
            "total_query_dim": total_query_dim,
            "item_ids": item_ids,
        }
        if targeting_meta is not None:
            meta["targeting"] = targeting_meta
        if knn_meta_list:
            meta["knn_leaves"] = knn_meta_list

        return pipeline, meta

    @staticmethod
    def save_meta(meta: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

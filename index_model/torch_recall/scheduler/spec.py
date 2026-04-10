from __future__ import annotations

from torch_recall.schema import Schema


class RecallSpec:
    """Base class for all recall composition spec nodes.

    A spec tree is pure data describing how recall methods combine.
    PipelineBuilder.build() compiles a spec tree into a single nn.Module.
    """


class Targeting(RecallSpec):
    """Leaf: targeting (boolean-rule) recall."""

    def __init__(
        self,
        schema: Schema,
        max_preds_per_conj: int | None = None,
        max_conj_per_item: int | None = None,
    ):
        self.schema = schema
        self.max_preds_per_conj = max_preds_per_conj
        self.max_conj_per_item = max_conj_per_item


class KNN(RecallSpec):
    """Leaf: K-nearest-neighbor (embedding) recall."""

    def __init__(self, metric: str = "cosine", weight: float = 1.0):
        self.metric = metric
        self.weight = weight


class And(RecallSpec):
    """Combiner: intersection — ``score = sum(children)``.

    Because targeting scores are 0 (match) / -inf (no match), summing
    with KNN scores naturally implements hard filtering: -inf poisons
    the total, so unmatched items are excluded after topk.
    """

    def __init__(self, *children: RecallSpec):
        if len(children) < 2:
            raise ValueError("And requires at least 2 children")
        self.children = list(children)


class Or(RecallSpec):
    """Combiner: union — ``score = max(children)``.

    An item is included if *any* child gives it a finite score.
    """

    def __init__(self, *children: RecallSpec):
        if len(children) < 2:
            raise ValueError("Or requires at least 2 children")
        self.children = list(children)

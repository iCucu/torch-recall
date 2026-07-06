"""Builder for DiffusionRecall.

Mirrors :class:`autoregressive.builder.GenerativeBuilder` in structure
but produces a :class:`DiffusionRecall` backed by a :class:`SidPathFilter`
instead of a Trie.  The Trie is not required here: the path-vector matrix
is built directly from the raw SID paths.
"""

from __future__ import annotations

import json

import torch

from torch_recall.schema import Item, Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.diffusion.path_filter import SidPathFilter


class DiffusionBuilder:
    """Builds a :class:`DiffusionRecall` from items with ``sid_path``."""

    def __init__(
        self,
        schema: Schema,
        decoder: torch.nn.Module,
        beam_width: int = 10,
        sid_vocab_size: int = 2048,
        path_length: int = 5,
    ):
        self.schema = schema
        self.decoder = decoder
        self.beam_width = beam_width
        self.sid_vocab_size = sid_vocab_size
        self.path_length = path_length

    # -- public API --------------------------------------------------------

    def build(self, items: list[Item]) -> tuple[torch.nn.Module, dict]:
        from torch_recall.recall_method.diffusion.recall import DiffusionRecall

        self._validate_items(items)
        N = len(items)

        targeting_model, targeting_meta = TargetingBuilder(self.schema).build(items)

        path_vectors = torch.tensor(
            [item.sid_path for item in items], dtype=torch.long
        )  # [N, L]

        path_filter = SidPathFilter(path_vectors, self.sid_vocab_size)

        user_dim = getattr(self.decoder, "user_dim", 64)

        model = DiffusionRecall(
            targeting=targeting_model,
            path_filter=path_filter,
            decoder=self.decoder,
            beam_width=self.beam_width,
            num_items=N,
            num_preds=targeting_meta["num_preds"],
            user_dim=user_dim,
        )

        meta: dict = {
            "num_items": N,
            "num_preds": targeting_meta["num_preds"],
            "beam_width": self.beam_width,
            "sid_vocab_size": self.sid_vocab_size,
            "path_length": self.path_length,
            "user_dim": user_dim,
            "targeting": targeting_meta,
            "item_ids": [item.id for item in items] if items[0].id else None,
        }
        return model, meta

    def save_meta(self, meta: dict, path: str) -> None:
        serializable = {k: v for k, v in meta.items() if k != "targeting"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # -- validation --------------------------------------------------------

    def _validate_items(self, items: list[Item]) -> None:
        for i, item in enumerate(items):
            if item.sid_path is None:
                raise ValueError(f"Item {i}: sid_path is None")
            if len(item.sid_path) != self.path_length:
                raise ValueError(
                    f"Item {i}: sid_path length {len(item.sid_path)} != {self.path_length}"
                )
            for j, sid in enumerate(item.sid_path):
                if not (0 <= sid < self.sid_vocab_size):
                    raise ValueError(
                        f"Item {i}, position {j}: sid {sid} not in [0, {self.sid_vocab_size})"
                    )

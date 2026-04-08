import json
import math
from pathlib import Path

import torch

from torch_recall.schema import Schema
from torch_recall.model import InvertedIndexModel
from torch_recall.tokenizer import WhitespaceTokenizer, Tokenizer


class IndexBuilder:
    def __init__(self, schema: Schema, tokenizer: Tokenizer | None = None):
        schema.validate()
        self.schema = schema
        self.tokenizer = tokenizer or WhitespaceTokenizer()

    def build(self, items: list[dict]) -> tuple[InvertedIndexModel, dict]:
        N = len(items)
        L = math.ceil(N / 64)

        discrete_dicts: dict[str, dict[str, int]] = {}
        text_dicts: dict[str, dict[str, int]] = {}

        # Pass 1: scan items to build dictionaries + per-item encodings
        discrete_encoded: dict[str, list[int]] = {}
        text_encoded: dict[str, list[set[int]]] = {}

        for field_name in self.schema.discrete_fields:
            vals: set[str] = set()
            for item in items:
                vals.add(str(item.get(field_name, "")))
            vdict = {v: i for i, v in enumerate(sorted(vals))}
            discrete_dicts[field_name] = vdict
            discrete_encoded[field_name] = [
                vdict[str(item.get(field_name, ""))] for item in items
            ]

        for field_name in self.schema.text_fields:
            all_terms: set[str] = set()
            per_item: list[set[str]] = []
            for item in items:
                text = str(item.get(field_name, ""))
                terms = set(self.tokenizer.tokenize(text))
                terms.discard("")
                per_item.append(terms)
                all_terms.update(terms)
            tdict = {t: i for i, t in enumerate(sorted(all_terms))}
            text_dicts[field_name] = tdict
            text_encoded[field_name] = [
                {tdict[t] for t in terms if t in tdict} for terms in per_item
            ]

        # Pass 2: allocate all bitmaps at once, fill via single item scan
        bitmap_layout: list[tuple[str, str, int]] = []  # (type_prefix, field, val_id)
        bitmap_lookup: dict[str, dict[int, int]] = {}

        for field_name in self.schema.discrete_fields:
            field_lookup: dict[int, int] = {}
            for val_id in range(len(discrete_dicts[field_name])):
                field_lookup[val_id] = len(bitmap_layout)
                bitmap_layout.append(("d", field_name, val_id))
            bitmap_lookup[f"d:{field_name}"] = field_lookup

        for field_name in self.schema.text_fields:
            field_lookup = {}
            for term_id in range(len(text_dicts[field_name])):
                field_lookup[term_id] = len(bitmap_layout)
                bitmap_layout.append(("t", field_name, term_id))
            bitmap_lookup[f"t:{field_name}"] = field_lookup

        num_bitmaps = len(bitmap_layout) if bitmap_layout else 1
        bitmaps = torch.zeros(num_bitmaps, L, dtype=torch.int64)

        # Single scan over items: O(N * avg_fields_per_item)
        for idx in range(N):
            word_idx = idx >> 6  # idx // 64
            bit_val = 1 << (idx & 63)  # 1 << (idx % 64)

            for field_name in self.schema.discrete_fields:
                val_id = discrete_encoded[field_name][idx]
                global_idx = bitmap_lookup[f"d:{field_name}"][val_id]
                bitmaps[global_idx, word_idx] |= bit_val

            for field_name in self.schema.text_fields:
                term_ids = text_encoded[field_name][idx]
                fl = bitmap_lookup[f"t:{field_name}"]
                for tid in term_ids:
                    bitmaps[fl[tid], word_idx] |= bit_val

        # Build numeric columns
        if self.schema.numeric_fields:
            numeric_data = torch.zeros(
                len(self.schema.numeric_fields), N, dtype=torch.float32
            )
            for fi, field_name in enumerate(self.schema.numeric_fields):
                numeric_data[fi] = torch.tensor(
                    [float(item.get(field_name, 0.0)) for item in items],
                    dtype=torch.float32,
                )
        else:
            numeric_data = torch.zeros(1, N, dtype=torch.float32)

        item_ids = torch.arange(N, dtype=torch.int64)

        model = InvertedIndexModel(
            bitmaps=bitmaps,
            numeric_data=numeric_data,
            item_ids=item_ids,
            num_items=N,
        )

        meta = {
            "schema": {
                "discrete": self.schema.discrete_fields,
                "numeric": self.schema.numeric_fields,
                "text": self.schema.text_fields,
            },
            "discrete_dicts": discrete_dicts,
            "text_dicts": text_dicts,
            "bitmap_lookup": {
                k: {str(ki): vi for ki, vi in v.items()}
                for k, v in bitmap_lookup.items()
            },
            "num_items": N,
            "bitmap_len": L,
            "num_bitmaps": bitmaps.shape[0],
        }

        return model, meta

    def save_meta(self, meta: dict, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

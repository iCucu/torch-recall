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

        for field_name in self.schema.discrete_fields:
            vals = sorted({str(item.get(field_name, "")) for item in items})
            discrete_dicts[field_name] = {v: i for i, v in enumerate(vals)}

        for field_name in self.schema.text_fields:
            all_terms: set[str] = set()
            for item in items:
                text = str(item.get(field_name, ""))
                all_terms.update(self.tokenizer.tokenize(text))
            all_terms.discard("")
            text_dicts[field_name] = {t: i for i, t in enumerate(sorted(all_terms))}

        bitmaps_list: list[torch.Tensor] = []
        bitmap_lookup: dict[str, dict[int, int]] = {}

        for field_name in self.schema.discrete_fields:
            vdict = discrete_dicts[field_name]
            field_lookup: dict[int, int] = {}
            for val_str, val_id in vdict.items():
                bitmap = torch.zeros(L, dtype=torch.int64)
                for idx, item in enumerate(items):
                    if str(item.get(field_name, "")) == val_str:
                        word_idx = idx // 64
                        bit_idx = idx % 64
                        bitmap[word_idx] |= 1 << bit_idx
                global_idx = len(bitmaps_list)
                field_lookup[val_id] = global_idx
                bitmaps_list.append(bitmap)
            bitmap_lookup[f"d:{field_name}"] = field_lookup

        for field_name in self.schema.text_fields:
            tdict = text_dicts[field_name]
            field_lookup = {}
            tokenized_items = []
            for item in items:
                text = str(item.get(field_name, ""))
                tokenized_items.append(set(self.tokenizer.tokenize(text)))

            for term_str, term_id in tdict.items():
                bitmap = torch.zeros(L, dtype=torch.int64)
                for idx, terms in enumerate(tokenized_items):
                    if term_str in terms:
                        word_idx = idx // 64
                        bit_idx = idx % 64
                        bitmap[word_idx] |= 1 << bit_idx
                global_idx = len(bitmaps_list)
                field_lookup[term_id] = global_idx
                bitmaps_list.append(bitmap)
            bitmap_lookup[f"t:{field_name}"] = field_lookup

        if not bitmaps_list:
            bitmaps_list.append(torch.zeros(L, dtype=torch.int64))
        bitmaps = torch.stack(bitmaps_list)

        if self.schema.numeric_fields:
            numeric_data = torch.zeros(
                len(self.schema.numeric_fields), N, dtype=torch.float32
            )
            for fi, field_name in enumerate(self.schema.numeric_fields):
                for idx, item in enumerate(items):
                    numeric_data[fi, idx] = float(item.get(field_name, 0.0))
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

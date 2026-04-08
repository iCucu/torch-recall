# Bitmap Inverted Index — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a packed-bitmap inverted index as a PyTorch nn.Module, export to .pt2 via AOTInductor, and provide C++ inference code for < 20ms queries over 1M items.

**Architecture:** Items are stored as packed int64 bitmaps (discrete + text fields) and float32 columns (numeric fields). Queries are converted to DNF, encoded as tensors, and evaluated in a single forward pass of static tensor operations. The model is exported via `torch.export` + `aoti_compile_and_package` to .pt2, loaded in C++ with `AOTIModelPackageLoader`.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, CMake 3.18+, LibTorch (C++)

---

## File Structure

### Create:
- `offline/pyproject.toml` — Python project config, dependencies
- `offline/torch_recall/__init__.py` — package init, public API
- `offline/torch_recall/schema.py` — field type definitions, validation
- `offline/torch_recall/model.py` — InvertedIndexModel(nn.Module)
- `offline/torch_recall/builder.py` — IndexBuilder: items → model
- `offline/torch_recall/tokenizer.py` — text tokenization
- `offline/torch_recall/query.py` — query parser, DNF converter, tensor encoder
- `offline/torch_recall/exporter.py` — torch.export + AOTInductor pipeline
- `offline/tests/test_model.py`
- `offline/tests/test_builder.py`
- `offline/tests/test_query.py`
- `offline/tests/test_e2e.py`
- `offline/benchmarks/bench_recall.py`
- `online/CMakeLists.txt`
- `online/include/torch_recall/common.h` — shared types, constants
- `online/include/torch_recall/index_meta.h` — metadata loading
- `online/include/torch_recall/query_parser.h`
- `online/include/torch_recall/dnf_converter.h`
- `online/include/torch_recall/tensor_encoder.h`
- `online/include/torch_recall/model_runner.h`
- `online/include/torch_recall/result_decoder.h`
- `online/src/index_meta.cpp`
- `online/src/query_parser.cpp`
- `online/src/dnf_converter.cpp`
- `online/src/tensor_encoder.cpp`
- `online/src/model_runner.cpp`
- `online/src/result_decoder.cpp`
- `online/src/main.cpp`
- `online/tests/test_recall.cpp`
- `README.md`

---

## Task 1: Project Scaffold + Schema

**Files:**
- Create: `offline/pyproject.toml`
- Create: `offline/torch_recall/__init__.py`
- Create: `offline/torch_recall/schema.py`
- Test: `offline/tests/test_model.py` (just schema tests for now)

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "torch-recall"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.6.0",
    "jieba>=0.42.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["."]
```

- [ ] **Step 2: Create schema.py**

```python
from dataclasses import dataclass, field
from enum import IntEnum


class FieldType(IntEnum):
    DISCRETE = 0
    NUMERIC = 1
    TEXT = 2


class NumericOp(IntEnum):
    EQ = 0
    LT = 1
    GT = 2
    LE = 3
    GE = 4


@dataclass
class Schema:
    discrete_fields: list[str] = field(default_factory=list)
    numeric_fields: list[str] = field(default_factory=list)
    text_fields: list[str] = field(default_factory=list)

    def validate(self) -> None:
        all_names = self.discrete_fields + self.numeric_fields + self.text_fields
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate field names across types")
        if not all_names:
            raise ValueError("Schema must have at least one field")

    @property
    def discrete_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.discrete_fields)}

    @property
    def numeric_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.numeric_fields)}

    @property
    def text_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.text_fields)}


MAX_BP = 32
MAX_NP = 16
MAX_CONJ = 16
P_TOTAL = MAX_BP + MAX_NP
```

- [ ] **Step 3: Create __init__.py**

```python
from torch_recall.schema import Schema, FieldType, NumericOp, MAX_BP, MAX_NP, MAX_CONJ
```

- [ ] **Step 4: Run tests**

Create `offline/tests/test_model.py`:

```python
import pytest
from torch_recall.schema import Schema


def test_schema_validation_ok():
    s = Schema(discrete_fields=["city"], numeric_fields=["price"], text_fields=["title"])
    s.validate()


def test_schema_empty_raises():
    s = Schema()
    with pytest.raises(ValueError, match="at least one field"):
        s.validate()


def test_schema_duplicate_raises():
    s = Schema(discrete_fields=["city"], numeric_fields=["city"])
    with pytest.raises(ValueError, match="Duplicate"):
        s.validate()


def test_field_index():
    s = Schema(discrete_fields=["city", "gender"], numeric_fields=["price"])
    assert s.discrete_field_index == {"city": 0, "gender": 1}
    assert s.numeric_field_index == {"price": 0}
```

Run: `cd offline && pip install -e ".[dev]" && pytest tests/test_model.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add offline/pyproject.toml offline/torch_recall/ offline/tests/test_model.py
git commit -m "feat: project scaffold with schema definitions"
```

---

## Task 2: InvertedIndexModel (nn.Module)

**Files:**
- Create: `offline/torch_recall/model.py`
- Modify: `offline/torch_recall/__init__.py`
- Test: `offline/tests/test_model.py` (add model tests)

- [ ] **Step 1: Write failing tests for model forward pass**

Append to `offline/tests/test_model.py`:

```python
import torch
from torch_recall.model import InvertedIndexModel


def _make_small_model():
    """4 items, 3 bitmap entries, 1 numeric field."""
    N = 4
    L = 1  # ceil(4/64) = 1
    bitmaps = torch.zeros(3, L, dtype=torch.int64)
    # bitmap 0: items {0, 2} → bits 0,2 → value = 0b0101 = 5
    bitmaps[0, 0] = 5
    # bitmap 1: items {1, 3} → bits 1,3 → value = 0b1010 = 10
    bitmaps[1, 0] = 10
    # bitmap 2: items {0, 1, 2} → bits 0,1,2 → value = 0b0111 = 7
    bitmaps[2, 0] = 7

    numeric_data = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # 1 field, 4 items
    item_ids = torch.arange(4, dtype=torch.int64)

    return InvertedIndexModel(
        bitmaps=bitmaps,
        numeric_data=numeric_data,
        item_ids=item_ids,
        num_items=N,
    )


def test_single_bitmap_predicate():
    model = _make_small_model()
    # Query: bitmap[0] → items {0, 2}
    bitmap_indices = torch.zeros(32, dtype=torch.int64)
    bitmap_valid = torch.zeros(32, dtype=torch.bool)
    bitmap_indices[0] = 0
    bitmap_valid[0] = True

    numeric_fields = torch.zeros(16, dtype=torch.int64)
    numeric_ops = torch.zeros(16, dtype=torch.int64)
    numeric_values = torch.zeros(16, dtype=torch.float32)
    numeric_valid = torch.zeros(16, dtype=torch.bool)

    negation_mask = torch.zeros(48, dtype=torch.bool)

    conj_matrix = torch.zeros(16, 48, dtype=torch.bool)
    conj_matrix[0, 0] = True  # conj 0 uses pred 0
    conj_valid = torch.zeros(16, dtype=torch.bool)
    conj_valid[0] = True

    result = model(
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid,
    )
    # result should have bits 0,2 set → value 5
    assert result[0].item() == 5


def test_and_two_bitmaps():
    model = _make_small_model()
    # Query: bitmap[0] AND bitmap[2] → {0,2} ∩ {0,1,2} = {0,2} → 5
    bitmap_indices = torch.zeros(32, dtype=torch.int64)
    bitmap_valid = torch.zeros(32, dtype=torch.bool)
    bitmap_indices[0] = 0
    bitmap_indices[1] = 2
    bitmap_valid[0] = True
    bitmap_valid[1] = True

    numeric_fields = torch.zeros(16, dtype=torch.int64)
    numeric_ops = torch.zeros(16, dtype=torch.int64)
    numeric_values = torch.zeros(16, dtype=torch.float32)
    numeric_valid = torch.zeros(16, dtype=torch.bool)
    negation_mask = torch.zeros(48, dtype=torch.bool)

    conj_matrix = torch.zeros(16, 48, dtype=torch.bool)
    conj_matrix[0, 0] = True
    conj_matrix[0, 1] = True
    conj_valid = torch.zeros(16, dtype=torch.bool)
    conj_valid[0] = True

    result = model(
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid,
    )
    assert result[0].item() == 5


def test_or_two_bitmaps():
    model = _make_small_model()
    # Query: bitmap[0] OR bitmap[1] → {0,2} ∪ {1,3} = {0,1,2,3} → 15
    bitmap_indices = torch.zeros(32, dtype=torch.int64)
    bitmap_valid = torch.zeros(32, dtype=torch.bool)
    bitmap_indices[0] = 0
    bitmap_indices[1] = 1
    bitmap_valid[0] = True
    bitmap_valid[1] = True

    numeric_fields = torch.zeros(16, dtype=torch.int64)
    numeric_ops = torch.zeros(16, dtype=torch.int64)
    numeric_values = torch.zeros(16, dtype=torch.float32)
    numeric_valid = torch.zeros(16, dtype=torch.bool)
    negation_mask = torch.zeros(48, dtype=torch.bool)

    # Two conjunctions, each with one predicate → OR
    conj_matrix = torch.zeros(16, 48, dtype=torch.bool)
    conj_matrix[0, 0] = True  # conj 0: bitmap[0]
    conj_matrix[1, 1] = True  # conj 1: bitmap[1]
    conj_valid = torch.zeros(16, dtype=torch.bool)
    conj_valid[0] = True
    conj_valid[1] = True

    result = model(
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid,
    )
    assert result[0].item() == 15


def test_not_bitmap():
    model = _make_small_model()
    # Query: NOT bitmap[0] → NOT {0,2} = {1,3} → 10
    # But all_ones for 4 items is 0b1111 = 15, NOT 5 = 15 XOR 5 = 10
    bitmap_indices = torch.zeros(32, dtype=torch.int64)
    bitmap_valid = torch.zeros(32, dtype=torch.bool)
    bitmap_indices[0] = 0
    bitmap_valid[0] = True

    numeric_fields = torch.zeros(16, dtype=torch.int64)
    numeric_ops = torch.zeros(16, dtype=torch.int64)
    numeric_values = torch.zeros(16, dtype=torch.float32)
    numeric_valid = torch.zeros(16, dtype=torch.bool)

    negation_mask = torch.zeros(48, dtype=torch.bool)
    negation_mask[0] = True  # negate predicate 0

    conj_matrix = torch.zeros(16, 48, dtype=torch.bool)
    conj_matrix[0, 0] = True
    conj_valid = torch.zeros(16, dtype=torch.bool)
    conj_valid[0] = True

    result = model(
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid,
    )
    # NOT on packed int64 flips ALL 64 bits. We need a valid_items_mask
    # to mask out bits beyond N. With N=4, valid bits = 0b1111 = 15.
    # ~5 in int64 is all bits flipped. We AND with valid_mask (15) → 10
    assert result[0].item() == 10


def test_numeric_less_than():
    model = _make_small_model()
    # Query: numeric_field[0] < 25 → items with values [10,20,30,40] < 25 → {0,1} → 0b0011 = 3
    bitmap_indices = torch.zeros(32, dtype=torch.int64)
    bitmap_valid = torch.zeros(32, dtype=torch.bool)

    numeric_fields = torch.zeros(16, dtype=torch.int64)
    numeric_ops = torch.zeros(16, dtype=torch.int64)
    numeric_values = torch.zeros(16, dtype=torch.float32)
    numeric_valid = torch.zeros(16, dtype=torch.bool)
    numeric_fields[0] = 0
    numeric_ops[0] = 1  # LT
    numeric_values[0] = 25.0
    numeric_valid[0] = True

    negation_mask = torch.zeros(48, dtype=torch.bool)

    conj_matrix = torch.zeros(16, 48, dtype=torch.bool)
    conj_matrix[0, 32] = True  # pred index 32 = first numeric pred (after MAX_BP=32)
    conj_valid = torch.zeros(16, dtype=torch.bool)
    conj_valid[0] = True

    result = model(
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid,
    )
    assert result[0].item() == 3
```

Run: `cd offline && pytest tests/test_model.py -v`
Expected: FAIL (model.py doesn't exist yet)

- [ ] **Step 2: Implement InvertedIndexModel**

Create `offline/torch_recall/model.py`:

```python
import torch
import torch.nn as nn

from torch_recall.schema import MAX_BP, MAX_NP, MAX_CONJ, P_TOTAL


class InvertedIndexModel(nn.Module):
    def __init__(
        self,
        bitmaps: torch.Tensor,
        numeric_data: torch.Tensor,
        item_ids: torch.Tensor,
        num_items: int,
    ):
        super().__init__()
        L = bitmaps.shape[1]

        self.register_buffer("bitmaps", bitmaps)
        self.register_buffer("numeric_data", numeric_data)
        self.register_buffer("item_ids", item_ids)

        all_ones = torch.full((L,), -1, dtype=torch.int64)
        self.register_buffer("all_ones", all_ones)

        valid_mask = self._make_valid_mask(num_items, L)
        self.register_buffer("valid_mask", valid_mask)

        pack_powers = torch.tensor([1 << i for i in range(63)] + [-(1 << 63)], dtype=torch.int64)
        self.register_buffer("pack_powers", pack_powers)

        self.num_items = num_items
        self.bitmap_len = L

    @staticmethod
    def _make_valid_mask(num_items: int, L: int) -> torch.Tensor:
        """Bitmap mask with only the first num_items bits set."""
        mask = torch.zeros(L, dtype=torch.int64)
        full_words = num_items // 64
        remainder = num_items % 64
        if full_words > 0:
            mask[:full_words] = -1  # all bits set
        if remainder > 0:
            mask[full_words] = (1 << remainder) - 1
        return mask

    def _pack_bool_to_bitmap(self, bool_tensor: torch.Tensor) -> torch.Tensor:
        """Pack [B, N] bool tensor to [B, L] int64 packed bitmap."""
        B = bool_tensor.shape[0]
        N = bool_tensor.shape[1]
        pad_size = self.bitmap_len * 64 - N
        if pad_size > 0:
            bool_tensor = torch.nn.functional.pad(bool_tensor, (0, pad_size))
        reshaped = bool_tensor.view(B, self.bitmap_len, 64).long()
        packed = (reshaped * self.pack_powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return packed

    def forward(
        self,
        bitmap_indices: torch.Tensor,
        bitmap_valid: torch.Tensor,
        numeric_fields: torch.Tensor,
        numeric_ops: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_valid: torch.Tensor,
        negation_mask: torch.Tensor,
        conj_matrix: torch.Tensor,
        conj_valid: torch.Tensor,
    ) -> torch.Tensor:
        L = self.bitmap_len

        # 1. Fetch bitmap predicates
        bp = self.bitmaps[bitmap_indices]  # [MAX_BP, L]

        # 2. Evaluate numeric predicates
        cols = self.numeric_data[numeric_fields]  # [MAX_NP, N]
        vals = numeric_values.unsqueeze(1)  # [MAX_NP, 1]
        ops = numeric_ops  # [MAX_NP]

        numeric_bool = (
            ((cols == vals) & (ops == 0).unsqueeze(1))
            | ((cols < vals) & (ops == 1).unsqueeze(1))
            | ((cols > vals) & (ops == 2).unsqueeze(1))
            | ((cols <= vals) & (ops == 3).unsqueeze(1))
            | ((cols >= vals) & (ops == 4).unsqueeze(1))
        )  # [MAX_NP, N]

        np_bitmap = self._pack_bool_to_bitmap(numeric_bool)  # [MAX_NP, L]

        # 3. Stack all predicate bitmaps
        all_bm = torch.cat([bp, np_bitmap], dim=0)  # [P, L]

        # 4. Apply negation: XOR with all_ones, then AND with valid_mask
        neg = negation_mask[:P_TOTAL].unsqueeze(1)  # [P, 1]
        all_bm = torch.where(neg, (all_bm ^ self.all_ones) & self.valid_mask, all_bm)

        # 5. Apply validity: invalid predicates → all_ones (identity for AND)
        all_valid = torch.cat([bitmap_valid, numeric_valid])  # [P]
        all_bm = torch.where(
            all_valid.unsqueeze(1),
            all_bm,
            self.all_ones.unsqueeze(0).expand(P_TOTAL, L),
        )

        # 6. DNF combine (unrolled loops)
        pred_parts = all_bm.unbind(0)  # P tensors of shape [L]

        result = torch.zeros(L, dtype=torch.int64, device=all_bm.device)

        for c in range(MAX_CONJ):
            conj_row = conj_matrix[c]  # [P]
            conj_result = self.all_ones.clone()
            for p in range(P_TOTAL):
                selected = torch.where(conj_row[p], pred_parts[p], self.all_ones)
                conj_result = conj_result & selected
            conj_result = torch.where(conj_valid[c], conj_result, torch.zeros_like(conj_result))
            result = result | conj_result

        # Mask to valid items only
        result = result & self.valid_mask

        return result
```

- [ ] **Step 3: Update __init__.py**

```python
from torch_recall.schema import Schema, FieldType, NumericOp, MAX_BP, MAX_NP, MAX_CONJ
from torch_recall.model import InvertedIndexModel
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd offline && pytest tests/test_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add offline/torch_recall/model.py offline/torch_recall/__init__.py offline/tests/test_model.py
git commit -m "feat: InvertedIndexModel with packed bitmap DNF evaluation"
```

---

## Task 3: Index Builder + Tokenizer

**Files:**
- Create: `offline/torch_recall/tokenizer.py`
- Create: `offline/torch_recall/builder.py`
- Test: `offline/tests/test_builder.py`

- [ ] **Step 1: Write failing tests**

Create `offline/tests/test_builder.py`:

```python
import torch
from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder


def test_build_discrete_only():
    schema = Schema(discrete_fields=["city", "gender"])
    items = [
        {"city": "北京", "gender": "男"},
        {"city": "上海", "gender": "女"},
        {"city": "北京", "gender": "女"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert model.bitmaps.shape[0] > 0
    assert model.num_items == 3
    assert "city" in meta["discrete_dicts"]
    assert "北京" in meta["discrete_dicts"]["city"]


def test_build_with_numeric():
    schema = Schema(discrete_fields=["city"], numeric_fields=["price"])
    items = [
        {"city": "北京", "price": 10.0},
        {"city": "上海", "price": 20.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert model.numeric_data.shape == (1, 2)
    assert model.numeric_data[0, 0].item() == 10.0


def test_build_with_text():
    schema = Schema(text_fields=["title"])
    items = [
        {"title": "游戏 攻略"},
        {"title": "美食 推荐"},
        {"title": "游戏 推荐"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert "title" in meta["text_dicts"]
    term_dict = meta["text_dicts"]["title"]
    assert "游戏" in term_dict
    assert "推荐" in term_dict
```

Run: `cd offline && pytest tests/test_builder.py -v`
Expected: FAIL

- [ ] **Step 2: Implement tokenizer.py**

```python
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.strip().split()


class JiebaTokenizer:
    def __init__(self):
        import jieba
        self._jieba = jieba

    def tokenize(self, text: str) -> list[str]:
        return list(self._jieba.cut(text))
```

- [ ] **Step 3: Implement builder.py**

```python
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
        bitmap_entries: list[tuple[str, str, int, int]] = []
        # (field_type, field_name, field_specific_id, global_bitmap_idx)

        # Pass 1: build value dictionaries
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

        # Pass 2: build bitmaps
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

        # Build numeric columns
        if self.schema.numeric_fields:
            numeric_data = torch.zeros(len(self.schema.numeric_fields), N, dtype=torch.float32)
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
            "bitmap_lookup": {k: {str(ki): vi for ki, vi in v.items()} for k, v in bitmap_lookup.items()},
            "num_items": N,
            "bitmap_len": L,
            "num_bitmaps": bitmaps.shape[0],
        }

        return model, meta

    def save_meta(self, meta: dict, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd offline && pytest tests/test_builder.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add offline/torch_recall/tokenizer.py offline/torch_recall/builder.py offline/tests/test_builder.py
git commit -m "feat: IndexBuilder with bitmap construction and text tokenization"
```

---

## Task 4: Query Parser + DNF Converter + Tensor Encoder

**Files:**
- Create: `offline/torch_recall/query.py`
- Test: `offline/tests/test_query.py`

- [ ] **Step 1: Write failing tests**

Create `offline/tests/test_query.py`:

```python
import pytest
import torch
from torch_recall.query import parse_expr, to_dnf, encode_query, And, Or, Not, Predicate


def test_parse_simple_eq():
    expr = parse_expr('city == "北京"')
    assert isinstance(expr, Predicate)
    assert expr.field == "city"
    assert expr.op == "=="
    assert expr.value == "北京"


def test_parse_and():
    expr = parse_expr('city == "北京" AND gender == "男"')
    assert isinstance(expr, And)
    assert len(expr.children) == 2


def test_parse_or():
    expr = parse_expr('city == "北京" OR city == "上海"')
    assert isinstance(expr, Or)


def test_parse_not():
    expr = parse_expr('NOT category == "体育"')
    assert isinstance(expr, Not)


def test_parse_nested():
    expr = parse_expr('(city == "北京" OR city == "上海") AND gender == "男"')
    assert isinstance(expr, And)


def test_parse_numeric():
    expr = parse_expr("price < 100.0")
    assert isinstance(expr, Predicate)
    assert expr.op == "<"
    assert expr.value == 100.0


def test_parse_contains():
    expr = parse_expr('title contains "游戏"')
    assert isinstance(expr, Predicate)
    assert expr.op == "contains"


def test_dnf_simple_and():
    expr = parse_expr('city == "北京" AND gender == "男"')
    dnf = to_dnf(expr)
    assert len(dnf) == 1  # one conjunction
    assert len(dnf[0]) == 2  # two predicates


def test_dnf_or_distributes():
    expr = parse_expr('(city == "北京" OR city == "上海") AND gender == "男"')
    dnf = to_dnf(expr)
    assert len(dnf) == 2  # two conjunctions


def test_dnf_not_pushdown():
    expr = parse_expr('NOT (city == "北京" AND gender == "男")')
    dnf = to_dnf(expr)
    # De Morgan: NOT city=北京 OR NOT gender=男 → 2 conjunctions
    assert len(dnf) == 2


def test_encode_query():
    meta = {
        "schema": {"discrete": ["city", "gender"], "numeric": ["price"], "text": ["title"]},
        "discrete_dicts": {
            "city": {"北京": 0, "上海": 1},
            "gender": {"男": 0, "女": 1},
        },
        "text_dicts": {"title": {"游戏": 0, "美食": 1}},
        "bitmap_lookup": {
            "d:city": {"0": 0, "1": 1},
            "d:gender": {"0": 2, "1": 3},
            "t:title": {"0": 4, "1": 5},
        },
        "num_items": 100,
        "bitmap_len": 2,
        "num_bitmaps": 6,
    }
    tensors = encode_query('city == "北京" AND price < 50.0', meta)
    assert tensors["bitmap_indices"].shape[0] == 32
    assert tensors["bitmap_valid"][0].item() is True
    assert tensors["numeric_valid"][0].item() is True
    assert tensors["conj_valid"][0].item() is True
```

Run: `cd offline && pytest tests/test_query.py -v`
Expected: FAIL

- [ ] **Step 2: Implement query.py — AST nodes**

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

import torch

from torch_recall.schema import MAX_BP, MAX_NP, MAX_CONJ, P_TOTAL, NumericOp


# --- AST Nodes ---

@dataclass
class Predicate:
    field: str
    op: str        # ==, !=, <, >, <=, >=, contains
    value: object  # str or float

@dataclass
class And:
    children: list[Expr]

@dataclass
class Or:
    children: list[Expr]

@dataclass
class Not:
    child: Expr

Expr = Union[Predicate, And, Or, Not]


# --- Negated Predicate (used in DNF) ---

@dataclass
class LiteralPred:
    pred: Predicate
    negated: bool = False
```

- [ ] **Step 3: Implement query.py — parser**

Append to `query.py`:

```python
# --- Tokenizer for expression strings ---

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (AND|OR|NOT|contains)          # keywords
        |([<>!=]=?|==)                 # operators
        |"([^"]*)"                     # quoted string
        |'([^']*)'                     # single-quoted string
        |(\()                          # lparen
        |(\))                          # rparen
        |([^\s()'"<>!=]+)              # bare word (field name or number)
    )\s*
    """,
    re.VERBOSE,
)

def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        if m.group(1):
            tokens.append(("KW", m.group(1)))
        elif m.group(2):
            tokens.append(("OP", m.group(2)))
        elif m.group(3) is not None:
            tokens.append(("STR", m.group(3)))
        elif m.group(4) is not None:
            tokens.append(("STR", m.group(4)))
        elif m.group(5):
            tokens.append(("LPAREN", "("))
        elif m.group(6):
            tokens.append(("RPAREN", ")"))
        elif m.group(7):
            tokens.append(("WORD", m.group(7)))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        children = [left]
        while self.peek() and self.peek() == ("KW", "OR"):
            self.consume()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else Or(children)

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        children = [left]
        while self.peek() and self.peek() == ("KW", "AND"):
            self.consume()
            children.append(self._parse_not())
        return children[0] if len(children) == 1 else And(children)

    def _parse_not(self) -> Expr:
        if self.peek() and self.peek() == ("KW", "NOT"):
            self.consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        if self.peek() and self.peek()[0] == "LPAREN":
            self.consume()
            expr = self.parse_expr()
            self.consume("RPAREN")
            return expr
        return self._parse_predicate()

    def _parse_predicate(self) -> Predicate:
        field_tok = self.consume("WORD")
        field_name = field_tok[1]

        next_tok = self.peek()
        if next_tok and next_tok == ("KW", "contains"):
            self.consume()
            val_tok = self.consume("STR")
            return Predicate(field_name, "contains", val_tok[1])

        op_tok = self.consume("OP")
        val_tok = self.consume()
        if val_tok[0] == "STR":
            value = val_tok[1]
        elif val_tok[0] == "WORD":
            try:
                value = float(val_tok[1])
            except ValueError:
                value = val_tok[1]
        else:
            raise ValueError(f"Unexpected value token: {val_tok}")

        return Predicate(field_name, op_tok[1], value)


def parse_expr(expr_str: str) -> Expr:
    tokens = _tokenize(expr_str)
    parser = _Parser(tokens)
    result = parser.parse_expr()
    if parser.pos != len(tokens):
        raise ValueError(f"Unexpected tokens after position {parser.pos}")
    return result
```

- [ ] **Step 4: Implement query.py — DNF converter**

Append to `query.py`:

```python
Conjunction = list[LiteralPred]
DNF = list[Conjunction]


def to_dnf(expr: Expr) -> DNF:
    return _to_dnf(expr)


def _to_dnf(expr: Expr) -> DNF:
    if isinstance(expr, Predicate):
        return [[LiteralPred(expr, negated=False)]]
    elif isinstance(expr, Not):
        return _negate_dnf(expr.child)
    elif isinstance(expr, And):
        result: DNF = [[]]
        for child in expr.children:
            child_dnf = _to_dnf(child)
            new_result: DNF = []
            for existing_conj in result:
                for child_conj in child_dnf:
                    new_result.append(existing_conj + child_conj)
            result = new_result
            if len(result) > MAX_CONJ:
                raise ValueError(
                    f"DNF exceeds {MAX_CONJ} conjunctions. Simplify the query."
                )
        return result
    elif isinstance(expr, Or):
        result = []
        for child in expr.children:
            result.extend(_to_dnf(child))
            if len(result) > MAX_CONJ:
                raise ValueError(
                    f"DNF exceeds {MAX_CONJ} conjunctions. Simplify the query."
                )
        return result
    else:
        raise TypeError(f"Unknown expr type: {type(expr)}")


def _negate_dnf(expr: Expr) -> DNF:
    if isinstance(expr, Predicate):
        return [[LiteralPred(expr, negated=True)]]
    elif isinstance(expr, Not):
        return _to_dnf(expr.child)
    elif isinstance(expr, And):
        # De Morgan: NOT (A AND B) = NOT A OR NOT B
        negated_or = Or([Not(c) for c in expr.children])
        return _to_dnf(negated_or)
    elif isinstance(expr, Or):
        # De Morgan: NOT (A OR B) = NOT A AND NOT B
        negated_and = And([Not(c) for c in expr.children])
        return _to_dnf(negated_and)
    else:
        raise TypeError(f"Unknown expr type: {type(expr)}")
```

- [ ] **Step 5: Implement query.py — tensor encoder**

Append to `query.py`:

```python
def encode_query(expr_str: str, meta: dict) -> dict[str, torch.Tensor]:
    expr = parse_expr(expr_str)
    dnf = to_dnf(expr)
    return _encode_dnf(dnf, meta)


def _encode_dnf(dnf: DNF, meta: dict) -> dict[str, torch.Tensor]:
    schema = meta["schema"]
    discrete_fields = set(schema["discrete"])
    numeric_fields_list = schema["numeric"]
    text_fields = set(schema["text"])
    numeric_field_idx = {name: i for i, name in enumerate(numeric_fields_list)}

    OP_MAP = {"==": 0, "<": 1, ">": 2, "<=": 3, ">=": 4}

    bitmap_indices = torch.zeros(MAX_BP, dtype=torch.int64)
    bitmap_valid = torch.zeros(MAX_BP, dtype=torch.bool)
    numeric_fields_t = torch.zeros(MAX_NP, dtype=torch.int64)
    numeric_ops = torch.zeros(MAX_NP, dtype=torch.int64)
    numeric_values = torch.zeros(MAX_NP, dtype=torch.float32)
    numeric_valid = torch.zeros(MAX_NP, dtype=torch.bool)
    negation_mask = torch.zeros(P_TOTAL, dtype=torch.bool)
    conj_matrix = torch.zeros(MAX_CONJ, P_TOTAL, dtype=torch.bool)
    conj_valid = torch.zeros(MAX_CONJ, dtype=torch.bool)

    # Collect unique predicates across all conjunctions
    all_bitmap_preds: list[tuple[str, object, bool]] = []  # (field, value, negated)
    all_numeric_preds: list[tuple[str, str, float, bool]] = []  # (field, op, value, negated)

    pred_key_to_bp_idx: dict[tuple, int] = {}
    pred_key_to_np_idx: dict[tuple, int] = {}
    bp_count = 0
    np_count = 0

    for conj in dnf:
        for lit in conj:
            pred = lit.pred
            neg = lit.negated

            if pred.op == "!=" or (pred.op == "==" and neg):
                actual_neg = not neg if pred.op == "!=" else neg
                pred = Predicate(pred.field, "==", pred.value)
                neg = actual_neg if pred.op != "!=" else not neg

            if pred.op == "!=" :
                pred = Predicate(pred.field, "==", pred.value)
                neg = not neg

            if pred.field in discrete_fields:
                val_id = meta["discrete_dicts"][pred.field].get(str(pred.value))
                if val_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"d:{pred.field}"].get(str(val_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key not in pred_key_to_bp_idx:
                    if bp_count >= MAX_BP:
                        raise ValueError(f"Exceeds MAX_BP={MAX_BP}")
                    idx = bp_count
                    bitmap_indices[idx] = int(bitmap_key)
                    bitmap_valid[idx] = True
                    negation_mask[idx] = neg
                    pred_key_to_bp_idx[key] = idx
                    bp_count += 1

            elif pred.field in text_fields and pred.op == "contains":
                term_id = meta["text_dicts"].get(pred.field, {}).get(str(pred.value))
                if term_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"t:{pred.field}"].get(str(term_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key not in pred_key_to_bp_idx:
                    if bp_count >= MAX_BP:
                        raise ValueError(f"Exceeds MAX_BP={MAX_BP}")
                    idx = bp_count
                    bitmap_indices[idx] = int(bitmap_key)
                    bitmap_valid[idx] = True
                    negation_mask[idx] = neg
                    pred_key_to_bp_idx[key] = idx
                    bp_count += 1

            elif pred.field in numeric_field_idx and pred.op in OP_MAP:
                key = ("numeric", pred.field, pred.op, float(pred.value), neg)
                if key not in pred_key_to_np_idx:
                    if np_count >= MAX_NP:
                        raise ValueError(f"Exceeds MAX_NP={MAX_NP}")
                    idx = np_count
                    numeric_fields_t[idx] = numeric_field_idx[pred.field]
                    numeric_ops[idx] = OP_MAP[pred.op]
                    numeric_values[idx] = float(pred.value)
                    numeric_valid[idx] = True
                    negation_mask[MAX_BP + idx] = neg
                    pred_key_to_np_idx[key] = idx
                    np_count += 1
            else:
                raise ValueError(f"Unknown field or op: {pred.field} {pred.op}")

    # Build conjunction matrix
    for ci, conj in enumerate(dnf):
        if ci >= MAX_CONJ:
            raise ValueError(f"Exceeds MAX_CONJ={MAX_CONJ}")
        conj_valid[ci] = True
        for lit in conj:
            pred = lit.pred
            neg = lit.negated

            if pred.op == "!=":
                pred = Predicate(pred.field, "==", pred.value)
                neg = not neg

            if pred.field in discrete_fields:
                val_id = meta["discrete_dicts"][pred.field].get(str(pred.value))
                if val_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"d:{pred.field}"].get(str(val_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key in pred_key_to_bp_idx:
                    conj_matrix[ci, pred_key_to_bp_idx[key]] = True

            elif pred.field in text_fields and pred.op == "contains":
                term_id = meta["text_dicts"].get(pred.field, {}).get(str(pred.value))
                if term_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"t:{pred.field}"].get(str(term_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key in pred_key_to_bp_idx:
                    conj_matrix[ci, pred_key_to_bp_idx[key]] = True

            elif pred.field in numeric_field_idx and pred.op in OP_MAP:
                key = ("numeric", pred.field, pred.op, float(pred.value), neg)
                if key in pred_key_to_np_idx:
                    conj_matrix[ci, MAX_BP + pred_key_to_np_idx[key]] = True

    return {
        "bitmap_indices": bitmap_indices,
        "bitmap_valid": bitmap_valid,
        "numeric_fields": numeric_fields_t,
        "numeric_ops": numeric_ops,
        "numeric_values": numeric_values,
        "numeric_valid": numeric_valid,
        "negation_mask": negation_mask,
        "conj_matrix": conj_matrix,
        "conj_valid": conj_valid,
    }
```

- [ ] **Step 6: Run tests**

Run: `cd offline && pytest tests/test_query.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add offline/torch_recall/query.py offline/tests/test_query.py
git commit -m "feat: query parser, DNF converter, and tensor encoder"
```

---

## Task 5: Exporter

**Files:**
- Create: `offline/torch_recall/exporter.py`
- Modify: `offline/torch_recall/__init__.py`

- [ ] **Step 1: Write failing test**

Add to `offline/tests/test_e2e.py`:

```python
import json
import tempfile
from pathlib import Path

import torch
from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query
from torch_recall.exporter import export_model


def test_export_and_reload():
    schema = Schema(discrete_fields=["city"], numeric_fields=["price"])
    items = [
        {"city": "北京", "price": 10.0},
        {"city": "上海", "price": 20.0},
        {"city": "北京", "price": 30.0},
        {"city": "上海", "price": 40.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "index_meta.json"
        builder.save_meta(meta, meta_path)

        pt2_path = Path(tmpdir) / "model.pt2"
        export_model(model, meta, str(pt2_path))

        assert pt2_path.exists()


def test_e2e_query():
    schema = Schema(
        discrete_fields=["city", "gender"],
        numeric_fields=["price"],
        text_fields=["title"],
    )
    items = [
        {"city": "北京", "gender": "男", "price": 10.0, "title": "游戏 攻略"},
        {"city": "上海", "gender": "女", "price": 20.0, "title": "美食 推荐"},
        {"city": "北京", "gender": "女", "price": 30.0, "title": "游戏 推荐"},
        {"city": "上海", "gender": "男", "price": 40.0, "title": "美食 攻略"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()

    # Query: city == "北京" → items 0, 2
    tensors = encode_query('city == "北京"', meta)
    with torch.no_grad():
        result = model(**tensors)

    # Decode result bitmap
    bits = []
    for word_idx in range(result.shape[0]):
        val = result[word_idx].item()
        for bit in range(64):
            if val & (1 << bit):
                bits.append(word_idx * 64 + bit)
    matching_ids = [b for b in bits if b < len(items)]
    assert sorted(matching_ids) == [0, 2]


def test_e2e_complex_query():
    schema = Schema(
        discrete_fields=["city", "gender"],
        numeric_fields=["price"],
    )
    items = [
        {"city": "北京", "gender": "男", "price": 10.0},
        {"city": "上海", "gender": "女", "price": 20.0},
        {"city": "北京", "gender": "女", "price": 30.0},
        {"city": "上海", "gender": "男", "price": 40.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()

    # Query: (city == "北京" OR city == "上海") AND price < 25.0
    # DNF: (city=北京 AND price<25) OR (city=上海 AND price<25)
    # city=北京: {0,2}, city=上海: {1,3}, price<25: {0,1}
    # Result: ({0,2} ∩ {0,1}) ∪ ({1,3} ∩ {0,1}) = {0} ∪ {1} = {0,1}
    tensors = encode_query('(city == "北京" OR city == "上海") AND price < 25.0', meta)
    with torch.no_grad():
        result = model(**tensors)

    bits = []
    for word_idx in range(result.shape[0]):
        val = result[word_idx].item()
        for bit in range(64):
            if val & (1 << bit):
                bits.append(word_idx * 64 + bit)
    matching_ids = [b for b in bits if b < len(items)]
    assert sorted(matching_ids) == [0, 1]
```

Run: `cd offline && pytest tests/test_e2e.py -v`
Expected: FAIL (exporter doesn't exist yet, but e2e query tests may pass after model+query are done)

- [ ] **Step 2: Implement exporter.py**

```python
import torch
import torch._inductor

from torch_recall.schema import MAX_BP, MAX_NP, MAX_CONJ, P_TOTAL
from torch_recall.model import InvertedIndexModel


def create_example_inputs(model: InvertedIndexModel, device: str = "cpu") -> tuple:
    return (
        torch.zeros(MAX_BP, dtype=torch.int64, device=device),
        torch.zeros(MAX_BP, dtype=torch.bool, device=device),
        torch.zeros(MAX_NP, dtype=torch.int64, device=device),
        torch.zeros(MAX_NP, dtype=torch.int64, device=device),
        torch.zeros(MAX_NP, dtype=torch.float32, device=device),
        torch.zeros(MAX_NP, dtype=torch.bool, device=device),
        torch.zeros(P_TOTAL, dtype=torch.bool, device=device),
        torch.zeros(MAX_CONJ, P_TOTAL, dtype=torch.bool, device=device),
        torch.zeros(MAX_CONJ, dtype=torch.bool, device=device),
    )


def export_model(model: InvertedIndexModel, meta: dict, output_path: str) -> str:
    model.eval()
    device = next(model.buffers()).device

    example_inputs = create_example_inputs(model, device=str(device))

    with torch.no_grad():
        exported = torch.export.export(model, example_inputs)

    path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
    )
    return path
```

- [ ] **Step 3: Update __init__.py**

```python
from torch_recall.schema import Schema, FieldType, NumericOp, MAX_BP, MAX_NP, MAX_CONJ
from torch_recall.model import InvertedIndexModel
from torch_recall.builder import IndexBuilder
from torch_recall.query import parse_expr, to_dnf, encode_query
from torch_recall.exporter import export_model
```

- [ ] **Step 4: Run all tests**

Run: `cd offline && pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add offline/torch_recall/exporter.py offline/torch_recall/__init__.py offline/tests/test_e2e.py
git commit -m "feat: torch.export + AOTInductor exporter and e2e tests"
```

---

## Task 6: Benchmark

**Files:**
- Create: `offline/benchmarks/bench_recall.py`

- [ ] **Step 1: Write benchmark script**

```python
"""Benchmark: build index and query 1M items."""
import time
import random
import string

import torch
from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query


def generate_items(n: int, schema: Schema) -> list[dict]:
    cities = [f"city_{i}" for i in range(50)]
    genders = ["男", "女"]
    categories = [f"cat_{i}" for i in range(100)]
    items = []
    for _ in range(n):
        item = {}
        for f in schema.discrete_fields:
            if f == "city":
                item[f] = random.choice(cities)
            elif f == "gender":
                item[f] = random.choice(genders)
            elif f == "category":
                item[f] = random.choice(categories)
            else:
                item[f] = f"val_{random.randint(0, 99)}"
        for f in schema.numeric_fields:
            item[f] = random.uniform(0, 1000)
        for f in schema.text_fields:
            words = [f"term_{random.randint(0, 999)}" for _ in range(5)]
            item[f] = " ".join(words)
        items.append(item)
    return items


def main():
    N = 1_000_000
    schema = Schema(
        discrete_fields=["city", "gender", "category"],
        numeric_fields=["price", "score"],
        text_fields=["title"],
    )

    print(f"Generating {N} items...")
    t0 = time.time()
    items = generate_items(N, schema)
    print(f"  Generated in {time.time() - t0:.1f}s")

    print("Building index...")
    t0 = time.time()
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")
    print(f"  Bitmaps: {model.bitmaps.shape}")
    print(f"  Numeric: {model.numeric_data.shape}")

    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")

    # Warmup
    query_str = 'city == "city_0" AND gender == "男" AND price < 500.0'
    tensors = encode_query(query_str, meta)
    tensors = {k: v.to(device) for k, v in tensors.items()}
    with torch.no_grad():
        _ = model(**tensors)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    num_queries = 100
    times = []
    for _ in range(num_queries):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model(**tensors)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    print(f"\nQuery benchmark ({num_queries} runs):")
    print(f"  Avg: {avg_ms:.3f}ms")
    print(f"  P50: {p50:.3f}ms")
    print(f"  P99: {p99:.3f}ms")

    # Count results
    bits = []
    packed = result.cpu()
    for word_idx in range(packed.shape[0]):
        val = packed[word_idx].item()
        for bit in range(64):
            if val & (1 << bit):
                bits.append(word_idx * 64 + bit)
    matching = [b for b in bits if b < N]
    print(f"  Matching items: {len(matching)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark**

Run: `cd offline && python benchmarks/bench_recall.py`
Expected: Avg latency < 20ms (target: ~0.1ms GPU, ~3ms CPU)

- [ ] **Step 3: Commit**

```bash
git add offline/benchmarks/bench_recall.py
git commit -m "feat: 1M item benchmark script"
```

---

## Task 7: C++ Model Runner + Result Decoder

**Files:**
- Create: `online/CMakeLists.txt`
- Create: `online/include/torch_recall/common.h`
- Create: `online/include/torch_recall/index_meta.h`
- Create: `online/include/torch_recall/model_runner.h`
- Create: `online/include/torch_recall/result_decoder.h`
- Create: `online/src/index_meta.cpp`
- Create: `online/src/model_runner.cpp`
- Create: `online/src/result_decoder.cpp`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_recall_online CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS include/*.h)

add_executable(torch_recall_cli ${SOURCES})
target_include_directories(torch_recall_cli PRIVATE include)
target_link_libraries(torch_recall_cli "${TORCH_LIBRARIES}")

set_property(TARGET torch_recall_cli PROPERTY CXX_STANDARD 17)

if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_recall_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:torch_recall_cli>)
endif()
```

- [ ] **Step 2: Create common.h**

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace torch_recall {

constexpr int MAX_BP = 32;
constexpr int MAX_NP = 16;
constexpr int MAX_CONJ = 16;
constexpr int P_TOTAL = MAX_BP + MAX_NP;

enum class NumericOp : int64_t {
    EQ = 0, LT = 1, GT = 2, LE = 3, GE = 4
};

}  // namespace torch_recall
```

- [ ] **Step 3: Create index_meta.h and index_meta.cpp**

`online/include/torch_recall/index_meta.h`:
```cpp
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch_recall {

struct IndexMetadata {
    std::vector<std::string> discrete_fields;
    std::vector<std::string> numeric_fields;
    std::vector<std::string> text_fields;

    std::unordered_set<std::string> discrete_field_set;
    std::unordered_set<std::string> text_field_set;
    std::unordered_map<std::string, int> numeric_field_index;

    // field_name -> {value_string -> value_id}
    std::unordered_map<std::string, std::unordered_map<std::string, int>> discrete_dicts;
    // field_name -> {term_string -> term_id}
    std::unordered_map<std::string, std::unordered_map<std::string, int>> text_dicts;
    // "d:field" or "t:field" -> {value_id_str -> bitmap_global_idx}
    std::unordered_map<std::string, std::unordered_map<std::string, int>> bitmap_lookup;

    int64_t num_items = 0;
    int64_t bitmap_len = 0;
    int64_t num_bitmaps = 0;

    static IndexMetadata load(const std::string& json_path);
};

}  // namespace torch_recall
```

`online/src/index_meta.cpp`:
```cpp
#include "torch_recall/index_meta.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace torch_recall {

IndexMetadata IndexMetadata::load(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + json_path);
    }
    nlohmann::json j;
    f >> j;

    IndexMetadata meta;

    for (auto& v : j["schema"]["discrete"]) meta.discrete_fields.push_back(v.get<std::string>());
    for (auto& v : j["schema"]["numeric"]) meta.numeric_fields.push_back(v.get<std::string>());
    for (auto& v : j["schema"]["text"]) meta.text_fields.push_back(v.get<std::string>());

    for (auto& f : meta.discrete_fields) meta.discrete_field_set.insert(f);
    for (auto& f : meta.text_fields) meta.text_field_set.insert(f);
    for (int i = 0; i < (int)meta.numeric_fields.size(); i++) {
        meta.numeric_field_index[meta.numeric_fields[i]] = i;
    }

    for (auto& [field, dict] : j["discrete_dicts"].items()) {
        for (auto& [val, id] : dict.items()) {
            meta.discrete_dicts[field][val] = id.get<int>();
        }
    }
    for (auto& [field, dict] : j["text_dicts"].items()) {
        for (auto& [term, id] : dict.items()) {
            meta.text_dicts[field][term] = id.get<int>();
        }
    }
    for (auto& [key, dict] : j["bitmap_lookup"].items()) {
        for (auto& [id_str, gidx] : dict.items()) {
            meta.bitmap_lookup[key][id_str] = gidx.get<int>();
        }
    }

    meta.num_items = j["num_items"].get<int64_t>();
    meta.bitmap_len = j["bitmap_len"].get<int64_t>();
    meta.num_bitmaps = j["num_bitmaps"].get<int64_t>();

    return meta;
}

}  // namespace torch_recall
```

Note: depends on `nlohmann/json.hpp`. Add to CMakeLists.txt:

```cmake
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(torch_recall_cli nlohmann_json::nlohmann_json "${TORCH_LIBRARIES}")
```

- [ ] **Step 4: Create model_runner.h and model_runner.cpp**

`online/include/torch_recall/model_runner.h`:
```cpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <torch/torch.h>

namespace torch_recall {

class ModelRunner {
public:
    explicit ModelRunner(const std::string& pt2_path);

    torch::Tensor run(
        torch::Tensor bitmap_indices,
        torch::Tensor bitmap_valid,
        torch::Tensor numeric_fields,
        torch::Tensor numeric_ops,
        torch::Tensor numeric_values,
        torch::Tensor numeric_valid,
        torch::Tensor negation_mask,
        torch::Tensor conj_matrix,
        torch::Tensor conj_valid
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace torch_recall
```

`online/src/model_runner.cpp`:
```cpp
#include "torch_recall/model_runner.h"
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace torch_recall {

struct ModelRunner::Impl {
    torch::inductor::AOTIModelPackageLoader loader;
    explicit Impl(const std::string& path) : loader(path) {}
};

ModelRunner::ModelRunner(const std::string& pt2_path)
    : impl_(std::make_unique<Impl>(pt2_path)) {}

torch::Tensor ModelRunner::run(
    torch::Tensor bitmap_indices,
    torch::Tensor bitmap_valid,
    torch::Tensor numeric_fields,
    torch::Tensor numeric_ops,
    torch::Tensor numeric_values,
    torch::Tensor numeric_valid,
    torch::Tensor negation_mask,
    torch::Tensor conj_matrix,
    torch::Tensor conj_valid
) {
    std::vector<torch::Tensor> inputs = {
        bitmap_indices, bitmap_valid,
        numeric_fields, numeric_ops, numeric_values, numeric_valid,
        negation_mask, conj_matrix, conj_valid
    };
    auto outputs = impl_->loader.run(inputs);
    return outputs[0];
}

}  // namespace torch_recall
```

- [ ] **Step 5: Create result_decoder.h and result_decoder.cpp**

`online/include/torch_recall/result_decoder.h`:
```cpp
#pragma once

#include <cstdint>
#include <vector>
#include <torch/torch.h>

namespace torch_recall {

class ResultDecoder {
public:
    explicit ResultDecoder(int64_t num_items);

    std::vector<int64_t> decode(const torch::Tensor& packed_bitmap) const;

private:
    int64_t num_items_;
};

}  // namespace torch_recall
```

`online/src/result_decoder.cpp`:
```cpp
#include "torch_recall/result_decoder.h"

namespace torch_recall {

ResultDecoder::ResultDecoder(int64_t num_items) : num_items_(num_items) {}

std::vector<int64_t> ResultDecoder::decode(const torch::Tensor& packed_bitmap) const {
    auto accessor = packed_bitmap.accessor<int64_t, 1>();
    int64_t L = accessor.size(0);
    std::vector<int64_t> result;
    result.reserve(1024);

    for (int64_t w = 0; w < L; ++w) {
        int64_t word = accessor[w];
        if (word == 0) continue;
        int64_t base = w * 64;
        while (word != 0) {
            int bit = __builtin_ctzll(static_cast<unsigned long long>(word));
            int64_t item_idx = base + bit;
            if (item_idx < num_items_) {
                result.push_back(item_idx);
            }
            word &= word - 1;  // clear lowest set bit
        }
    }
    return result;
}

}  // namespace torch_recall
```

- [ ] **Step 6: Commit**

```bash
git add online/
git commit -m "feat: C++ model runner, result decoder, and metadata loader"
```

---

## Task 8: C++ Query Parser + DNF Converter + Tensor Encoder

**Files:**
- Create: `online/include/torch_recall/query_parser.h`
- Create: `online/include/torch_recall/dnf_converter.h`
- Create: `online/include/torch_recall/tensor_encoder.h`
- Create: `online/src/query_parser.cpp`
- Create: `online/src/dnf_converter.cpp`
- Create: `online/src/tensor_encoder.cpp`

- [ ] **Step 1: Create query_parser.h and query_parser.cpp**

`online/include/torch_recall/query_parser.h`:
```cpp
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <variant>

namespace torch_recall {

struct Predicate {
    std::string field;
    std::string op;  // ==, !=, <, >, <=, >=, contains
    std::variant<std::string, double> value;
};

struct AndExpr;
struct OrExpr;
struct NotExpr;

using Expr = std::variant<
    Predicate,
    std::shared_ptr<AndExpr>,
    std::shared_ptr<OrExpr>,
    std::shared_ptr<NotExpr>
>;

struct AndExpr { std::vector<Expr> children; };
struct OrExpr  { std::vector<Expr> children; };
struct NotExpr { Expr child; };

Expr parse_expression(const std::string& input);

}  // namespace torch_recall
```

`online/src/query_parser.cpp`:
```cpp
#include "torch_recall/query_parser.h"
#include <regex>
#include <stdexcept>

namespace torch_recall {

enum class TokenType { KW, OP, STR, LPAREN, RPAREN, WORD };

struct Token {
    TokenType type;
    std::string value;
};

static std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;
    std::regex re(
        R"(\s*(?:(AND|OR|NOT|contains)|([<>!=]=?|==)|"([^"]*)"|'([^']*)'|(\()|(\))|([^\s()'"<>!=]+))\s*)"
    );
    auto begin = std::sregex_iterator(input.begin(), input.end(), re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        auto& m = *it;
        if (m[1].matched) tokens.push_back({TokenType::KW, m[1].str()});
        else if (m[2].matched) tokens.push_back({TokenType::OP, m[2].str()});
        else if (m[3].matched) tokens.push_back({TokenType::STR, m[3].str()});
        else if (m[4].matched) tokens.push_back({TokenType::STR, m[4].str()});
        else if (m[5].matched) tokens.push_back({TokenType::LPAREN, "("});
        else if (m[6].matched) tokens.push_back({TokenType::RPAREN, ")"});
        else if (m[7].matched) tokens.push_back({TokenType::WORD, m[7].str()});
    }
    return tokens;
}

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    Expr parse() {
        auto result = parse_or();
        if (pos_ != tokens_.size()) {
            throw std::runtime_error("Unexpected tokens at position " + std::to_string(pos_));
        }
        return result;
    }

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;

    const Token* peek() const {
        return pos_ < tokens_.size() ? &tokens_[pos_] : nullptr;
    }

    Token consume() { return tokens_[pos_++]; }

    bool match_kw(const std::string& kw) const {
        auto* t = peek();
        return t && t->type == TokenType::KW && t->value == kw;
    }

    Expr parse_or() {
        auto left = parse_and();
        std::vector<Expr> children = {std::move(left)};
        while (match_kw("OR")) {
            consume();
            children.push_back(parse_and());
        }
        if (children.size() == 1) return std::move(children[0]);
        return std::make_shared<OrExpr>(OrExpr{std::move(children)});
    }

    Expr parse_and() {
        auto left = parse_not();
        std::vector<Expr> children = {std::move(left)};
        while (match_kw("AND")) {
            consume();
            children.push_back(parse_not());
        }
        if (children.size() == 1) return std::move(children[0]);
        return std::make_shared<AndExpr>(AndExpr{std::move(children)});
    }

    Expr parse_not() {
        if (match_kw("NOT")) {
            consume();
            return std::make_shared<NotExpr>(NotExpr{parse_not()});
        }
        return parse_primary();
    }

    Expr parse_primary() {
        auto* t = peek();
        if (t && t->type == TokenType::LPAREN) {
            consume();
            auto expr = parse_or();
            if (!peek() || peek()->type != TokenType::RPAREN) {
                throw std::runtime_error("Expected ')'");
            }
            consume();
            return expr;
        }
        return parse_predicate();
    }

    Expr parse_predicate() {
        auto field_tok = consume();
        if (field_tok.type != TokenType::WORD) {
            throw std::runtime_error("Expected field name, got: " + field_tok.value);
        }

        if (match_kw("contains")) {
            consume();
            auto val = consume();
            return Predicate{field_tok.value, "contains", val.value};
        }

        auto op_tok = consume();
        auto val_tok = consume();

        std::variant<std::string, double> value;
        if (val_tok.type == TokenType::STR) {
            value = val_tok.value;
        } else {
            try {
                value = std::stod(val_tok.value);
            } catch (...) {
                value = val_tok.value;
            }
        }
        return Predicate{field_tok.value, op_tok.value, value};
    }
};

Expr parse_expression(const std::string& input) {
    auto tokens = tokenize(input);
    Parser parser(std::move(tokens));
    return parser.parse();
}

}  // namespace torch_recall
```

- [ ] **Step 2: Create dnf_converter.h and dnf_converter.cpp**

`online/include/torch_recall/dnf_converter.h`:
```cpp
#pragma once

#include "torch_recall/query_parser.h"
#include "torch_recall/common.h"
#include <vector>

namespace torch_recall {

struct LiteralPred {
    Predicate pred;
    bool negated = false;
};

using Conjunction = std::vector<LiteralPred>;
using DNF = std::vector<Conjunction>;

DNF to_dnf(const Expr& expr);

}  // namespace torch_recall
```

`online/src/dnf_converter.cpp`:
```cpp
#include "torch_recall/dnf_converter.h"
#include <stdexcept>

namespace torch_recall {

static DNF convert(const Expr& expr);
static DNF negate(const Expr& expr);

static DNF convert(const Expr& expr) {
    if (auto* p = std::get_if<Predicate>(&expr)) {
        return {{LiteralPred{*p, false}}};
    }
    if (auto* n = std::get_if<std::shared_ptr<NotExpr>>(&expr)) {
        return negate((*n)->child);
    }
    if (auto* a = std::get_if<std::shared_ptr<AndExpr>>(&expr)) {
        DNF result = {{}};
        for (auto& child : (*a)->children) {
            auto child_dnf = convert(child);
            DNF new_result;
            for (auto& existing : result) {
                for (auto& child_conj : child_dnf) {
                    auto merged = existing;
                    merged.insert(merged.end(), child_conj.begin(), child_conj.end());
                    new_result.push_back(std::move(merged));
                }
            }
            result = std::move(new_result);
            if ((int)result.size() > MAX_CONJ) {
                throw std::runtime_error("DNF exceeds MAX_CONJ. Simplify the query.");
            }
        }
        return result;
    }
    if (auto* o = std::get_if<std::shared_ptr<OrExpr>>(&expr)) {
        DNF result;
        for (auto& child : (*o)->children) {
            auto child_dnf = convert(child);
            result.insert(result.end(), child_dnf.begin(), child_dnf.end());
            if ((int)result.size() > MAX_CONJ) {
                throw std::runtime_error("DNF exceeds MAX_CONJ. Simplify the query.");
            }
        }
        return result;
    }
    throw std::runtime_error("Unknown expression type");
}

static DNF negate(const Expr& expr) {
    if (auto* p = std::get_if<Predicate>(&expr)) {
        return {{LiteralPred{*p, true}}};
    }
    if (auto* n = std::get_if<std::shared_ptr<NotExpr>>(&expr)) {
        return convert((*n)->child);
    }
    if (auto* a = std::get_if<std::shared_ptr<AndExpr>>(&expr)) {
        // De Morgan: NOT (A AND B) = NOT A OR NOT B
        std::vector<Expr> negated_children;
        for (auto& c : (*a)->children) {
            negated_children.push_back(std::make_shared<NotExpr>(NotExpr{c}));
        }
        auto or_expr = std::make_shared<OrExpr>(OrExpr{std::move(negated_children)});
        return convert(or_expr);
    }
    if (auto* o = std::get_if<std::shared_ptr<OrExpr>>(&expr)) {
        // De Morgan: NOT (A OR B) = NOT A AND NOT B
        std::vector<Expr> negated_children;
        for (auto& c : (*o)->children) {
            negated_children.push_back(std::make_shared<NotExpr>(NotExpr{c}));
        }
        auto and_expr = std::make_shared<AndExpr>(AndExpr{std::move(negated_children)});
        return convert(and_expr);
    }
    throw std::runtime_error("Unknown expression type in negate");
}

DNF to_dnf(const Expr& expr) {
    return convert(expr);
}

}  // namespace torch_recall
```

- [ ] **Step 3: Create tensor_encoder.h and tensor_encoder.cpp**

`online/include/torch_recall/tensor_encoder.h`:
```cpp
#pragma once

#include "torch_recall/dnf_converter.h"
#include "torch_recall/index_meta.h"
#include <torch/torch.h>
#include <unordered_map>

namespace torch_recall {

struct QueryTensors {
    torch::Tensor bitmap_indices;
    torch::Tensor bitmap_valid;
    torch::Tensor numeric_fields;
    torch::Tensor numeric_ops;
    torch::Tensor numeric_values;
    torch::Tensor numeric_valid;
    torch::Tensor negation_mask;
    torch::Tensor conj_matrix;
    torch::Tensor conj_valid;
};

QueryTensors encode_dnf(const DNF& dnf, const IndexMetadata& meta);

}  // namespace torch_recall
```

`online/src/tensor_encoder.cpp`:
```cpp
#include "torch_recall/tensor_encoder.h"
#include <stdexcept>

namespace torch_recall {

static const std::unordered_map<std::string, int64_t> OP_MAP = {
    {"==", 0}, {"<", 1}, {">", 2}, {"<=", 3}, {">=", 4}
};

QueryTensors encode_dnf(const DNF& dnf, const IndexMetadata& meta) {
    QueryTensors qt;
    qt.bitmap_indices = torch::zeros({MAX_BP}, torch::kInt64);
    qt.bitmap_valid = torch::zeros({MAX_BP}, torch::kBool);
    qt.numeric_fields = torch::zeros({MAX_NP}, torch::kInt64);
    qt.numeric_ops = torch::zeros({MAX_NP}, torch::kInt64);
    qt.numeric_values = torch::zeros({MAX_NP}, torch::kFloat32);
    qt.numeric_valid = torch::zeros({MAX_NP}, torch::kBool);
    qt.negation_mask = torch::zeros({P_TOTAL}, torch::kBool);
    qt.conj_matrix = torch::zeros({MAX_CONJ, P_TOTAL}, torch::kBool);
    qt.conj_valid = torch::zeros({MAX_CONJ}, torch::kBool);

    struct PredKey {
        std::string type;
        int64_t id;
        bool neg;
        bool operator==(const PredKey& o) const {
            return type == o.type && id == o.id && neg == o.neg;
        }
    };
    struct PredKeyHash {
        size_t operator()(const PredKey& k) const {
            return std::hash<std::string>()(k.type) ^ (std::hash<int64_t>()(k.id) << 1) ^ (std::hash<bool>()(k.neg) << 2);
        }
    };

    std::unordered_map<PredKey, int, PredKeyHash> bp_map, np_map;
    int bp_count = 0, np_count = 0;

    auto resolve_bitmap = [&](const Predicate& pred, bool neg) -> int {
        std::string prefix;
        std::string val_str;

        if (meta.discrete_field_set.count(pred.field)) {
            prefix = "d:" + pred.field;
            auto val = std::get<std::string>(pred.value);
            auto dit = meta.discrete_dicts.find(pred.field);
            if (dit == meta.discrete_dicts.end()) return -1;
            auto vit = dit->second.find(val);
            if (vit == dit->second.end()) return -1;
            val_str = std::to_string(vit->second);
        } else if (meta.text_field_set.count(pred.field) && pred.op == "contains") {
            prefix = "t:" + pred.field;
            auto term = std::get<std::string>(pred.value);
            auto tit = meta.text_dicts.find(pred.field);
            if (tit == meta.text_dicts.end()) return -1;
            auto vit = tit->second.find(term);
            if (vit == tit->second.end()) return -1;
            val_str = std::to_string(vit->second);
        } else {
            return -1;
        }

        auto bit = meta.bitmap_lookup.find(prefix);
        if (bit == meta.bitmap_lookup.end()) return -1;
        auto git = bit->second.find(val_str);
        if (git == bit->second.end()) return -1;
        int64_t global_idx = git->second;

        PredKey key{"bitmap", global_idx, neg};
        if (bp_map.count(key)) return bp_map[key];
        if (bp_count >= MAX_BP) throw std::runtime_error("Exceeds MAX_BP");

        int idx = bp_count++;
        qt.bitmap_indices[idx] = global_idx;
        qt.bitmap_valid[idx] = true;
        qt.negation_mask[idx] = neg;
        bp_map[key] = idx;
        return idx;
    };

    auto resolve_numeric = [&](const Predicate& pred, bool neg) -> int {
        auto nit = meta.numeric_field_index.find(pred.field);
        if (nit == meta.numeric_field_index.end()) return -1;
        auto oit = OP_MAP.find(pred.op);
        if (oit == OP_MAP.end()) return -1;
        double val = std::get<double>(pred.value);

        PredKey key{"numeric", static_cast<int64_t>(nit->second * 100 + oit->second), neg};
        if (np_map.count(key)) return np_map[key];
        if (np_count >= MAX_NP) throw std::runtime_error("Exceeds MAX_NP");

        int idx = np_count++;
        qt.numeric_fields[idx] = nit->second;
        qt.numeric_ops[idx] = oit->second;
        qt.numeric_values[idx] = static_cast<float>(val);
        qt.numeric_valid[idx] = true;
        qt.negation_mask[MAX_BP + idx] = neg;
        np_map[key] = idx;
        return idx;
    };

    for (int ci = 0; ci < (int)dnf.size() && ci < MAX_CONJ; ++ci) {
        qt.conj_valid[ci] = true;
        for (auto& lit : dnf[ci]) {
            auto pred = lit.pred;
            bool neg = lit.negated;

            if (pred.op == "!=") {
                pred.op = "==";
                neg = !neg;
            }

            if (meta.discrete_field_set.count(pred.field) ||
                (meta.text_field_set.count(pred.field) && pred.op == "contains")) {
                int idx = resolve_bitmap(pred, neg);
                if (idx >= 0) qt.conj_matrix[ci][idx] = true;
            } else if (meta.numeric_field_index.count(pred.field)) {
                int idx = resolve_numeric(pred, neg);
                if (idx >= 0) qt.conj_matrix[ci][MAX_BP + idx] = true;
            }
        }
    }

    return qt;
}

}  // namespace torch_recall
```

- [ ] **Step 4: Commit**

```bash
git add online/include/ online/src/query_parser.cpp online/src/dnf_converter.cpp online/src/tensor_encoder.cpp
git commit -m "feat: C++ query parser, DNF converter, and tensor encoder"
```

---

## Task 9: C++ Main CLI + Integration

**Files:**
- Create: `online/src/main.cpp`

- [ ] **Step 1: Implement main.cpp**

```cpp
#include <iostream>
#include <chrono>
#include <torch/torch.h>

#include "torch_recall/index_meta.h"
#include "torch_recall/model_runner.h"
#include "torch_recall/query_parser.h"
#include "torch_recall/dnf_converter.h"
#include "torch_recall/tensor_encoder.h"
#include "torch_recall/result_decoder.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.pt2> <index_meta.json> <query>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string meta_path = argv[2];
    std::string query_str = argv[3];

    try {
        c10::InferenceMode guard;

        std::cout << "Loading metadata..." << std::endl;
        auto meta = torch_recall::IndexMetadata::load(meta_path);

        std::cout << "Loading model..." << std::endl;
        torch_recall::ModelRunner runner(model_path);

        std::cout << "Parsing query: " << query_str << std::endl;
        auto expr = torch_recall::parse_expression(query_str);
        auto dnf = torch_recall::to_dnf(expr);
        std::cout << "  DNF conjunctions: " << dnf.size() << std::endl;

        auto qt = torch_recall::encode_dnf(dnf, meta);

        std::cout << "Running query..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto result = runner.run(
            qt.bitmap_indices, qt.bitmap_valid,
            qt.numeric_fields, qt.numeric_ops,
            qt.numeric_values, qt.numeric_valid,
            qt.negation_mask, qt.conj_matrix, qt.conj_valid
        );

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        torch_recall::ResultDecoder decoder(meta.num_items);
        auto ids = decoder.decode(result);

        std::cout << "Results:" << std::endl;
        std::cout << "  Matching items: " << ids.size() << std::endl;
        std::cout << "  Query time: " << ms << " ms" << std::endl;

        if (ids.size() <= 20) {
            std::cout << "  IDs:";
            for (auto id : ids) std::cout << " " << id;
            std::cout << std::endl;
        } else {
            std::cout << "  First 20 IDs:";
            for (int i = 0; i < 20; i++) std::cout << " " << ids[i];
            std::cout << " ..." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

- [ ] **Step 2: Update CMakeLists.txt with full dependencies**

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_recall_online CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(torch_recall_cli ${SOURCES})
target_include_directories(torch_recall_cli PRIVATE include)
target_link_libraries(torch_recall_cli
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)
```

- [ ] **Step 3: Build and verify**

```bash
cd online
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . --config Release
```

Expected: builds successfully

- [ ] **Step 4: Commit**

```bash
git add online/src/main.cpp online/CMakeLists.txt
git commit -m "feat: C++ CLI for query inference"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# torch-recall

High-performance bitmap inverted index built with PyTorch. Exports to `.pt2` via AOTInductor for C++ inference.

## Features

- Packed int64 bitmap inverted index for discrete and text fields
- Columnar storage with vectorized comparison for numeric fields
- Full boolean expressions (AND / OR / NOT, arbitrary nesting) via DNF conversion
- 1M items, < 20ms query latency (GPU ~0.1ms, CPU ~3ms)
- PyTorch model export to `.pt2` via `torch.export` + AOTInductor
- C++ inference with `AOTIModelPackageLoader`

## Quick Start

### Build Index (Python)

```bash
cd offline
pip install -e ".[dev]"
python -c "
from torch_recall import Schema, IndexBuilder, export_model

schema = Schema(
    discrete_fields=['city', 'gender'],
    numeric_fields=['price'],
    text_fields=['title'],
)

items = [
    {'city': '北京', 'gender': '男', 'price': 10.0, 'title': '游戏 攻略'},
    {'city': '上海', 'gender': '女', 'price': 20.0, 'title': '美食 推荐'},
]

builder = IndexBuilder(schema)
model, meta = builder.build(items)
builder.save_meta(meta, 'index_meta.json')
export_model(model, meta, 'model.pt2')
"
```

### Query (C++)

```bash
cd online
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . --config Release
./torch_recall_cli model.pt2 index_meta.json 'city == "北京" AND price < 15.0'
```

### Query (Python)

```python
from torch_recall import Schema, IndexBuilder, encode_query
import torch

builder = IndexBuilder(schema)
model, meta = builder.build(items)
model.eval()

tensors = encode_query('city == "北京" AND price < 15.0', meta)
with torch.no_grad():
    result = model(**tensors)
```

## Query Syntax

```
city == "北京"
price < 100.0
title contains "游戏"
city == "北京" AND gender == "男"
(city == "北京" OR city == "上海") AND price < 100.0
NOT category == "体育"
```

Operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `contains`
Connectors: `AND`, `OR`, `NOT`, `(`, `)`

## Benchmark

```bash
cd offline
python benchmarks/bench_recall.py
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

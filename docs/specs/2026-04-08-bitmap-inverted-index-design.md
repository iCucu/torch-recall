# Bitmap Inverted Index — Design Spec

## Goal

Build a high-performance inverted index using PyTorch tensor operations, export as `.pt2` via AOTInductor, and provide C++ inference code. Target: 1M items, < 20ms latency, GPU + CPU.

## Requirements

| # | Requirement | Detail |
|---|-------------|--------|
| R1 | Query types | Discrete equality, numeric range comparison, text term matching |
| R2 | Boolean logic | Full boolean expressions: arbitrary AND / OR / NOT nesting |
| R3 | Scale | 1M items, ~50 discrete fields (cardinality < 1000), ~10 numeric fields, ~10K text terms |
| R4 | Latency | < 20ms (GPU ~0.1ms, CPU ~3ms expected) |
| R5 | Artifact | `.pt2` file via `torch._inductor.aoti_compile_and_package` |
| R6 | C++ inference | Load `.pt2` with `AOTIModelPackageLoader`, run queries |
| R7 | Platforms | CUDA GPU + CPU |

## Architecture

```
offline (Python, 离线建索引)                 online (C++, 在线推理)
┌──────────────────────────────┐            ┌──────────────────────────────┐
│ items (dict/JSON)            │            │ query string                 │
│       ↓                      │            │       ↓                      │
│ IndexBuilder                 │            │ QueryParser                  │
│  - encode discrete → int     │            │  - tokenize expression       │
│  - tokenize text → term IDs  │            │       ↓                      │
│  - build packed bitmaps      │            │ DNFConverter                 │
│  - stack numeric columns     │            │  - expr tree → DNF form      │
│       ↓                      │            │       ↓                      │
│ InvertedIndexModel(Module)   │   .pt2     │ TensorEncoder                │
│  - bitmaps buffer [E, L]     │ ────────>  │  - DNF → padded tensors      │
│  - numeric_data buffer       │            │       ↓                      │
│  - forward() = eval + DNF    │            │ AOTIModelPackageLoader       │
│       ↓                      │            │  - loader.run(inputs)        │
│ torch.export.export          │            │       ↓                      │
│       ↓                      │            │ ResultDecoder                │
│ aoti_compile_and_package     │            │  - unpack bitmap → item IDs  │
│       ↓                      │            └──────────────────────────────┘
│ model.pt2 + index_meta.json  │
└──────────────────────────────┘
```

## Data Model

### Item Schema

User defines fields by type:

```python
schema = {
    "discrete": ["city", "gender", "category", ...],   # up to ~50 fields
    "numeric": ["price", "age", "score", ...],          # up to ~10 fields
    "text": ["title", "description", ...]               # text fields, tokenized
}
```

### Internal Encoding

- **Discrete values**: string → int ID via per-field dictionary. E.g., `city="北京"` → `0`, `city="上海"` → `1`.
- **Text terms**: tokenized words → global term ID dictionary. E.g., `"游戏"` → `42`.
- **Numeric values**: stored as float32 directly.

### Bitmap Table

For each unique `(field, value)` pair in discrete fields and each unique `(field, term)` in text fields, a packed bitmap is precomputed:

- Bitmap length `L = ceil(N / 64)` int64 values.
- Bit `j` of word `j // 64` is set iff item `j` matches.
- For 1M items: `L = 15625`, each bitmap = ~122 KB.

All bitmaps are stacked into a single `[E, L]` int64 tensor, where `E` is the total number of entries.

### Memory Budget (medium scale)

| Component | Formula | Size |
|-----------|---------|------|
| Bitmaps (discrete) | 50 fields * 500 avg values * 122KB | ~3.0 GB |
| Bitmaps (text) | 10K terms * 122KB | ~1.2 GB |
| Numeric columns | 10 fields * 1M * 4B | ~40 MB |
| Item IDs | 1M * 8B | ~8 MB |
| **Total** | | **~4.3 GB** |

Fits in 8GB+ GPU or any server CPU.

## nn.Module Design

```python
class InvertedIndexModel(nn.Module):
    # Registered buffers (stored in .pt2):
    #   bitmaps:      Tensor[E, L]    int64  — packed bitmap table
    #   numeric_data: Tensor[F_n, N]  float32 — numeric columns
    #   item_ids:     Tensor[N]       int64  — item ID mapping
    #   all_ones:     Tensor[L]       int64  — 0xFFFFFFFFFFFFFFFF constant
    #   pack_powers:  Tensor[64]      int64  — [2^0, 2^1, ..., 2^63]

    # Compile-time constants:
    MAX_BP   = 32   # max bitmap predicates per query
    MAX_NP   = 16   # max numeric predicates per query
    MAX_CONJ = 16   # max conjunctions in DNF
    P = MAX_BP + MAX_NP  # total predicate slots
```

### Forward Signature

```python
def forward(self,
            bitmap_indices: Tensor,    # [MAX_BP] int64
            bitmap_valid: Tensor,      # [MAX_BP] bool
            numeric_fields: Tensor,    # [MAX_NP] int64
            numeric_ops: Tensor,       # [MAX_NP] int64  (0=eq,1=lt,2=gt,3=le,4=ge)
            numeric_values: Tensor,    # [MAX_NP] float32
            numeric_valid: Tensor,     # [MAX_NP] bool
            negation_mask: Tensor,     # [P] bool
            conj_matrix: Tensor,       # [MAX_CONJ, P] bool
            conj_valid: Tensor,        # [MAX_CONJ] bool
) -> Tensor:                           # [L] int64 — result packed bitmap
```

### Forward Logic

1. **Fetch bitmap predicates**: `self.bitmaps[bitmap_indices]` → `[MAX_BP, L]`
2. **Evaluate numeric predicates**:
   - `self.numeric_data[numeric_fields]` → `[MAX_NP, N]`
   - Apply all 5 comparison operators, select correct one via masking → `[MAX_NP, N]` bool
   - Pack bool to int64 bitmap → `[MAX_NP, L]`
3. **Stack**: `cat([bitmap_preds, numeric_packed])` → `[P, L]`
4. **Negate**: where `negation_mask`, XOR with `all_ones`
5. **Validity**: where invalid, replace with `all_ones` (identity for AND)
6. **DNF combine** (unrolled compile-time loops):
   - For each conjunction `c` in `0..MAX_CONJ`:
     - For each predicate `p` in `0..P`: select predicate if `conj_matrix[c, p]`, else `all_ones`
     - AND all → conjunction result
     - If `conj_valid[c]`: OR into final result
7. **Return** `[L]` packed bitmap

All loops have compile-time constant bounds → `torch.export` traces them into a static graph.

## Query Encoding

### Expression Syntax

```
(city == "北京" OR city == "上海") AND gender == "男" AND NOT category == "体育" AND price < 100.0
```

Supported operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `contains` (text).
Boolean connectors: `AND`, `OR`, `NOT`, parentheses for grouping.

Note: `!=` is syntactic sugar for `NOT (== value)`. The query encoder rewrites it as a negated equality predicate before DNF conversion. Similarly, `NOT contains "term"` negates the corresponding bitmap.

### Pipeline

1. **Parse**: recursive descent parser → expression tree (AST)
2. **DNF Convert**: distribute OR over AND, push NOT inward (De Morgan's) → flat list of conjunctions
3. **Encode**: map field names + values to bitmap indices (using `index_meta.json` dictionaries), map numeric predicates to (field_idx, op, value) tuples
4. **Pad**: pad to `MAX_BP`, `MAX_NP`, `MAX_CONJ` with validity masks
5. **Tensorize**: create input tensors matching the forward signature

### DNF Conversion

Any boolean expression can be converted to DNF: `(a1 AND a2 AND ...) OR (b1 AND b2 AND ...) OR ...`

- Push NOT inward using De Morgan's laws: `NOT (A AND B)` → `NOT A OR NOT B`
- Distribute AND over OR: `A AND (B OR C)` → `(A AND B) OR (A AND C)`
- Result: flat OR of AND-clauses, each containing only atomic predicates or negated atomic predicates

Worst case exponential blowup is mitigated by limiting `MAX_CONJ = 16`. If DNF exceeds this, the query is rejected with an error suggesting simplification.

## Export Pipeline

```python
import torch
import torch._inductor

model = build_inverted_index(items, schema)

example_inputs = create_example_query_tensors(model)

exported = torch.export.export(
    model,
    example_inputs,
)

torch._inductor.aoti_compile_and_package(
    exported,
    package_path="model.pt2",
)
```

The `index_meta.json` file is saved alongside, containing:
- Schema definition
- Per-field value dictionaries (string ↔ int ID)
- Text term dictionary (string ↔ int ID)
- Bitmap entry mapping: `(field_id, value_id)` → bitmap row index
- Model constants: `MAX_BP`, `MAX_NP`, `MAX_CONJ`, `N`, `L`

## C++ Inference

### API

```cpp
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

class TorchRecall {
public:
    TorchRecall(const std::string& model_pt2_path,
                const std::string& index_meta_json_path);

    // High-level: parse expression, return matching item IDs
    std::vector<int64_t> query(const std::string& expression);

    // Low-level: pre-encoded tensors
    torch::Tensor query_tensors(
        torch::Tensor bitmap_indices, torch::Tensor bitmap_valid,
        torch::Tensor numeric_fields, torch::Tensor numeric_ops,
        torch::Tensor numeric_values, torch::Tensor numeric_valid,
        torch::Tensor negation_mask, torch::Tensor conj_matrix,
        torch::Tensor conj_valid);

private:
    torch::inductor::AOTIModelPackageLoader loader_;
    IndexMetadata meta_;
    // internal helpers
    ExprTree parse(const std::string& expr);
    DNF to_dnf(const ExprTree& tree);
    QueryTensors encode(const DNF& dnf);
    std::vector<int64_t> decode_bitmap(const torch::Tensor& packed_bitmap);
};
```

### Components

1. **QueryParser**: recursive descent parser for expression strings → AST
2. **DNFConverter**: AST → DNF using De Morgan's + distribution
3. **TensorEncoder**: DNF → padded input tensors, using metadata dictionaries
4. **ModelRunner**: `AOTIModelPackageLoader` wrapping `.run(inputs)`
5. **ResultDecoder**: unpack int64 bitmap → extract set bit positions → map to item IDs

### Build System (CMake)

```cmake
cmake_minimum_required(VERSION 3.18)
project(torch_recall)
find_package(Torch REQUIRED)
# ... sources, linking against torch libraries
```

## Project File Structure

```
torch-recall/
├── offline/                        # 离线: 索引构建 + 模型导出 (Python)
│   ├── torch_recall/
│   │   ├── __init__.py
│   │   ├── schema.py               # field type definitions, validation
│   │   ├── builder.py              # items → InvertedIndexModel
│   │   ├── model.py                # InvertedIndexModel(nn.Module)
│   │   ├── query.py                # query parser + DNF converter + tensor encoder
│   │   ├── exporter.py             # torch.export + aoti_compile_and_package
│   │   └── tokenizer.py            # text field tokenization (jieba or simple split)
│   ├── tests/
│   │   ├── test_builder.py
│   │   ├── test_model.py
│   │   ├── test_query.py
│   │   └── test_e2e.py
│   ├── benchmarks/
│   │   └── bench_recall.py         # latency benchmark for 1M items
│   └── pyproject.toml
├── online/                         # 在线: 加载模型 + 查询推理 (C++)
│   ├── include/torch_recall/
│   │   ├── query_parser.h
│   │   ├── dnf_converter.h
│   │   ├── tensor_encoder.h
│   │   ├── model_runner.h
│   │   └── result_decoder.h
│   ├── src/
│   │   ├── query_parser.cpp
│   │   ├── dnf_converter.cpp
│   │   ├── tensor_encoder.cpp
│   │   ├── model_runner.cpp
│   │   ├── result_decoder.cpp
│   │   └── main.cpp                # CLI example
│   ├── tests/
│   │   └── test_query.cpp
│   └── CMakeLists.txt
├── docs/
│   └── specs/
│       └── 2026-04-08-bitmap-inverted-index-design.md
└── README.md
```

## Performance Analysis

### GPU (CUDA)

| Step | Data size | Time |
|------|-----------|------|
| Bitmap fetch (32 entries) | 32 * 122KB = 3.8MB | ~0.01ms |
| Numeric eval (16 preds) | 16 * 1M * 4B = 64MB | ~0.05ms |
| Pack bool→int64 | 16 * 1M | ~0.02ms |
| DNF combine (16 conj * 48 preds) | 48 * 122KB | ~0.01ms |
| **Total** | | **~0.1ms** |

### CPU (single core, AVX2)

| Step | Data size | Time |
|------|-----------|------|
| Bitmap fetch (32 entries) | 3.8MB (L3 cache) | ~0.1ms |
| Numeric eval (16 preds) | 64MB | ~2ms |
| Pack bool→int64 | 16M bools | ~0.5ms |
| DNF combine | 48 * 122KB | ~0.2ms |
| **Total** | | **~3ms** |

Both well under the 20ms requirement.

## Limitations and Mitigations

| Limitation | Mitigation |
|------------|------------|
| DNF exponential blowup | Cap at MAX_CONJ=16, reject over-complex queries |
| High cardinality discrete fields | Fields with >5000 unique values: fall back to columnar eval at build time |
| Text tokenization quality | Pluggable tokenizer (default: jieba for Chinese, whitespace for English) |
| Fixed max predicate/conjunction sizes | Configurable at build time, recompile .pt2 for different limits |
| Numeric field precision | float32; use int64 columns for exact integer comparison if needed |

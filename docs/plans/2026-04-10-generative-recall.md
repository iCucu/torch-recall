# Implementation Plan: Generative Recall

> Spec: [docs/specs/2026-04-10-generative-recall-design.md](../specs/2026-04-10-generative-recall-design.md)

## Step 1: Item.sid_path 字段

**文件**: [index/torch_recall/schema.py](../../index/torch_recall/schema.py)

- `Item` dataclass 增加 `sid_path: list[int] | None = None`
- 无需修改 Schema（sid_path 不是 schema 级别的字段定义）

**验证**: 现有测试不受影响（sid_path 是可选字段）

---

## Step 2: Trie 数据结构

**文件**: `index/torch_recall/recall_method/generative/trie.py`（新建）

### Trie — Dense/CSR 混合 nn.Module

采用全局 State ID 体系，无 `DenseTrieLevel`/`SparseTrieLevel` 子模块。所有状态（节点）按层连续编号，统一存储。

```python
class Trie(nn.Module):
    """Dense/CSR hybrid trie with global state IDs."""

    # ---- registered buffers ----
    start_mask:     [V] bool               # 合法首 token
    dense_mask:     [V, V] bool            # d_dense=2 的密集查表
    dense_states:   [V, V] int64           # d_dense=2 的 state 查表
    packed_csr:     [E + V, 2] int64       # CSR 全量边 [token, next_state]
    csr_indptr:     [num_states + 2] int64 # CSR 行指针
    leaf_item_ids:  [num_leaves, max_items_per_leaf] int64
    leaf_item_valid:[num_leaves, max_items_per_leaf] bool

    # ---- metadata (Python scalars / lists) ----
    level_start:   list[int]       # 每层起始 state ID
    level_count:   list[int]       # 每层节点数
    max_branch_factors: list[int]  # 每层最大分支因子
    num_items, num_states, d_dense, path_length, vocab_size

    # ---- methods ----
    propagate_validity(item_mask [B, N]) -> node_valid [B, num_states] bool
    resolve_leaves(beam_states [B, beam], scores [B, beam]) -> [B, N] float
```

#### State ID 分配

```
State 0       — 占位符（padding / OOB）
State 1..V    — L0 节点 (state = first_token + 1)
State V+1..   — L1, L2, … 节点
最后一段       — 叶子节点 (depth L-1)
```

#### CSR 格式

```python
# packed_csr[indptr[s] : indptr[s+1]] 为 state s 的所有子 (token, child_state) 对
# 末尾 V 行 OOB 填充 → gather 越界安全
```

#### 向上传播 (propagate_validity)

利用 CSR burst-read，从叶子逐层向上聚合子节点有效性。每步：
`node_valid[:, parent_states] = (node_valid[:, child_states] & struct_mask).any(dim=-1)`

#### resolve_leaves

beam 叶子 state → leaf_item_ids 映射 → scatter_reduce(amax) 到 `[B, N]`。

**验证**: 单测用小型 trie（5 items, 3 unique paths），验证 CSR 结构、state ID 和传播正确性。

---

## Step 3: Trie Builder

**文件**: `index/torch_recall/recall_method/generative/builder.py`（新建）

采用纯 NumPy 向量化构建（参考 STATIC 的 `build_static_index`）：

```python
class GenerativeBuilder:
    def __init__(self, schema, decoder, beam_width=10, dense_levels=2,
                 sid_vocab_size=2048, path_length=5):

    def build(self, items) -> (GenerativeRecall, meta):
        1. 校验: 每个 item 必须有 sid_path, len == path_length, 值 ∈ [0, vocab_size)
        2. 调用 TargetingBuilder(schema).build(items) → targeting_model, targeting_meta
        3. _build_static_index(sorted_sids, V, d_dense):
           a. np.lexsort → 字典序排序
           b. diff-scan → 识别每层新前缀
           c. cumulative 分配全局 State ID
           d. 收集所有 (parent, token, child) 边
           e. np.bincount + np.cumsum → CSR 压缩
           f. 合成 dense_mask, dense_states, start_mask
           g. 计算 max_branch_factors
        4. _build_leaf_mapping: leaf state → item index
        5. 组装 Trie
        6. 构建 GenerativeRecall(targeting, trie, decoder, ...)
        7. 返回 (model, meta)
```

**验证**: 构建后检查 num_states, CSR indptr shape, leaf mapping。

---

## Step 4: MockDecoder

**文件**: `index/torch_recall/recall_method/generative/decoder.py`（新建）

```python
class MockDecoder(nn.Module):
    def __init__(self, user_dim: int, vocab_size: int = 2048):
        self.proj = nn.Linear(user_dim, vocab_size)

    def forward(self, user_repr, partial_paths):
        return self.proj(user_repr)  # [B*beam, 2048]
```

简单线性映射，不依赖 partial_paths（纯 mock）。

---

## Step 5: GenerativeRecall

**文件**: `index/torch_recall/recall_method/generative/recall.py`（新建）

```python
class GenerativeRecall(RecallOp):
    def __init__(self, targeting, trie, decoder, beam_width, num_items,
                 num_preds, user_dim, query_offset):
        self.targeting = targeting      # TargetingRecall
        self.trie = trie                # Trie
        self.decoder = decoder          # nn.Module
        # ...

    def forward(self, pred_satisfied, query):
        user_repr = query[:, query_offset : query_offset + user_dim]

        # Phase 1: targeting → item_mask
        # Phase 2: trie.propagate_validity(item_mask) → node_valid [B, S]
        # Phase 3: beam search
        #   Step 0: start_mask + node_valid → 首 token topk
        #   Step 1..d_dense-1: _step_dense (dense_mask/dense_states lookup)
        #   Step d_dense..L-1: _step_csr (CSR burst-read)
        #   Decoder logits 经 log_softmax 后 mask
        #   beam 更新用 _gather_beams
        # Phase 4: trie.resolve_leaves → [B, N]
```

**验证**: 小数据 E2E forward，检查输出 shape `[B, N]`，合法 item 有限分数。

---

## Step 6: Spec 与 Pipeline 集成

### 6a. Generative spec 节点

**文件**: [index/torch_recall/scheduler/spec.py](../../index/torch_recall/scheduler/spec.py)

```python
@dataclass
class Generative(RecallSpec):
    schema: Schema
    decoder: nn.Module
    beam_width: int = 10
    dense_levels: int = 2
```

### 6b. PipelineBuilder 扩展

**文件**: [index/torch_recall/scheduler/pipeline_builder.py](../../index/torch_recall/scheduler/pipeline_builder.py)

- `_collect_leaves` 增加 `generative_leaves` 收集
- `build()` 中遇到 `Generative` 叶子时调用 `GenerativeBuilder.build(items)`
- `total_query_dim` 加上 `user_dim`，通过 `query_offset` 切片

### 6c. Encoder 扩展

**文件**: [index/torch_recall/scheduler/encoder.py](../../index/torch_recall/scheduler/encoder.py)

- `encode_pipeline_inputs` 支持 `generative` 部分（user_repr 作为 query 的一段）

### 6d. 导出

- `scheduler/__init__.py` 导出 `Generative`
- `__init__.py` 顶层导出

---

## Step 7: 测试

**文件**: `index/tests/test_generative.py`（新建）

### 测试数据

```python
SCHEMA = Schema(discrete_fields=["city"], numeric_fields=["age"])
ITEMS = [
    Item(id="a", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="b", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),  # 共享 path
    Item(id="c", targeting_rule='city == "上海"', sid_path=[0, 1, 5, 6, 7]),
    Item(id="d", targeting_rule='age > 18',       sid_path=[0, 2, 8, 9, 10]),
    Item(id="e", targeting_rule='city == "北京"', sid_path=[0, 2, 8, 9, 11]),
]
```

### 测试用例

1. **TestBuilder**: Trie 构建正确性（num_states, CSR indptr, dense_mask, leaf mapping）
2. **TestForward**: 用 MockDecoder + 小 trie → forward 输出 `[B, N]` shape，分数有限
3. **TestTargetingMask**: 被 targeting 过滤的 item 分数为 -inf
4. **TestBatch**: B=3 batch 结果正确
5. **TestPipelineGenerativeOnly**: 独立 `Generative` pipeline 正确工作
6. **TestPipelineOrGenerativeKNN**: `Or(Generative, KNN)` 组合正确工作

---

## Step 8: Example

**文件**: `examples/05_generative_recall.py`（新建）

完整示例：构建 Generative pipeline → Trie 结构概览 → 查询 → 打印结果。

---

## 实施顺序

```
Step 1 (schema)
    ↓
Step 2 (trie.py) + Step 4 (decoder.py)     ← 可并行
    ↓
Step 3 (builder.py)
    ↓
Step 5 (recall.py)
    ↓
Step 6 (spec + pipeline integration)
    ↓
Step 7 (tests) + Step 8 (example)           ← 可并行
```

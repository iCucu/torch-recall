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

### 2a. DenseTrieLevel

```python
class DenseTrieLevel(nn.Module):
    """Trie level with dense [num_nodes, V] representation."""
    buffers:
        valid:          [num_nodes, V] bool
        child_id:       [num_nodes, V] int64
        prop_children:  [num_nodes, max_ch] int64
        prop_mask:      [num_nodes, max_ch] bool

    methods:
        get_children(node_ids [B, beam]) -> (valid [B, beam, V], child_ids [B, beam, V])
        propagate(child_valid [B, num_children]) -> [B, num_nodes] bool
```

### 2b. SparseTrieLevel

```python
class SparseTrieLevel(nn.Module):
    """Trie level with padded sparse representation."""
    buffers:
        children_sid:   [num_nodes, max_ch] int64
        children_node:  [num_nodes, max_ch] int64
        children_valid: [num_nodes, max_ch] bool

    methods:
        get_children(node_ids [B, beam]) -> (sids [B, beam, max_ch], nodes [B, beam, max_ch], valid [B, beam, max_ch])
        propagate(child_valid [B, num_children]) -> [B, num_nodes] bool
```

注意: `propagate` 复用 `prop_children` / `prop_mask` buffers（与 SparseTrieLevel 的 children buffers 形状一致，直接复用）。

### 2c. HybridTrie

```python
class HybridTrie(nn.Module):
    levels: nn.ModuleList          # [DenseTrieLevel, DenseTrieLevel, SparseTrieLevel, ...]
    leaf_item_ids:   [num_leaves, max_items_per_leaf] int64
    leaf_item_valid: [num_leaves, max_items_per_leaf] bool
    num_nodes_per_level: list[int]

    methods:
        propagate_validity(item_mask [B, N]) -> list[Tensor]  # node_valid per level
        resolve_leaves(leaf_nodes [B, beam], scores [B, beam]) -> [B, N] float
```

**验证**: 单测用小型 trie（5 items, 3 unique paths），验证节点结构和传播正确性。

---

## Step 3: Trie Builder

**文件**: `index/torch_recall/recall_method/generative/builder.py`（新建）

```python
class GenerativeBuilder:
    def __init__(self, schema, decoder, beam_width=10, dense_levels=2,
                 sid_vocab_size=2048, path_length=5):

    def build(self, items) -> (GenerativeRecall, meta):
        1. 校验: 每个 item 必须有 sid_path, len == path_length, 值 ∈ [0, vocab_size)
        2. 校验: 有 targeting_rule 的 item 才参与 targeting 构建
        3. 调用 TargetingBuilder(schema).build(items) → targeting_model, targeting_meta
        4. 从 items 提取 sid_paths，构建 trie:
           a. 收集每层的 unique prefixes → 分配 node ids
           b. 前 dense_levels 层: 构建 [num_nodes, V] valid/child_id tensors
           c. 后续 sparse 层: 构建 padded children tensors
           d. 叶子层: 构建 leaf→item 映射
           e. 每层构建 propagation buffers (padded children lists)
        5. 组装 HybridTrie
        6. 构建 GenerativeRecall(targeting, trie, decoder, ...)
        7. 返回 (model, meta)

    def save_meta(self, meta, path): ...
```

**验证**: 构建后检查 trie 节点数、层数、leaf 映射。

---

## Step 4: MockDecoder

**文件**: `index/torch_recall/recall_method/generative/decoder.py`（新建）

```python
class MockDecoder(nn.Module):
    def __init__(self, user_dim: int, vocab_size: int = 2048):
        self.proj = nn.Linear(user_dim, vocab_size)

    def forward(self, user_repr, partial_paths):
        # user_repr: [B*beam, D], partial_paths: [B*beam, 5]
        return self.proj(user_repr)  # [B*beam, 2048]
```

简单线性映射，不依赖 partial_paths（纯 mock）。

---

## Step 5: GenerativeRecall

**文件**: `index/torch_recall/recall_method/generative/recall.py`（新建）

```python
class GenerativeRecall(RecallOp):
    def __init__(self, targeting, trie, decoder, beam_width, num_items, num_preds, user_dim):
        self.targeting = targeting      # TargetingRecall
        self.trie = trie                # HybridTrie
        self.decoder = decoder          # nn.Module
        self.beam_width = beam_width
        self.num_items = num_items
        self.num_preds = num_preds
        self.user_dim = user_dim

    def forward(self, pred_satisfied, query):
        # Phase 1: targeting mask
        targeting_scores = self.targeting(pred_satisfied, query)
        item_mask = targeting_scores > float("-inf")

        # Phase 2: trie validity propagation
        node_valid_per_level = self.trie.propagate_validity(item_mask)

        # Phase 3: beam search (5 steps, unrolled)
        B = pred_satisfied.shape[0]
        beam_nodes = zeros(B, beam_width)  # all start at root (node 0)
        beam_scores = zeros(B, beam_width)
        beam_paths = zeros(B, beam_width, 5)

        for step in range(5):
            level = self.trie.levels[step]
            node_valid_next = node_valid_per_level[step + 1] if step < 4 else leaf_valid

            if isinstance(level, DenseTrieLevel):
                # dense beam step
                ...
            else:
                # sparse beam step
                ...

            # topk → update beam_nodes, beam_scores, beam_paths

        # Phase 4: resolve to [B, N]
        return self.trie.resolve_leaves(beam_nodes, beam_scores, self.num_items)

    def example_inputs(self, device="cpu"):
        return (
            zeros(1, self.num_preds, dtype=torch.bool, device=device),
            randn(1, self.user_dim, device=device),
        )
```

注意: `for step in range(5)` 中 `isinstance` 检查在 tracing 时是静态的（ModuleList 中每个元素类型已知），`torch.export` 兼容。

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
- `_compile` 中返回构建好的 `GenerativeRecall` 模块
- `total_query_dim` 加上 `user_dim`（decoder 的输入维度需要在 Generative spec 中指定）

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

1. **TestTrieBuilder**: trie 构建正确性（节点数、边数、leaf mapping）
2. **TestTriePropagate**: 给定 item_mask，node_valid 正确传播
3. **TestBeamSearch**: MockDecoder + 小 trie → beam 输出都是合法 sid path
4. **TestForwardShape**: `forward()` 输出 `[B, N]`，分数范围正确
5. **TestTargetingMask**: 被 targeting 过滤的 item 分数为 -inf
6. **TestBatch**: B=3 batch 结果与逐条一致
7. **TestExport**: `export_recall_model` 成功导出 `.pt2`
8. **TestPipelineIntegration**: `Or(Generative(...), KNN(...))` 组合正确工作

---

## Step 8: Example

**文件**: `examples/05_generative_recall.py`（新建）

完整示例：构建 Generative pipeline → 查询 → 导出。

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

预计工作量: 每 step 约 1 个 commit，总共 ~8 commits。

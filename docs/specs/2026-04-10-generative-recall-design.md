# Generative Recall — 技术设计

> 基于 Trie-Constrained Beam Search 的生成式召回

## 概述

在现有 Targeting (定向过滤) 和 KNN (向量召回) 之外，新增 **Generative Recall** 召回方法。核心思路：

1. 每个 item 拥有一个 **sid path**（语义 ID 路径），长度 5，每个 sid ∈ [0, 2048)
2. 离线将所有 item 的 sid path 构建成一棵 **Trie**
3. 在线时，先由 Targeting 计算 item mask，向上传播到 trie 得到每个节点的有效性
4. 外部 **Decoder**（自回归 nn.Module）通过 5 轮 beam search，在有效 trie 分支中选取 top-K sid path
5. 反解 sid path 到 item index，输出 `[B, N]` 分数

GenerativeRecall 实现 `RecallOp` 接口，可通过 `And`/`Or` 与其他召回方法组合。

## 数据模型

### Item 扩展

`Item` dataclass 新增 `sid_path` 可选字段：

```python
@dataclass
class Item:
    id: str | None = None
    targeting_rule: str | None = None
    embedding: list[float] | None = None
    sid_path: list[int] | None = None      # 长度 5，每个值 ∈ [0, 2048)
```

sid path 与 item 的关系是 **多对一**：多个 item 可共享同一 sid path（叶子节点存多个 item）。

### Decoder 协议

外部传入的自回归 decoder 遵循以下接口：

```python
class GenerativeDecoder(nn.Module):
    def forward(
        self,
        user_repr: torch.Tensor,      # [B * beam, D]
        partial_paths: torch.Tensor,   # [B * beam, 5]  (padded, 未到达的位置为 0)
    ) -> torch.Tensor:                 # [B * beam, 2048] logits
```

- `user_repr`: 用户表征向量，由 `query` 输入提供
- `partial_paths`: 当前已选的 sid path 前缀，固定长度 5，未选位置填 0
- 返回对所有 2048 个候选 sid 的 logits

项目内提供 `MockDecoder`（返回随机 logits）用于测试。

## 架构

### forward 四阶段

```
forward(pred_satisfied [B, P], query [B, D]) -> [B, N] float

Phase 1: Targeting Mask
    targeting_scores = self.targeting(pred_satisfied, query)   # [B, N]
    item_mask = targeting_scores > float("-inf")                # [B, N] bool

Phase 2: Trie Validity Propagation (bottom-up)
    leaf_valid   ← item_mask + leaf→item mapping               # [B, num_leaves]
    node_valid_4 ← leaf_valid + L4 propagation buffers         # [B, num_L4_nodes]
    node_valid_3 ← node_valid_4 + L3 propagation buffers
    node_valid_2 ← node_valid_3 + L2 propagation buffers
    node_valid_1 ← node_valid_2 + L1 propagation buffers

Phase 3: Beam Search (5 steps)
    遍历 L0→L4，每步:
    - 从 trie 取候选 children
    - 叠加 node_valid 约束
    - decoder 打分
    - topk 选 beam

Phase 4: Resolve to [B, N]
    叶子 → item 映射 + scatter_reduce(max)
```

- `pred_satisfied` 用于 Phase 1（targeting mask 计算）
- `query` 作为 `user_repr` 传给 decoder（用户表征向量）

所有操作均为固定形状 tensor 运算，兼容 `torch.export`。

### 模块组成

```
GenerativeRecall(RecallOp)
├── targeting: TargetingRecall       # 复用现有模块，计算 item_mask
├── trie: HybridTrie                 # Trie 结构 + 传播 buffers
│   ├── DenseTrieLevel (L0)
│   ├── DenseTrieLevel (L1)
│   ├── SparseTrieLevel (L2)
│   ├── SparseTrieLevel (L3)
│   └── SparseTrieLevel (L4)
│   └── leaf→item buffers
└── decoder: nn.Module               # 外部传入
```

## HybridTrie — 混合稠密/稀疏 Trie

### Dense Level（L0, L1）

```python
class DenseTrieLevel(nn.Module):
    valid:    [num_nodes, 2048] bool     # sid j 是否有子节点
    child_id: [num_nodes, 2048] int64    # 对应的子节点 id（-1 = 不存在）

    # 向上传播用
    prop_children: [num_nodes, max_ch] int64   # 子节点列表（padded）
    prop_mask:     [num_nodes, max_ch] bool     # padding mask
```

- L0: 1 个节点（root），L1: 最多 2048 个节点
- 恒定 ~36 MB，不随 item 数增长

### Sparse Level（L2, L3, L4）

```python
class SparseTrieLevel(nn.Module):
    children_sid:   [num_nodes, max_ch] int64    # 边的 sid 值
    children_node:  [num_nodes, max_ch] int64    # 子节点 id
    children_valid: [num_nodes, max_ch] bool     # padding mask
```

- `max_ch` 由构建时的实际数据决定（L2 ≈ 32, L3 ≈ 16, L4 ≈ 8）
- padded 格式保证固定形状 gather

### Leaf → Item 映射

```python
leaf_item_ids:   [num_leaves, max_items_per_leaf] int64
leaf_item_valid: [num_leaves, max_items_per_leaf] bool
```

### 内存估算

| 组件 | N = 1M | N = 10M |
|------|--------|---------|
| Dense L0 + L1 | 36 MB | 36 MB |
| Sparse L2-L4 | ~54 MB | ~400 MB |
| Leaf mapping | ~15 MB | ~130 MB |
| Propagation buffers | ~50 MB | ~300 MB |
| **合计** | **~155 MB** | **~866 MB** |

## Targeting Mask 向上传播

从 `item_mask [B, N]` 逐层推导每个 trie 节点子树下是否有合法 item：

```python
def propagate_validity(self, item_mask):
    # Leaf validity: 叶子下有任一合法 item 即为 valid
    items = item_mask[:, self.leaf_item_ids]              # [B, num_leaves, max_items]
    leaf_valid = (items & self.leaf_item_valid).any(2)    # [B, num_leaves]

    # Bottom-up: L4 ← leaves, L3 ← L4, L2 ← L3, L1 ← L2
    node_valid = leaf_valid
    for level in [self.level4, self.level3, self.level2, self.level1]:
        ch = node_valid[:, level.prop_children]           # [B, num_nodes, max_ch]
        node_valid = (ch & level.prop_mask).any(2)        # [B, num_nodes]

    return node_valid_per_level  # dict: level -> [B, num_nodes] bool
```

每一步都是 `gather → bitwise AND → any`，标准 tensor 操作。

## Beam Search

### Dense 层（Step 0, 1）

直接 gather `[num_nodes, 2048]` 的 valid/child_id 矩阵：

```python
valid = level.valid[beam_nodes]                           # [B, beam, 2048]
child_ids = level.child_id[beam_nodes]                    # [B, beam, 2048]
valid = valid & node_valid_next[:, child_ids]             # 叠加 targeting 约束

logits = decoder(query_expanded, beam_paths)              # [B, beam, 2048]
masked = logits.masked_fill(~valid, float("-inf"))

flat = (beam_scores.unsqueeze(-1) + masked).view(B, -1)  # [B, beam * 2048]
top_scores, top_flat = flat.topk(beam_width)              # [B, beam_width]
beam_idx = top_flat // 2048
sid_idx  = top_flat % 2048
```

### Sparse 层（Step 2, 3, 4）

使用紧凑的 padded children（max_ch << 2048），只 gather 有效 sid 的 logits：

```python
ch_sids  = level.children_sid[beam_nodes]                 # [B, beam, max_ch]
ch_nodes = level.children_node[beam_nodes]                # [B, beam, max_ch]
ch_valid = level.children_valid[beam_nodes]               # [B, beam, max_ch]
ch_valid = ch_valid & node_valid_next[:, ch_nodes]        # 叠加 targeting 约束

full_logits = decoder(query_expanded, beam_paths)         # [B, beam, 2048]
compact = full_logits.gather(2, ch_sids)                  # [B, beam, max_ch]
compact = compact.masked_fill(~ch_valid, float("-inf"))

flat = (beam_scores.unsqueeze(-1) + compact).view(B, -1) # [B, beam * max_ch]
top_scores, top_flat = flat.topk(beam_width)
beam_idx = top_flat // max_ch
ch_idx   = top_flat % max_ch
```

### Phase 4: 结果映射

```python
# beam_nodes 是叶子节点: [B, beam_width]
items = self.leaf_item_ids[beam_nodes]                    # [B, beam, max_items]
valid = self.leaf_item_valid[beam_nodes]                  # [B, beam, max_items]
scores_exp = beam_scores.unsqueeze(-1).expand_as(items)

output = torch.full((B, N), float("-inf"))
flat_items = items.view(B, -1)
flat_scores = scores_exp.masked_fill(~valid, float("-inf")).view(B, -1)
output.scatter_reduce_(1, flat_items, flat_scores, reduce="amax")
```

## 文件结构

```
index/torch_recall/recall_method/generative/
├── __init__.py
├── recall.py          # GenerativeRecall(RecallOp)
├── builder.py         # GenerativeBuilder
├── trie.py            # HybridTrie, DenseTrieLevel, SparseTrieLevel
└── decoder.py         # MockDecoder
```

修改的现有文件：
- `schema.py` — Item 增加 `sid_path` 字段
- `scheduler/spec.py` — 增加 `Generative` spec 节点
- `scheduler/pipeline_builder.py` — 处理 `Generative` 叶子
- `scheduler/__init__.py` — 导出新符号
- `__init__.py` — 顶层导出

## Spec 与 Pipeline 集成

新增 `Generative` spec 节点：

```python
@dataclass
class Generative(RecallSpec):
    schema: Schema
    decoder: nn.Module
    beam_width: int = 10
    dense_levels: int = 2
```

使用方式：

```python
# 独立使用
spec = Generative(schema, decoder, beam_width=10)

# 与 KNN 组合（取并集）
spec = Or(Generative(schema, decoder), KNN(metric="cosine"))
```

`PipelineBuilder` 遇到 `Generative` 叶子时：
1. 调用 `GenerativeBuilder(spec.schema, spec.decoder, spec.beam_width, spec.dense_levels).build(items)`
2. 得到 `GenerativeRecall` 模块
3. 其 `query` 输入占用 `total_query_dim` 中的 D 维度（用户表征维度）

## 测试计划

1. **TrieBuilder 单测**: 构建 trie，验证节点数、边数、leaf 映射正确性
2. **Targeting mask 传播单测**: 给定 item_mask，验证各层 node_valid 正确
3. **Beam search 单测**: 用 MockDecoder + 小 trie，验证 beam search 输出合法 sid path
4. **E2E 单测**: 构建完整 pipeline，验证 forward 输出 shape 和分数范围
5. **Batch 测试**: B > 1 时结果与逐条一致
6. **Export 测试**: `torch.export` + `.pt2` 导出成功
7. **Pipeline 组合测试**: `Or(Generative, KNN)` 正确工作

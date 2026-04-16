# Generative Recall — 技术设计

> 基于 Trie-Constrained Beam Search 的生成式召回
>
> 参考: [STATIC — Sparse Transition-Accelerated Trie Index for Constrained Decoding](https://arxiv.org/abs/2602.22647) (Su et al., 2026)

## 概述

在现有 Targeting (定向过滤) 和 KNN (向量召回) 之外，新增 **Generative Recall** 召回方法。核心思路：

1. 每个 item 拥有一个 **sid path**（语义 ID 路径），长度 5，每个 sid ∈ [0, 2048)
2. 离线将所有 item 的 sid path 构建成一棵 **Trie**（Dense/CSR 混合表示）
3. 在线时，先由 Targeting 计算 item mask，通过 CSR 向上传播到 trie 得到每个节点的有效性
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

Phase 2: Trie Validity Propagation (bottom-up via CSR)
    node_valid [B, num_states] bool
    叶子 ← item_mask + leaf→item mapping
    逐层通过 CSR burst-read 聚合子节点有效性，向上传播至 L0

Phase 3: Beam Search (L steps)
    Step 0: start_mask 过滤 + L0 node_valid → 选首 token
    Steps 1..d_dense-1: Dense 查表 + node_valid → topk
    Steps d_dense..L-1: CSR burst-read + node_valid → topk

Phase 4: Resolve to [B, N]
    叶子 state → item 映射 + scatter_reduce(amax)
```

- `pred_satisfied` 用于 Phase 1（targeting mask 计算）
- `query` 作为 `user_repr` 传给 decoder（用户表征向量）
- Decoder logits 经过 `log_softmax` 归一化后再 mask，beam scores 累加为真正的 log-probability

所有操作均为固定形状 tensor 运算，兼容 `torch.export`。

### 模块组成

```
GenerativeRecall(RecallOp)
├── targeting: TargetingRecall       # 复用现有模块，计算 item_mask
├── trie: Trie                       # Dense/CSR 混合 Trie
│   ├── start_mask [V] bool          # 合法首 token
│   ├── dense_mask [V, V] bool       # Dense 层 (d_dense=2)
│   ├── dense_states [V, V] int64    # Dense 层 state ID
│   ├── packed_csr [E+V, 2] int64    # CSR 转移表 (全量边)
│   ├── csr_indptr [S+2] int64       # CSR 行指针
│   └── leaf→item buffers
└── decoder: nn.Module               # 外部传入
```

## Trie — Dense/CSR 混合表示

### 全局 State ID 体系

所有 trie 节点共享一个全局 state ID 空间，按层连续分配：

```
State 0       — 填充/未使用
State 1..V    — L0 节点 (state = first_token + 1)
State V+1..   — L1, L2, … 节点
最后一段       — 叶子节点 (depth L-1)
```

优势：
- 无 `isinstance` 类型分支，`torch.export` 友好
- CSR 统一编码所有边，propagation 和 beam search 共用同一数据

### Dense 层 (前 d_dense 层)

为 trie 的 "热头" 提供 O(1) 查表加速：

```python
start_mask:   [V] bool          # 合法首 token
dense_mask:   [V, V] bool       # (d_dense=2) 给定 t0，合法的 t1
dense_states: [V, V] int64      # (d_dense=2) 跟随 (t0, t1) 后的 state ID
```

- 恒定 ~36 MB，不随 item 数增长
- Dense 层是 CSR 的冗余加速，CSR 完整包含所有边

### CSR 稀疏层 (全量)

采用 Compressed Sparse Row 格式，零填充浪费：

```python
packed_csr: [num_edges + V, 2] int64   # 每行 [token_id, next_state_id]
csr_indptr: [num_states + 2] int64     # 行指针，indptr[s]..indptr[s+1] 为 state s 的子节点

# 末尾 V 行为 OOB 安全填充 (token=V, state=0)
```

beam search 的 CSR 查询（burst-read）：

```python
starts = csr_indptr[flat_states]
offsets = torch.arange(limit)
gather_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
gathered = packed_csr[gather_idx.clamp(max=...)]  # [B*beam, K, 2]
```

- 每次查询 O(K) 而非 O(V)，K = 该层最大分支因子
- I/O 复杂度与约束集总量无关

### Leaf → Item 映射

```python
leaf_item_ids:   [num_leaves, max_items_per_leaf] int64
leaf_item_valid: [num_leaves, max_items_per_leaf] bool
```

### 内存估算

| 组件 | N = 1M | N = 10M |
|------|--------|---------|
| Dense (start_mask + dense_mask + dense_states) | 36 MB | 36 MB |
| CSR (packed_csr + csr_indptr) | ~30 MB | ~200 MB |
| Leaf mapping | ~15 MB | ~130 MB |
| **合计** | **~81 MB** | **~366 MB** |

CSR 相比 padded sparse 节省显著内存（无 max_ch 填充浪费）。

## Targeting Mask 向上传播

从 `item_mask [B, N]` 通过 CSR 逐层推导每个 trie 节点子树下是否有合法 item：

```python
def propagate_validity(self, item_mask):
    node_valid = zeros(B, num_states, dtype=bool)

    # 叶子: 有任一合法 item 即为 valid
    leaf_valid = (item_mask[:, leaf_item_ids] & leaf_item_valid).any(2)
    node_valid[:, leaf_start:leaf_end] = leaf_valid

    # 逐层向上 (depth L-2 → 0): CSR burst-read 聚合子节点
    for d in reversed(range(path_length - 1)):
        states = arange(level_start[d], level_start[d] + level_count[d])
        starts = csr_indptr[states]
        actual_lens = csr_indptr[states + 1] - starts

        offsets = arange(max_branch_factors[d + 1])
        gather_idx = (starts[:, None] + offsets[None, :]).clamp(max=...)

        ch_states = packed_csr[gather_idx, 1]           # [s_count, max_br]
        struct_ok = offsets[None, :] < actual_lens[:, None]

        child_v = node_valid[:, ch_states] & struct_ok   # [B, s_count, max_br]
        node_valid[:, level_start[d]:level_end[d]] = child_v.any(2)

    return node_valid  # [B, num_states]
```

每步均为 `CSR gather → bitwise AND → reduce_any`，标准 tensor 操作。

## Beam Search

### Step 0: 首 token

```python
logits = decoder(query, zeros_paths)        # [B, V]
logprobs = log_softmax(logits, dim=-1)

valid = start_mask & node_valid[:, 1:V+1]   # [B, V]
logprobs = logprobs.masked_fill(~valid, -inf)

top_lp, top_tok = logprobs.topk(beam)       # [B, beam]
beam_states = top_tok + 1                    # L0 state IDs
```

### Dense 步 (Step 1, d_dense=2)

```python
parent_tok = flat_states - 1                            # 恢复首 token
masks = dense_mask[parent_tok]                          # [B*beam, V]
state_table = dense_states[parent_tok]                  # [B*beam, V]

nv = node_valid[batch_idx, state_table]                 # [B*beam, V]
combined = masks & nv

logprobs = log_softmax(decoder_logits, -1)
logprobs = logprobs.masked_fill(~combined, -inf)

# topk → beam update
```

### CSR 步 (Step 2, 3, 4)

```python
starts = csr_indptr[flat_states]
actual_lens = csr_indptr[flat_states + 1] - starts
limit = max_branch_factors[step]

offsets = arange(limit)
gather_idx = (starts[:, None] + offsets[None, :]).clamp(max=...)

gathered = packed_csr[gather_idx]                       # [B*beam, K, 2]
cand_tokens = gathered[..., 0]
cand_states = gathered[..., 1]

struct_ok = offsets[None, :] < actual_lens[:, None]
cand_nv = node_valid[batch_idx, cand_states]
valid = struct_ok & cand_nv

logprobs = log_softmax(decoder_logits, -1)
cand_logprobs = logprobs.gather(1, cand_tokens.clamp(max=V-1))
cand_logprobs = cand_logprobs.masked_fill(~valid, -inf)

# topk → beam update via _gather_beams
```

### Phase 4: 结果映射

```python
local = beam_states - leaf_start                        # [B, beam]
items = leaf_item_ids[local]                            # [B, beam, max_items]
valid = leaf_item_valid[local]

output = full((B, N), -inf)
flat_items = items.view(B, -1)
flat_scores = scores.unsqueeze(-1).masked_fill(~valid, -inf).view(B, -1)
output.scatter_reduce_(1, flat_items, flat_scores, reduce="amax")
```

## 向量化 Index 构建

构建过程采用纯 NumPy 向量化操作（参考 STATIC），百万级 item 毫秒级完成：

```python
# 1. 排序 SID paths → lexicographic order
sort_idx = np.lexsort(sid_paths.T[::-1])
sorted_sids = sid_paths[sort_idx]

# 2. diff-scan 识别唯一前缀
diff = sorted_sids[1:] != sorted_sids[:-1]
first_diff = diff.argmax(axis=1)
is_new[1:, depth] = first_diff <= depth

# 3. 分配全局 State ID (cumulative)
state_ids[mask, depth] = arange(cur, cur + n_new)
state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])

# 4. 收集边 → CSR packing
counts = np.bincount(parents, minlength=num_states)
indptr[1:] = np.cumsum(counts)
packed_csr = np.vstack([tokens, children]).T

# 5. 合成 Dense 查表
dense_mask[t0, t1] = True
dense_states[t0, t1] = state_ids[:, d_dense - 1]
```

## 文件结构

```
index/torch_recall/recall_method/generative/
├── __init__.py
├── recall.py          # GenerativeRecall(RecallOp) — beam search
├── builder.py         # GenerativeBuilder — vectorized NumPy index construction
├── trie.py            # Trie — Dense/CSR hybrid with propagation
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
3. 其 `query` 输入占用 `total_query_dim` 中的 D 维度（用户表征维度），通过 `query_offset` 切片

## 测试计划

1. **Builder 单测**: 构建 trie，验证节点数、CSR 结构、leaf 映射正确性
2. **Trie 传播单测**: 给定 item_mask，验证 `propagate_validity` 结果正确
3. **Forward 单测**: 用 MockDecoder + 小 trie，验证 beam search 输出 shape 和分数范围
4. **Batch 测试**: B > 1 时结果与逐条一致
5. **Pipeline 组合测试**: `Or(Generative, KNN)` 正确工作
6. **Export 测试**: `torch.export` + `.pt2` 导出成功

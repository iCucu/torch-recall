# 生成式召回 (Generative Recall)

> 基于 Trie 约束 Beam Search 的生成式召回方法

## 目录

- [1. 问题定义](#1-问题定义)
- [2. 核心思路](#2-核心思路)
- [3. 概念与术语](#3-概念与术语)
- [4. 整体架构](#4-整体架构)
- [5. 数据模型](#5-数据模型)
- [6. Trie 数据结构](#6-trie-数据结构)
- [7. 离线构建](#7-离线构建)
- [8. 在线推理](#8-在线推理)
- [9. 与 Pipeline 集成](#9-与-pipeline-集成)
- [10. 代码结构](#10-代码结构)
- [11. 使用示例](#11-使用示例)
- [12. 内存估算](#12-内存估算)
- [13. 性能分析](#13-性能分析)
- [14. 参考资料](#14-参考资料)
- **[端到端示例 (walkthrough.md)](walkthrough.md)** — 用 7 个 item 完整走一遍构建和推理

---

## 1. 问题定义

**任务**：给定用户的属性和表征向量，从 N 个 item 中召回与用户最相关的 top-K 个。

传统做法（KNN）将 item 映射为 embedding，用向量相似度检索。生成式召回换了一种思路：将 item 编码为**语义 ID 路径**（Semantic ID path），用自回归 decoder 逐步"生成"出 item 的语义编码，再反解回 item。

**为什么需要生成式召回？**

| 方法 | 原理 | 适用场景 | 局限 |
|------|------|---------|------|
| Targeting | 布尔规则匹配 | 精确定向过滤 | 无法排序 |
| KNN | embedding 内积 | 语义相似度检索 | 表达能力受限于静态 embedding |
| **Generative (AR)** | Trie 约束自回归解码 | 复杂交互建模 | 需要训练 decoder；生成顺序固定 |
| **Generative (Diffusion)** | 路径位图约束并行解码 | 与 AR 互补，生成顺序自适应 | 需要训练双向 decoder |

生成式召回通过自回归模型捕捉用户与 item 之间的复杂交互关系，能力上限高于固定 embedding 的内积匹配。

---

## 2. 核心思路

```
离线:
  1. 每个 item 被聚类/编码为一条 sid_path: [s0, s1, s2, s3, s4]
     其中每个 si ∈ [0, 2048)
  2. 所有 item 的 sid_path 构建成一棵 Trie (前缀树)

在线 (每次请求):
  3. Targeting 过滤 → item_mask → 向上传播到 Trie 得到 node_valid
  4. Decoder 自回归生成 5 个 token (sid)
     每步受 Trie 结构 + node_valid 双重约束
     使用 beam search 保留 top-beam 条路径
  5. 最终 beam 的叶子节点 → 反解为 item → 输出 [B, N] 分数
```

关键约束：整个过程必须是**纯 tensor 操作**（无 Python 循环在数据维度上、无动态形状），以兼容 `torch.export` 导出为 `.pt2` 模型。

---

## 3. 概念与术语

### Semantic ID Path (sid_path)

每个 item 拥有一条长度为 L（默认 5）的语义 ID 路径。例如：

```
item "北京烤鸭": sid_path = [0, 1, 2, 3, 4]
item "全聚德":    sid_path = [0, 1, 2, 3, 4]   ← 与"北京烤鸭"共享路径
item "连锁火锅": sid_path = [0, 2, 8, 9, 10]
```

sid_path 通常由离线聚类（如 RQ-VAE、hierarchical k-means）生成。路径的每一层编码了不同粒度的语义：

```
depth 0: 大类    (如"食品" vs "服务")
depth 1: 中类    (如"中餐" vs "西餐")
depth 2: 小类
depth 3: 子类
depth 4: 叶子    (最细粒度)
```

多个 item 可共享同一 sid_path（多对一关系）。

### 词表 (Vocabulary)

每层 sid 的取值范围为 `[0, V)`，默认 `V = 2048`。

### Trie (前缀树)

所有 item 的 sid_path 构成一棵 Trie。相同前缀的路径共享节点：

```
                    root
                   /    \
                 0       1
                / \       \
              1    2       3
             / \    \       \
            2   5    8      12
            |   |    |       \
            3   6    9       ...
            |   |   / \
            4   7  10  11
```

Trie 的每个节点有一个全局唯一的 **State ID**（整数），用于在 tensor 中索引。

### Beam Search (束搜索)

从 root 开始，每步选择 top-beam 个最优的部分路径，逐步扩展至完整路径。

```
Step 0: root → 选 beam 个最优的 depth-0 token
Step 1: 对每个 beam，选 beam 个最优的 depth-1 token (共 beam×V 候选，保留 beam 个)
...
Step 4: 到达叶子，每个 beam 路径对应一个或多个 item
```

### Decoder (解码器)

外部传入的 `nn.Module`，自回归地输出 logits：

```
输入: (user_repr [B*beam, D], partial_paths [B*beam, L])
输出: next_logits [B*beam, V]
```

Decoder 根据用户表征和已选的前缀，预测下一层 sid 的概率分布。本项目提供 `MockDecoder`（随机线性映射）用于测试。

---

## 4. 整体架构

```
GenerativeRecall(RecallOp)
│
├── targeting: TargetingRecall
│     输入: pred_satisfied [B, P] → 输出: item_mask [B, N]
│
├── trie: Trie (nn.Module)
│     结构数据 (registered buffers):
│     ├── start_mask     [V]           首 token 合法性
│     ├── dense_mask     [V, V]        L0→L1 的密集查表
│     ├── dense_states   [V, V]        L0→L1 的 state 查表
│     ├── packed_csr     [E+V, 2]      全量 CSR 边表
│     ├── csr_indptr     [S+2]         CSR 行指针
│     ├── leaf_item_ids  [nL, M]       叶子→item 映射
│     └── leaf_item_valid[nL, M]       叶子→item 有效性
│
│     方法:
│     ├── propagate_validity(item_mask) → node_valid [B, S]
│     └── resolve_leaves(states, scores) → [B, N]
│
└── decoder: nn.Module (外部传入)
      forward(user_repr, partial_paths) → logits [B*beam, V]
```

`GenerativeRecall` 是 `RecallOp` 的子类，`forward` 签名统一：

```python
forward(pred_satisfied: [B, P] bool, query: [B, D] float) -> [B, N] float
```

---

## 5. 数据模型

### Item 扩展

`Item` dataclass 增加了 `sid_path` 字段：

```python
@dataclass
class Item:
    id: str | None = None
    targeting_rule: str | None = None
    embedding: list[float] | None = None
    sid_path: list[int] | None = None      # 长度 L，每个值 ∈ [0, V)
```

item 只需填充它参与的召回方法所需的字段。例如纯 Generative 召回只需 `targeting_rule` + `sid_path`。

### Decoder 接口协议

```python
class MyDecoder(nn.Module):
    def forward(
        self,
        user_repr: torch.Tensor,      # [B * beam, D]   用户表征
        partial_paths: torch.Tensor,   # [B * beam, L]   已选 sid 前缀 (0 填充)
    ) -> torch.Tensor:                 # [B * beam, V]   下一步 logits
```

- `user_repr` 来自 pipeline 的 `query` 输入（通过 `query_offset` 切片）
- `partial_paths` 固定长度 L，未到达的位置填 0
- 返回值是原始 logits（GenerativeRecall 内部会做 `log_softmax`）

---

## 6. Trie 数据结构

### 6.1 设计思路

Trie 需要支持两类操作：

1. **Beam Search 查询**：给定当前 beam 的 state，快速获取其所有合法子节点
2. **Validity Propagation**：给定叶子的有效性，向上传播到每个节点

传统 Trie 用指针/字典实现，无法 tensor 化。我们采用 **Dense/CSR 混合表示**，灵感来自 [STATIC](https://arxiv.org/abs/2602.22647) (Su et al., 2026)。

### 6.2 全局 State ID

所有 Trie 节点共享一个连续的整数 ID 空间：

```
State 0         → 占位符 (padding / OOB 安全)
State 1 … V    → L0 节点 (depth-0，共 V 个槽位，state = token + 1)
State V+1 …    → L1 节点
State ...       → L2, L3 节点
最后一段         → L4 叶子节点 (depth L-1)
```

每层的起始位置和节点数由 `level_start` 和 `level_count` 记录：

```python
level_start = [1, 2049, 2053, 2060, ...]
level_count = [2048, 4, 7, ...]
```

优势：
- 所有节点用单个 `[B, num_states]` 的 `node_valid` 张量表示有效性
- 无 `isinstance` 类型判断，`torch.export` 友好
- CSR 统一编码所有层的边，propagation 和 beam search 共用

### 6.3 Dense 层

Trie 的前 `d_dense` 层（默认 2 层）使用**密集查表**加速：

```python
start_mask:   [V] bool        # L0: 哪些首 token 在 Trie 中存在
dense_mask:   [V, V] bool     # L0→L1: dense_mask[t0, t1] 表示路径 (t0, t1) 是否存在
dense_states: [V, V] int64    # L0→L1: dense_states[t0, t1] 是对应节点的 state ID
```

dense 层是 CSR 的**冗余加速**。CSR 包含全量边（含 dense 层），dense 表只是为了 beam search 时 O(1) 查表。

对于 L0→L1：给定 beam 中一个 state 在 L0 层（state = token + 1），恢复 token 后直接用 `dense_mask[token]` 获取 V 维的合法性向量，无需遍历子节点。

内存开销恒定：`V × V × (1 + 8) ≈ 36 MB`（V=2048），不随 item 数增长。

### 6.4 CSR 稀疏层

深层（L2 往后）节点多但每个节点分支少，用 **Compressed Sparse Row (CSR)** 格式紧凑存储所有边：

```python
packed_csr: [num_edges + V, 2] int64
#   每行 = [token_id, child_state_id]
#   末尾 V 行是 OOB 安全填充 (token=V, state=0)

csr_indptr: [num_states + 2] int64
#   state s 的子节点: packed_csr[indptr[s] : indptr[s+1]]
```

**CSR 工作原理图解：**

```
假设 Trie 有以下边:
  state 1 → (token=1, child=2049), (token=2, child=2050)
  state 2 → (token=3, child=2051)
  state 3 → (无子节点)

packed_csr:
  idx: | 0       | 1       | 2       | ...
  val: | [1,2049]| [2,2050]| [3,2051]| ...

csr_indptr:
       state: |  0  |  1  |  2  |  3  |  4  | ...
       value: |  0  |  0  |  2  |  3  |  3  | ...
                     ↑         ↑         ↑
                state 1 的边: idx 0..1   state 2 的边: idx 2..2
                                         state 3 的边: idx 3..3 (空)
```

查询 state `s` 的所有子节点：

```python
start = csr_indptr[s]
end   = csr_indptr[s + 1]
children = packed_csr[start:end]  # 每行 [token, child_state]
```

beam search 中的**批量查询**（burst-read）：

```python
starts = csr_indptr[flat_states]               # [B*beam]
offsets = torch.arange(max_branch_factor)       # [K]
gather_idx = starts[:, None] + offsets[None, :] # [B*beam, K]
gathered = packed_csr[gather_idx.clamp(...)]    # [B*beam, K, 2]
```

### 6.5 Leaf → Item 映射

叶子节点到 item 的映射：

```python
leaf_item_ids:   [num_leaves, max_items_per_leaf] int64
leaf_item_valid: [num_leaves, max_items_per_leaf] bool
```

多个 item 可共享同一 sid_path，因此同一叶子可关联多个 item。`max_items_per_leaf` 由实际数据决定，不足的位置用 `valid=False` 标记。

---

## 7. 离线构建

`GenerativeBuilder` 负责离线构建 Trie 和 `GenerativeRecall` 模型。

### 7.1 构建流程

```
items (list[Item])
    ↓
1. 校验 sid_path (长度、值域)
    ↓
2. 构建 Targeting 模型 (复用 TargetingBuilder)
    ↓
3. 构建 Trie 索引 (_build_static_index)
    ↓
4. 构建 leaf→item 映射 (_build_leaf_mapping)
    ↓
5. 组装 Trie + GenerativeRecall
    ↓
(model, meta)
```

### 7.2 向量化 Trie 构建

核心函数 `_build_static_index` 使用纯 NumPy 向量化操作，百万级 item 毫秒级完成。

**Step 1: 字典序排序**

```python
sort_idx = np.lexsort(sid_paths.T[::-1])
sorted_sids = sid_paths[sort_idx]
```

排序后相同前缀的路径相邻排列，为后续 diff-scan 做准备。

**Step 2: diff-scan 识别唯一前缀**

```python
# 比较相邻行，找到每行首个不同的 depth
diff = sorted_sids[1:] != sorted_sids[:-1]
first_diff = diff.argmax(axis=1)

# is_new[i, d] = True 表示第 i 行在 depth d 产生了新前缀
for depth in range(L):
    is_new[1:, depth] = (first_diff <= depth)
```

示例：

```
sorted_sids:           is_new:
[0, 1, 2, 3, 4]       [T, T, T, T, T]   (第一行全新)
[0, 1, 2, 3, 4]       [F, F, F, F, F]   (完全相同)
[0, 1, 5, 6, 7]       [F, F, T, T, T]   (depth 2 开始不同)
[0, 2, 8, 9, 10]      [F, T, T, T, T]   (depth 1 开始不同)
[0, 2, 8, 9, 11]      [F, F, F, F, T]   (depth 4 不同)
```

**Step 3: 分配全局 State ID**

```python
# L0: state = token + 1 (固定映射)
state_ids[:, 0] = sorted_sids[:, 0] + 1

# L1 起: 只有 is_new 的行分配新 ID，其余继承前一行
for depth in range(1, L):
    state_ids[is_new[:, depth], depth] = arange(cur, cur + n_new)
    state_ids[:, depth] = np.maximum.accumulate(state_ids[:, depth])
    cur += n_new
```

`np.maximum.accumulate` 用于"填充"：非新前缀的行继承上方最近一个新前缀的 state ID。

**Step 4: 收集边 → CSR 压缩**

```python
# 每个新前缀产生一条 parent→child 边
for depth in range(1, L):
    parents.append(state_ids[is_new[:, depth], depth - 1])
    tokens.append(sorted_sids[is_new[:, depth], depth])
    children.append(state_ids[is_new[:, depth], depth])

# CSR: bincount → cumsum → 排列
counts = np.bincount(parents, minlength=num_states)
indptr[1:] = np.cumsum(counts)
```

**Step 5: 合成 Dense 表**

```python
dense_mask[t0, t1] = True
dense_states[t0, t1] = state_ids[:, d_dense - 1]
```

### 7.3 完整构建示例

详见 [walkthrough.md 第 2 节](walkthrough.md#2-离线构建)，以 7 个 item 为例完整展示：排序 → diff-scan → State ID 分配 → CSR 压缩 → Dense 表合成 → Leaf 映射，以及每个向量的最终内容。

### 7.4 构建输出

- **model** (`GenerativeRecall`): 可直接 `forward()` 的 `nn.Module`
- **meta** (`dict`): 元信息，用于在线编码（targeting 谓词映射等）

---

## 8. 在线推理

### 8.1 Forward 四阶段

```python
def forward(pred_satisfied: [B, P], query: [B, D]) -> [B, N]:
```

#### Phase 1: Targeting Mask

```python
targeting_scores = self.targeting(pred_satisfied, query)  # [B, N]
item_mask = targeting_scores > float("-inf")               # [B, N] bool
```

利用已有的 TargetingRecall 模块，判断每个 item 对每个用户是否通过定向过滤。

#### Phase 2: Validity Propagation

```python
node_valid = self.trie.propagate_validity(item_mask)      # [B, num_states] bool
```

从叶子到根，逐层向上传播有效性。如果一个节点的子树下存在任何一个合法 item，该节点就是 valid。

传播过程：

```
1. leaf_valid = item_mask → leaf_item_ids 的 gather → any
   "叶子节点关联的 item 中，是否有至少一个通过了 targeting？"

2. for depth = L-2 → 0:
     对当前层每个节点，通过 CSR 查到其子节点
     node_valid[parent] = any(node_valid[children])
     "至少一个子节点是 valid → 父节点也 valid"
```

**为什么需要 propagation？**

因为 beam search 是从 root 向下逐层推进的。如果不传播，beam 可能选择一条看似高分的路径，但到叶子时发现所有对应 item 都被 targeting 过滤了，浪费了 beam 预算。提前传播确保 beam 只走有效路径。

### 8.2 推理示例

详见 [walkthrough.md 第 3 节](walkthrough.md#3-在线推理)，沿用 7.3 的 Trie（7 个 item, beam_width=3），以用户 `{city: "北京", age: 25}` 为例完整展示：
- Targeting 过滤掉 item c, f, g
- Validity Propagation 剪掉 token=1 的整棵子树和 item c 的分支
- 5 步 Beam Search（Step 0 首 token → Step 1 Dense → Steps 2~4 CSR）
- Resolve Leaves 映射回 item，最终结果 item d > item e > item a/b

### 8.3 推理逻辑详解

#### Phase 3: Beam Search

beam search 从 depth 0 执行到 depth L-1，每步：

1. 调用 decoder 获取 logits，经 `log_softmax` 归一化
2. 用 Trie 结构 + `node_valid` 生成 mask
3. 将 mask 外的 logits 设为 `-inf`
4. 累加 beam score + 当前 step log-prob
5. 在 `beam × candidates` 中取 top-beam

**Step 0: 首 token**

```python
logits = decoder(query, zeros_paths)        # [B, V]
lp = log_softmax(logits, dim=-1)

valid = start_mask & node_valid[:, 1:V+1]   # [B, V]
lp = lp.masked_fill(~valid, -inf)

top_lp, top_tok = lp.topk(beam)             # [B, beam]
beam_states = top_tok + 1                    # L0 state IDs
```

在 V 个候选中，只保留 Trie 中存在 **且** 子树下有合法 item 的 token。

**Dense 步 (step < d_dense, 默认 step=1)**

```python
# 查 dense 表: [B*beam] → [B*beam, V]
masks = dense_mask[parent_tok]                # Trie 中是否存在
st_table = dense_states[parent_tok]           # 对应的 child state ID
nv = node_valid[batch_idx, st_table]          # child state 是否 valid

combined = masks & nv
lp = lp.masked_fill(~combined, -inf)

# B 个 batch × beam 个 beam × V 个候选 → topk(beam)
cand = beam_scores[:, :, None] + lp.view(B, beam, V)
top_sc, top_flat = cand.view(B, -1).topk(beam)
```

dense 表将 O(1) 查表替代了 CSR 的间接寻址。

**CSR 步 (step >= d_dense)**

```python
# CSR burst-read: 读取每个 beam state 的子节点
starts = csr_indptr[flat_states]
offsets = arange(max_branch_factor)
gather_idx = (starts[:, None] + offsets[None, :]).clamp(...)
gathered = packed_csr[gather_idx]             # [B*beam, K, 2]
cand_tokens = gathered[..., 0]
cand_states = gathered[..., 1]

# 结构性掩码 + 有效性掩码
struct_ok = offsets < actual_lens
cand_nv = node_valid[batch_idx, cand_states]
valid = struct_ok & cand_nv

# 从 decoder logits 中 gather 对应 token 的 log-prob
cand_lp = full_logprobs.gather(1, cand_tokens).masked_fill(~valid, -inf)

# B × beam × K 候选 → topk(beam)
```

CSR 步只考虑 K 个实际子节点（K = 该层最大分支因子），远小于 V=2048。

**Beam 重排**

每步 topk 后需要重排 `beam_paths`，使用 `_gather_beams` 工具函数：

```python
def _gather_beams(x: [B, old_beam, ...], beam_idx: [B, new_beam]) -> [B, new_beam, ...]:
    # 用 gather 高效重排
```

#### Phase 4: Resolve Leaves

```python
# beam_states 此时是叶子节点的 state ID
local = beam_states - leaf_start                     # → leaf 的 local index
items = leaf_item_ids[local]                          # [B, beam, max_items]
valid = leaf_item_valid[local]                        # [B, beam, max_items]

# 展开 beam scores 到 items
scores_exp = beam_scores[:, :, None].expand_as(items)
scores_exp = scores_exp.masked_fill(~valid, -inf)

# scatter_reduce: 同一 item 取最大分数
output = full((B, N), -inf)
output.scatter_reduce_(1, flat_items, flat_scores, reduce="amax")
```

如果多条 beam 路径到达同一个叶子（或不同叶子指向同一 item），取分数最大的。

### 8.4 Beam Score 语义

beam score 是**累积 log-probability**：

```
score(path) = log_softmax(logits_0)[t0]
            + log_softmax(logits_1)[t1]
            + ...
            + log_softmax(logits_4)[t4]
```

使用 `log_softmax` 而非原始 logits 的好处：
- 分数有明确的概率解释
- 不同步数的分数可比较
- 数值稳定性更好（避免大 logit 值溢出）

---

## 9. 与 Pipeline 集成

### 9.1 Spec 声明

```python
from torch_recall.scheduler import Generative, Or, KNN, PipelineBuilder

# 独立使用
spec = Generative(schema, decoder, beam_width=10, dense_levels=2)

# 与 KNN 取并集
spec = Or(
    Generative(schema, decoder, beam_width=10),
    KNN(metric="cosine"),
)
```

### 9.2 Pipeline 编译

`PipelineBuilder.build()` 将 spec 树编译为 `nn.Module` 树：

```
RecallPipeline
└── root: OrModule
    ├── GenerativeRecall (query_offset=0, user_dim=D)
    │   ├── targeting
    │   ├── trie
    │   └── decoder
    └── KNNRecall (query_offset=D, dim=D2)
```

`GenerativeRecall` 通过 `query_offset` 从 pipeline 的 `query` tensor 中切出自己需要的 `user_repr` 部分。

### 9.3 分数组合

| 组合 | 语义 | 效果 |
|------|------|------|
| `Or(Generative, KNN)` | `max(gen_score, knn_score)` | 两种召回的并集 |
| `And(Generative, Targeting)` | `gen_score + tgt_score` | Generative 结果再做定向过滤 |

最终由 `RecallPipeline` 做 `topk(K)`，输出 `(scores [B, K], indices [B, K])`。

---

## 10. 代码结构

```
index/torch_recall/recall_method/
├── autoregressive/              ← 自回归生成式召回（Trie 约束）
│   ├── __init__.py              导出 GenerativeRecall, GenerativeBuilder, MockDecoder, Trie
│   ├── trie.py                  Trie: Dense/CSR 混合前缀树，propagation + resolve
│   ├── builder.py               GenerativeBuilder: 向量化 NumPy 构建
│   ├── recall.py                GenerativeRecall(RecallOp): beam search 主逻辑
│   └── decoder.py               MockDecoder: 测试用 decoder
└── diffusion/                   ← 离散扩散生成式召回（路径位图约束）
    ├── __init__.py              导出 DiffusionRecall, DiffusionBuilder, MockDiffusionDecoder, SidPathFilter
    ├── path_filter.py           SidPathFilter: 路径位图，init_beam_mask / collect_valid_tokens / filter_beam_paths
    ├── builder.py               DiffusionBuilder: 从 sid_path 直接构造 path_vectors，无需 Trie
    ├── recall.py                DiffusionRecall(RecallOp): 自适应顺序解码主逻辑
    └── model.py                 MockDiffusionDecoder: 测试用双向 decoder
```

两个子目录共用：

- `recall_method/base.py` — `RecallOp` 基类（统一 `forward` 接口）
- `recall_method/targeting/` — `TargetingRecall` + `TargetingBuilder`（两者都在 Phase 1 调用）

### 各文件职责

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `autoregressive/trie.py` | `Trie` | 存储 Trie 结构（buffers），提供 `propagate_validity` 和 `resolve_leaves` |
| `autoregressive/builder.py` | `GenerativeBuilder` | 离线构建入口，校验 items，调用 `_build_static_index` |
| `autoregressive/builder.py` | `_build_static_index` | 纯 NumPy 向量化 Trie 构建（排序、diff-scan、CSR 压缩） |
| `autoregressive/builder.py` | `_build_leaf_mapping` | 叶子 state → item index 的映射表 |
| `autoregressive/recall.py` | `GenerativeRecall` | `RecallOp` 实现：targeting + propagation + beam search + resolve |
| `autoregressive/recall.py` | `_gather_beams` | beam 重排工具函数 |
| `autoregressive/decoder.py` | `MockDecoder` | 简单线性映射，用于测试 |
| `diffusion/path_filter.py` | `SidPathFilter` | 路径位图核心，3 个操作：`init_beam_mask` / `collect_valid_tokens` / `filter_beam_paths` |
| `diffusion/builder.py` | `DiffusionBuilder` | 离线构建：`item.sid_path` → `path_vectors [N,L]`，无需 Trie |
| `diffusion/recall.py` | `DiffusionRecall` | `RecallOp` 实现：targeting + 自适应顺序 L 步解码 + beam_mask resolve |
| `diffusion/model.py` | `MockDiffusionDecoder` | 并行输出 `[B_beam, L, V]` logits，用于测试 |

---

## 11. 使用示例

### 11.1 最小示例

```python
from torch_recall.schema import Schema, Item
from torch_recall.scheduler import Generative, PipelineBuilder, encode_pipeline_inputs
from torch_recall.recall_method.autoregressive.decoder import MockDecoder

# 1. 定义 schema
schema = Schema(discrete_fields=["city"], numeric_fields=["age"])

# 2. 准备 items
items = [
    Item(id="北京烤鸭", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="全聚德",    targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="上海小笼包", targeting_rule='city == "上海"', sid_path=[0, 1, 5, 6, 7]),
    Item(id="连锁火锅",  targeting_rule="age > 18",      sid_path=[0, 2, 8, 9, 10]),
    Item(id="学生套餐",  targeting_rule="age > 10",      sid_path=[0, 2, 8, 9, 11]),
]

# 3. 创建 decoder (真实场景替换为训练好的模型)
decoder = MockDecoder(user_dim=32)

# 4. 构建 pipeline
spec = Generative(schema, decoder, beam_width=4, dense_levels=2)
builder = PipelineBuilder(spec, k=3)
pipeline, meta = builder.build(items)
pipeline.eval()

# 5. 在线查询
from torch_recall.recall_method.targeting.encoder import encode_user
import torch

gen_meta = meta["generative_leaves"][0]
pred = encode_user({"city": "北京", "age": 25}, gen_meta["targeting"]).unsqueeze(0)
query = torch.randn(1, meta["total_query_dim"])

with torch.no_grad():
    top_scores, top_indices = pipeline(pred, query)

# top_indices[0] → tensor([0, 1, 4]) 之类的结果
item_ids = meta["item_ids"]
for i in range(3):
    idx = top_indices[0, i].item()
    score = top_scores[0, i].item()
    print(f"#{i+1} {item_ids[idx]}  score={score:.4f}")
```

### 11.2 与 KNN 组合

```python
spec = Or(
    Generative(schema, decoder, beam_width=4),
    KNN(metric="cosine"),
)
builder = PipelineBuilder(spec, k=5)
pipeline, meta = builder.build(items_with_embeddings)
```

### 11.3 导出 .pt2

```python
from torch_recall.scheduler.exporter import export_recall_model
export_recall_model(pipeline, "generative_pipeline.pt2")
```

完整示例见 [examples/05_generative_recall.py](../../examples/05_generative_recall.py)。

---

## 12. 内存估算

### Trie 结构

| 组件 | 大小公式 | N=1M | N=10M |
|------|---------|------|-------|
| `start_mask` [V] | V bytes | 2 KB | 2 KB |
| `dense_mask` [V, V] | V² bytes | 4 MB | 4 MB |
| `dense_states` [V, V] | V² × 8 bytes | 32 MB | 32 MB |
| `packed_csr` [E+V, 2] | (E+V) × 16 bytes | ~30 MB | ~200 MB |
| `csr_indptr` [S+2] | (S+2) × 8 bytes | ~8 MB | ~40 MB |
| `leaf_item_ids` [nL, M] | nL × M × 8 bytes | ~15 MB | ~130 MB |
| `leaf_item_valid` [nL, M] | nL × M bytes | ~2 MB | ~16 MB |
| **合计** | | **~91 MB** | **~422 MB** |

其中 E = 总边数（约等于总节点数），S = 总 state 数，nL = 叶子数，M = 每叶最大 item 数。

### 运行时

| 组件 | 大小 |
|------|------|
| `node_valid` [B, S] | B × S bytes |
| beam tensors | B × beam × L × 8 bytes |
| decoder 中间状态 | 取决于 decoder 架构 |

---

## 13. 性能分析

### 时间复杂度

每次请求的计算：

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| Targeting | O(B × P × C) | P=谓词数, C=conjunction数 |
| Propagation | O(B × S) | S=总节点数，每节点 CSR 读 + reduce |
| Beam Search | O(L × B × beam × decoder_cost) | L=5步，每步调 decoder + topk |
| Resolve | O(B × beam × M) | M=每叶最大 item 数 |

**瓶颈在 Decoder**：beam search 每步调用一次 decoder，decoder（通常是 Transformer）的前向传播占总时间 80-90%。Trie 操作（gather、mask、CSR 读取）开销极小。

### 优化方向

1. **Decoder 优化**：量化（INT8/FP16）、FlashAttention、KV-cache
2. **算子融合**：用 Triton 将 `gather + mask + log_softmax` 融合为单个 kernel
3. **减小 beam**：beam_width 直接影响 decoder 调用的 batch size
4. **增大 d_dense**：更多层用 dense 表（内存换速度）

---

## 14. 参考资料

1. **STATIC**: Su et al., "Sparse Transition-Accelerated Trie Index for Constrained Decoding", arXiv:2602.22647, 2026.
   - CSR 表示、全局 state ID、向量化构建方法的灵感来源
   - https://github.com/youtube/static-constraint-decoding

2. **DSI**: Tay et al., "Transformer Memory as a Differentiable Search Index", NeurIPS 2022.
   - 生成式检索的开创性工作，将 doc ID 作为 target sequence

3. **TIGER**: Rajput et al., "Recommender Systems with Generative Retrieval", NeurIPS 2023.
   - 将生成式检索应用于推荐系统，使用 semantic ID

4. **RQ-VAE**: Lee et al., "Autoregressive Image Generation using Residual Quantization", CVPR 2022.
   - 一种生成 semantic ID (codebook path) 的方法

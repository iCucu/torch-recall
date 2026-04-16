# 生成式召回 — 端到端示例

> 用 7 个 item 完整走一遍离线构建和在线推理，展示每个向量的具体内容。

本文档是 [README.md](README.md) 的配套示例。参数：V=2048, L=5, d_dense=2, beam_width=3。

---

## 1. 输入数据

7 个 item，使用了**两个不同的首 token（0 和 1）**，以体现 L0 层如何预分配 V 个 state 槽位。

```
item a: sid_path = [0, 1, 2,  3,  4]      targeting_rule: city == "北京"
item b: sid_path = [0, 1, 2,  3,  4]      targeting_rule: city == "北京"  ← 和 a 共享路径
item c: sid_path = [0, 1, 5,  6,  7]      targeting_rule: city == "上海"
item d: sid_path = [0, 2, 8,  9,  10]     targeting_rule: age > 18
item e: sid_path = [0, 2, 8,  9,  11]     targeting_rule: city == "北京"
item f: sid_path = [1, 3, 12, 13, 14]     targeting_rule: city == "广州"
item g: sid_path = [1, 4, 17, 18, 19]     targeting_rule: city == "深圳"
```

---

## Tensor 速查

构建完成后 `Trie` 中存储了以下 tensor（registered buffer），推理时全部参与计算：

| Tensor | 形状 | 含义 | 用在哪 |
|--------|------|------|--------|
| `start_mask` | [V] bool | 哪些首 token 在 Trie 中存在。`start_mask[t]=True` 表示至少有一个 item 以 token `t` 开头 | Beam Search Step 0：过滤首 token |
| `dense_mask` | [V, V] bool | 前 d_dense 层的密集查表。`dense_mask[t0, t1]=True` 表示前缀 `(t0, t1)` 在 Trie 中存在 | Beam Search Dense 步：O(1) 判断子节点是否存在 |
| `dense_states` | [V, V] int64 | 前 d_dense 层的 state 查表。`dense_states[t0, t1]` 是前缀 `(t0, t1)` 对应的 L1 节点 state ID | Beam Search Dense 步：获取子节点的 state ID |
| `packed_csr` | [E+V, 2] int64 | CSR 格式的全量边表。每行 `[token, child_state]`，末尾 V 行是 OOB 安全填充 | Beam Search CSR 步 + Validity Propagation：读取子节点 |
| `csr_indptr` | [S+2] int64 | CSR 行指针。state `s` 的子节点 = `packed_csr[indptr[s] : indptr[s+1]]` | 配合 `packed_csr` 使用，定位每个 state 的子节点区间 |
| `leaf_item_ids` | [nL, M] int64 | 叶子→item 映射。`leaf_item_ids[leaf_idx]` 是该叶子关联的 item index 列表（padded） | Phase 4 Resolve：叶子 state → item index |
| `leaf_item_valid` | [nL, M] bool | 叶子→item 有效性。`leaf_item_valid[leaf_idx, j]=True` 表示第 j 个 item 是真实的（非 padding） | Phase 4 Resolve：区分真实 item 和 padding |

推理过程中**动态生成**的 tensor：

| Tensor | 形状 | 含义 | 用在哪 |
|--------|------|------|--------|
| `item_mask` | [B, N] bool | 每个用户对每个 item 的 targeting 匹配结果 | Phase 1 → Phase 2 输入 |
| `node_valid` | [B, S] bool | 每个用户视角下每个 trie 节点是否有效（子树下有合法 item） | Phase 2 输出 → Phase 3 每步 mask |
| `beam_states` | [B, beam] int64 | 当前 beam 中每条路径的 trie state ID | Phase 3：跟踪 beam 在 trie 中的位置 |
| `beam_scores` | [B, beam] float | 当前 beam 中每条路径的累积 log-probability | Phase 3：累积分数，用于 topk 排序 |
| `beam_paths` | [B, beam, L] int64 | 当前 beam 中每条路径已选的 token 序列（未到达位置填 0） | Phase 3：传给 decoder 作为 partial_paths |

其中：V=词表大小, L=路径长度, E=总边数, S=总 state 数, nL=叶子数, M=每叶最大 item 数, B=batch size。

---

## 2. 离线构建

### 2.1 排序

`np.lexsort` 按字典序排列。本例已有序：

```
sorted_sids:                sort_idx:
row 0: [0, 1, 2,  3,  4]      0 (item a)
row 1: [0, 1, 2,  3,  4]      1 (item b)
row 2: [0, 1, 5,  6,  7]      2 (item c)
row 3: [0, 2, 8,  9,  10]     3 (item d)
row 4: [0, 2, 8,  9,  11]     4 (item e)
row 5: [1, 3, 12, 13, 14]     5 (item f)
row 6: [1, 4, 17, 18, 19]     6 (item g)
```

### 2.2 diff-scan

比较每行与上一行，找首个不同的 depth：

```
row 0 vs —:     首行，全新
row 1 vs row 0: 完全相同          → first_diff = 5 (无差异)
row 2 vs row 1: depth 2 不同      → first_diff = 2
row 3 vs row 2: depth 1 不同      → first_diff = 1
row 4 vs row 3: depth 4 不同      → first_diff = 4
row 5 vs row 4: depth 0 就不同!   → first_diff = 0  ← 首 token 变了
row 6 vs row 5: depth 1 不同      → first_diff = 1
```

`is_new[i, d] = True` 当 `first_diff <= d`：

```
          d0     d1     d2     d3     d4
row 0:    T      T      T      T      T
row 1:    F      F      F      F      F     ← 完全相同
row 2:    F      F      T      T      T     ← depth 2 起新分支
row 3:    F      T      T      T      T     ← depth 1 起新分支
row 4:    F      F      F      F      T     ← 仅 depth 4 新叶子
row 5:    T      T      T      T      T     ← 首 token 变了，全部都新!
row 6:    F      T      T      T      T     ← depth 1 起新分支
```

### 2.3 分配 State ID

**depth 0**：固定映射 `state = token + 1`

```
item a~e 首 token = 0 → State 1
item f~g 首 token = 1 → State 2

state_ids[:, 0] = [1, 1, 1, 1, 1, 2, 2]
```

L0 预分配 V=2048 个槽位（State 1 到 State 2048），**不管数据只用了 2 个**：

```
State 1    → token 0 (有 item)
State 2    → token 1 (有 item)
State 3    → token 2 (空，无 item 使用此 token)
State 4    → token 3 (空)
...
State 2048 → token 2047 (空)

level_start[0] = 1, level_count[0] = 2048, cur = 2049
```

> **这就是为什么 L1 从 State 2049 开始**——前面 2048 个 state 已被 L0 占用，
> 即使绝大部分是空的。这是 dense 层用空间换 O(1) 查表的代价。

**depth 1**：is_new = [T, F, F, T, F, T, T]，4 个新前缀

```
row 0 → state 2049   (前缀 [0, 1])
row 3 → state 2050   (前缀 [0, 2])
row 5 → state 2051   (前缀 [1, 3])
row 6 → state 2052   (前缀 [1, 4])

accumulate 填充:
state_ids[:, 1] = [2049, 2049, 2049, 2050, 2050, 2051, 2052]

level_start[1] = 2049, level_count[1] = 4, cur = 2053
```

**depth 2**：is_new = [T, F, T, T, F, T, T]，5 个新前缀

```
row 0 → state 2053   (前缀 [0, 1, 2])
row 2 → state 2054   (前缀 [0, 1, 5])
row 3 → state 2055   (前缀 [0, 2, 8])
row 5 → state 2056   (前缀 [1, 3, 12])
row 6 → state 2057   (前缀 [1, 4, 17])

state_ids[:, 2] = [2053, 2053, 2054, 2055, 2055, 2056, 2057]

level_start[2] = 2053, level_count[2] = 5, cur = 2058
```

**depth 3**：5 个新前缀

```
row 0 → state 2058   (前缀 [0, 1, 2, 3])
row 2 → state 2059   (前缀 [0, 1, 5, 6])
row 3 → state 2060   (前缀 [0, 2, 8, 9])
row 5 → state 2061   (前缀 [1, 3, 12, 13])
row 6 → state 2062   (前缀 [1, 4, 17, 18])

state_ids[:, 3] = [2058, 2058, 2059, 2060, 2060, 2061, 2062]

level_start[3] = 2058, level_count[3] = 5, cur = 2063
```

**depth 4 (叶子)**：6 个新叶子

```
row 0 → state 2063   (路径 [0, 1, 2, 3, 4])
row 2 → state 2064   (路径 [0, 1, 5, 6, 7])
row 3 → state 2065   (路径 [0, 2, 8, 9, 10])
row 4 → state 2066   (路径 [0, 2, 8, 9, 11])
row 5 → state 2067   (路径 [1, 3, 12, 13, 14])
row 6 → state 2068   (路径 [1, 4, 17, 18, 19])

state_ids[:, 4] = [2063, 2063, 2064, 2065, 2066, 2067, 2068]
                   ↑a     ↑b 继承

level_start[4] = 2063, level_count[4] = 6, cur = 2069
```

### 2.4 完整 state_ids

```
          d0    d1     d2     d3     d4
row 0(a): 1     2049   2053   2058   2063
row 1(b): 1     2049   2053   2058   2063   ← 和 a 完全一样
row 2(c): 1     2049   2054   2059   2064
row 3(d): 1     2050   2055   2060   2065
row 4(e): 1     2050   2055   2060   2066
row 5(f): 2     2051   2056   2061   2067
row 6(g): 2     2052   2057   2062   2068
```

num_states = **2069**

**State ID 全局布局**：

```
区间                   层       用途
──────────────────────────────────────────
State 0                —       占位符 (padding / OOB)
State 1 … 2048        L0      2048 个槽位 (仅 State 1, 2 有 item)
State 2049 … 2052     L1      4 个节点
State 2053 … 2057     L2      5 个节点
State 2058 … 2062     L3      5 个节点
State 2063 … 2068     L4      6 个叶子
```

### 2.5 Trie 树形图

```
   State 1 (tok=0)        State 2 (tok=1)       States 3…2048 (空)
   /          \             /          \
St 2049    St 2050      St 2051     St 2052
(tok=1)    (tok=2)      (tok=3)     (tok=4)
 /    \       \            \            \
St2053 St2054  St2055    St2056      St2057
(t=2)  (t=5)  (t=8)     (t=12)      (t=17)
 |      |      |           |            |
St2058 St2059  St2060   St2061      St2062
(t=3)  (t=6)  (t=9)    (t=13)      (t=18)
 |      |     /   \        |            |
St2063 St2064 St2065 St2066 St2067   St2068
(t=4)  (t=7) (t=10) (t=11) (t=14)   (t=19)
[a,b]   [c]   [d]    [e]    [f]      [g]
```

### 2.6 packed_csr — 边表

共 20 条边：

```
idx:  [token, child_state]      来源
──────────────────────────────────────────
 0:   [  1,  2049]              State 1 → 2049
 1:   [  2,  2050]              State 1 → 2050
 2:   [  3,  2051]              State 2 → 2051
 3:   [  4,  2052]              State 2 → 2052
 4:   [  2,  2053]              State 2049 → 2053
 5:   [  5,  2054]              State 2049 → 2054
 6:   [  8,  2055]              State 2050 → 2055
 7:   [ 12,  2056]              State 2051 → 2056
 8:   [ 17,  2057]              State 2052 → 2057
 9:   [  3,  2058]              State 2053 → 2058
10:   [  6,  2059]              State 2054 → 2059
11:   [  9,  2060]              State 2055 → 2060
12:   [ 13,  2061]              State 2056 → 2061
13:   [ 18,  2062]              State 2057 → 2062
14:   [  4,  2063]              State 2058 → 2063
15:   [  7,  2064]              State 2059 → 2064
16:   [ 10,  2065]              State 2060 → 2065
17:   [ 11,  2066]              State 2060 → 2066
18:   [ 14,  2067]              State 2061 → 2067
19:   [ 19,  2068]              State 2062 → 2068
20…2067: [2048, 0]              OOB 安全填充 (共 2048 行)
```

### 2.7 csr_indptr — 行指针

```
indptr[0]       = 0
indptr[1]       = 0      ← State 0: 无边
indptr[2]       = 2      ← State 1: idx 0..1  (2 条边)
indptr[3]       = 4      ← State 2: idx 2..3  (2 条边)
indptr[4]       = 4  ┐
indptr[5]       = 4  │
  ...                │   ← States 3 ~ 2048: 全部无边!
indptr[2049]    = 4  ┘     这 2046 个 state 是 L0 的空槽位
indptr[2050]    = 6      ← State 2049: idx 4..5
indptr[2051]    = 7      ← State 2050: idx 6
indptr[2052]    = 8      ← State 2051: idx 7
indptr[2053]    = 9      ← State 2052: idx 8
indptr[2054]    = 10     ← State 2053: idx 9
indptr[2055]    = 11     ← State 2054: idx 10
indptr[2056]    = 12     ← State 2055: idx 11
indptr[2057]    = 13     ← State 2056: idx 12
indptr[2058]    = 14     ← State 2057: idx 13
indptr[2059]    = 15     ← State 2058: idx 14
indptr[2060]    = 16     ← State 2059: idx 15
indptr[2061]    = 18     ← State 2060: idx 16..17 (2 条边!)
indptr[2062]    = 19     ← State 2061: idx 18
indptr[2063]    = 20     ← State 2062: idx 19
indptr[2064..2069] = 20  ← 叶子无出边
indptr[2070]    = 2068   ← OOB 填充末尾
```

> 注意 indptr[4] 到 indptr[2049] 这一大段全是 4——这就是 L0 预分配 2048
> 个 state 的直接体现。虽然只有 State 1 和 State 2 有边，但 State 3~2048
> 也各占了 indptr 中的一个位置（值都相同，不占 packed_csr 空间）。

验证查询 State 2060 的子节点：

```python
start = indptr[2060]      # = 16
end   = indptr[2061]      # = 18
packed_csr[16:18] → [[10, 2065], [11, 2066]]
# ✓ token 10 → State 2065 (item d)
# ✓ token 11 → State 2066 (item e)
```

### 2.8 Dense 表

**start_mask** [2048]：

```
start_mask[0] = True     ← token 0 对应 State 1 (有 item)
start_mask[1] = True     ← token 1 对应 State 2 (有 item)
其余 2046 个 = False
```

**dense_mask** [2048, 2048]（4 个 True）：

```
dense_mask[0, 1] = True    ← 前缀 (0, 1) 存在
dense_mask[0, 2] = True    ← 前缀 (0, 2) 存在
dense_mask[1, 3] = True    ← 前缀 (1, 3) 存在
dense_mask[1, 4] = True    ← 前缀 (1, 4) 存在
```

**dense_states** [2048, 2048]（4 个非零）：

```
dense_states[0, 1] = 2049   ← 前缀 (0, 1) → State 2049
dense_states[0, 2] = 2050   ← 前缀 (0, 2) → State 2050
dense_states[1, 3] = 2051   ← 前缀 (1, 3) → State 2051
dense_states[1, 4] = 2052   ← 前缀 (1, 4) → State 2052
```

### 2.9 Leaf → Item 映射

**leaf_item_ids** [6, 2] 和 **leaf_item_valid** [6, 2]：

```
                ids       valid
leaf 0 (2063):  [0, 1]    [T, T]     ← item a 和 b 共享路径
leaf 1 (2064):  [2, 0]    [T, F]     ← 只有 item c
leaf 2 (2065):  [3, 0]    [T, F]     ← 只有 item d
leaf 3 (2066):  [4, 0]    [T, F]     ← 只有 item e
leaf 4 (2067):  [5, 0]    [T, F]     ← 只有 item f
leaf 5 (2068):  [6, 0]    [T, F]     ← 只有 item g
```

max_items_per_leaf = 2（因为 leaf 0 关联了 2 个 item）。

### 2.10 构建结果汇总

| 向量 | 形状 | 关键内容 |
|------|------|---------|
| `start_mask` | [2048] bool | `[0]` 和 `[1]` 为 True，其余 False |
| `dense_mask` | [2048, 2048] bool | 4 个 True: `[0,1]`, `[0,2]`, `[1,3]`, `[1,4]` |
| `dense_states` | [2048, 2048] int64 | 4 个非零: 对应 State 2049~2052 |
| `packed_csr` | [2068, 2] int64 | 20 条真实边 + 2048 条 OOB 填充 |
| `csr_indptr` | [2071] int64 | States 3~2048 全为同一值（空 L0 槽位） |
| `leaf_item_ids` | [6, 2] int64 | 叶子→item 映射 |
| `leaf_item_valid` | [6, 2] bool | 叶子→item 有效性 |
| `level_start` | [1, 2049, 2053, 2058, 2063] | 各层首 state ID |
| `level_count` | [2048, 4, 5, 5, 6] | 各层节点数（L0 固定 2048） |
| `max_branch_factors` | [2, 2, 2, 1, 2] | 各层最大分支因子 |
| `num_states` | 2069 | 总 state 数 |

---

## 3. 在线推理

### 3.1 场景设定

用户属性：`{city: "北京", age: 25}`

item 的 targeting 匹配结果：

```
item a: city == "北京"         ✓ 匹配
item b: city == "北京"         ✓ 匹配
item c: city == "上海"         ✗ 不匹配
item d: age > 18               ✓ 匹配
item e: city == "北京"         ✓ 匹配
item f: city == "广州"         ✗ 不匹配
item g: city == "深圳"         ✗ 不匹配
```

### 3.2 Phase 1: Targeting Mask

```
targeting_scores = [0.0, 0.0, -inf, 0.0, 0.0, -inf, -inf]
                    a✓   b✓   c✗    d✓   e✓   f✗    g✗

item_mask = [T, T, F, T, T, F, F]
```

### 3.3 Phase 2: Validity Propagation

从叶子向上传播。只要子树下有一个 valid item，节点就 valid。

**叶子层（L4，States 2063~2068）**：

```
leaf 0 (St 2063) → item a✓, b✓ → 有 valid item → T
leaf 1 (St 2064) → item c✗     → 无 valid item → F
leaf 2 (St 2065) → item d✓     → 有 valid item → T
leaf 3 (St 2066) → item e✓     → 有 valid item → T
leaf 4 (St 2067) → item f✗     → 无 valid item → F
leaf 5 (St 2068) → item g✗     → 无 valid item → F
```

**L3 层（States 2058~2062）**，CSR 查子节点：

```
St 2058 → 子: St 2063 (T) → T     (通往 a, b)
St 2059 → 子: St 2064 (F) → F     (通往 c，被过滤)
St 2060 → 子: St 2065 (T), St 2066 (T) → T  (通往 d, e)
St 2061 → 子: St 2067 (F) → F     (通往 f，被过滤)
St 2062 → 子: St 2068 (F) → F     (通往 g，被过滤)
```

**L2 层（States 2053~2057）**：

```
St 2053 → 子: St 2058 (T) → T
St 2054 → 子: St 2059 (F) → F     ← item c 的整条分支被剪掉
St 2055 → 子: St 2060 (T) → T
St 2056 → 子: St 2061 (F) → F     ← item f 的整条分支被剪掉
St 2057 → 子: St 2062 (F) → F     ← item g 的整条分支被剪掉
```

**L1 层（States 2049~2052）**：

```
St 2049 → 子: St 2053 (T), St 2054 (F) → T  (至少一个 valid)
St 2050 → 子: St 2055 (T)              → T
St 2051 → 子: St 2056 (F)              → F   ← token 1 下的整棵子树被剪掉!
St 2052 → 子: St 2057 (F)              → F   ← 同上
```

**L0 层（States 1~2048）**：

```
St 1 (token=0) → 子: St 2049 (T), St 2050 (T) → T
St 2 (token=1) → 子: St 2051 (F), St 2052 (F) → F  ← token=1 整个被剪掉!
St 3 ~ St 2048 → 无子节点 → F
```

**node_valid 最终结果**（只列 True 的 state）：

```
T: 1, 2049, 2050, 2053, 2055, 2058, 2060, 2063, 2065, 2066
F: 其余所有 (包括 State 2 和它下面的整棵子树)
```

对应到 Trie 图（✓ = valid, ✗ = pruned）：

```
   State 1 ✓ (tok=0)      State 2 ✗ (tok=1)
   /          \             /          \
St 2049 ✓  St 2050 ✓    St 2051 ✗   St 2052 ✗
(tok=1)    (tok=2)      (tok=3)     (tok=4)
 /    \       \            \            \
St2053✓ St2054✗ St2055✓  St2056✗     St2057✗
(t=2)  (t=5)   (t=8)    (t=12)      (t=17)
 |      |       |          |            |
St2058✓ St2059✗ St2060✓  St2061✗     St2062✗
(t=3)  (t=6)   (t=9)    (t=13)      (t=18)
 |      |      /   \        |            |
St2063✓ St2064✗ St2065✓ St2066✓ St2067✗ St2068✗
(t=4)  (t=7)  (t=10) (t=11) (t=14)   (t=19)
[a,b]✓  [c]✗   [d]✓   [e]✓   [f]✗     [g]✗
```

> 只剩 3 条有效路径：`[0,1,2,3,4]`（a,b）、`[0,2,8,9,10]`（d）、`[0,2,8,9,11]`（e）。
> 整个 token=1 的子树（item f, g）和 token=0 下 item c 的分支都被剪掉了。

### 3.4 Phase 3: Beam Search (beam_width=3)

假设 decoder 的 `log_softmax` 输出如下（简化为只列相关 token 的值）。

#### Step 0: 首 token

```
decoder logits → log_softmax:
  token 0: -0.5    token 1: -1.2    其余: 各种值

valid = start_mask & node_valid[:, 1:V+1]:
  token 0: T & T = T
  token 1: T & F = F  ← 被 propagation 剪掉!
  token 2~2047: F

mask 后: token 0 = -0.5, 其余全 -inf

topk(3):
  beam 0: token=0, score=-0.5, state=1
  beam 1: score=-inf  (只有 1 个合法 token)
  beam 2: score=-inf
```

实际上只有 1 个合法首 token（token=0），3 个 beam 中只有 beam 0 有效。

```
beam_states = [1, -, -]
beam_scores = [-0.5, -inf, -inf]
beam_paths  = [[0,0,0,0,0], ...]
```

#### Step 1: Dense 步

只有 beam 0 有效（state=1），查 dense 表：

```
parent_tok = state - 1 = 0
dense_mask[0] → V 维向量，[1]=True, [2]=True, 其余 False
dense_states[0] → [1]=2049, [2]=2050

叠加 node_valid:
  token 1 → state 2049 → valid? T → 保留
  token 2 → state 2050 → valid? T → 保留

decoder logits → log_softmax:
  token 1: -0.8    token 2: -0.3

累积分数 = beam_score + log_prob:
  (beam0, token1): -0.5 + (-0.8) = -1.3 → state 2049
  (beam0, token2): -0.5 + (-0.3) = -0.8 → state 2050
  其余全 -inf

topk(3):
  beam 0: tok=2, score=-0.8, state=2050    ← 最优
  beam 1: tok=1, score=-1.3, state=2049
  beam 2: score=-inf
```

```
beam_states = [2050, 2049, -]
beam_scores = [-0.8, -1.3, -inf]
beam_paths  = [[0,2,0,0,0], [0,1,0,0,0], ...]
```

#### Step 2: CSR 步

d_dense=2，从 step 2 开始用 CSR。max_branch_factor = 2。

对 beam 0 (state=2050) 和 beam 1 (state=2049) 做 CSR burst-read：

```
beam 0, state 2050:
  indptr[2050] = 6, indptr[2051] = 7 → packed_csr[6] = [8, 2055]
  子节点: token=8, child=2055
  node_valid[2055] = T → 保留

beam 1, state 2049:
  indptr[2049] = 4, indptr[2050] = 6 → packed_csr[4..5] = [[2,2053],[5,2054]]
  子节点 1: token=2, child=2053, node_valid[2053] = T → 保留
  子节点 2: token=5, child=2054, node_valid[2054] = F → 剪掉!

decoder log_softmax (只列相关 token):
  beam 0: token 8 = -0.4
  beam 1: token 2 = -0.6, token 5 = -0.9

累积分数:
  (beam0, tok=8):  -0.8 + (-0.4) = -1.2 → state 2055
  (beam1, tok=2):  -1.3 + (-0.6) = -1.9 → state 2053
  (beam1, tok=5):  masked → -inf          (node_valid=F)

topk(3):
  beam 0: score=-1.2, state=2055, path=[0,2,8,0,0]
  beam 1: score=-1.9, state=2053, path=[0,1,2,0,0]
  beam 2: score=-inf
```

#### Step 3: CSR 步

max_branch_factor = 1，每个 state 只有 1 个子节点：

```
beam 0, state 2055 → child: token=9, state=2060, valid=T
beam 1, state 2053 → child: token=3, state=2058, valid=T

decoder log_softmax:
  beam 0: token 9 = -0.2
  beam 1: token 3 = -0.5

累积:
  beam 0: -1.2 + (-0.2) = -1.4 → state 2060, path=[0,2,8,9,0]
  beam 1: -1.9 + (-0.5) = -2.4 → state 2058, path=[0,1,2,3,0]
```

#### Step 4: CSR 步（最后一步，到达叶子）

max_branch_factor = 2：

```
beam 0, state 2060 → 子: [10, 2065] 和 [11, 2066], 都 valid
beam 1, state 2058 → 子: [4, 2063], valid

decoder log_softmax:
  beam 0: token 10 = -0.3, token 11 = -0.7
  beam 1: token 4 = -0.4

累积:
  (beam0, tok=10): -1.4 + (-0.3) = -1.7 → state 2065 (leaf, item d)
  (beam0, tok=11): -1.4 + (-0.7) = -2.1 → state 2066 (leaf, item e)
  (beam1, tok=4):  -2.4 + (-0.4) = -2.8 → state 2063 (leaf, item a/b)

topk(3):
  beam 0: score=-1.7, state=2065, path=[0,2,8,9,10]
  beam 1: score=-2.1, state=2066, path=[0,2,8,9,11]
  beam 2: score=-2.8, state=2063, path=[0,1,2,3,4]
```

### 3.5 Phase 4: Resolve Leaves

将 beam 的叶子 state 映射回 item：

```
beam 0: state 2065, local = 2065-2063 = 2 → leaf_item_ids[2] = [3], item d
beam 1: state 2066, local = 2066-2063 = 3 → leaf_item_ids[3] = [4], item e
beam 2: state 2063, local = 2063-2063 = 0 → leaf_item_ids[0] = [0, 1], item a 和 b!
```

scatter_reduce（amax）聚合到 `[1, 7]` 输出：

```
output = [-inf] * 7

beam 2 → item 0 (a): score = -2.8
beam 2 → item 1 (b): score = -2.8    ← 共享叶子，同分
beam 0 → item 3 (d): score = -1.7
beam 1 → item 4 (e): score = -2.1

最终 output:
  item a:  -2.8
  item b:  -2.8
  item c:  -inf    ← targeting 过滤
  item d:  -1.7    ← 最高分!
  item e:  -2.1
  item f:  -inf    ← targeting 过滤
  item g:  -inf    ← targeting 过滤
```

### 3.6 最终结果

Pipeline 做 `topk(K)` 返回排序结果：

```
#1  item d (连锁火锅)   score = -1.7
#2  item e (学生套餐)   score = -2.1
#3  item a (北京烤鸭)   score = -2.8
#4  item b (全聚德)      score = -2.8
#5  item c              score = -inf  (无匹配)
#6  item f              score = -inf  (无匹配)
#7  item g              score = -inf  (无匹配)
```

被 targeting 过滤的 item c, f, g 自始至终不会出现在结果中——它们在 Phase 2 就被从 Trie 中剪掉了，beam search 根本不会探索到它们。

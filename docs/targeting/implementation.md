# 定向召回: 模块实现

本文档详解定向召回 (Targeting Recall) 的各模块实现。

> 框架共享组件参见 [../architecture.md](../architecture.md)，设计原理参见 [design.md](design.md)，端到端示例参见 [walkthrough.md](walkthrough.md)

---

## 整体流程

```
离线 (一次性)                              在线 (每次用户请求)
─────────────                            ─────────────
rules: list[str]                         user_attrs: dict
      │                                        │
      ▼                                        ▼
 TargetingBuilder.build()                encode_user(user_attrs, meta)
      │                                        │
      ├→ 解析规则 → DNF                         ▼
      ├→ 构建谓词注册表                    pred_satisfied: [P] bool
      ├→ 构建 conj-pred / item-conj 张量         │
      │                                        ▼
      ▼                                  model.forward(pred_satisfied)
 TargetingRecall (nn.Module)                   │  ┌────────────────────────┐
      │                                        │  │ gather pred → [C, K]   │
      ▼                                        │  │ XOR negated, OR ~valid │
 torch.export → AOTInductor                    │  │ all(dim=1) → [C] conj  │
      │                                        │  │ gather conj → [N, J]   │
      ▼                                        │  │ AND valid, any(dim=1)  │
 model.pt2 + targeting_meta.json               │  └────────────────────────┘
                                               ▼
                                         item_ok: [N] bool
```

---

## 1. TargetingBuilder：索引构建

**文件**: `index/torch_recall/recall_method/targeting/builder.py`

TargetingBuilder 将 `list[str]` 的定向规则转换为 `TargetingRecall` 模型 + 元数据 JSON。

### 构建流程

```
rules: list[str]
    │
    ├─ Phase 1: 逐条解析 → AST → DNF
    │   ├─ 收集所有原子谓词，去重分配 pred_id
    │   ├─ _register_predicate(): 统一 != → == + negated
    │   └─ 记录每个 item 的 conjunction 列表
    │
    ├─ Phase 2: 构建 conjunction 张量
    │   conj_pred_ids:     [C, K] int64  → 每个 conj 引用的谓词 ID
    │   conj_pred_negated: [C, K] bool   → 是否取反
    │   conj_pred_valid:   [C, K] bool   → 是否有效 (vs padding)
    │
    ├─ Phase 3: 构建 item-conjunction 张量
    │   item_conj_ids:     [N, J] int64  → 每个 item 引用的 conj ID
    │   item_conj_valid:   [N, J] bool   → 是否有效 (vs padding)
    │
    └─ Phase 4: 构建 meta 字典
        ├─ predicate_registry.discrete: {field: {value: pred_id}}
        ├─ predicate_registry.numeric:  [{field, op, value, pred_id}]
        └─ predicate_registry.text:     {field: {term: pred_id}}
```

### 关键设计

**谓词去重**: 不同 item 的规则中出现的相同谓词（如多个 item 都用 `city == "北京"`）共享同一个 pred_id，避免冗余评估。

**`!=` 标准化**: `city != "广州"` 被注册为 `city == "广州"` + negated=True。这样谓词注册表中不需要区分 `==` 和 `!=`，取反在 forward() 中用 XOR 处理。

**Conjunction 扁平化**: 所有 item 的 conjunction 被收集到一个扁平数组 `all_conj_preds[C]`，每个 item 只记录引用的 conjunction 偏移量。这使得 conj_pred_ids 是一个紧凑的 [C, K] 张量。

---

## 2. TargetingRecall：推理模型

**文件**: `index/torch_recall/recall_method/targeting/recall.py`

### 模型状态 (Buffers)

```python
self.conj_pred_ids      # [C, K] int64  → conjunction 引用的谓词 ID
self.conj_pred_negated  # [C, K] bool   → 是否取反
self.conj_pred_valid    # [C, K] bool   → 有效性掩码
self.item_conj_ids      # [N, J] int64  → item 引用的 conjunction ID
self.item_conj_valid    # [N, J] bool   → 有效性掩码
```

### forward() 详解

```python
def forward(self, pred_satisfied: torch.Tensor) -> torch.Tensor:
    # pred_satisfied: [P] bool

    # Step 1: Conjunction 匹配
    sat = pred_satisfied[self.conj_pred_ids]  # [C, K] — gather
    sat = sat ^ self.conj_pred_negated        # 处理 NOT
    sat = sat | ~self.conj_pred_valid         # padding → True (AND 单位元)
    conj_ok = sat.all(dim=1)                  # [C] — AND 规约

    # Step 2: Item 匹配
    item_conj_ok = conj_ok[self.item_conj_ids]  # [N, J] — gather
    item_conj_ok = item_conj_ok & self.item_conj_valid  # padding → False (OR 零元)
    item_ok = item_conj_ok.any(dim=1)           # [N] — OR 规约

    return item_ok
```

**6 行核心代码**，全部是 tensor 操作：

| 操作 | 语义 | 张量形状 |
|------|------|---------|
| `pred_satisfied[conj_pred_ids]` | gather：取出 conj 引用的谓词 | [C,K] |
| `^ conj_pred_negated` | XOR：处理 NOT | [C,K] |
| `\| ~conj_pred_valid` | padding 位设为 True (AND 单位元) | [C,K] |
| `.all(dim=1)` | AND 规约：conj 内所有谓词都满足? | [C] |
| `conj_ok[item_conj_ids]` | gather：取出 item 引用的 conj | [N,J] |
| `& item_conj_valid` | padding 位设为 False (OR 零元) | [N,J] |
| `.any(dim=1)` | OR 规约：任一 conj 满足? | [N] |

**为什么 padding 处理不同**：
- Conjunction 内是 AND → padding 应为 True（不影响 AND 结果）→ `| ~valid`
- Item 内是 OR → padding 应为 False（不影响 OR 结果）→ `& valid`

---

## 3. encode_user：用户属性编码

**文件**: `index/torch_recall/recall_method/targeting/encoder.py`

将用户属性 dict 转换为 `[P] bool` 的 `pred_satisfied` 张量：

```python
def encode_user(user_attrs: dict, meta: dict) -> torch.Tensor:
    pred_satisfied = torch.zeros(P, dtype=torch.bool)

    # 离散: user_attrs["city"] == "北京" → pred_satisfied[pred_id] = True
    # 数值: user_attrs["age"] > 18 → pred_satisfied[pred_id] = True
    # 文本: "游戏" in tokenize(user_attrs["tags"]) → pred_satisfied[pred_id] = True

    return pred_satisfied
```

逻辑简单直接——对照谓词注册表逐个评估。字段缺失时对应位保持 False。

这个编码在 Python 侧完成（非 torch.export 路径），复杂度 O(P)，对于典型场景（几百个谓词）耗时可忽略。

---

## 4. 数据流全景图

```
                           离线构建 (Python)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  rules: list[str]   (每条 item 的定向规则)                            │
│       │                                                             │
│       ▼                                                             │
│  TargetingBuilder.build()                                           │
│       │                                                             │
│       ├── 逐条解析: parse_expr → AST → to_dnf → DNF                  │
│       │                                                             │
│       ├── 构建谓词注册表:                                             │
│       │   pred_id 0: discrete city=="北京"                           │
│       │   pred_id 1: discrete gender=="男"                           │
│       │   pred_id 2: numeric  age > 18                              │
│       │   ...                                                       │
│       │                                                             │
│       ├── 构建 conjunction 张量:                                      │
│       │   conj_pred_ids[C, K], conj_pred_negated[C, K]              │
│       │                                                             │
│       ├── 构建 item-conjunction 张量:                                  │
│       │   item_conj_ids[N, J], item_conj_valid[N, J]                │
│       │                                                             │
│       ▼                                                             │
│  TargetingRecall (nn.Module)                                        │
│       │                                                             │
│       ▼                                                             │
│  torch.export → AOTInductor → model.pt2                             │
│  save_meta → targeting_meta.json                                    │
│                                                                     │
└───────┼─────────────────────────────────────────────────────────────┘
        │  model.pt2 + targeting_meta.json
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       ▼                    用户编码 (Python)                         │
│                                                                     │
│  user_attrs: {"city": "北京", "gender": "男", "age": 30}             │
│       │                                                             │
│       ▼                                                             │
│  encode_user(user_attrs, meta)                                      │
│       │  对照谓词注册表逐个评估:                                      │
│       │  city=="北京" → True                                         │
│       │  gender=="男" → True                                         │
│       │  age > 18    → True                                         │
│       │  ...                                                        │
│       ▼                                                             │
│  pred_satisfied: [P] bool                                           │
│       │                                                             │
│  save_user_tensors → tensors.pt (for C++)                           │
│                                                                     │
└───────┼─────────────────────────────────────────────────────────────┘
        │  tensors.pt
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       ▼                    推理 (Python or C++)                      │
│                                                                     │
│  model.forward(pred_satisfied)                                      │
│       │  ┌────────────────────────────────────────────────┐         │
│       │  │ Step 1: Conjunction 匹配                        │         │
│       │  │   gather pred → [C, K]                         │         │
│       │  │   XOR negated → handle NOT                     │         │
│       │  │   OR ~valid → padding = True                   │         │
│       │  │   all(dim=1) → conj_ok [C]                     │         │
│       │  │                                                │         │
│       │  │ Step 2: Item 匹配                               │         │
│       │  │   gather conj → [N, J]                         │         │
│       │  │   AND valid → padding = False                  │         │
│       │  │   any(dim=1) → item_ok [N]                     │         │
│       │  └────────────────────────────────────────────────┘         │
│       ▼                                                             │
│  item_ok: [N] bool → 匹配的 item                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附: 设计决策一览

| 决策 | 选择 | 理由 |
|------|------|------|
| 匹配方向 | 反向：item 规则 ↔ 用户属性 | 广告定向等场景的自然表达 |
| 索引结构 | Two-level gather + reduce | 全 tensor 操作，可被 AOTInductor 编译 |
| 规则标准化 | DNF (OR of ANDs) | 统一表示任意布尔表达式 |
| NOT 实现 | 谓词级 negated + XOR | De Morgan 推到叶节点，最小化复杂度 |
| Conjunction padding | `\| ~valid` (AND 单位元 True) | 无效槽不影响 AND 规约 |
| Item padding | `& valid` (OR 零元 False) | 无效槽不影响 OR 规约 |
| 谓词去重 | 全局注册表 + pred_id | 相同谓词只评估一次 |
| 固定形状张量 | K=8, J=16 + validity mask | 满足 torch.export 静态形状约束 |
| 用户编码 | Python 侧 O(P) 评估 | 不在 forward() 中，保持 trace 纯净 |

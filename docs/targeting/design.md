# 定向召回: 设计与优化策略

> 框架总览参见 [../architecture.md](../architecture.md)

---

## 1. 问题定义

**输入**:
- N 个 item，每个 item 有一条定向规则（布尔表达式），如 `city == "北京" AND age > 18`
- 一个用户的属性向量，如 `{city: "北京", gender: "男", age: 30}`

**输出**: 所有定向规则被该用户满足的 item 集合。

**规模**: N > 1M items，数百个唯一谓词，需要亚毫秒级在线延迟。

---

## 2. 核心架构：Two-Level Gather + Reduce

整个匹配逻辑被归结为两次 gather + 两次 reduce，全部基于 PyTorch 张量操作：

```
pred_satisfied: [P] bool    (用户满足哪些谓词)
       │
       │  Level 1: Conjunction 匹配
       ▼
  gather → [C, K]            取出每个 conjunction 引用的 K 个谓词
  XOR negated → [C, K]       处理 NOT
  OR ~valid → [C, K]         padding = True (AND 单位元)
  all(dim=1) → [C] bool      AND 规约: 所有谓词都满足?
       │
       │  Level 2: Item 匹配
       ▼
  gather → [N, J]            取出每个 item 引用的 J 个 conjunction
  AND valid → [N, J]         padding = False (OR 零元)
  any(dim=1) → [N] bool      OR 规约: 任一 conjunction 满足?
       │
       ▼
  item_ok: [N] bool           最终结果
```

**为什么不用 packed bitmap?**

前一版架构使用 packed int64 bitmap 表示 item 集合，通过位运算做 AND/OR。这在「查询→item」的正向召回中很高效，但定向召回的方向相反——需要逐 item 检查其规则是否满足，而非逐查询条件筛选 item。Two-level gather+reduce 更自然地表达这种逻辑，且同样可以被 AOTInductor 编译。

---

## 3. 离线索引设计

### 3.1 谓词注册表

所有 item 规则中出现的原子谓词被收集、去重、分配连续 ID：

```
pred_id 0: discrete  city == "北京"
pred_id 1: discrete  city == "上海"
pred_id 2: numeric   age > 18
pred_id 3: numeric   age >= 25
pred_id 4: text      tags contains "游戏"
...
```

好处：
- 相同谓词只评估一次（去重）
- 整数 ID 便于张量 gather
- `!=` 统一为 `==` + negated，减少谓词数量

### 3.2 DNF 标准化

每条 item 规则被解析为 AST，然后转换为析取范式 (DNF = OR of ANDs)：

```
规则: (city=="北京" OR city=="上海") AND age>=25
DNF:  [[city=="北京", age>=25], [city=="上海", age>=25]]
```

DNF 是二级索引结构的基础——每个 conjunction 对应一行 conj_pred_ids 张量。

### 3.3 张量索引

两组固定形状的张量：

| 张量 | 形状 | 含义 |
|------|------|------|
| `conj_pred_ids` | [C, K] int64 | conjunction→谓词 映射 |
| `conj_pred_negated` | [C, K] bool | 是否取反 |
| `conj_pred_valid` | [C, K] bool | 有效性掩码 |
| `item_conj_ids` | [N, J] int64 | item→conjunction 映射 |
| `item_conj_valid` | [N, J] bool | 有效性掩码 |

K=MAX_PREDS_PER_CONJ=8, J=MAX_CONJ_PER_ITEM=16。不足的部分用 valid=False 标记。

---

## 4. 在线匹配设计

### 4.1 用户编码

在 Python 侧完成（不经过 torch.export 路径）：

```python
pred_satisfied = torch.zeros(P, dtype=torch.bool)
for pred in registry:
    if evaluate(pred, user_attrs):
        pred_satisfied[pred.id] = True
```

复杂度 O(P)。对于典型场景（几百个谓词），耗时 <0.1ms。

### 4.2 模型推理

`forward(pred_satisfied)` → `item_ok`，6 行 tensor 操作，详见 [implementation.md](implementation.md#2-targetingrecall推理模型)，完整执行过程参见 [walkthrough.md](walkthrough.md)。

---

## 5. 内存估算

对于 N 个 item、P 个谓词、C 个 conjunction：

| 张量 | 大小 |
|------|------|
| conj_pred_ids [C, K] int64 | C × 8 × 8 bytes |
| conj_pred_negated [C, K] bool | C × 8 × 1 byte |
| conj_pred_valid [C, K] bool | C × 8 × 1 byte |
| item_conj_ids [N, J] int64 | N × 16 × 8 bytes |
| item_conj_valid [N, J] bool | N × 16 × 1 byte |

典型场景 (N=1M, 平均 1.5 conj/item → C≈1.5M):

| 组件 | 大小 |
|------|------|
| conjunction 张量 | ~24 MB |
| item 张量 | ~144 MB |
| **总计** | **~168 MB** |

可以通过减小 K 和 J 或使用 int32 进一步优化。

---

## 6. 设计限制与缓解

| 限制 | 影响 | 缓解策略 |
|------|------|---------|
| DNF 展开可能指数膨胀 | 极端嵌套规则 conjunction 数爆炸 | MAX_CONJ_PER_ITEM=16 限制，超出拒绝 |
| 固定 K 和 J | 超过限制的规则无法表达 | K=8, J=16 覆盖绝大多数实际定向规则 |
| 全量 item 扫描 | 延迟与 N 线性相关 | AOTInductor 编译后向量化，1M item 仍可亚毫秒 |
| 静态索引 | 新增/修改 item 需重建 | 批量更新 + 热加载 .pt2 |
| 文本分词质量 | 影响 text contains 匹配 | 可插拔分词器，默认空格，可替换 jieba |

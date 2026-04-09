# 端到端示例详解

用 5 条规则和 1 个用户，完整走一遍从离线构建到在线匹配的全过程。涵盖 `!=` 取反 (XOR negated) 的处理。

> 前置阅读: [design.md](design.md) — 设计原理，[implementation.md](implementation.md) — 模块实现

---

## 样例数据

```python
schema = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age"],
    text_fields=[],
)

rules = [
    'city == "北京" AND gender == "男"',         # item 0
    'city == "上海"',                            # item 1
    "age > 18",                                  # item 2
    '(city == "北京" OR city == "上海") AND age >= 25',  # item 3
    'city != "广州" AND age > 18',               # item 4  ← 含 != (取反)
]

user = {"city": "北京", "gender": "男", "age": 30}
```

---

## 第一步：离线构建 (TargetingBuilder.build)

### Phase 1 — 解析规则 → DNF，构建谓词注册表

逐条规则解析为 AST → DNF，收集所有出现的原子谓词：

```
item 0: city=="北京" AND gender=="男"
  DNF = [[city=="北京", gender=="男"]]    → 1 conjunction, 2 predicates

item 1: city=="上海"
  DNF = [[city=="上海"]]                  → 1 conjunction, 1 predicate

item 2: age > 18
  DNF = [[age>18]]                        → 1 conjunction, 1 predicate

item 3: (city=="北京" OR city=="上海") AND age >= 25
  DNF = [[city=="北京", age>=25],          → 2 conjunctions
         [city=="上海", age>=25]]

item 4: city != "广州" AND age > 18
  DNF = [[city=="广州"(negated!), age>18]]  → 1 conjunction, 2 predicates
  注意: != 被标准化为 == + negated=True
  构建时 _register_predicate 处理:
    city != "广州" → 注册 city == "广州" (pred_id 5), negated=True
    age > 18       → 复用已有的 pred_id 3, negated=False
```

谓词注册表 (去重后):

```
pred_id 0: discrete  city == "北京"
pred_id 1: discrete  gender == "男"
pred_id 2: discrete  city == "上海"
pred_id 3: numeric   age > 18
pred_id 4: numeric   age >= 25
pred_id 5: discrete  city == "广州"    ← 新增 (由 != 标准化而来)
```

P=6 个唯一谓词。注意 `city != "广州"` 并没有在注册表中产生一条 `!=` 谓词，而是注册为 `city == "广州"`，取反信息记录在 conjunction 的 `negated` 标志中。

### Phase 2 — 构建 conjunction 张量

Phase 1 的结果是每个 item 的 conjunction 列表（每个 conjunction 又是一组 `(pred_id, negated)` 对）:

```
item_conjs (Phase 1 的输出):
  item 0: [[(0,F), (1,F)]]                   ← 1 个 conj: pred0 AND pred1
  item 1: [[(2,F)]]                           ← 1 个 conj: pred2
  item 2: [[(3,F)]]                           ← 1 个 conj: pred3
  item 3: [[(0,F), (4,F)], [(2,F), (4,F)]]   ← 2 个 conj: (pred0 AND pred4) OR (pred2 AND pred4)
  item 4: [[(5,T), (3,F)]]                   ← 1 个 conj: pred5(negated!) AND pred3
                                                 ↑ (5,T) 表示 pred 5 取反: NOT (city=="广州")
```

目标是把这些 conjunction 铺平成一个统一的 `[C, K]` 张量，其中 C 是所有 item 的 conjunction 总数，K=MAX_PREDS_PER_CONJ=8 是每条 conjunction 的固定宽度。

**Step 2a — 扁平化: 所有 conjunction 收集到一个数组，记录每个 item 的偏移量**

```python
all_conj_preds = []        # 扁平的 conjunction 列表
item_conj_offsets = []     # 每个 item 的 conjunction 在 all_conj_preds 中的位置

for conjs in item_conjs:
    offsets = []
    for conj in conjs:
        offsets.append(len(all_conj_preds))   # 当前 conj 的全局编号
        all_conj_preds.append(conj)
    item_conj_offsets.append(offsets)
```

逐步执行:

```
处理 item 0: conjs = [[(0,F), (1,F)]]
  conj [(0,F), (1,F)] → 编号 0 (= len(all_conj_preds))
  all_conj_preds = [[(0,F), (1,F)]]
  item_conj_offsets = [[0]]

处理 item 1: conjs = [[(2,F)]]
  conj [(2,F)] → 编号 1
  all_conj_preds = [[(0,F), (1,F)], [(2,F)]]
  item_conj_offsets = [[0], [1]]

处理 item 2: conjs = [[(3,F)]]
  conj [(3,F)] → 编号 2
  all_conj_preds = [[(0,F), (1,F)], [(2,F)], [(3,F)]]
  item_conj_offsets = [[0], [1], [2]]

处理 item 3: conjs = [[(0,F), (4,F)], [(2,F), (4,F)]]
  conj [(0,F), (4,F)] → 编号 3
  conj [(2,F), (4,F)] → 编号 4
  all_conj_preds = [[(0,F),(1,F)], [(2,F)], [(3,F)], [(0,F),(4,F)], [(2,F),(4,F)]]
  item_conj_offsets = [[0], [1], [2], [3, 4]]

处理 item 4: conjs = [[(5,T), (3,F)]]
  conj [(5,T), (3,F)] → 编号 5
  all_conj_preds = [...之前 5 个..., [(5,T), (3,F)]]
  item_conj_offsets = [[0], [1], [2], [3, 4], [5]]
```

最终: C = len(all_conj_preds) = 6 个 conjunction。

item 3 拥有 2 个 conjunction (编号 3 和 4)，其余 item 各 1 个。注意 conj 5 (来自 item 4) 中 `(5,T)` 的 T 表示 negated=True。这个偏移量信息 `item_conj_offsets` 在 Phase 3 用于构建 item→conjunction 映射。

**Step 2b — 填充固定宽度张量 [C, K]**

分配三个全零张量:

```python
conj_pred_ids     = torch.zeros(C=6, K=8, dtype=int64)     # 谓词 ID
conj_pred_negated = torch.zeros(C=6, K=8, dtype=bool)      # 是否取反
conj_pred_valid   = torch.zeros(C=6, K=8, dtype=bool)      # 有效/padding
```

逐 conjunction 逐 predicate 填入:

```python
for ci, preds in enumerate(all_conj_preds):
    for pi, (pred_id, negated) in enumerate(preds):
        conj_pred_ids[ci, pi] = pred_id
        conj_pred_negated[ci, pi] = negated
        conj_pred_valid[ci, pi] = True          # 标记为有效
        # 未填到的 pi 保持默认值: ids=0, negated=False, valid=False
```

逐步执行:

```
ci=0, preds=[(0,F), (1,F)]:
  pi=0: ids[0,0]=0, neg[0,0]=F, valid[0,0]=T   ← pred 0: city=="北京"
  pi=1: ids[0,1]=1, neg[0,1]=F, valid[0,1]=T   ← pred 1: gender=="男"
  pi=2~7: 未填, 保持 ids=0, neg=F, valid=F     ← padding

ci=1, preds=[(2,F)]:
  pi=0: ids[1,0]=2, neg[1,0]=F, valid[1,0]=T   ← pred 2: city=="上海"
  pi=1~7: padding

ci=2, preds=[(3,F)]:
  pi=0: ids[2,0]=3, neg[2,0]=F, valid[2,0]=T   ← pred 3: age > 18
  pi=1~7: padding

ci=3, preds=[(0,F), (4,F)]:
  pi=0: ids[3,0]=0, neg[3,0]=F, valid[3,0]=T   ← pred 0: city=="北京"
  pi=1: ids[3,1]=4, neg[3,1]=F, valid[3,1]=T   ← pred 4: age >= 25
  pi=2~7: padding

ci=4, preds=[(2,F), (4,F)]:
  pi=0: ids[4,0]=2, neg[4,0]=F, valid[4,0]=T   ← pred 2: city=="上海"
  pi=1: ids[4,1]=4, neg[4,1]=F, valid[4,1]=T   ← pred 4: age >= 25
  pi=2~7: padding

ci=5, preds=[(5,T), (3,F)]:                     ← 来自 item 4: city != "广州" AND age > 18
  pi=0: ids[5,0]=5, neg[5,0]=T, valid[5,0]=T   ← pred 5: city=="广州", negated=True!
  pi=1: ids[5,1]=3, neg[5,1]=F, valid[5,1]=T   ← pred 3: age > 18
  pi=2~7: padding
```

**最终结果: 三个 [6, 8] 张量**

```
conj_pred_ids (哪些谓词):
           k=0  k=1  k=2  k=3  k=4  k=5  k=6  k=7
  conj 0: [ 0,   1,   0,   0,   0,   0,   0,   0 ]   ← city北京, gender男
  conj 1: [ 2,   0,   0,   0,   0,   0,   0,   0 ]   ← city上海
  conj 2: [ 3,   0,   0,   0,   0,   0,   0,   0 ]   ← age>18
  conj 3: [ 0,   4,   0,   0,   0,   0,   0,   0 ]   ← city北京, age>=25
  conj 4: [ 2,   4,   0,   0,   0,   0,   0,   0 ]   ← city上海, age>=25
  conj 5: [ 5,   3,   0,   0,   0,   0,   0,   0 ]   ← city广州(取反!), age>18

conj_pred_negated (是否取反):
           k=0  k=1  k=2  k=3  k=4  k=5  k=6  k=7
  conj 0: [ F,   F,   F,   F,   F,   F,   F,   F ]
  conj 1: [ F,   F,   F,   F,   F,   F,   F,   F ]
  conj 2: [ F,   F,   F,   F,   F,   F,   F,   F ]
  conj 3: [ F,   F,   F,   F,   F,   F,   F,   F ]
  conj 4: [ F,   F,   F,   F,   F,   F,   F,   F ]
  conj 5: [ T,   F,   F,   F,   F,   F,   F,   F ]   ← k=0 位 negated=True!

conj_pred_valid (哪些槽位有效):
           k=0  k=1  k=2  k=3  k=4  k=5  k=6  k=7
  conj 0: [ T,   T,   F,   F,   F,   F,   F,   F ]   ← 2 个有效谓词
  conj 1: [ T,   F,   F,   F,   F,   F,   F,   F ]   ← 1 个有效谓词
  conj 2: [ T,   F,   F,   F,   F,   F,   F,   F ]
  conj 3: [ T,   T,   F,   F,   F,   F,   F,   F ]
  conj 4: [ T,   T,   F,   F,   F,   F,   F,   F ]
  conj 5: [ T,   T,   F,   F,   F,   F,   F,   F ]   ← 2 个有效谓词
```

注意 padding 槽位 (valid=F) 中的 `ids=0` 是无意义的默认值——forward() 会通过 `| ~valid` 将这些位置设为 True，使它们不影响 AND 规约结果。

**为什么需要固定宽度 K=8?** 因为 `torch.export` 要求所有张量形状在编译时确定。如果 conjunction 的谓词数不固定，就无法导出为 `.pt2`。K=8 意味着单条 conjunction 最多 8 个谓词，超过则报错。实际广告定向规则通常 2-4 个谓词，8 足够。

### Phase 3 — 构建 item-conjunction 张量

用 Phase 2 记录的 `item_conj_offsets` 构建 item→conjunction 的映射:

```python
item_conj_ids   = torch.zeros(N=5, J=16, dtype=int64)
item_conj_valid = torch.zeros(N=5, J=16, dtype=bool)

for item_idx, offsets in enumerate(item_conj_offsets):
    for ji, conj_idx in enumerate(offsets):
        item_conj_ids[item_idx, ji] = conj_idx
        item_conj_valid[item_idx, ji] = True
```

```
item_conj_offsets = [[0], [1], [2], [3, 4], [5]]

item 0: offsets=[0]    → ids[0,0]=0, valid[0,0]=T         ← 引用 conj 0
item 1: offsets=[1]    → ids[1,0]=1, valid[1,0]=T         ← 引用 conj 1
item 2: offsets=[2]    → ids[2,0]=2, valid[2,0]=T         ← 引用 conj 2
item 3: offsets=[3,4]  → ids[3,0]=3, valid[3,0]=T         ← 引用 conj 3
                         ids[3,1]=4, valid[3,1]=T         ← 引用 conj 4
item 4: offsets=[5]    → ids[4,0]=5, valid[4,0]=T         ← 引用 conj 5 (含取反)
```

最终:

```
item_conj_ids:
          j=0  j=1  j=2 ... j=15
  item 0: [ 0,   0,   0, ...,  0 ]   ← 只有 j=0 有效
  item 1: [ 1,   0,   0, ...,  0 ]
  item 2: [ 2,   0,   0, ...,  0 ]
  item 3: [ 3,   4,   0, ...,  0 ]   ← j=0 和 j=1 有效 (OR 关系)
  item 4: [ 5,   0,   0, ...,  0 ]

item_conj_valid:
          j=0  j=1  j=2 ... j=15
  item 0: [ T,   F,   F, ...,  F ]
  item 1: [ T,   F,   F, ...,  F ]
  item 2: [ T,   F,   F, ...,  F ]
  item 3: [ T,   T,   F, ...,  F ]   ← 2 个有效 conjunction
  item 4: [ T,   F,   F, ...,  F ]
```

J=16 的逻辑与 K=8 相同: 固定宽度 + validity mask，满足 `torch.export` 静态形状约束。

### 构建总结: 两级索引结构

```
谓词注册表            conj 张量 [C=6, K=8]         item 张量 [N=5, J=16]
┌─────────────┐      ┌───────────────────┐        ┌──────────────────┐
│ 0: city=北京 │◄──── │ conj 0: [0, 1]    │◄────── │ item 0: [0]      │
│ 1: gender=男 │      │ conj 1: [2]       │        │ item 1: [1]      │
│ 2: city=上海 │      │ conj 2: [3]       │        │ item 2: [2]      │
│ 3: age > 18  │      │ conj 3: [0, 4]    │◄──┐    │ item 3: [3, 4]   │
│ 4: age >= 25 │      │ conj 4: [2, 4]    │◄──┘    │ item 4: [5]      │
│ 5: city=广州 │◄──── │ conj 5: [5̄, 3]    │◄────── │                  │
└─────────────┘      └───────────────────┘        └──────────────────┘
   P=6 个谓词          C=6 个 conjunction            N=5 个 item
                       5̄ = pred 5 取反 (negated)

forward() 时:
  Level 1: pred_satisfied[P] → gather by conj_pred_ids → [C, K] → XOR neg → AND → conj_ok[C]
  Level 2: conj_ok[C]        → gather by item_conj_ids → [N, J] → OR  → item_ok[N]
```

---

## 第二步：在线编码 (encode_user)

用户: `{"city": "北京", "gender": "男", "age": 30}`

逐谓词评估:

```
pred 0: city=="北京"  → 用户 city="北京" → True  ✓
pred 1: gender=="男"  → 用户 gender="男" → True  ✓
pred 2: city=="上海"  → 用户 city="北京" → False ✗
pred 3: age > 18     → 30 > 18          → True  ✓
pred 4: age >= 25    → 30 >= 25         → True  ✓
pred 5: city=="广州"  → 用户 city="北京" → False ✗

pred_satisfied = [True, True, False, True, True, False]
```

注意 pred 5 评估的是 `city == "广州"` (而非 `!=`)，结果为 False。取反逻辑在 forward() 的 XOR negated 步骤处理。

---

## 第三步：forward() 执行

### Step 1: Conjunction 匹配

**1a. gather — 取出每个 conjunction 引用的谓词结果**

```python
sat = pred_satisfied[conj_pred_ids]   # gather → [C=6, K=8]
```

用 `conj_pred_ids` 中的 ID 从 `pred_satisfied` 中查表:

```
conj 0: ids=[0,1,0,...]  → sat = [T, T, F, F, F, F, F, F]   ← pred[0]=T, pred[1]=T
conj 1: ids=[2,0,0,...]  → sat = [F, F, F, F, F, F, F, F]   ← pred[2]=F
conj 2: ids=[3,0,0,...]  → sat = [T, F, F, F, F, F, F, F]   ← pred[3]=T
conj 3: ids=[0,4,0,...]  → sat = [T, T, F, F, F, F, F, F]   ← pred[0]=T, pred[4]=T
conj 4: ids=[2,4,0,...]  → sat = [F, T, F, F, F, F, F, F]   ← pred[2]=F, pred[4]=T
conj 5: ids=[5,3,0,...]  → sat = [F, T, F, F, F, F, F, F]   ← pred[5]=F, pred[3]=T
```

**1b. XOR negated — 处理取反**

```python
sat = sat ^ conj_pred_negated   # 逐元素异或
```

XOR 的作用: 如果 `negated=True`，则将谓词的评估结果翻转。

```
conj_pred_negated:
  conj 0: [F, F, F, F, F, F, F, F]   ← 无取反
  conj 1: [F, F, F, F, F, F, F, F]
  conj 2: [F, F, F, F, F, F, F, F]
  conj 3: [F, F, F, F, F, F, F, F]
  conj 4: [F, F, F, F, F, F, F, F]
  conj 5: [T, F, F, F, F, F, F, F]   ← k=0 位取反!
```

XOR 逐元素计算:

```
conj 0~4: negated 全为 F → XOR 不改变任何值

conj 5:  sat = [F, T, F, F, F, F, F, F]
         neg = [T, F, F, F, F, F, F, F]
         XOR = [T, T, F, F, F, F, F, F]
               ↑
               F XOR T = T ← 关键! pred 5 (city=="广州") 原始结果 False,
                             取反后变成 True, 表示 "用户不在广州" 成立
```

XOR 后:

```
conj 0: [T, T, F, F, F, F, F, F]     (不变)
conj 1: [F, F, F, F, F, F, F, F]     (不变)
conj 2: [T, F, F, F, F, F, F, F]     (不变)
conj 3: [T, T, F, F, F, F, F, F]     (不变)
conj 4: [F, T, F, F, F, F, F, F]     (不变)
conj 5: [T, T, F, F, F, F, F, F]     ← k=0 从 F 翻转为 T!
```

**XOR 取反的原理:**

| pred_satisfied | negated | XOR 结果 | 含义 |
|:-:|:-:|:-:|---|
| True | False | True | 谓词满足，不取反 → 通过 |
| False | False | False | 谓词不满足，不取反 → 不通过 |
| True | True | False | 谓词满足，但要取反 → 不通过 (如: 用户在广州，但规则要求 NOT 广州) |
| **False** | **True** | **True** | **谓词不满足，取反 → 通过 (如: 用户不在广州，规则要求 NOT 广州)** |

**1c. OR ~valid — padding 位设为 True**

```python
sat = sat | ~conj_pred_valid   # padding 位 → True (AND 单位元)
```

```
conj 0: [T, T, T, T, T, T, T, T]   ← valid=[T,T,F,...] → padding 设 T
conj 1: [F, T, T, T, T, T, T, T]   ← valid=[T,F,...] → 只有 k=0 是真实值
conj 2: [T, T, T, T, T, T, T, T]
conj 3: [T, T, T, T, T, T, T, T]
conj 4: [F, T, T, T, T, T, T, T]
conj 5: [T, T, T, T, T, T, T, T]   ← XOR 后两个真实位都是 T
```

**1d. all(dim=1) — AND 规约**

```
conj_ok = [True, False, True, True, False, True]
```

验证:
- conj 0: city=北京(T) AND gender=男(T) → ✓
- conj 1: city=上海(F) → ✗
- conj 2: age>18(T) → ✓
- conj 3: city=北京(T) AND age>=25(T) → ✓
- conj 4: city=上海(F) AND age>=25(T) → ✗
- conj 5: NOT city=广州(F→T) AND age>18(T) → **✓** ← 取反后通过!

### Step 2: Item 匹配

```python
item_conj_ok = conj_ok[item_conj_ids]   # gather → [N=5, J=16]
```

```
item 0: [True,  ...(padding)]       ← conj 0 = True
item 1: [False, ...(padding)]       ← conj 1 = False
item 2: [True,  ...(padding)]       ← conj 2 = True
item 3: [True, False, ...(padding)] ← conj 3=True, conj 4=False
item 4: [True,  ...(padding)]       ← conj 5 = True
```

AND item_conj_valid (清零 padding):

```
item 0: [True,  False, False, ...]
item 1: [False, False, False, ...]
item 2: [True,  False, False, ...]
item 3: [True,  False, False, ...]  ← conj 3=T, conj 4 valid=T 但值=F
item 4: [True,  False, False, ...]  ← conj 5 取反后通过
```

any(dim=1):

```
item_ok = [True, False, True, True, True]
```

**最终结果: 匹配 item [0, 2, 3, 4]**

---

## 验证

```
       │ item 规则                               │ 用户: 北京,男,30    │ 匹配 │
───────┼─────────────────────────────────────────┼────────────────────┼──────┤
item 0 │ city=="北京" AND gender=="男"             │ 北京✓ 男✓          │ ✓    │
item 1 │ city=="上海"                              │ 北京≠上海          │ ✗    │
item 2 │ age > 18                                  │ 30>18 ✓           │ ✓    │
item 3 │ (city=="北京" OR city=="上海") AND age>=25 │ 北京✓ 30>=25✓      │ ✓    │
item 4 │ city!="广州" AND age>18                   │ 北京≠广州✓ 30>18✓  │ ✓    │
```

结果 [0, 2, 3, 4] 与手工验证一致。

**item 4 的关键路径:**
`city != "广州"` → 注册为 `city == "广州"` (pred 5) + negated=True
→ encode_user: pred 5 = False (用户不在广州)
→ gather: sat[5,0] = False
→ XOR negated: False XOR True = **True** (取反后通过!)
→ AND age>18(True) → conj 5 通过 → item 4 匹配

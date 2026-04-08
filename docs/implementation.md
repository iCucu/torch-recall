# 逐模块实现详解

本文档按代码模块逐一解析 torch-recall 的实现。适合想要深入理解代码或进行二次开发的读者。

---

## 整体流程

torch-recall 将「item 集合上的布尔过滤」转化为「固定形状的张量运算」，再通过 PyTorch 的 AOTInductor 编译为原生 C++ 内核，从而在无 Python 解释器的环境中以亚毫秒延迟执行。

整个系统分为**离线**和**在线**两个阶段：

```
离线 (一次性)                              在线 (每次查询)
─────────────                            ─────────────
items (list[dict])                       查询字符串
      │                                        │
      ▼                                        ▼
 IndexBuilder.build()                     解析 → AST
      │                                        │
      ├→ 离散字段: 每个值建一条 bitmap            ▼
      ├→ 文本字段: 分词后每个 term 建一条 bitmap  DNF 转换 (De Morgan + 分配律)
      ├→ 数值字段: float32 列存储                │
      │                                        ▼
      ▼                                  张量编码: DNF → 固定尺寸 tensor
 InvertedIndexModel (nn.Module)                │
      │                                        ▼
      ▼                                  model.forward(tensors) → [L] bitmap
 torch.export → AOTInductor                    │  ┌──────────────────────────┐
      │                                        │  │ bitmap 查表 + 数值比较    │
      ▼                                        │  │ → 向量化 AND (24 次)     │
 model.pt2 + index_meta.json                   │  │ → 树规约 OR (3 步)       │
                                               │  └──────────────────────────┘
                                               ▼
                                         位扫描解码 → 匹配 item IDs
```

### 核心逻辑

**1. 所有数据用 packed int64 bitmap 表示集合。** 每个离散值/文本 term 一条 bitmap，1M item 只需 15,625 个 int64 (~122 KB)。集合运算变成位运算：AND=交集，OR=并集，XOR+mask=补集。一次 int64 AND 同时处理 64 个 item。

**2. 查询被标准化为 DNF (析取范式)。** 任意嵌套的布尔表达式通过 De Morgan 定律和分配律展开为 "OR of ANDs"。每个 conjunction (AND 子句) 内做 bitmap 交集，最终所有 conjunction 做并集。这将复杂的逻辑统一为固定的 AND+OR 两步。

**3. forward() 是固定形状的静态计算图。** 查询被编码为固定尺寸的张量（MAX_BP=16 bitmap 谓词 + MAX_NP=8 数值谓词 + CONJ_PER_PASS=8 conjunction），无效部分用 padding + validity mask 处理。这满足 `torch.export` 的静态图约束，使模型可以被编译为 C++ 内核。

**4. 计算量恒定，与查询复杂度和选择率无关。** 无论查询用了 1 个还是 8 个谓词，forward() 的操作数相同。这是 bitmap 架构相比 posting list 的核心 trade-off——用固定开销换来可预测延迟。

**5. 大查询通过多 pass 线性扩展。** 超过 8 个 conjunction 时，query encoder 自动分批，每批调用一次 forward()，结果 OR 合并。系统级支持最多 1024 conjunction。

---

## 端到端示例

用 6 个 item 完整走一遍从索引构建到查询执行的全过程。

### 样例数据

```python
schema = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["price"],
    text_fields=[],
)

items = [
    {"city": "北京", "gender": "男", "price": 68.0},    # item 0
    {"city": "上海", "gender": "女", "price": 128.0},   # item 1
    {"city": "北京", "gender": "女", "price": 299.0},   # item 2
    {"city": "广州", "gender": "男", "price": 35.0},    # item 3
    {"city": "北京", "gender": "男", "price": 15.0},    # item 4
    {"city": "上海", "gender": "男", "price": 88.0},    # item 5
]
```

### 第一步：索引构建 (IndexBuilder.build)

N=6, L=⌈6/64⌉=1 (只需 1 个 int64 word)

**Pass 1 — 建字典 + 编码**

```
discrete_dicts:
  city:   {"上海": 0, "北京": 1, "广州": 2}    (字母序排序)
  gender: {"女": 0, "男": 1}

discrete_encoded:
  city:   [1, 0, 1, 2, 1, 0]    (每 item 的 val_id)
  gender: [1, 0, 0, 1, 1, 1]
```

**分配 bitmap 全局行号**

```
bitmap 行 0: city=上海     (d:city val_id=0)
bitmap 行 1: city=北京     (d:city val_id=1)
bitmap 行 2: city=广州     (d:city val_id=2)
bitmap 行 3: gender=女     (d:gender val_id=0)
bitmap 行 4: gender=男     (d:gender val_id=1)
```

**Pass 2 — 逐 item 设置位**

每个 item 的位位置 = `1 << item_idx` (因为 L=1，只有 word 0)

```
item 0 (北京, 男, 68):  bitmap[1] |= 0b000001, bitmap[4] |= 0b000001
item 1 (上海, 女, 128): bitmap[0] |= 0b000010, bitmap[3] |= 0b000010
item 2 (北京, 女, 299): bitmap[1] |= 0b000100, bitmap[3] |= 0b000100
item 3 (广州, 男, 35):  bitmap[2] |= 0b001000, bitmap[4] |= 0b001000
item 4 (北京, 男, 15):  bitmap[1] |= 0b010000, bitmap[4] |= 0b010000
item 5 (上海, 男, 88):  bitmap[0] |= 0b100000, bitmap[4] |= 0b100000
```

**最终 bitmap 表** (bitmaps 形状 [5, 1])

```
行 0  city=上海:  0b100010  = {item 1, 5}
行 1  city=北京:  0b010101  = {item 0, 2, 4}
行 2  city=广州:  0b001000  = {item 3}
行 3  gender=女:  0b000110  = {item 1, 2}
行 4  gender=男:  0b111001  = {item 0, 3, 4, 5}
```

**数值列** (numeric_data 形状 [1, 6])

```
price: [68.0, 128.0, 299.0, 35.0, 15.0, 88.0]
         i0     i1      i2    i3    i4    i5
```

**辅助 buffer**

```
all_ones   = [0b111111]           (L=1, 6 个 1)
valid_mask = [0b111111]           (6 个有效 item, 低 6 位全 1)
pack_powers = [1, 2, 4, 8, 16, 32, ...]  (用于 bool→int64 打包)
```

**meta (保存为 JSON)**

```json
{
  "discrete_dicts": {"city": {"上海": 0, "北京": 1, "广州": 2}, "gender": {"女": 0, "男": 1}},
  "bitmap_lookup": {"d:city": {"0": 0, "1": 1, "2": 2}, "d:gender": {"0": 3, "1": 4}},
  "num_items": 6,
  "bitmap_len": 1,
  "num_bitmaps": 5
}
```

### 第二步：查询处理

查询: `(city == "北京" OR city == "上海") AND price < 100`

#### 2a. 解析 → AST

```
And(
  Or(
    Predicate("city", "==", "北京"),
    Predicate("city", "==", "上海")
  ),
  Predicate("price", "<", 100.0)
)
```

#### 2b. DNF 转换

分配律: `AND(OR(A, B), C)` → `(A AND C) OR (B AND C)`

```
DNF = [
  conjunction 0: [city=北京, price<100],
  conjunction 1: [city=上海, price<100],
]
```

2 个 conjunction ≤ CONJ_PER_PASS=8，单 pass 即可。

#### 2c. 张量编码

**Step 1: 收集谓词** (`_collect_predicates`)

```
bitmap 谓词:
  bp_idx 0 → city=北京, bitmap_indices[0]=1 (行 1), bitmap_valid[0]=True
  bp_idx 1 → city=上海, bitmap_indices[1]=0 (行 0), bitmap_valid[1]=True
  bp_idx 2..15 → padding, bitmap_valid=False

数值谓词:
  np_idx 0 → price < 100, numeric_fields[0]=0, numeric_ops[0]=1(LT), numeric_values[0]=100.0, numeric_valid[0]=True
  np_idx 1..7 → padding, numeric_valid=False

negation_mask = [F, F, ..., F]  (无取反)
```

**Step 2: 填充 conj_matrix** (`_fill_conj_matrix`)

P_TOTAL=24 列，前 16 列是 bitmap 谓词，后 8 列是数值谓词。

```
                bp0  bp1  bp2..15  np0  np1..7
conj 0 (北京&价格): [ 1,   0,   0..0,   1,   0..0 ]  ← 使用 bp0 和 np0
conj 1 (上海&价格): [ 0,   1,   0..0,   1,   0..0 ]  ← 使用 bp1 和 np0
conj 2..7:         [ 0,   0,   0..0,   0,   0..0 ]  ← padding

conj_valid = [T, T, F, F, F, F, F, F]
```

### 第三步：forward() 执行

#### 阶段 1: 收集谓词 bitmap

```
bp = bitmaps[bitmap_indices]
  bp[0] = bitmaps[1] = 0b010101  (city=北京: item 0,2,4)
  bp[1] = bitmaps[0] = 0b100010  (city=上海: item 1,5)
  bp[2..15] = bitmaps[0] = 0b100010  (padding, 后续 valid=False 会覆盖为 all_ones)
```

数值比较: `numeric_data[0] < 100.0` → `[68<100, 128<100, 299<100, 35<100, 15<100, 88<100]`

```
numeric_bool[0] = [T, F, F, T, T, T]    (item 0,3,4,5 价格<100)
numeric_bool[1..7] = [F, F, F, F, F, F]  (padding)
```

打包: `_pack_bool_to_bitmap`

```
np_bitmap[0] = 0b110001 · 2^0 + ... = 0b111001  → {item 0, 3, 4, 5}
np_bitmap[1..7] = 0b000000
```

合并:

```
all_bm[0]  = bp[0]        = 0b010101  (city=北京)
all_bm[1]  = bp[1]        = 0b100010  (city=上海)
all_bm[2..15] = padding
all_bm[16] = np_bitmap[0] = 0b111001  (price<100)
all_bm[17..23] = padding
```

#### 阶段 2: 有效性掩码

无效谓词 (valid=False) 替换为 all_ones (0b111111)：

```
all_bm[0]  = 0b010101  (有效, 保留)
all_bm[1]  = 0b100010  (有效, 保留)
all_bm[2]  = 0b111111  (无效 → all_ones)
...
all_bm[15] = 0b111111  (无效 → all_ones)
all_bm[16] = 0b111001  (有效, 保留)
all_bm[17] = 0b111111  (无效 → all_ones)
...
all_bm[23] = 0b111111  (无效 → all_ones)
```

#### 阶段 3: 向量化 AND

初始: `conj_results[0..7] = 0b111111` (AND 单位元, 全 1)

24 次迭代中，只有 p=0, p=1, p=16 的 conj_matrix 中有 True，其余谓词 all_bm = all_ones，AND 不改变结果。等价于：

```
p=0 (city=北京, 0b010101):
  conj 0 使用 (matrix[0,0]=T): conj_results[0] &= 0b010101 → 0b010101
  conj 1 不使用 (matrix[1,0]=F): conj_results[1] &= 0b111111 → 不变

p=1 (city=上海, 0b100010):
  conj 0 不使用: 不变
  conj 1 使用 (matrix[1,1]=T): conj_results[1] &= 0b100010 → 0b100010

p=16 (price<100, 0b111001):
  conj 0 使用 (matrix[0,16]=T): conj_results[0] &= 0b111001 → 0b010001
  conj 1 使用 (matrix[1,16]=T): conj_results[1] &= 0b111001 → 0b100000

p=2..15, p=17..23: all_bm = all_ones, AND 不改变结果
```

AND 完成后:

```
conj_results[0] = 0b010001 = {item 0, 4}    ← 北京 AND 价格<100 ✓
conj_results[1] = 0b100000 = {item 5}       ← 上海 AND 价格<100 ✓
conj_results[2..7] = 0b111111               ← padding
```

#### 阶段 4: 置零无效 + OR 树规约

无效 conjunction (conj_valid=False) 置零:

```
conj_results[0] = 0b010001  (valid)
conj_results[1] = 0b100000  (valid)
conj_results[2] = 0b000000  (zeroed)
...
conj_results[7] = 0b000000  (zeroed)
```

树规约 (3 步):

```
Step 1: [8] → [4]
  cr[0] = conj_results[0] | conj_results[1] = 0b010001 | 0b100000 = 0b110001
  cr[1] = conj_results[2] | conj_results[3] = 0b000000 | 0b000000 = 0b000000
  cr[2] = conj_results[4] | conj_results[5] = 0b000000
  cr[3] = conj_results[6] | conj_results[7] = 0b000000

Step 2: [4] → [2]
  cr[0] = 0b110001 | 0b000000 = 0b110001
  cr[1] = 0b000000 | 0b000000 = 0b000000

Step 3: [2] → [1]
  result = 0b110001 | 0b000000 = 0b110001
```

最终: `result & valid_mask = 0b110001 & 0b111111 = 0b110001`

### 第四步：结果解码

```
result = 0b110001

word = 0b110001
  ctz(0b110001) = 0 → item 0 (北京, 男, 68)  ✓ 北京且价格<100
  word &= word-1  → 0b110000
  ctz(0b110000) = 4 → item 4 (北京, 男, 15)  ✓ 北京且价格<100
  word &= word-1  → 0b100000
  ctz(0b100000) = 5 → item 5 (上海, 男, 88)  ✓ 上海且价格<100
  word &= word-1  → 0b000000 → 结束
```

**最终结果: [0, 4, 5]** — 正确匹配了所有 (北京 OR 上海) AND 价格<100 的 item。

### 验证

```
item 0: city=北京 ✓, price=68<100 ✓  → 匹配
item 1: city=上海 ✓, price=128≥100 ✗ → 不匹配
item 2: city=北京 ✓, price=299≥100 ✗ → 不匹配
item 3: city=广州 ✗                  → 不匹配
item 4: city=北京 ✓, price=15<100 ✓  → 匹配
item 5: city=上海 ✓, price=88<100 ✓  → 匹配
```

---

## 目录

1. [Schema 与常量定义](#1-schema-与常量定义)
2. [IndexBuilder：索引构建](#2-indexbuilder索引构建)
3. [InvertedIndexModel：推理模型](#3-invertedindexmodel推理模型)
4. [查询处理管线：解析 → DNF → 张量编码](#4-查询处理管线解析--dnf--张量编码)
5. [Exporter：torch.export + AOTInductor 导出](#5-exportertorchexport--aotinductor-导出)
6. [C++ 在线推理引擎](#6-c-在线推理引擎)
7. [数据流全景图](#7-数据流全景图)

---

## 1. Schema 与常量定义

**文件**: `offline/torch_recall/schema.py`

### Schema

```python
@dataclass
class Schema:
    discrete_fields: list[str]   # 离散字段 (city, gender, ...)
    numeric_fields:  list[str]   # 数值字段 (price, score, ...)
    text_fields:     list[str]   # 文本字段 (title, desc, ...)
```

Schema 定义了三种字段类型，决定了索引构建和查询处理的方式：

- **discrete**: 等值匹配。每个唯一值建一条 bitmap（如 `city=北京` 一条、`city=上海` 一条）
- **numeric**: 范围比较 (`==`, `<`, `>`, `<=`, `>=`)。存储原始浮点值，查询时实时向量化比较
- **text**: 分词后按 term 建 bitmap。支持 `contains` 操作

### 编译期常量

```python
MAX_BP = 16           # 单条查询最大 bitmap 谓词数 (离散 + 文本)
MAX_NP = 8            # 单条查询最大数值谓词数
P_TOTAL = 24          # MAX_BP + MAX_NP
MAX_CONJ = 1024       # 系统级: 单条查询最大 DNF conjunction 数
CONJ_PER_PASS = 8     # 模型级: 单次 forward() 处理的 conjunction 数
CONJ_PASS_LEVELS = 3  # log2(CONJ_PER_PASS), 树规约层数
```

这些常量是 `torch.export` 的**静态形状约束**。所有查询都被填充到这些固定尺寸。较小的值减少 padding 浪费，但限制了单 pass 容量——超出 `CONJ_PER_PASS` 的查询通过多次 `forward()` + OR 合并处理。

---

## 2. IndexBuilder：索引构建

**文件**: `offline/torch_recall/builder.py`

IndexBuilder 将 `list[dict]` 的原始数据转换为 `InvertedIndexModel` + 元数据 JSON。

### 构建流程

```
items (list[dict])
    │
    ├─ Pass 1: 扫描数据，构建值字典 + 每 item 编码
    │   ├─ discrete_dicts: {"city": {"北京": 0, "上海": 1, ...}}
    │   ├─ text_dicts:     {"title": {"游戏": 0, "美食": 1, ...}}
    │   ├─ discrete_encoded: {"city": [0, 1, 0, 2, ...]}  (每 item 的 val_id)
    │   └─ text_encoded:     {"title": [{0,3}, {1,2}, ...]}  (每 item 的 term_id 集合)
    │
    ├─ 分配 bitmap 全局编号 (bitmap_layout)
    │   bitmap 0: d:city val_id=0 ("北京")
    │   bitmap 1: d:city val_id=1 ("上海")
    │   ...
    │   bitmap K: t:title term_id=0 ("游戏")
    │   ...
    │
    ├─ Pass 2: 单次扫描，逐 item 设置 bitmap 位
    │   for idx in range(N):
    │       word_idx = idx >> 6          # 第几个 int64
    │       bit_val  = 1 << (idx & 63)   # 该 int64 中的哪一位
    │       bitmaps[global_idx, word_idx] |= bit_val
    │
    ├─ 构建数值列: numeric_data[field_idx, item_idx] = float_value
    │
    └─ 输出:
        ├─ InvertedIndexModel (bitmaps, numeric_data, item_ids)
        └─ meta (字典映射、bitmap_lookup、num_items 等)
```

### 关键设计

**bitmap 内存布局**: `bitmaps` 是一个 `[num_bitmaps, L]` 的 int64 tensor，其中 `L = ⌈N/64⌉`。每行代表一个值（如 "city=北京"），每列是 64 个 item 的存在性位。这个二维 tensor 是连续存储的，顺序读取 cache 友好。

**bitmap_lookup 字典**: 记录 `(field, value) → bitmap 行号` 的映射关系。查询时通过这个字典将谓词 `city == "北京"` 映射到 bitmap 表中的具体行号。这个映射保存在 `meta` JSON 中，离线/在线共享。

**分词器接口**: `Tokenizer` 是一个 Protocol，默认实现 `WhitespaceTokenizer` 按空格分词。可替换为 `JiebaTokenizer` 处理中文。分词在构建时一次性完成，每个 term 建一条 bitmap。

---

## 3. InvertedIndexModel：推理模型

**文件**: `offline/torch_recall/model.py`

这是整个系统的核心——一个 `nn.Module`，其 `forward()` 接收固定尺寸的查询张量，返回一条 packed int64 bitmap，表示所有匹配的 item。

### 模型状态 (Buffers)

```python
self.bitmaps       # [num_bitmaps, L]  离散+文本值的 bitmap 表
self.numeric_data  # [F, N]            数值列 (F 个字段 × N 个 item)
self.item_ids      # [N]               item ID 序列
self.all_ones      # [L]               全 1 bitmap (AND 运算的单位元)
self.valid_mask    # [L]               有效 item 掩码 (最后一个 word 可能不满 64 位)
self.pack_powers   # [64]              [1, 2, 4, ..., 2^62, -2^63] 用于 bool→int64 打包
```

所有数据通过 `register_buffer` 注册，会被 `torch.export` 捕获并嵌入到 `.pt2` 中。

### forward() 详解

`forward()` 的输入是 9 个固定尺寸的张量，描述了一个 DNF 查询的一个批次（最多 `CONJ_PER_PASS=8` 个 conjunction）。

```
输入:
  bitmap_indices  [16]       → 查询引用的 bitmap 行号
  bitmap_valid    [16]       → 哪些 bitmap 谓词是有效的
  numeric_fields  [8]        → 数值字段索引
  numeric_ops     [8]        → 比较操作 (0=eq, 1=lt, 2=gt, 3=le, 4=ge)
  numeric_values  [8]        → 比较阈值
  numeric_valid   [8]        → 哪些数值谓词是有效的
  negation_mask   [24]       → 哪些谓词需要取反
  conj_matrix     [8, 24]    → conjunction×predicate 布尔矩阵
  conj_valid      [8]        → 哪些 conjunction 是有效的

输出:
  result          [L]        → packed int64 bitmap, 匹配 item 的位设为 1
```

#### 阶段 1: 谓词 bitmap 收集

```python
# 离散/文本谓词: 从 bitmap 表中按索引取出
bp = self.bitmaps[bitmap_indices]        # [16, L]

# 数值谓词: 取出列数据，做 5-way 无分支向量化比较
cols = self.numeric_data[numeric_fields]  # [8, N]
numeric_bool = (
    ((cols == vals) & (ops == 0).unsqueeze(1))    # eq
    | ((cols < vals) & (ops == 1).unsqueeze(1))   # lt
    | ((cols > vals) & (ops == 2).unsqueeze(1))   # gt
    | ((cols <= vals) & (ops == 3).unsqueeze(1))   # le
    | ((cols >= vals) & (ops == 4).unsqueeze(1))   # ge
)
# bool[8, N] → packed int64[8, L]
np_bitmap = self._pack_bool_to_bitmap(numeric_bool)

# 合并为统一的谓词 bitmap 表
all_bm = cat([bp, np_bitmap])  # [24, L]
```

5-way OR 的设计是为了避免 `if/switch` 动态控制流（`torch.export` 不允许）。对于实际的数值谓词，只有对应的 `ops == X` 条件为 True，其他 4 个分支产生全 False 不影响结果。

#### 阶段 2: 取反和有效性掩码

```python
# 对 negated 谓词取反: XOR all_ones 再 AND valid_mask
all_bm = where(neg, (all_bm ^ all_ones) & valid_mask, all_bm)

# 无效谓词替换为 all_ones (AND 运算单位元, 不影响交集结果)
all_bm = where(all_valid, all_bm, all_ones)
```

`valid_mask` 处理 N 不是 64 的整数倍的情况——最后一个 int64 word 中超出 N 的位必须被清零，否则 NOT 操作会引入幽灵 item。

#### 阶段 3: 向量化 conjunction 评估

这是计算的核心。每个 conjunction 是若干谓词的 AND。`conj_matrix[c, p]` 表示 conjunction c 是否使用谓词 p。

```python
conj_results = all_ones.expand(8, L)  # 初始化为 AND 单位元

for p in range(24):  # 循环谓词 (不是循环 conjunction)
    mask_p = conj_matrix[:, p].unsqueeze(1)       # [8, 1] 哪些 conj 使用谓词 p
    pred_bm = all_bm[p].unsqueeze(0)               # [1, L] 谓词 p 的 bitmap
    selected = where(mask_p, pred_bm, all_ones)     # 不使用则保持 all_ones
    conj_results &= selected                        # 并行 AND 到所有 conj
```

**设计关键**: 外循环是谓词 (24 次)，内部在 8 个 conjunction 上并行。这比双循环 (conj × pred) 产生的静态图节点少得多：24 个 `where` + 24 个 `&` = 48 个节点，而非 24×8=192 个。

#### 阶段 4: 无效 conjunction 置零 + OR 树规约

```python
# 无效 conjunction 设为全 0 (OR 单位元)
conj_results = where(conj_valid, conj_results, zeros)

# 树规约: [8,L] → [4,L] → [2,L] → [1,L]
for _ in range(3):
    cr = cr.view(size//2, 2, L)
    cr = cr[:, 0, :] | cr[:, 1, :]

result = cr.squeeze(0) & valid_mask  # [L]
```

树规约用固定 3 步将 8 个 conjunction 的结果 OR 合并为一条 bitmap。无效 conjunction 已被置零（OR 单位元），不影响结果。

#### _pack_bool_to_bitmap: bool → packed int64

```python
def _pack_bool_to_bitmap(self, bool_tensor):
    # bool_tensor: [B, N]
    # 补齐到 L*64 宽度，reshape 为 [B, L, 64]
    reshaped = bool_tensor.view(B, L, 64).long()
    # 每个 64-bit word: sum(bit_i * 2^i)
    packed = (reshaped * self.pack_powers).sum(dim=2)
    return packed  # [B, L]
```

`pack_powers` 是 `[1, 2, 4, ..., 2^62, -2^63]`。最后一个是负数是因为 int64 的最高位是符号位——`1 << 63` 在 int64 中表示为 `-2^63`。乘法 + 求和等价于将 64 个 bool 打包为一个 int64。

---

## 4. 查询处理管线：解析 → DNF → 张量编码

**文件**: `offline/torch_recall/query.py`

查询处理分为三个阶段：字符串解析 → DNF 转换 → 张量编码。

### 4.1 递归下降解析器

将查询字符串解析为 AST (抽象语法树)。

```
输入: '(city == "北京" OR city == "上海") AND price < 100'

词法分析 (_tokenize):
  [WORD:"city", OP:"==", STR:"北京", KW:"OR", WORD:"city", OP:"==", STR:"上海", ...]

语法分析 (_Parser):
  And(
    Or(
      Predicate("city", "==", "北京"),
      Predicate("city", "==", "上海")
    ),
    Predicate("price", "<", 100.0)
  )
```

解析器支持的语法：
- 运算符优先级: `NOT` > `AND` > `OR`
- 括号分组
- `contains` 关键字（文本匹配）
- `!=` 被转换为 `== + negated`

### 4.2 DNF 转换器

将任意嵌套的 AST 展开为**析取范式** (Disjunctive Normal Form): OR of ANDs。

```
输入 AST:
  AND(OR(city=北京, city=上海), price<100)

DNF 输出 (分配律展开):
  [
    [city=北京 AND price<100],      ← conjunction 0
    [city=上海 AND price<100],      ← conjunction 1
  ]
```

转换规则：
- `AND(A, B)`: 对 A 和 B 的 DNF 做笛卡尔积
- `OR(A, B)`: 合并 A 和 B 的 DNF
- `NOT(AND(...))`: De Morgan 定律 → `OR(NOT(...))`，然后递归
- `NOT(OR(...))`: De Morgan 定律 → `AND(NOT(...))`，然后递归
- `NOT(NOT(X))`: 双重否定消去

DNF 展开可能指数膨胀。系统在过程中检查 `len(result) > MAX_CONJ` 并及时拒绝。

### 4.3 张量编码器

将 DNF 编码为 `forward()` 所需的固定尺寸张量。分两步：

**Step 1: `_collect_predicates`** — 扫描所有 conjunction，收集去重后的谓词集合：

```
DNF 中出现的谓词:
  city=北京 (bitmap #5)     → bp_idx 0
  city=上海 (bitmap #8)     → bp_idx 1
  price < 100 (field 0, op 1) → np_idx 0

填充共享张量:
  bitmap_indices = [5, 8, 0, 0, ..., 0]  (MAX_BP=16 个)
  bitmap_valid   = [T, T, F, F, ..., F]
  numeric_fields = [0, 0, ..., 0]        (MAX_NP=8 个)
  numeric_ops    = [1, 0, ..., 0]
  numeric_values = [100.0, 0, ..., 0]
  numeric_valid  = [T, F, ..., F]
  negation_mask  = [F, F, ..., F]        (P_TOTAL=24 个)
```

**Step 2: `_fill_conj_matrix`** — 为每个 conjunction 标记引用了哪些谓词：

```
conj_matrix (CONJ_PER_PASS=8 × P_TOTAL=24):
  conj 0: [1, 0, ..., 0 | 1, 0, ..., 0]  ← city=北京 AND price<100
  conj 1: [0, 1, ..., 0 | 1, 0, ..., 0]  ← city=上海 AND price<100
  conj 2-7: [0, 0, ..., 0]                ← padding (conj_valid=False)
```

**多 pass 分批**: 当 DNF > 8 个 conjunction 时，`_encode_dnf_multi` 按 `CONJ_PER_PASS` 分块。共享谓词张量（bitmap_indices 等）只收集一次，每个 batch 生成不同的 `conj_matrix` 和 `conj_valid`。

---

## 5. Exporter：torch.export + AOTInductor 导出

**文件**: `offline/torch_recall/exporter.py`

```python
def export_model(model, meta, output_path):
    example_inputs = create_example_inputs(model)  # 9 个全零张量，形状固定
    exported = torch.export.export(model, example_inputs)  # trace 静态计算图
    torch._inductor.aoti_compile_and_package(exported, package_path=output_path)
```

导出过程：

1. **`torch.export.export`**: 以 `example_inputs` trace `forward()`，生成一个完全静态的计算图 (ExportedProgram)。图中不包含任何 Python 控制流——所有 `for` 循环被展开，所有 `torch.where` 保留为图中的节点。模型的 buffer (bitmaps, numeric_data 等) 被捕获为图的常量。

2. **`aoti_compile_and_package`**: 将 ExportedProgram 编译为 C++ 内核代码，打包为 `.pt2` 文件。`.pt2` 包含编译后的共享库 (`.so`/`.dylib`) + 序列化的张量常量 + 元信息。

`.pt2` 文件可以在没有 Python 的环境中由 C++ 的 `AOTIModelPackageLoader` 直接加载执行。

---

## 6. C++ 在线推理引擎

### 6.1 类型系统

**文件**: `online/include/torch_recall/query_parser.h`

C++ 端用 `std::variant` 实现与 Python 端等价的 AST 类型：

```cpp
struct Predicate { string field; string op; variant<string, double> value; };
struct AndExpr   { vector<Expr> children; };
struct OrExpr    { vector<Expr> children; };
struct NotExpr   { Expr child; };

using Expr = variant<Predicate, shared_ptr<AndExpr>, shared_ptr<OrExpr>, shared_ptr<NotExpr>>;
```

`shared_ptr` 用于递归结构（variant 不能直接嵌套自身）。

### 6.2 查询解析器

**文件**: `online/src/query_parser.cpp`

与 Python 完全对等的递归下降解析器。用 `std::regex` 做词法分析，`Parser` 类做语法分析。支持相同的语法和优先级规则。

### 6.3 DNF 转换器

**文件**: `online/src/dnf_converter.cpp`

与 Python 的 `to_dnf` 逻辑完全对等：AND 做笛卡尔积，OR 做拼接，NOT 用 De Morgan 定律递归展开。同样检查 `MAX_CONJ` 上限。

### 6.4 张量编码器

**文件**: `online/src/tensor_encoder.cpp`

与 Python 的 `_collect_predicates` + `_fill_conj_matrix` 逻辑对等。关键差异：

- C++ 端直接操作 `torch::Tensor` (LibTorch API)
- 通过 `IndexMetadata` 结构体访问字典映射（从 `index_meta.json` 加载）
- 返回 `vector<QueryTensors>` 支持多 pass

### 6.5 模型执行器

**文件**: `online/src/model_runner.cpp`

```cpp
struct ModelRunner::Impl {
    torch::inductor::AOTIModelPackageLoader loader;
};

torch::Tensor ModelRunner::run(...) {
    vector<torch::Tensor> inputs = { ... };
    auto outputs = impl_->loader.run(inputs);  // 直接调用编译后的 C++ 内核
    return outputs[0];
}
```

`AOTIModelPackageLoader` 是 PyTorch 提供的 C++ API，加载 `.pt2` 中预编译的内核代码并执行。没有 Python 解释器、没有 JIT 编译、没有图优化——纯粹的原生函数调用。

### 6.6 结果解码器

**文件**: `online/src/result_decoder.cpp`

从 packed int64 bitmap 中提取匹配的 item ID：

```cpp
while (word != 0) {
    int bit = __builtin_ctzll(word);   // 硬件 count-trailing-zeros, ~1 cycle
    result.push_back(base + bit);
    word &= word - 1;                  // 清除最低位的 1
}
```

`__builtin_ctzll` 是 GCC/Clang 内建函数，编译为单条 CPU 指令 (`tzcnt` / `rbit+clz`)。

### 6.7 CLI 主入口

**文件**: `online/src/main.cpp`

将以上组件串联：

```
argv[query_str]
  → parse_expression()     → Expr AST
  → to_dnf()               → DNF (vector<vector<LiteralPred>>)
  → encode_dnf()           → vector<QueryTensors> (可能多个 batch)
  → for each batch:
      runner.run(tensors)   → torch::Tensor [L]
      result |= pass_result
  → ResultDecoder::decode() → vector<int64_t> (匹配 item IDs)
```

多 pass 的结果通过逐位 OR (`result | pass_result`) 合并。

---

## 7. 数据流全景图

```
                           离线 (Python)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  items: list[dict]                                                  │
│       │                                                             │
│       ▼                                                             │
│  IndexBuilder.build()                                               │
│       │                                                             │
│       ├── Pass 1: 扫描 → discrete_dicts, text_dicts                 │
│       │                   (值字典: "北京"→0, "上海"→1, ...)           │
│       │                                                             │
│       ├── 分配 bitmap_layout                                        │
│       │   bitmap 行 0: d:city val_id=0                              │
│       │   bitmap 行 1: d:city val_id=1                              │
│       │   ...                                                       │
│       │   bitmap 行 K: t:title term_id=0                            │
│       │                                                             │
│       ├── Pass 2: 逐 item 位设置                                     │
│       │   bitmaps[global_idx, idx>>6] |= 1<<(idx&63)               │
│       │                                                             │
│       ├── 数值列: numeric_data[field_idx, item_idx] = value          │
│       │                                                             │
│       ▼                                                             │
│  InvertedIndexModel(bitmaps, numeric_data, item_ids)                │
│       │                                                             │
│       ▼                                                             │
│  torch.export.export(model, example_inputs)                         │
│       │  trace forward() → 静态计算图 (ExportedProgram)              │
│       ▼                                                             │
│  aoti_compile_and_package → model.pt2                               │
│       │  编译为 C++ 内核 + 打包 buffer 数据                          │
│       │                                                             │
│  builder.save_meta → index_meta.json                                │
│       │  保存字典映射和元信息                                        │
│       │                                                             │
└───────┼─────────────────────────────────────────────────────────────┘
        │
        │  model.pt2 + index_meta.json
        │
┌───────┼─────────────────────────────────────────────────────────────┐
│       ▼                         在线 (C++ / Python)                 │
│                                                                     │
│  query_str: 'city == "北京" AND price < 100'                        │
│       │                                                             │
│       ▼                                                             │
│  parse_expression(query_str) → AST                                  │
│       │  递归下降: OR > AND > NOT > primary > predicate             │
│       │                                                             │
│       ▼                                                             │
│  to_dnf(AST) → DNF                                                 │
│       │  De Morgan + 分配律:                                        │
│       │  AND(OR(A,B), C) → [(A,C), (B,C)]                          │
│       │                                                             │
│       ▼                                                             │
│  encode_dnf(DNF, meta) → vector<QueryTensors>                      │
│       │  Step 1: 收集去重谓词 → 共享张量                             │
│       │  Step 2: 分批填充 conj_matrix                               │
│       │                                                             │
│       ▼                                                             │
│  for each batch:                                                    │
│    model.forward(tensors) → [L] int64 bitmap                       │
│       │  ┌────────────────────────────────────────────────┐         │
│       │  │ 1. 取 bitmap 谓词: bitmaps[indices] → [16, L]  │         │
│       │  │ 2. 数值比较+打包: cols vs vals → [8, L]        │         │
│       │  │ 3. 合并: [24, L]                              │         │
│       │  │ 4. 取反+有效性掩码                             │         │
│       │  │ 5. 向量化 AND: 24 次循环, 8 个 conj 并行       │         │
│       │  │ 6. 树规约 OR: [8,L]→[4,L]→[2,L]→[1,L]       │         │
│       │  │ 7. & valid_mask → result [L]                  │         │
│       │  └────────────────────────────────────────────────┘         │
│    result |= pass_result                                            │
│       │                                                             │
│       ▼                                                             │
│  ResultDecoder.decode(result) → [item_id_0, item_id_1, ...]        │
│       │  逐 word: __builtin_ctzll 提取设为 1 的位                    │
│       │                                                             │
│       ▼                                                             │
│  返回匹配的 item ID 列表                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附: 关键设计决策一览

| 决策 | 选择 | 理由 |
|------|------|------|
| 集合表示 | Dense packed int64 bitmap | 计算量固定、无分支、cache 友好、可被 AOTInductor 编译 |
| 数值处理 | 运行时 5-way 向量化比较 | 值空间连续无法预建 bitmap；5-way OR 避免动态控制流 |
| 查询标准化 | DNF (OR of ANDs) | 统一表示任意布尔表达式，直接映射到 bitmap AND+OR 操作 |
| NOT 实现 | 谓词级取反 (XOR all_ones) | 在 DNF 中 NOT 被 De Morgan 推到叶节点，只需谓词级 XOR |
| 固定尺寸输入 | padding + validity mask | torch.export 要求完全静态形状 |
| conjunction 循环方向 | 外循环谓词，内循环 conjunction | 图节点数 = P_TOTAL (24) 而非 P_TOTAL × CONJ (192) |
| OR 合并 | 树规约 | 固定 log₂(CONJ_PER_PASS) 步，比循环 OR 产生更少图节点 |
| 多 pass | 分批 conj_matrix + 结果 OR | 支持 1024 conjunctions 同时保持小模型图 |
| int64 最高位 | pack_powers 最后元素 = -(1<<63) | int64 符号位特殊处理，保证打包正确 |
| Python 位解码 | `& ((1<<64)-1)` | Python 无限精度整数，负值会导致 `word &= word-1` 无限循环 |

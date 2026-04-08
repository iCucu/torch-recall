# 实现细节与优化手段

本文档详细介绍 torch-recall 的核心数据结构、查询处理流程、编译兼容性设计，以及各项性能优化策略。

---

## 1. 核心数据结构

### 1.1 Packed int64 Bitmap

每个离散值或文本 term 对应一条 bitmap，记录哪些 item 拥有该属性。Bitmap 用 `int64` 紧密打包——每个 int64 存储 64 个 item 的状态：

```
bitmap["city=北京"] = [0b...0101]   →  item 0, 2 是北京
bitmap["gender=男"]  = [0b...1001]   →  item 0, 3 是男
```

100 万 item 只需 `⌈1,000,000 / 64⌉ = 15,625` 个 int64（~122 KB/bitmap）。

查询时用位运算完成集合操作，一条指令同时处理 64 个 item：

| 操作 | 位运算 | 含义 | 计算量 (1M items) |
|------|--------|------|-------------------|
| AND | `a & b` | 交集 | 15,625 次 int64 AND |
| OR | `a \| b` | 并集 | 15,625 次 int64 OR |
| NOT | `~a & valid_mask` | 补集 | 15,625 次 XOR + AND |

### 1.2 数值列存储

数值字段以 `[F, N]` 的 float32 矩阵存储（F 个字段 × N 个 item）。查询时做向量化比较：

```python
bool_mask = numeric_data[field_idx] < threshold   # 一次性比较 N 个 item
packed_bitmap = pack_bool_to_int64(bool_mask)      # 打包为 bitmap，参与位运算
```

支持 5 种数值比较 `==`, `<`, `>`, `<=`, `>=`，在 forward 中通过 5-way OR 实现无分支向量化比较，所有 item 的比较结果一次性打包成 bitmap 参与后续位运算。

---

## 2. 查询处理流程

任意布尔表达式通过 4 步转换为静态张量输入：

```
查询字符串                          递归下降解析器
  (city == "北京" OR city == "上海") AND price < 100
        ↓
AST 表达式树                        DNF 转换器 (De Morgan + 分配律)
  AND(OR(city=北京, city=上海), price<100)
        ↓
DNF 析取范式                        平坦化为 "OR of ANDs"
  (city=北京 AND price<100) OR (city=上海 AND price<100)
        ↓
张量编码                            分批填充到 CONJ_PER_PASS=64
  [batch_0, batch_1, ...]          每批生成一组 forward() 输入
```

**DNF (析取范式)** 是关键：将任意嵌套的布尔表达式展开为 "OR of ANDs" 的平坦形式，每个 conjunction（AND 子句）内部只做 bitmap 交集，最终对所有 conjunction 做并集。

---

## 3. 两级 Conjunction 处理

为了同时支持大规模 DNF（最多 1024 conjunctions）和高效编译，系统采用两级设计：

| 层级 | 常量 | 用途 |
|------|------|------|
| **模型级** | `CONJ_PER_PASS=8` | 单次 forward() 处理的 conjunction 数，决定 .pt2 编译图大小 |
| **系统级** | `MAX_CONJ=1024` | 单条查询允许的最大 conjunction 数 |

当 DNF 超过 8 个 conjunction 时，query encoder 自动分批，每批调用一次 forward()，结果 OR 合并：

```
DNF (30 conjunctions)
  → batch 0 (conj 0-7)   → forward() → result_0
  → batch 1 (conj 8-15)  → forward() → result_1
  → batch 2 (conj 16-23) → forward() → result_2
  → batch 3 (conj 24-29) → forward() → result_3
  → final = result_0 | result_1 | result_2 | result_3
```

CONJ_PER_PASS 设为较小的值 (8) 是刻意的优化选择。典型查询只有 1-4 个 conjunction，较小的 pass 尺寸避免了对大量无效 padding 的计算。复杂查询通过多 pass 线性扩展，总计算量反而更少（P_TOTAL×CONJ_PER_PASS = 24×8 = 192 单位/pass，远小于旧方案的 96×64 = 6,144 单位/pass）。

---

## 4. torch.export 兼容性设计

`torch.export` 要求完全静态的计算图。以下设计保证兼容性：

1. **固定大小输入**: 所有查询张量填充到编译期常量 `MAX_BP=16, MAX_NP=8, CONJ_PER_PASS=8`，用 validity mask 标记实际有效位
2. **向量化 AND + 树规约 OR**: forward 中对 `CONJ_PER_PASS` 个 conjunction 做向量化 AND（循环 `P_TOTAL=24` 次），再用 `log₂(8)=3` 级树规约做 OR 合并——整个图仅约 ~30 个静态节点
3. **返回固定形状**: 返回 `[L]` 的 packed bitmap（而非变长 ID 列表），bitmap → item ID 的解码在模型外部完成
4. **无动态控制流**: 所有条件逻辑用 `torch.where` 替代 `if`，确保 trace 路径唯一

### .pt2 导出与 C++ 推理

```python
# 离线：Python 导出
exported = torch.export.export(model, example_inputs)
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
```

```cpp
// 在线：C++ 加载
torch::inductor::AOTIModelPackageLoader loader("model.pt2");
auto outputs = loader.run(inputs);  // 直接调用 AOTInductor 编译后的内核
```

`.pt2` 包含 AOTInductor 预编译的 C++ 内核代码，加载后直接执行，不经过 Python 解释器，也无需 TorchScript。

---

## 5. 性能优化策略

### 5.1 位图打包：64x 数据并行

传统做法用 `bool[]` 数组存储集合，需要 N 次逐元素操作。Packed int64 将 64 个 bool 压缩到一个 int64，一次 `&` 运算等价于 64 次 bool AND。

对于 1M items，一次 AND 操作只需 15,625 次 int64 运算，而非 1,000,000 次 bool 运算。

### 5.2 内存布局：连续存储 + Cache 友好

- **Bitmap 表**: `[num_entries, L]` 的连续 int64 tensor，每个 entry 122 KB，顺序读取命中 L1/L2 cache
- **数值列**: `[F, N]` 的连续 float32 tensor，列存布局保证同字段的所有 item 值连续

### 5.3 数值谓词：向量化比较 + 动态打包

数值比较不预建 bitmap（值空间连续，无法穷举），而是运行时生成：

```python
# 5-way 无分支比较，利用 PyTorch 向量化
numeric_bool = (
    ((cols == vals) & (ops == 0).unsqueeze(1))
    | ((cols < vals) & (ops == 1).unsqueeze(1))
    | ((cols > vals) & (ops == 2).unsqueeze(1))
    | ((cols <= vals) & (ops == 3).unsqueeze(1))
    | ((cols >= vals) & (ops == 4).unsqueeze(1))
)
# 运行时打包: bool[N] → int64[L]
np_bitmap = pack_bool_to_bitmap(numeric_bool)
```

### 5.4 向量化 Conjunction 评估 + 树规约

旧方案用双层循环（`MAX_CONJ × P_TOTAL`）逐个处理 conjunction，扩展到 1024 时图节点爆炸。

新方案反转思路——**循环谓词，向量化 conjunction**：

```python
# 初始化: 所有 conjunction 以 all_ones 开始 (AND 单位元)
conj_results = all_ones.expand(CONJ_PER_PASS, L)  # [8, L]

# AND 循环: P_TOTAL=24 次迭代，每次在 8 个 conjunction 上并行
for p in range(P_TOTAL):
    mask = conj_matrix[:, p]              # [8] → 哪些 conjunction 使用谓词 p
    selected = where(mask, all_bm[p], all_ones)  # [8, L]
    conj_results &= selected

# OR 树规约: 3 级折叠  [8, L] → [4, L] → [2, L] → [1, L]
for level in range(3):
    cr = cr.view(size//2, 2, L)
    cr = cr[:, 0, :] | cr[:, 1, :]
```

| 方案 | 每 pass 计算量 | 说明 |
|------|--------------|------|
| 旧 (P_TOTAL=96, CONJ=64) | 96 × 64 = 6,144 | 典型查询 (1 谓词 1 conj) 浪费 99.9% |
| **新 (P_TOTAL=24, CONJ=8)** | **24 × 8 = 192** | **减少 32 倍，典型查询浪费 ~87%** |

### 5.5 torch.compile / AOTInductor 内核融合

`torch.compile` 和 AOTInductor 将向量化 AND 循环 + 树规约融合为高效 C++ 内核，相比 Eager 模式提速约 8 倍。

| 模式 | 处理方式 | 延迟 (1M items) |
|------|---------|-----------------|
| Eager | Python 循环 + ATen 调用 | ~20ms |
| torch.compile | Triton/C++ 内核融合 | ~3.3ms |
| AOTInductor (.pt2) | 预编译 C++ 代码 | <0.1ms |

### 5.6 索引构建优化

构建过程对 N 个 item 执行单次扫描（O(N × 平均字段数)），直接通过位操作写入预分配的 bitmap tensor，避免中间数据结构：

```python
for idx in range(N):
    word_idx = idx >> 6           # idx // 64
    bit_val = 1 << (idx & 63)    # 直接位设置，无需打包
    bitmaps[global_idx, word_idx] |= bit_val
```

### 5.7 结果解码：硬件位扫描

解码阶段从 packed bitmap 提取匹配 item ID。C++ 端使用硬件 `ctz`（count trailing zeros）指令逐位提取：

```cpp
while (word) {
    int bit = __builtin_ctzll(word);   // 硬件指令，1 cycle
    ids.push_back(base + bit);
    word &= word - 1;                  // 清除最低位
}
```

Python 端需注意 `int64` 的 `.item()` 返回无限精度整数，必须 `& ((1 << 64) - 1)` 截断到 64 位，否则负值会导致无限循环。

---

## 6. 与主流倒排索引的架构对比

torch-recall 与 Elasticsearch (Lucene) 等主流搜索引擎在数据结构和执行模型上有本质差异：

| | torch-recall | Elasticsearch / Lucene |
|---|---|---|
| **索引结构** | Dense packed int64 bitmap — 每个值一条定长 bitmap | Posting list — 变长压缩 doc ID 列表 + skip list |
| **集合运算** | 整条 bitmap 位运算 (AND/OR/NOT) | 迭代器模型，逐 doc ID 推进 + 跳表加速交集 |
| **选择率敏感性** | 不敏感——bitmap 长度固定，计算量与命中数无关 | 敏感——posting list 长度与匹配数正相关 |
| **稀疏值效率** | 低——bitmap 定长，稀疏值浪费空间 | 高——只存匹配的 doc ID，变长编码压缩 |
| **执行层** | AOTInductor → 预编译 C++ 内核，无解释器 | JVM 解释 / C2 JIT + 迭代器链 |
| **更新能力** | 静态（需重建） | 近实时（段合并） |

**设计选择的本质**：torch-recall 用空间换时间 + 确定性。不压缩 bitmap 意味着更多内存，但换来了固定计算量、无分支、CPU cache 友好，以及可被 AOTInductor 完整编译为原生代码的能力。这在推荐系统的静态 item 池过滤场景中是合理的 trade-off。

详细性能对比数据见 [benchmark.md](benchmark.md#与主流系统的对比)。

---

## 7. 设计限制与缓解

| 限制 | 影响 | 缓解策略 |
|------|-----|---------|
| DNF 展开可能指数膨胀 | 极端嵌套查询超出 MAX_CONJ=1024 | 上限 1024，超出拒绝；多 pass 设计使延迟线性退化 |
| 高基数离散字段 | 每个值一条 bitmap，内存线性增长 | 基数 >5000 建议退化为数值或分桶 |
| 文本分词质量 | 影响 text contains 召回率 | 可插拔分词器接口，默认空格，可替换 jieba |
| Bitmap 内存 | ~122 KB/entry，1152 entries = 137 MB | 内存与 (基数总和 × item 数) 线性相关 |
| Python 解码性能 | `.item()` 返回无限精度整数 | 必须 `& ((1<<64)-1)` 截断，否则负值死循环 |
| CONJ_PER_PASS 与延迟的权衡 | 值越大单 pass 越慢，总 pass 数越少 | 默认 8，典型查询最优；可调整后重新导出 |

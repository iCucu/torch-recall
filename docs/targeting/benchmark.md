# 性能测试结果

测试环境：Apple M 系列 CPU，Python 3.12，PyTorch 2.x。

测试数据：合成定向规则，Schema 含 3 个离散字段 (基数 50+2+100) + 2 个数值字段 + 1 个文本字段 (~200 terms)。每条规则 1-4 个谓词，约 30% 规则包含 OR（2 个 conjunction）。

---

## 运行 Benchmark

```bash
cd torch-recall
source .venv/bin/activate

# 核心性能测试 (构建、内存、延迟)
PYTHONPATH=index python index/benchmarks/bench_targeting.py

# 方案对比 (暴力Python / pyroaring / torch-recall)
PYTHONPATH=index python index/benchmarks/bench_comparison.py

# 加 --large 运行 1M 规模
PYTHONPATH=index python index/benchmarks/bench_targeting.py --large
PYTHONPATH=index python index/benchmarks/bench_comparison.py --large
```

---

## 查询延迟 — 各规模

### forward() 延迟

| 规模 (N items) | Eager | torch.compile | 编码 (encode_user) | E2E (encode+eager) |
|---------------|-------|--------------|--------------------|--------------------|
| 1K | 0.11ms | 0.12ms | 0.05ms | 0.17ms |
| 10K | 0.36ms | 0.15ms | 0.05ms | 0.43ms |
| 100K | 1.02ms | 0.37ms | 0.05ms | 1.10ms |
| **1M** | **6.67ms** | **2.65ms** | **0.05ms** | **7.05ms** |

关键观察：
- `encode_user` 耗时恒定 ~0.05ms，与 item 数无关（只遍历谓词注册表，~428 个谓词）
- `forward()` 延迟与 N 近似线性，因为核心操作是 [C, K] 和 [N, J] 的 gather + reduce
- `torch.compile` 相比 Eager 提速 **2.5–2.8x**（100K–1M 规模）

---

## 索引构建

| 规模 | 构建时间 | 内存占用 | 谓词数 | Conjunction 数 |
|------|---------|---------|--------|---------------|
| 1K | 0.02s | 0.2 MB | 416 | 1,321 |
| 10K | 0.21s | 2.4 MB | 428 | 13,024 |
| 100K | 2.18s | 23.7 MB | 428 | 130,214 |
| **1M** | **22.3s** | **236.5 MB** | **428** | **1,299,622** |

谓词数在 ~428 处饱和（由离散值空间 + 数值阈值 + 文本词表决定），不随 item 数增长。Conjunction 数与 item 数线性增长（~1.3 conj/item）。

### 内存构成

以 1M items 为例 (C≈1.3M, K=8, J=16)：

| 张量 | 大小 | 占比 |
|------|------|-----|
| conj_pred_ids [C, K] int64 | 78 MB | 33% |
| conj_pred_negated [C, K] bool | 10 MB | 4% |
| conj_pred_valid [C, K] bool | 10 MB | 4% |
| item_conj_ids [N, J] int64 | 122 MB | 52% |
| item_conj_valid [N, J] bool | 15 MB | 6% |
| **总计** | **~237 MB** | |

---

## 方案对比

三种「反向匹配」方案在同一数据集上的实测对比。

### 对比方案

| 方案 | 实现方式 |
|------|---------|
| **暴力 Python** | 逐 item 评估预解析的 DNF，纯 Python 循环 |
| **pyroaring 倒排索引** | 倒排索引结构 (predicate → conjunction 映射)，Python 集合运算 |
| **torch-recall (eager)** | Two-Level Gather+Reduce，PyTorch 张量操作，无编译 |
| **torch-recall (compiled)** | 同上 + torch.compile 内核融合 |

注意：所有方案的结果经过一致性验证，在所有测试用例上命中完全一致。

### 查询延迟 (ms, avg)

| 规模 | 暴力 Python | pyroaring | torch-recall (eager) | torch-recall (compiled) |
|------|-----------|-----------|---------------------|------------------------|
| 1K | 0.47 | 0.23 | 0.19 | 0.18 |
| 10K | 4.51 | 1.91 | 0.58 | **0.22** |
| 100K | 47.26 | 22.44 | 1.51 | **0.44** |

### 加速比 (相对暴力 Python)

| 规模 | pyroaring | torch-recall (eager) | torch-recall (compiled) |
|------|-----------|---------------------|------------------------|
| 1K | 2.0x | 2.4x | 2.6x |
| 10K | 2.4x | 7.8x | **20.2x** |
| 100K | 2.1x | **31.3x** | **107.9x** |

### 构建时间 (s)

| 规模 | 暴力 Python (预解析) | pyroaring | torch-recall |
|------|-------------------|-----------|-------------|
| 1K | 0.04 | 0.01 | 0.02 |
| 10K | 0.11 | 0.07 | 0.26 |
| 100K | 0.91 | 0.80 | 2.22 |

torch-recall 构建较慢（需创建张量），但这是一次性离线操作，查询延迟的优势远大于构建开销。

---

## 各执行模式延迟总结

| 模式 | 延迟 (100K items) | 延迟 (1M items) | 说明 |
|------|-----------------|----------------|------|
| 暴力 Python | 47ms | ~470ms (估) | 纯解释器循环 |
| pyroaring 倒排 | 22ms | ~220ms (估) | C 库集合运算 + Python 循环 |
| torch-recall (eager) | 1.5ms | 6.7ms | PyTorch tensor 操作 |
| **torch-recall (compiled)** | **0.44ms** | **2.65ms** | **torch.compile 内核融合** |
| torch-recall (AOTInductor) | TBD | TBD | 预编译 C++ 内核 (预期 <1ms) |

---

## 结果分析

### 1. torch-recall 的优势随规模增大而显著

小规模 (1K) 时各方案延迟接近（均 <0.5ms）。到 100K 时，torch-recall compiled 比暴力 Python 快 **108x**，比 pyroaring 快 **51x**。

原因：Python 循环的开销与 N 线性增长，而 torch-recall 的 gather+reduce 被 PyTorch 底层的向量化内核处理——单次 ATen 调用处理所有 C 个 conjunction 或 N 个 item。

### 2. pyroaring 在反向匹配中优势有限

传统倒排索引（pyroaring/ES/Lucene）在**正向检索**中表现出色：查询条件少，posting list 短，跳表加速交集运算。

但在**反向匹配**中，每个 item 有独立的 DNF 规则，必须逐 item 评估 conjunction 是否满足。pyroaring 虽然减少了谓词评估开销（共享谓词结果），但仍需 Python 循环遍历所有 item 的 conjunction 结构。这个循环是瓶颈。

torch-recall 将整个 item × conjunction 评估张量化为一次 gather + reduce 操作，消除了 Python 循环。

### 3. encode_user 耗时可忽略

用户编码 ~0.05ms，与 item 数无关（仅遍历谓词注册表）。即使在 1M 规模下，编码时间也只占总延迟的 <1%。

### 4. torch.compile 带来显著加速

在 10K+ 规模下 torch.compile 比 Eager 快 2-3x。内核融合将多次独立的 gather/reduce 调用合并为一次高效的 C++ 内核调用。

AOTInductor 导出后预期进一步消除 Python 框架开销。

---

## 与传统倒排索引的架构对比

| 维度 | torch-recall | 传统倒排索引 (ES/Lucene/pyroaring) |
|------|-------------|--------------------------------|
| **问题方向** | 反向：item规则 → 用户属性 | 正向：查询条件 → item 集合 |
| **索引结构** | 两级张量：conj→pred + item→conj | Posting list：term → doc ID 列表 |
| **评估方式** | 全量张量 gather + reduce | 迭代器模型 + 跳表交集 |
| **Python 循环** | 无 (全部委托给 PyTorch 内核) | 必需 (遍历 item conjunction 结构) |
| **编译优化** | torch.compile / AOTInductor | 无 (依赖 Python / C 库原生速度) |
| **选择率敏感** | 不敏感 (固定 gather 尺寸) | 敏感 (posting list 长度) |
| **动态更新** | 不支持 (需重建) | 支持 |
| **部署形态** | .pt2 嵌入式 / C++ | 独立服务 / 嵌入式库 |

### 核心差异

**传统倒排索引解决的是正向问题**：给定少量查询条件，从大量 item 中筛选匹配的。通过 posting list 跳表，可以跳过大量不匹配的 item，复杂度接近 O(k) 而非 O(N)。

**定向召回是反向问题**：每个 item 有自己的规则，必须对所有 item 评估其规则是否被用户满足。没有有效的跳过机制——除非用户不满足某个谓词，且该谓词被大量 item 使用（这需要更复杂的倒排结构，如 BE-tree）。

torch-recall 选择了不同的策略：不试图跳过 item，而是将全量评估**向量化**。通过 PyTorch 的高效 tensor 操作（底层 BLAS/SIMD），一次 forward 调用处理所有 N 个 item，单次操作的 per-item 开销极低。

---

## 适用场景

| 场景 | 推荐方案 |
|------|---------|
| **广告定向 / 受众匹配**：静态 item 规则池 + 用户属性匹配 | **torch-recall** — 亚毫秒延迟，可编译导出 |
| **全文检索 + 布尔过滤**：query → item 正向检索 | **ES / Lucene** — 功能完整，生态成熟 |
| **实时数据 + 频繁更新** | **ES** — 近实时索引，自动分片 |
| **嵌入推理管线**：与 PyTorch 模型串联 | **torch-recall** — tensor 原生接口，compile/export/C++ 全链路 |
| **纯集合运算** (无 conjunction 结构) | **pyroaring** — 最快，压缩率高，SIMD |

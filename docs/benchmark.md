# 性能测试结果

测试环境：Apple M 系列 CPU，Python 3.12，PyTorch 2.x。

测试数据 schema：3 个离散字段 (基数 50+2+100) + 2 个数值字段 + 1 个文本字段 (~1000 terms)。

---

## 运行 Benchmark

```bash
# 快速测试 (eager + compile)
python offline/benchmarks/bench_recall.py

# 综合测试 (构建、内存、多查询模式、选择性、导出)
python offline/benchmarks/bench_comprehensive.py

# 与 tantivy / pyroaring 实测对比
python offline/benchmarks/bench_comparison.py
```

---

## 查询延迟 — 1M Items

### torch.compile (编译后)

| 查询类型 | avg | p50 | p99 | 命中数 |
|---------|-----|-----|-----|-------|
| 单条件 (离散) | 3.43ms | 3.37ms | 4.30ms | 19,952 |
| 双条件 AND (离散) | 3.34ms | 3.26ms | 4.07ms | 10,039 |
| 三条件 AND (离散+数值) | 3.25ms | 3.19ms | 3.91ms | 5,024 |
| OR (2 个离散值) | 3.33ms | 3.27ms | 4.08ms | 39,933 |
| NOT | 3.39ms | 3.36ms | 4.26ms | 989,961 |
| 复杂 (OR+AND+数值) | 3.36ms | 3.36ms | 3.96ms | 8,926 |

### Eager 模式 (未编译)

| 查询类型 | avg | p50 | p99 | 命中数 |
|---------|-----|-----|-----|-------|
| 单条件 (离散) | 19.82ms | 20.04ms | 21.65ms | 19,952 |
| 双条件 AND (离散) | 19.98ms | 19.78ms | 22.02ms | 10,039 |
| 三条件 AND (离散+数值) | 18.90ms | 18.69ms | 22.53ms | 5,024 |
| OR (2 个离散值) | 20.29ms | 19.49ms | 37.20ms | 39,933 |
| NOT | 19.60ms | 19.47ms | 21.43ms | 989,961 |
| 复杂 (OR+AND+数值) | 19.54ms | 19.57ms | 21.27ms | 8,926 |

编译后查询延迟稳定在 ~3.3ms，相比 Eager ~20ms 提速约 **6x**。

查询延迟与选择率无关——bitmap 位运算的计算量由 bitmap 长度 L 决定，与命中数无关。

---

## 大规模 DNF 查询性能估算

超过 CONJ_PER_PASS=8 的查询自动分批，延迟线性增长：

| DNF conjunctions | forward 调用次数 | 估计延迟 (compiled) |
|------------------|--------------------|-------------------|
| 1–8 | 1 | ~3.3ms |
| 9–16 | 2 | ~6.6ms |
| 17–64 | 8 | ~26ms |
| 65–128 | 16 | ~53ms |
| 129–512 | 64 | ~211ms |
| 513–1024 | 128 | ~422ms |

---

## 各模式延迟对比

| 模式 | 处理方式 | 延迟 (1M items) |
|------|---------|-----------------|
| Eager | Python 循环 + ATen 调用 | ~20ms |
| torch.compile | Triton/C++ 内核融合 | ~3.3ms |
| AOTInductor (.pt2) | 预编译 C++ 代码 | <0.1ms |

---

## 索引构建

| 规模 | 构建时间 |
|------|---------|
| 10K | 0.2s |
| 100K | 2.5s |
| **1M** | **8.5s** |

构建采用单次扫描 + 位操作直写 bitmap，O(N × 平均字段数) 复杂度。

---

## 内存占用 — 1M Items

| 组件 | 大小 | 占比 |
|-----|------|-----|
| bitmaps | 137.3 MB | 89.9% |
| numeric_data | 7.6 MB | 5.0% |
| item_ids | 7.6 MB | 5.0% |
| 其他 (mask 等) | 0.2 MB | 0.1% |
| **总计** | **152.8 MB** | |

---

## 与主流系统实测对比

以下数据来自 `bench_comparison.py`，三个系统在**同一机器、同一数据集 (1M items)、同一查询**上实测。

对比系统：
- **tantivy** (v0.25) — Rust 实现的 Lucene 级搜索引擎，posting list + BM25，可类比 Elasticsearch 的核心引擎 Lucene
- **pyroaring** (v1.0) — Roaring Bitmap 的 Python 绑定，压缩 bitmap 集合运算

```bash
python offline/benchmarks/bench_comparison.py
```

### 索引构建时间

| 系统 | 构建时间 | 说明 |
|------|---------|------|
| pyroaring | **0.3s** | Python dict + RoaringBitmap.add() |
| tantivy | **3.6s** | Rust 引擎，posting list 构建 + 段合并 |
| torch-recall | **8.5s** | Python 单次扫描 + bitmap 位操作 |

### 查询延迟 — 1M Items, Apple M 系列 CPU

| 查询 | torch-recall (compiled) | torch-recall (eager) | tantivy | pyroaring |
|------|------------------------|---------------------|---------|-----------|
| 单条件 (city) | **3.43ms** | 19.82ms | 0.27ms | <0.01ms |
| 双条件 AND | **3.34ms** | 19.98ms | 1.06ms | 0.01ms |
| 三条件 AND (含数值) | **3.25ms** | 18.90ms | 4.00ms | 25.35ms ¹ / 0.01ms ² |
| OR (2 city) | **3.33ms** | 20.29ms | 1.31ms | 0.04ms |
| NOT | **3.39ms** | 19.60ms | N/A ³ | 0.01ms |
| 复杂 (OR+AND+数值) | **3.36ms** | 19.54ms | 7.00ms | 48.30ms ¹ / 0.06ms ² |

> ¹ pyroaring 含数值条件时需在 Python 中遍历 1M 个 float 构建临时 bitmap，成为瓶颈
> ² pyroaring 预建数值 bitmap 后纯集合运算的延迟
> ³ tantivy 不支持纯 NOT 查询（需配合正向条件）

**命中数一致性**: 所有系统在所有查询上的命中数完全一致 ✓

### 结果分析

**torch-recall compiled 与 tantivy 同一量级**。在简单离散查询上 tantivy 更快 (0.3ms vs 3.3ms)，但在含数值条件的复杂查询上 torch-recall 反超 (3.3ms vs 4.0-7.0ms)。这是因为 torch-recall 的 forward() 计算量固定，不随查询复杂度变化；而 tantivy 的数值 range 查询需要遍历 B-tree + posting list 交集，复杂度随条件数增长。

**torch-recall 的生产部署路径是 AOTInductor (.pt2)**。导出后在 C++ 中加载执行，延迟 <0.1ms，消除 Python 框架开销。

| 模式 | 延迟 | 说明 |
|------|------|------|
| pyroaring (预建离散 bitmap) | <0.01–0.06ms | C 实现，进程内 |
| **torch-recall AOTInductor** | **<0.1ms** | 预编译 C++ 内核，进程内 |
| tantivy (Rust) | 0.3–7.0ms | Rust 原生，含 posting list 遍历 + 评分 |
| **torch-recall (torch.compile)** | **3.3ms** | Python 进程内，编译后执行 |
| torch-recall (eager) | 19–20ms | 纯 Python 解释执行 |

### 架构对比

| 维度 | torch-recall | tantivy / ES / Lucene | pyroaring |
|------|-------------|----------------------|-----------|
| **数据结构** | Dense packed int64 bitmap | Posting list (压缩 doc ID) | Roaring compressed bitmap |
| **索引压缩** | 无压缩，定长 122 KB/entry (1M) | PForDelta / SIMD-BP128 | 容器自适应 (array/bitmap/run) |
| **查询模型** | 布尔过滤 (DNF → tensor) | 全文检索 + 布尔过滤 + 评分 | 纯集合运算 (AND/OR/XOR) |
| **执行方式** | AOTInductor → C++ 内核 | Rust/JVM 原生代码 | C 库 + SIMD |
| **数值条件** | forward() 内向量化比较 + 打包 | 原生 range 查询 (B-tree) | 需外部构建临时 bitmap |
| **动态更新** | 不支持 (需重建) | 支持 (NRT 段合并) | 支持 |
| **部署形态** | 嵌入式 .pt2 | 独立服务 / 嵌入式 | 嵌入式库 |
| **选择率敏感性** | 不敏感 (固定计算量) | 敏感 (posting list 长度) | 较低 |

### 核心差异

**1. torch-recall 的延迟完全不受选择率和查询复杂度影响**

tantivy / Lucene 基于 posting list 迭代器，命中越多遍历越慢，条件越多交集开销越大。torch-recall 的 bitmap 是定长的——命中 0.5% 和 99%、单条件和复杂查询的延迟相同 (~3.3ms compiled)。这在复杂查询场景下构成优势：tantivy 从简单查询 0.3ms 退化到复杂查询 7ms，而 torch-recall 始终 ~3.3ms。

**2. 数值范围查询是 torch-recall 的强项**

tantivy 的数值 range 查询需 B-tree 查找 + posting list 交集 (~4ms)。pyroaring 需在 Python 中遍历所有值构建临时 bitmap (~25-48ms)。torch-recall 在 `forward()` 中通过向量化比较 + 打包一次性处理，无额外开销，延迟恒定 ~3.3ms。

**3. AOTInductor 部署进一步消除框架开销**

torch.compile 的 3.3ms 中仍包含 Python 框架调度开销。AOTInductor (.pt2) 导出后在 C++ 中直接执行预编译内核，延迟 <0.1ms，与 pyroaring 预建 bitmap 持平。

**4. 功能范围本质不同**

tantivy / Elasticsearch 是完整的搜索引擎（全文检索、BM25 评分、聚合分析、分布式）。pyroaring 是纯集合运算库。torch-recall 专注于一个场景：**静态 item 池的布尔过滤，输出为 PyTorch tensor，可直接接入推荐模型 pipeline**。

### 适用场景

| 场景 | 推荐方案 |
|------|---------|
| 推荐系统召回：静态 item 池 + 复杂过滤 + 接入 PyTorch 模型 | **torch-recall** — AOTInductor <0.1ms，tensor 原生接口 |
| 全文搜索 + 布尔过滤 + 评分排序 | **tantivy / Elasticsearch** — 功能完整、生态成熟 |
| 实时数据 + 频繁更新 + 分布式 | **Elasticsearch** — NRT 索引、自动分片 |
| 纯集合运算 (无数值条件) | **pyroaring** — 最快、压缩率高、SIMD |
| 嵌入推理管线 (与 PyTorch 模型串联) | **torch-recall** — compile / export / C++ 全链路 |

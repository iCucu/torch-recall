# torch-recall

基于 PyTorch 的高性能 bitmap 倒排索引。离线构建索引并通过 AOTInductor 导出为 `.pt2` 模型包，在线用 C++（或 Python）加载推理。

**性能指标**: 100 万 item，查询延迟 < 20ms（compiled CPU ~6ms，AOTInductor C++ < 0.1ms）。

---

## 实现原理

### 整体架构

```
离线 (Python)                              在线 (C++ / Python)
┌──────────────────────────────┐          ┌──────────────────────────────┐
│  Items (dict/JSON)           │          │  查询字符串                   │
│       ↓                      │          │       ↓                      │
│  IndexBuilder                │          │  QueryParser  (递归下降解析)   │
│   • 离散字段 → 值字典 → bitmap │          │       ↓                      │
│   • 文本字段 → 分词 → bitmap   │          │  DNFConverter (De Morgan 展开)│
│   • 数值字段 → float32 列存储  │          │       ↓                      │
│       ↓                      │  .pt2    │  TensorEncoder (DNF → 张量)   │
│  InvertedIndexModel (Module) │ ───────> │       ↓                      │
│       ↓                      │          │  AOTIModelPackageLoader       │
│  torch.export + AOTInductor  │          │   → model.forward() 推理      │
│       ↓                      │          │       ↓                      │
│  model.pt2 + index_meta.json │          │  ResultDecoder (bitmap → IDs) │
└──────────────────────────────┘          └──────────────────────────────┘
```

### 核心数据结构：Packed int64 Bitmap

每个离散值或文本 term 对应一个 bitmap，记录哪些 item 拥有该属性：

```
bitmap["city=北京"] = [0b...0101]   →  item 0, 2 是北京
bitmap["gender=男"]  = [0b...1001]   →  item 0, 3 是男
```

Bitmap 用 int64 紧密打包：每个 int64 存储 64 个 item 的状态。100 万 item 只需 15,625 个 int64（~122 KB/bitmap）。

查询时用位运算完成集合操作：

| 操作 | 位运算 | 含义 |
|------|--------|------|
| AND | `a & b` | 交集 |
| OR | `a \| b` | 并集 |
| NOT | `~a & valid_mask` | 补集 |

一条位运算同时处理 64 个 item，天然高效。

### 数值字段：Columnar 列存储

数值字段以 `[F, N]` 的 float32 矩阵存储（F 个字段 × N 个 item）。查询时做向量化比较：

```python
bool_mask = numeric_data[field_idx] < threshold   # 一次性比较所有 item
packed_bitmap = pack_bool_to_int64(bool_mask)      # 打包为 bitmap，参与位运算
```

### 查询处理流程

任意布尔表达式通过 4 步转换为静态张量输入：

```
查询字符串                          递归下降解析器
  (city == "北京" OR city == "上海") AND price < 100
        ↓
AST 表达式树                        DNF 转换器 (De Morgan + 分配律)
  AND(OR(city=北京, city=上海), price<100)
        ↓
DNF 析取范式                        平坦化为"OR of ANDs"
  (city=北京 AND price<100) OR (city=上海 AND price<100)
        ↓
张量编码                            填充到固定大小，生成 forward() 输入
  bitmap_indices, numeric_ops, conj_matrix, ...
```

### torch.export 兼容性

`torch.export` 要求静态计算图。本项目的设计要点：

1. **固定大小输入**: 所有查询张量填充到编译期常量 `MAX_BP=32, MAX_NP=16, MAX_CONJ=16`，用 validity mask 标记有效位
2. **循环常量边界**: forward 中的 `for c in range(MAX_CONJ): for p in range(P_TOTAL):` 会被 `torch.export` 完全展开为静态图
3. **返回固定形状**: 返回 `[L]` 的 packed bitmap（而非变长 ID 列表），解码在模型外部完成

### .pt2 导出与 C++ 推理

```python
# 离线导出
exported = torch.export.export(model, example_inputs)
torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")
```

```cpp
// C++ 在线加载
torch::inductor::AOTIModelPackageLoader loader("model.pt2");
auto outputs = loader.run(inputs);  // 直接调用编译后的优化内核
```

---

## 环境准备

```bash
# 克隆项目
git clone <repo> && cd torch-recall

# 创建虚拟环境 + 安装依赖
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch pytest jieba
uv pip install -e "offline[dev]"
```

## 快速上手

### 1. 离线构建索引 + 导出模型

```python
from torch_recall import Schema, IndexBuilder, export_model

schema = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["price"],
    text_fields=["title"],
)

items = [
    {"city": "北京", "gender": "男", "price": 68.0, "title": "开放世界 冒险 游戏"},
    {"city": "上海", "gender": "女", "price": 128.0, "title": "上海 美食 推荐"},
    {"city": "北京", "gender": "女", "price": 299.0, "title": "Python 编程 入门"},
    {"city": "广州", "gender": "男", "price": 35.0, "title": "广州 早茶 美食"},
]

builder = IndexBuilder(schema)
model, meta = builder.build(items)

builder.save_meta(meta, "index_meta.json")    # 字段字典、bitmap 映射
export_model(model, meta, "model.pt2")        # AOTInductor 编译后的模型
```

### 2. Python 在线查询

```python
import torch
from torch_recall import encode_query

tensors = encode_query('city == "北京" AND price < 100.0', meta)

model.eval()
with torch.no_grad():
    result = model(**tensors)    # 返回 [L] int64 packed bitmap

# 解码 bitmap → item IDs
for w in range(result.shape[0]):
    word = result[w].item()
    while word:
        bit = (word & -word).bit_length() - 1
        print(f"匹配 item {w * 64 + bit}")
        word &= word - 1
```

### 3. C++ 在线查询

```bash
# 编译
cd online && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . --config Release

# 查询
./torch_recall_cli model.pt2 index_meta.json 'city == "北京" AND price < 100.0'
```

### 4. 完整演示

```bash
python examples/01_build_index.py     # 离线: 构建索引 + 导出
python examples/02_query_python.py    # 在线: Python 推理
bash   examples/03_query_cpp.sh       # 在线: C++ 推理
```

## 查询语法

```
city == "北京"                                     # 离散等值
price < 100.0                                      # 数值范围
title contains "游戏"                               # 文本匹配
city == "北京" AND gender == "男"                    # AND
city == "北京" OR city == "上海"                     # OR
NOT category == "体育"                               # NOT
(city == "北京" OR city == "上海") AND price < 100   # 嵌套
```

**运算符**: `==`  `!=`  `<`  `>`  `<=`  `>=`  `contains`
**连接词**: `AND`  `OR`  `NOT`  `(`  `)`

## 性能测试

```bash
python offline/benchmarks/bench_recall.py
```

100 万 item 基准测试结果（Apple M 系列 CPU）：

| 模式 | 平均延迟 | P50 | P99 |
|------|---------|-----|-----|
| Eager (未编译) | ~30ms | ~29ms | ~35ms |
| torch.compile | ~6.6ms | ~6.6ms | ~7.8ms |
| AOTInductor C++ | <0.1ms | <0.06ms | <0.06ms |

## 项目结构

```
torch-recall/
├── offline/                        离线: 索引构建 + 模型导出 (Python)
│   ├── torch_recall/
│   │   ├── schema.py               字段类型定义、编译期常量
│   │   ├── model.py                InvertedIndexModel (nn.Module)
│   │   ├── builder.py              IndexBuilder: items → 模型 + 元数据
│   │   ├── tokenizer.py            文本分词 (空格 / jieba)
│   │   ├── query.py                解析器、DNF 转换器、张量编码器
│   │   └── exporter.py             torch.export + AOTInductor 导出
│   ├── tests/                      单元测试 + 集成测试
│   └── benchmarks/                 百万级性能测试
│
├── online/                         在线: 加载模型 + 查询推理 (C++)
│   ├── include/torch_recall/       头文件
│   ├── src/
│   │   ├── query_parser.cpp        递归下降解析器
│   │   ├── dnf_converter.cpp       DNF 转换 (De Morgan)
│   │   ├── tensor_encoder.cpp      DNF → 填充张量
│   │   ├── model_runner.cpp        AOTIModelPackageLoader 封装
│   │   ├── result_decoder.cpp      bitmap → item ID 提取
│   │   └── main.cpp                CLI 入口
│   └── CMakeLists.txt
│
├── examples/                       完整离线/在线演示
│   ├── 01_build_index.py
│   ├── 02_query_python.py
│   └── 03_query_cpp.sh
│
└── docs/specs/                     设计文档
```

## 设计限制

| 限制 | 缓解策略 |
|------|---------|
| DNF 展开可能指数膨胀 | 上限 MAX_CONJ=16，超出则拒绝并提示简化 |
| 高基数离散字段占用内存 | 基数 >5000 的字段建议拆分或退化为数值处理 |
| 文本分词质量 | 可插拔分词器，默认空格分词，可替换为 jieba |
| 固定 MAX 参数 | 编译期常量，可调整后重新导出 .pt2 |

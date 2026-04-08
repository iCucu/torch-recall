# torch-recall

基于 PyTorch 的高性能 bitmap 倒排索引。离线构建索引并通过 AOTInductor 导出为 `.pt2` 模型包，在线用 C++（或 Python）加载推理。

**核心指标**: 100 万 item · 编译后查询 ~3.3ms (CPU) · AOTInductor <0.1ms · 支持 1024 DNF conjunctions

## 原理概述

```
离线 (Python)                            在线 (C++ / Python)
┌────────────────────────────────┐      ┌────────────────────────────────┐
│  Items  →  IndexBuilder        │      │  查询字符串                     │
│   • 离散字段 → packed bitmap    │      │    ↓  解析 → DNF → 张量编码     │
│   • 文本字段 → 分词 → bitmap    │ .pt2 │    ↓  forward() × N → OR 合并  │
│   • 数值字段 → float32 列存储   │─────→│    ↓  bitmap → item IDs        │
│  torch.export + AOTInductor    │      │  AOTIModelPackageLoader        │
└────────────────────────────────┘      └────────────────────────────────┘
```

- **Packed int64 Bitmap**: 每个离散值 / 文本 term 一条 bitmap，64 个 item 压缩到一个 int64，位运算完成 AND / OR / NOT
- **列存储数值**: `[F, N]` float32 矩阵，运行时向量化比较后打包为 bitmap
- **DNF 查询**: 任意布尔表达式 → 递归下降解析 → De Morgan 展开为析取范式 → 固定大小张量输入
- **两级 conjunction**: 模型单次处理 64 个 conjunction (CONJ_PER_PASS)，系统级支持 1024 个，超限自动分批

详细实现原理与优化手段见 [docs/design.md](docs/design.md)，性能测试数据见 [docs/benchmark.md](docs/benchmark.md)。

## 环境准备

```bash
git clone <repo> && cd torch-recall

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch pytest jieba
uv pip install -e "offline[dev]"
```

## 快速上手

### 1. 离线构建索引 + 导出

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

builder.save_meta(meta, "index_meta.json")
export_model(model, meta, "model.pt2")
```

### 2. Python 查询

```python
import torch
from torch_recall import encode_query

tensors = encode_query('city == "北京" AND price < 100.0', meta)

model.eval()
with torch.no_grad():
    result = model(**tensors)

# 解码 bitmap → item IDs
for w in range(result.shape[0]):
    word = result[w].item() & ((1 << 64) - 1)
    while word:
        bit = (word & -word).bit_length() - 1
        print(f"匹配 item {w * 64 + bit}")
        word &= word - 1
```

### 3. C++ 查询

```bash
cd online && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . --config Release

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

## 项目结构

```
torch-recall/
├── offline/                        离线: Python
│   ├── torch_recall/
│   │   ├── schema.py               字段类型、编译期常量
│   │   ├── model.py                InvertedIndexModel (nn.Module)
│   │   ├── builder.py              IndexBuilder
│   │   ├── tokenizer.py            分词器 (空格 / jieba)
│   │   ├── query.py                解析、DNF、张量编码
│   │   └── exporter.py             torch.export + AOTInductor
│   ├── tests/                      测试
│   └── benchmarks/                 性能测试脚本
├── online/                         在线: C++
│   ├── include/torch_recall/       头文件
│   ├── src/                        实现
│   └── CMakeLists.txt
├── examples/                       端到端演示
└── docs/
    ├── design.md                   实现细节与优化手段
    └── benchmark.md                性能测试结果
```

## 系统参数

| 常量 | 值 | 含义 |
|------|---|------|
| `MAX_BP` | 16 | 单条查询最大 bitmap 谓词数 |
| `MAX_NP` | 8 | 单条查询最大数值谓词数 |
| `MAX_CONJ` | 1024 | 单条查询最大 conjunction 数 |
| `CONJ_PER_PASS` | 8 | 单次 forward() 处理的 conjunction 数 |

所有参数为编译期常量，修改后需重新导出 `.pt2` 并重编译 C++。

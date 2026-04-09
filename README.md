# torch-recall

基于 PyTorch 的召回框架。每种召回方法实现为 `nn.Module`，通过 AOTInductor 导出为 `.pt2` 模型包，支持 C++/Python 在线推理。

当前已实现 **定向召回 (Targeting Recall)**：item 携带定向规则（布尔表达式），给定用户标签，高效匹配所有符合条件的 item。

## 架构概览

```
共享组件                                    召回方法 (可扩展)
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│  Schema          字段类型定义     │      │  targeting/                  │
│  query/          布尔表达式解析   │      │    TargetingRecall (Module)  │
│  scheduler/      torch.export    │      │    TargetingBuilder          │
│  tokenizer       分词器          │      │    encode_user               │
│                                  │  .pt2 ├──────────────────────────────┤
│  inference_engine/               │──────→│  knn/ (planned)             │
│    通用 C++ 推理引擎              │      ├──────────────────────────────┤
│                                  │      │  ann/ (planned)             │
└──────────────────────────────────┘      └──────────────────────────────┘
```

每种召回方法遵循 **Builder → Module → Encoder** 模式：
1. **Builder**: 离线构建索引，生成 `nn.Module` + meta JSON
2. **Module** (`nn.Module`): `forward()` 纯 tensor 操作，可编译导出
3. **Encoder**: 将业务输入编码为 `forward()` 所需的 tensor

## 环境准备

```bash
git clone <repo> && cd torch-recall

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch pytest jieba
uv pip install -e "index_model[dev]"
```

## 快速上手 — 定向召回

### 1. 离线构建定向索引

```python
from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.scheduler.exporter import export_recall_model

schema = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)

rules = [
    'city == "北京" AND gender == "男"',
    'city == "上海"',
    "age > 18",
    '(city == "北京" OR city == "上海") AND age >= 25',
    'tags contains "游戏"',
    'price < 100.0 AND tags contains "美食"',
    'city != "广州"',
    '(city == "广州" AND gender == "女") OR age > 30',
]

builder = TargetingBuilder(schema)
model, meta = builder.build(rules)

builder.save_meta(meta, "targeting_meta.json")
export_recall_model(model, "targeting_model.pt2")
```

### 2. Python 查询

```python
model.eval()
result = model.query(
    {"city": "北京", "gender": "男", "age": 30, "tags": "游戏 科技"},
    meta,
)
matched = [i for i in range(len(rules)) if result[i].item()]
print(f"匹配的 item: {matched}")
# → 匹配的 item: [0, 2, 3, 4, 6]
```

### 3. C++ 查询

```bash
# 编译推理引擎
cd inference_engine && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
cmake --build . --config Release
```

```bash
# Python 侧编码用户属性为张量文件
python -m torch_recall encode-user \
    --user '{"city":"北京","gender":"男","age":30,"tags":"游戏 科技"}' \
    --meta targeting_meta.json \
    --output tensors.pt

# C++ 通用推理
./torch_recall_cli targeting_model.pt2 tensors.pt --num-items 8
```

### 4. 完整演示

```bash
PYTHONPATH=index_model python examples/01_build_targeting.py   # 离线: 构建索引 + 导出
PYTHONPATH=index_model python examples/02_query_targeting.py   # 在线: Python 推理
bash examples/03_targeting_cpp.sh                              # 在线: C++ 推理
```

## 规则语法

Item 定向规则使用布尔表达式语法:

```
city == "北京"                                     # 离散等值
city != "广州"                                     # 离散不等
age > 18                                           # 数值范围
tags contains "游戏"                                # 文本匹配
city == "北京" AND gender == "男"                    # AND
city == "北京" OR city == "上海"                     # OR
(city == "北京" OR city == "上海") AND age >= 25     # 嵌套
```

**运算符**: `==`  `!=`  `<`  `>`  `<=`  `>=`  `contains`
**连接词**: `AND`  `OR`  `NOT`  `(`  `)`

## 项目结构

```
torch-recall/
├── index_model/                         Index Model (索引模型)
│   ├── pyproject.toml
│   ├── torch_recall/
│   │   ├── __init__.py                  公共 API
│   │   ├── __main__.py                  CLI (encode-user)
│   │   ├── schema.py                    共享: 字段类型、编译期常量
│   │   ├── tokenizer.py                 共享: 分词器 (空格 / jieba)
│   │   ├── scheduler/
│   │   │   └── exporter.py              共享: 通用 .pt2 导出
│   │   ├── recall_method/
│   │   │   └── targeting/               定向召回
│   │   │       ├── recall.py            TargetingRecall (nn.Module)
│   │   │       ├── builder.py           TargetingBuilder
│   │   │       └── encoder.py           encode_user / save_user_tensors
│   │   └── query/                       共享: 查询解析
│   │       ├── parser.py                AST + 递归下降解析器
│   │       └── dnf.py                   DNF 转换
│   └── tests/
├── inference_engine/                    共享: 通用 C++ 推理引擎
│   ├── CMakeLists.txt                   Torch only (无第三方依赖)
│   ├── include/torch_recall/
│   │   ├── model_runner.h               run(vector<Tensor>) → Tensor
│   │   └── result_decoder.h             bool tensor → item IDs
│   └── src/
│       ├── main.cpp                     通用 CLI: load .pt2 + tensors.pt
│       ├── model_runner.cpp
│       └── result_decoder.cpp
├── examples/                            端到端演示
└── docs/
    ├── architecture.md                  框架架构
    └── targeting/                       定向召回
        ├── design.md                    设计与优化策略
        ├── implementation.md            模块实现参考
        ├── walkthrough.md              端到端示例详解
        └── benchmark.md                性能测试结果
```

## 系统参数 (Targeting)

| 常量 | 值 | 含义 |
|------|---|------|
| `MAX_PREDS_PER_CONJ` | 8 | 单条 conjunction 最大谓词数 |
| `MAX_CONJ_PER_ITEM` | 16 | 单个 item 最大 conjunction 数 |

所有参数为编译期常量，修改后需重新构建索引并导出 `.pt2`。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 框架架构、共享组件、扩展方式 |
| [docs/targeting/design.md](docs/targeting/design.md) | 定向召回设计与优化策略 |
| [docs/targeting/implementation.md](docs/targeting/implementation.md) | 定向召回模块实现参考 |
| [docs/targeting/walkthrough.md](docs/targeting/walkthrough.md) | 定向召回端到端示例详解 |
| [docs/targeting/benchmark.md](docs/targeting/benchmark.md) | 定向召回性能测试结果 |

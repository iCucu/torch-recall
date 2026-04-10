# torch-recall

基于 PyTorch 的召回框架。每种召回方法实现为统一的 `RecallOp`（`nn.Module`），通过声明式组合（`And` / `Or`）灵活编排，经 AOTInductor 导出为单一 `.pt2` 模型包，支持 C++/Python 在线推理。

## 架构概览

```
共享组件                              召回方法 (RecallOp)
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Schema / Item  数据定义     │    │  targeting/                 │
│  RecallOp       统一算子接口  │    │    TargetingRecall          │
│  query/         表达式解析   │    │    TargetingBuilder          │
│  tokenizer      分词器       │    │    encode_user               │
│                              │    ├─────────────────────────────┤
│  scheduler/     声明式组合   │    │  knn/                       │
│    And / Or     交集 / 并集  │    │    KNNRecall                │
│    Pipeline     topk 输出    │    │    KNNBuilder               │
│    exporter     .pt2 导出    │    │    encode_query             │
│                              │    ├─────────────────────────────┤
│  inference_engine/           │    │  ann/ (planned)             │
│    通用 C++ 推理引擎          │    │    ANNRecall                │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 统一算子接口

所有召回方法实现相同的 `forward` 签名：

```python
class RecallOp(nn.Module):
    def forward(self, pred_satisfied: Tensor, query: Tensor) -> Tensor:
        # [P] bool, [1, D] float -> [1, N] float
```

- **Targeting**: 返回 `0.0`（匹配）/ `-inf`（不匹配），忽略 `query`
- **KNN**: 返回相似度分数，忽略 `pred_satisfied`
- **And**: `sum(children)` — `-inf` 使不匹配项自然被排除
- **Or**: `max(children)` — 任一子节点匹配即包含

### Builder → RecallOp → Encoder 模式

| 组件 | 职责 |
|------|------|
| **Builder** | 离线构建索引，生成 `RecallOp` + meta JSON |
| **RecallOp** (`nn.Module`) | `forward()` 纯 tensor 操作，可编译导出 |
| **Encoder** | 将业务输入编码为 `forward()` 所需的 tensor |

## 环境准备

```bash
git clone <repo> && cd torch-recall

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch pytest jieba
uv pip install -e "index_model[dev]"
```

## 快速上手 — 声明式 Pipeline

最推荐的使用方式：通过 `And` / `Or` 声明式组合多种召回方法，导出为单一模型。

```python
from torch_recall.schema import Schema, Item
from torch_recall.scheduler import (
    And, Or, Targeting, KNN,
    PipelineBuilder, export_recall_model, encode_pipeline_inputs,
)

# 1. 定义 Schema
schema = Schema(discrete_fields=["city"], numeric_fields=["age"])

# 2. 准备 Item（同时携带定向规则和 embedding）
items = [
    Item(id="item-0", targeting_rule='city == "北京"', embedding=[0.9, 0.1, 0.0]),
    Item(id="item-1", targeting_rule='city == "上海"', embedding=[0.1, 0.9, 0.0]),
    Item(id="item-2", targeting_rule='age > 18',       embedding=[0.5, 0.5, 0.5]),
]

# 3. 声明组合方式
spec = And(Targeting(schema), KNN(metric="cosine"))  # 定向过滤 + KNN 排序

# 4. 构建 + 导出
builder = PipelineBuilder(spec, k=2)
pipeline, meta = builder.build(items)
export_recall_model(pipeline, "pipeline.pt2")

# 5. 在线查询
pred_satisfied, query = encode_pipeline_inputs(
    user_attrs={"city": "北京", "age": 25},
    query_vectors=[[0.8, 0.2, 0.0]],
    meta=meta,
)
top_scores, top_indices = pipeline(pred_satisfied, query)
```

### 更多组合方式

```python
# 多路 KNN 取并集
spec = Or(KNN(metric="cosine"), KNN(metric="l2"))

# 复杂嵌套: (定向 AND 文本KNN) OR 图像KNN
spec = Or(
    And(Targeting(schema), KNN(metric="cosine")),
    KNN(metric="inner_product"),
)
```

## 单独使用 — Targeting

```python
from torch_recall.schema import Schema, Item
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.scheduler.exporter import export_recall_model

schema = Schema(discrete_fields=["city", "gender"], numeric_fields=["age"])
items = [
    Item(id="ad-0", targeting_rule='city == "北京" AND gender == "男"'),
    Item(id="ad-1", targeting_rule='city == "上海"'),
    Item(id="ad-2", targeting_rule='age > 18'),
]

builder = TargetingBuilder(schema)
model, meta = builder.build(items)

result = model.query({"city": "北京", "gender": "男", "age": 30}, meta)
matched = [i for i in range(len(items)) if result[i].item()]
# → [0, 2]
```

## 单独使用 — KNN

```python
from torch_recall.schema import Item
from torch_recall.recall_method.knn.builder import KNNBuilder

items = [
    Item(id="doc-0", embedding=[1.0, 0.0, 0.0]),
    Item(id="doc-1", embedding=[0.0, 1.0, 0.0]),
    Item(id="doc-2", embedding=[0.7, 0.7, 0.0]),
]

builder = KNNBuilder(k=2, metric="cosine")
model, meta = builder.build(items)

scores, indices = model.query([0.9, 0.1, 0.0], meta)
# → indices: [0, 2]  (最相似的两个)
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

## 完整演示

```bash
# Targeting 单独使用
PYTHONPATH=index_model python examples/01_build_targeting.py
PYTHONPATH=index_model python examples/02_query_targeting.py

# Pipeline: Targeting + KNN 取交集
PYTHONPATH=index_model python examples/04_pipeline_and.py

# C++ 推理
bash examples/03_targeting_cpp.sh
```

## 项目结构

```
torch-recall/
├── index_model/
│   └── torch_recall/
│       ├── schema.py                    Schema, Item 数据定义
│       ├── recall_method/
│       │   ├── base.py                  RecallOp 统一算子基类
│       │   ├── targeting/               定向召回
│       │   │   ├── recall.py            TargetingRecall(RecallOp)
│       │   │   ├── builder.py           TargetingBuilder
│       │   │   └── encoder.py           encode_user
│       │   └── knn/                     K 近邻召回
│       │       ├── recall.py            KNNRecall(RecallOp)
│       │       ├── builder.py           KNNBuilder
│       │       └── encoder.py           encode_query
│       ├── scheduler/
│       │   ├── spec.py                  Targeting, KNN, And, Or 声明式 spec
│       │   ├── pipeline.py              AndModule, OrModule, RecallPipeline
│       │   ├── pipeline_builder.py      PipelineBuilder: spec → nn.Module
│       │   ├── encoder.py               encode_pipeline_inputs
│       │   └── exporter.py              通用 .pt2 导出
│       ├── query/                       布尔表达式解析 + DNF
│       └── tokenizer.py                 分词器
├── inference_engine/                    通用 C++ 推理引擎
├── examples/                            端到端演示
└── docs/
    ├── architecture.md                  框架架构
    └── targeting/                       定向召回文档
```

## 文档

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 框架架构、共享组件、扩展方式 |
| [docs/targeting/design.md](docs/targeting/design.md) | 定向召回设计与优化策略 |
| [docs/targeting/implementation.md](docs/targeting/implementation.md) | 定向召回模块实现参考 |
| [docs/targeting/walkthrough.md](docs/targeting/walkthrough.md) | 定向召回端到端示例详解 |
| [docs/targeting/benchmark.md](docs/targeting/benchmark.md) | 定向召回性能测试结果 |

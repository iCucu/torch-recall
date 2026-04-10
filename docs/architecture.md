# 框架架构

torch-recall 是基于 PyTorch 的召回框架。每种召回方法实现为统一的 `RecallOp`（`nn.Module`），通过声明式组合导出为 `.pt2` 模型包，在 C++/Python 中执行推理。

---

## 1. 整体架构

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
│  inference/           │    │  ann/ (planned)             │
│    通用 C++ 推理引擎          │    │    ANNRecall                │
└─────────────────────────────┘    └─────────────────────────────┘
```

---

## 2. 统一算子接口 — RecallOp

所有召回方法实现相同的基类和 `forward` 签名:

```python
class RecallOp(nn.Module):
    """forward(pred_satisfied: [P] bool, query: [1, D] float) -> [1, N] float"""
```

每个算子接收完整的用户表示 `(pred_satisfied, query)`，返回 `[1, N] float` 的逐 item 分数:

| 算子 | 输出含义 | 使用的输入 |
|------|---------|-----------|
| **TargetingRecall** | `0.0` (匹配) / `-inf` (不匹配) | `pred_satisfied` |
| **KNNRecall** | 相似度分数 (越高越相关) | `query` 的对应切片 |
| **AndModule** | `sum(children)` — `-inf` 传染，实现交集 | 透传给子节点 |
| **OrModule** | `max(children)` — 任一匹配即包含 | 透传给子节点 |

这个设计使得:
- 每种召回方法在自己的包中实现算子（`targeting/recall.py`、`knn/recall.py`）
- `scheduler/` 只做组合（`And` / `Or`），不包含方法特定逻辑
- 新增召回方法只需实现 `RecallOp`，即可参与组合

### Builder → RecallOp → Encoder 模式

| 组件 | 职责 | 约束 |
|------|------|------|
| **Builder** | 离线构建索引，生成 `RecallOp` + meta JSON | 一次性离线执行 |
| **RecallOp** (`nn.Module`) | `forward()` 接收 tensor，输出逐 item 分数 | 纯 tensor 操作，无动态控制流 |
| **Encoder** | 将业务对象编码为 `forward()` 所需的 tensor | Python 侧执行，不经过 torch.export |

---

## 3. 召回方法

### 3.1 Targeting — 定向召回

**场景**: 广告定向、受众匹配。每个 item 携带布尔定向规则，给定用户属性，找出所有规则被满足的 item。

**核心**: Two-Level Gather + Reduce。将 item 规则标准化为 DNF，构建谓词→conjunction→item 的两级张量索引，用 gather + bool 规约完成全量匹配。

**输入/输出**:
- 输入: `pred_satisfied [P] bool` — 用户满足了哪些谓词
- 输出: `[1, N] float` — `0.0` (匹配) / `-inf` (不匹配)

详见 [targeting/](targeting/) 目录。

### 3.2 KNN — K 近邻召回

**场景**: 向量检索。给定查询向量，从 item embedding 池中找出最相似的 item。

**核心**: 全量矩阵乘法。将 item embedding 注册为 buffer，`forward()` 根据 metric（内积/余弦/L2）计算 `[1, N]` 相似度分数。

**输入/输出**:
- 输入: `query [1, D] float` — 查询 embedding（通过 offset/dim 切片自己的部分）
- 输出: `[1, N] float` — 逐 item 相似度分数

**支持的距离度量**: `inner_product`, `cosine`, `l2`

### 3.3 ANN — 近似近邻召回 (planned)

**场景**: 大规模向量检索。在精度可接受的范围内加速 KNN。

**核心**: IVF / HNSW 等索引结构的 tensor 化实现，或混合 PyTorch + C++ 扩展。

---

## 4. 声明式组合 — Scheduler

`scheduler/` 提供声明式 API，将多种召回方法组合为单一模型:

```python
from torch_recall.scheduler import And, Or, Targeting, KNN, PipelineBuilder

spec = And(Targeting(schema), KNN(metric="cosine"))
builder = PipelineBuilder(spec, k=100)
pipeline, meta = builder.build(items)
```

### Spec 树（纯数据）

`Targeting`, `KNN`, `And`, `Or` 是纯数据节点，描述组合关系:

```
And
├── Targeting(schema)
└── Or
    ├── KNN("cosine")
    └── KNN("l2")
```

### 编译为 nn.Module 树

`PipelineBuilder.build()` 将 spec 树编译为 `RecallOp` 组成的 `nn.Module` 树:

```
RecallPipeline
└── root: AndModule
    ├── TargetingRecall
    └── OrModule
        ├── KNNRecall(offset=0, dim=D1)
        └── KNNRecall(offset=D1, dim=D2)
```

### 分数约定

| 组合 | 计算方式 | 效果 |
|------|---------|------|
| `And(A, B)` | `A + B` | `-inf` 传染 → 交集过滤 |
| `Or(A, B)` | `max(A, B)` | 任一有限分即保留 → 并集 |

最终由 `RecallPipeline` 对根节点输出做 `topk(K)`，返回 `(scores [1,K], indices [1,K])`。

### 输入路由

Pipeline 的 `forward(pred_satisfied, query)`:
- `pred_satisfied [P] bool`: 所有 Targeting 节点共享
- `query [1, D_total] float`: 多个 KNN 节点各自通过 `offset:offset+dim` 切片

---

## 5. 共享组件

### 5.1 Schema 与 Item

**文件**: `torch_recall/schema.py`

```python
@dataclass
class Schema:
    discrete_fields: list[str]   # 离散字段 (city, gender, ...)
    numeric_fields:  list[str]   # 数值字段 (age, price, ...)
    text_fields:     list[str]   # 文本字段 (tags, ...)

@dataclass
class Item:
    id: str | None = None
    targeting_rule: str | None = None
    embedding: list[float] | None = None
```

`Item` 是所有召回方法共享的输入格式。每个 item 只需填充它参与的召回方法所需的字段。

### 5.2 查询解析管线

**文件**: `torch_recall/query/parser.py`, `torch_recall/query/dnf.py`

布尔表达式解析为 AST，再转换为 DNF：

```
输入: '(city == "北京" OR city == "上海") AND age >= 25'
→ AST: And(Or(Pred("city","==","北京"), Pred("city","==","上海")), Pred("age",">=",25))
→ DNF: [[city=北京 AND age>=25], [city=上海 AND age>=25]]
```

### 5.3 Exporter: torch.export + AOTInductor

**文件**: `torch_recall/scheduler/exporter.py`

```python
def export_recall_model(model, output_path):
    example_inputs = model.example_inputs()
    exported = torch.export.export(model, example_inputs)
    torch._inductor.aoti_compile_and_package(exported, package_path=output_path)
```

通用导出函数，接受任何实现了 `example_inputs()` 的 `nn.Module`。单独的 `TargetingRecall`、`KNNRecall`、组合后的 `RecallPipeline` 都可以导出。

### 5.4 C++ 推理引擎

**文件**: `inference/`

通用的 tensor-in / tensor-out 执行器，可执行任何 `.pt2` 模型包:

```
torch_recall_cli <model.pt2> <inputs.pt> [--num-items N]
```

---

## 6. torch.export 兼容性设计

所有 `RecallOp` 的 `forward()` 都必须满足 `torch.export` 的约束：

| 约束 | 解决方案 |
|------|---------|
| 静态形状 | 固定维度 + validity mask 处理 padding |
| 无动态控制流 | 全部用 tensor 操作: gather, matmul, topk 等 |
| 无 Python 对象 | 业务编码在 forward() 外部完成 |
| Buffer 冻结 | 索引张量通过 register_buffer 注册，导出时嵌入 .pt2 |
| Python 属性 | metric, offset 等作为 trace-time 常量在导出时固化 |

---

## 7. 代码组织

```
torch-recall/
├── index/
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
├── inference/                    通用 C++ 推理引擎
├── examples/                            端到端演示
└── docs/
    ├── architecture.md                  ← 本文档
    └── targeting/                       定向召回文档
```

新增召回方法时:
1. 在 `recall_method/` 下新建目录，实现 `RecallOp` 子类 + `Builder` + `Encoder`
2. 在 `scheduler/spec.py` 中添加对应的 spec 节点
3. 在 `pipeline_builder.py` 中添加该 spec 的编译逻辑
4. 共享组件（Schema, exporter, C++ engine）无需修改

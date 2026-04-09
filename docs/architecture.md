# 框架架构

torch-recall 是基于 PyTorch 的召回框架。每种召回方法 (recall method) 实现为一个 `nn.Module`，通过 `torch.export` + AOTInductor 编译为 `.pt2` 模型包，在 C++/Python 中执行推理。

---

## 1. 整体架构

```
                 共享组件                              召回方法 (可扩展)
┌─────────────────────────────────────┐    ┌────────────────────────────────┐
│                                     │    │  targeting/                    │
│  Schema        字段类型定义          │    │    TargetingRecall (nn.Module) │
│  query/        布尔表达式解析 + DNF  │    │    TargetingBuilder            │
│  scheduler/    torch.export 导出    │    │    encode_user                 │
│  tokenizer     分词器              │    ├────────────────────────────────┤
│                                     │    │  knn/ (planned)               │
│  inference_engine/                  │    │    KNNRecall (nn.Module)       │
│    通用 C++ tensor-in/tensor-out    │    ├────────────────────────────────┤
│    推理引擎                          │    │  ann/ (planned)               │
│                                     │    │    ANNRecall (nn.Module)       │
└─────────────────────────────────────┘    └────────────────────────────────┘
```

每种召回方法遵循统一模式：

| 组件 | 职责 | 约束 |
|------|------|------|
| **Builder** | 离线构建索引，生成 `nn.Module` + meta JSON | 一次性离线执行 |
| **Recall** (`nn.Module`) | `forward()` 接收 tensor 输入，输出匹配结果 | 纯 tensor 操作，无动态控制流 |
| **Encoder** | 将业务对象（用户属性/查询/向量）编码为 `forward()` 所需的 tensor | Python 侧执行，不经过 torch.export |

---

## 2. 召回方法

### 2.1 Targeting — 定向召回 (已实现)

**场景**: 广告定向、受众匹配。每个 item 携带布尔定向规则，给定用户属性，找出所有规则被满足的 item。

**核心**: Two-Level Gather + Reduce。将 item 规则标准化为 DNF，构建谓词→conjunction→item 的两级张量索引，用 gather + bool 规约完成全量匹配。

详见 [targeting/](targeting/) 目录：

| 文档 | 内容 |
|------|------|
| [design.md](targeting/design.md) | 设计与优化策略 |
| [implementation.md](targeting/implementation.md) | 模块实现参考 |
| [walkthrough.md](targeting/walkthrough.md) | 端到端示例详解 |
| [benchmark.md](targeting/benchmark.md) | 性能测试结果 |

### 2.2 KNN — K 近邻召回 (planned)

**场景**: 向量检索。给定查询向量，从 item 向量池中找出 top-K 最近邻。

**核心**: 全量矩阵乘法 + top-K。将 item embedding 注册为 buffer，`forward()` 计算余弦/内积距离后取 top-K。

### 2.3 ANN — 近似近邻召回 (planned)

**场景**: 大规模向量检索。在精度可接受的范围内加速 KNN。

**核心**: IVF / HNSW 等索引结构的 tensor 化实现，或混合 PyTorch + C++ 扩展。

---

## 3. 共享组件

### 3.1 Schema 与常量

**文件**: `index_model/torch_recall/schema.py`

```python
@dataclass
class Schema:
    discrete_fields: list[str]   # 离散字段 (city, gender, ...)
    numeric_fields:  list[str]   # 数值字段 (age, price, ...)
    text_fields:     list[str]   # 文本字段 (tags, ...)
```

Schema 定义字段类型，决定谓词的注册和评估方式。所有召回方法共享同一个 Schema 定义。

编译期常量（targeting 特有的在 targeting 模块中定义）:

```python
MAX_PREDS_PER_CONJ = 8   # K: 单条 conjunction 最大谓词数
MAX_CONJ_PER_ITEM = 16   # J: 单个 item 最大 conjunction 数
```

### 3.2 查询解析管线

**文件**: `index_model/torch_recall/query/parser.py`, `index_model/torch_recall/query/dnf.py`

布尔表达式解析为 AST，再转换为 DNF：

```
输入: '(city == "北京" OR city == "上海") AND age >= 25'

词法分析 → [WORD, OP, STR, KW, ...]
语法分析 → And(Or(Pred("city","==","北京"), Pred("city","==","上海")), Pred("age",">=",25))
DNF 转换 → [[city=北京 AND age>=25], [city=上海 AND age>=25]]
```

DNF 转换规则：
- `AND(A, B)`: A 和 B 的 DNF 做笛卡尔积
- `OR(A, B)`: 合并 A 和 B 的 DNF
- `NOT`: De Morgan 推到叶节点，作为谓词级别的 negated 标记

查询解析管线被 targeting recall 使用。KNN/ANN 方法可能使用不同的输入格式，但 parser 本身作为共享工具可被复用。

### 3.3 Exporter：torch.export + AOTInductor

**文件**: `index_model/torch_recall/scheduler/exporter.py`

```python
def export_recall_model(model, output_path):
    example_inputs = model.example_inputs()
    exported = torch.export.export(model, example_inputs)
    torch._inductor.aoti_compile_and_package(exported, package_path=output_path)
```

通用导出函数，接受任何实现了 `example_inputs()` 方法的 `nn.Module`。所有召回方法共享此导出路径。

- `torch.export.export`: trace forward()，生成静态计算图，所有 buffer 被捕获为常量
- `aoti_compile_and_package`: 编译为 C++ 内核，打包为 `.pt2` 文件

### 3.4 C++ 推理引擎

**文件**: `inference_engine/`

通用的 tensor-in / tensor-out 执行器，不包含任何业务逻辑，可执行任何 `.pt2` 模型包。

```cpp
// model_runner.cpp
torch::Tensor ModelRunner::run(const std::vector<torch::Tensor>& inputs) {
    auto outputs = impl_->loader.run(inputs);
    return outputs[0];
}
```

CLI:

```
torch_recall_cli <model.pt2> <inputs.pt> [--num-items N]
```

Python 侧编码业务输入为 tensor 文件 (`tensors.pt`)，C++ 侧加载 `.pt2` + `tensors.pt` 执行推理。这个模式对所有召回方法通用。

---

## 4. torch.export 兼容性设计

所有召回方法的 `forward()` 都必须满足 `torch.export` 的约束：

| 约束 | 解决方案 |
|------|---------|
| 静态形状 | 固定维度 + validity mask 处理 padding |
| 无动态控制流 | 全部用 tensor 操作: gather, matmul, topk, 等 |
| 无 Python 对象 | 业务编码在 forward() 外部完成 |
| Buffer 冻结 | 索引张量通过 register_buffer 注册，导出时嵌入 .pt2 |

导出后生成 `.pt2`，可在 C++ 中通过 `AOTIModelPackageLoader` 加载，无需 Python 解释器。

---

## 5. 代码组织

```
torch-recall/
├── index_model/                         Index Model (索引模型)
│   └── torch_recall/
│       ├── schema.py                    共享: 字段类型定义
│       ├── tokenizer.py                 共享: 分词器
│       ├── query/                       共享: 布尔表达式解析 + DNF
│       ├── scheduler/
│       │   └── exporter.py              共享: 通用 .pt2 导出
│       └── recall_method/
│           ├── targeting/               定向召回
│           │   ├── recall.py            TargetingRecall (nn.Module)
│           │   ├── builder.py           TargetingBuilder
│           │   └── encoder.py           encode_user
│           ├── knn/                     (planned) K 近邻召回
│           └── ann/                     (planned) 近似近邻召回
├── inference_engine/                    共享: 通用 C++ 推理引擎
└── docs/
    ├── architecture.md                  ← 本文档
    └── targeting/                       定向召回文档
        ├── design.md                    设计与优化策略
        ├── implementation.md            模块实现参考
        ├── walkthrough.md              端到端示例详解
        └── benchmark.md                性能测试结果
```

新增召回方法时：
1. 在 `recall_method/` 下新建目录，实现 `Recall (nn.Module)` + `Builder` + `Encoder`
2. 在 `docs/` 下新建对应目录，编写 design / implementation / walkthrough / benchmark
3. 共享组件 (Schema, exporter, C++ engine) 无需修改

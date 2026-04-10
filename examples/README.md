# examples — 召回框架完整演示

## 示例列表

| 示例 | 说明 |
|------|------|
| `01_build_targeting.py` | 定向召回: 离线构建索引 + 导出 .pt2 |
| `02_query_targeting.py` | 定向召回: Python 在线匹配 |
| `03_targeting_cpp.sh` | 定向召回: C++ 推理 |
| `04_pipeline_and.py` | **Pipeline: Targeting + KNN 并行召回取交集** |

## 流程概览

### 单独使用 Targeting

```
01_build_targeting.py        02_query_targeting.py        03_targeting_cpp.sh
      │                            │                            │
      ▼                            ▼                            ▼
┌─────────────┐           ┌──────────────┐           ┌──────────────┐
│  定义 Schema │           │  重建模型     │           │  Python 编码  │
│  定义 Rules  │           │  加载 meta   │           │  encode_user  │
│  构建索引    │──output──▶│  encode_user │           │     ↓         │
│  导出 .pt2   │           │  model.query │           │  C++ 加载 .pt2│
│  保存 meta   │           │  解码结果    │           │  + tensors.pt │
└─────────────┘           └──────────────┘           │  通用推理     │
    (Python)                 (Python)                └──────────────┘
                                                      (Python + C++)
```

### Pipeline: Targeting + KNN 取交集 (推荐)

```
04_pipeline_and.py
      │
      ▼
┌──────────────────────────────────┐
│  定义 Schema                      │
│  准备 Item (规则 + embedding)     │
│  声明: And(Targeting, KNN)       │
│  PipelineBuilder.build(items)    │
│         ↓                         │
│  pipeline(pred_satisfied, query) │
│         ↓                         │
│  → top_scores, top_indices       │
│         ↓                         │
│  导出 pipeline.pt2               │
└──────────────────────────────────┘
```

`And(Targeting, KNN)` 的效果:
- **Targeting** 做硬过滤: 只保留定向规则匹配的 item（不匹配 → `-inf`）
- **KNN** 做相似度排序: 按 embedding 相似度打分
- **And** 取交集: 分数相加，`-inf` 使不匹配项被排除
- **topk**: 从匹配项中取分数最高的 K 个

## 运行步骤

```bash
# 0. 确保已安装依赖
cd torch-recall
source .venv/bin/activate

# 1. 离线: 构建定向索引 + 导出模型
PYTHONPATH=index python examples/01_build_targeting.py

# 2. 在线: Python 推理
PYTHONPATH=index python examples/02_query_targeting.py

# 3. 在线: C++ 推理 (需先编译)
bash examples/03_targeting_cpp.sh

# 4. Pipeline: Targeting + KNN 取交集
PYTHONPATH=index python examples/04_pipeline_and.py
```

## 产出文件

运行后会在 `examples/output/` 下生成:

| 文件 | 来源 | 说明 |
|---|---|---|
| `targeting_model.pt2` | 01 | 单独 Targeting 模型包 |
| `targeting_meta.json` | 01 | Targeting 谓词注册表 + 元信息 |
| `targeting_rules.json` | 01 | 原始定向规则 |
| `pipeline_and.pt2` | 04 | And(Targeting, KNN) 组合模型包 |
| `pipeline_and_meta.json` | 04 | Pipeline 元信息（含 targeting + knn） |

## 组合方式一览

```python
from torch_recall.scheduler import And, Or, Targeting, KNN, PipelineBuilder

# Targeting + KNN 取交集（定向过滤 + 排序）
spec = And(Targeting(schema), KNN(metric="cosine"))

# 多路 KNN 取并集
spec = Or(KNN(metric="cosine"), KNN(metric="l2"))

# 复杂嵌套: (定向 AND 文本KNN) OR 图像KNN
spec = Or(
    And(Targeting(schema), KNN(metric="cosine")),
    KNN(metric="inner_product"),
)

# 构建 + 导出
builder = PipelineBuilder(spec, k=100)
pipeline, meta = builder.build(items)
export_recall_model(pipeline, "pipeline.pt2")
```

#!/usr/bin/env python3
"""
Pipeline 示例 — Targeting + KNN 并行召回取交集
==============================================

演示 And(Targeting, KNN) 组合:
  - Targeting 做硬过滤: 只保留定向规则匹配的 item
  - KNN 做相似度排序: 按 embedding 相似度给 item 打分
  - And 取交集: 不匹配定向的 item 被排除, 剩余 item 按 KNN 分数排序

最终导出为一个 .pt2 模型, forward(pred_satisfied, query) -> (scores, indices).

运行方式:
    cd torch-recall
    PYTHONPATH=index_model python examples/04_pipeline_and.py
"""

import time
from pathlib import Path

import torch

from torch_recall.schema import Schema, Item
from torch_recall.scheduler import (
    And,
    Targeting,
    KNN,
    PipelineBuilder,
    export_recall_model,
    encode_pipeline_inputs,
)

# ── 1. 定义 Schema ──────────────────────────────────────────────────────────

schema = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age"],
    text_fields=[],
)

# ── 2. 准备 Item (同时携带定向规则和 embedding) ─────────────────────────────

items = [
    Item(id="北京烤鸭店",     targeting_rule='city == "北京"',                embedding=[0.9, 0.1, 0.0, 0.0]),
    Item(id="上海本帮菜馆",   targeting_rule='city == "上海"',                embedding=[0.1, 0.9, 0.0, 0.0]),
    Item(id="全国连锁火锅",   targeting_rule='age > 18',                     embedding=[0.5, 0.5, 0.5, 0.0]),
    Item(id="高端日料",       targeting_rule='city == "北京" AND age >= 25',   embedding=[0.8, 0.0, 0.2, 0.0]),
    Item(id="学生优惠套餐",   targeting_rule='age > 10',                      embedding=[0.3, 0.3, 0.3, 0.8]),
    Item(id="北京女性SPA",    targeting_rule='city == "北京" AND gender == "女"', embedding=[0.1, 0.0, 0.8, 0.2]),
]

# ── 3. 声明组合方式: Targeting AND KNN ───────────────────────────────────────

spec = And(
    Targeting(schema),           # 硬过滤: 只保留定向匹配的 item
    KNN(metric="inner_product"), # 相似度排序: 按 embedding 内积打分
)

K = 3  # 最终返回 top-K

# ── 4. 构建 Pipeline ────────────────────────────────────────────────────────

print("=" * 60)
print("Pipeline 示例: And(Targeting, KNN)")
print("=" * 60)

t0 = time.time()
builder = PipelineBuilder(spec, k=K)
pipeline, meta = builder.build(items)
pipeline.eval()
print(f"\n构建耗时: {time.time() - t0:.3f}s")
print(f"  item 数量: {meta['num_items']}")
print(f"  谓词数量: {meta['num_preds']}")
print(f"  query 维度: {meta['total_query_dim']}")

# ── 5. 在线查询 ─────────────────────────────────────────────────────────────

users = [
    (
        {"city": "北京", "gender": "男", "age": 30},
        [0.9, 0.0, 0.1, 0.0],  # 偏好北京美食
        "北京 30岁男性",
    ),
    (
        {"city": "北京", "gender": "女", "age": 28},
        [0.2, 0.0, 0.7, 0.1],  # 偏好 SPA/休闲
        "北京 28岁女性",
    ),
    (
        {"city": "上海", "gender": "男", "age": 22},
        [0.1, 0.8, 0.1, 0.5],  # 偏好上海菜+学生套餐
        "上海 22岁男性",
    ),
]

item_ids = meta["item_ids"]

print("\n" + "-" * 60)
for user_attrs, query_vec, desc in users:
    pred_satisfied, query = encode_pipeline_inputs(
        user_attrs, [query_vec], meta
    )
    with torch.no_grad():
        top_scores, top_indices = pipeline(pred_satisfied, query)

    print(f"\n【{desc}】 属性={user_attrs}")
    print(f"  query embedding={query_vec}")
    print(f"  召回 top-{K}:")
    for rank in range(K):
        idx = top_indices[0, rank].item()
        score = top_scores[0, rank].item()
        name = item_ids[idx] if score > float("-inf") else "(无匹配)"
        if score > float("-inf"):
            print(f"    #{rank+1}  {name:<12s}  score={score:.4f}  (item[{idx}])")
        else:
            print(f"    #{rank+1}  {name}  score=-inf")

# ── 6. 导出 .pt2 ────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

pt2_path = output_dir / "pipeline_and.pt2"
meta_path = output_dir / "pipeline_and_meta.json"

print(f"\n导出模型到 {pt2_path} ...")
t0 = time.time()
export_recall_model(pipeline, str(pt2_path))
builder.save_meta(meta, str(meta_path))
print(f"✓ 导出完成 ({time.time() - t0:.1f}s, {pt2_path.stat().st_size / 1024:.0f} KB)")

print(f"\n产出文件:")
print(f"  {pt2_path}")
print(f"  {meta_path}")
print("=" * 60)

#!/usr/bin/env python3
"""
Step 1 — 离线构建索引 + 导出 .pt2 模型
=======================================

演示数据：8 条广告 item，包含离散字段、数值字段和文本字段。

运行方式:
    cd torch-recall
    source .venv/bin/activate
    python examples/01_build_index.py

产出 (写入 examples/output/):
    - model.pt2        AOTInductor 编译后的模型包
    - index_meta.json  索引元数据 (字段映射、bitmap 字典等)
    - items.json       原始 items，方便后续对照
"""

import json
import sys
import time
from pathlib import Path

import torch

from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.exporter import export_model

# ── 1. 定义 Schema ──────────────────────────────────────────────────────────

schema = Schema(
    discrete_fields=["city", "gender", "category"],
    numeric_fields=["price", "score"],
    text_fields=["title"],
)

# ── 2. 准备 Items ───────────────────────────────────────────────────────────

items = [
    # 0
    {"city": "北京", "gender": "男", "category": "游戏",
     "price": 68.0,  "score": 4.5, "title": "开放世界 冒险 游戏 新手攻略"},
    # 1
    {"city": "上海", "gender": "女", "category": "美食",
     "price": 128.0, "score": 4.8, "title": "上海 本帮菜 美食 推荐"},
    # 2
    {"city": "北京", "gender": "女", "category": "教育",
     "price": 299.0, "score": 4.2, "title": "Python 编程 入门 教程"},
    # 3
    {"city": "广州", "gender": "男", "category": "美食",
     "price": 35.0,  "score": 4.9, "title": "广州 早茶 美食 攻略"},
    # 4
    {"city": "深圳", "gender": "女", "category": "科技",
     "price": 599.0, "score": 4.0, "title": "AI 大模型 技术 解读"},
    # 5
    {"city": "北京", "gender": "男", "category": "游戏",
     "price": 45.0,  "score": 3.8, "title": "策略 游戏 进阶 攻略"},
    # 6
    {"city": "上海", "gender": "男", "category": "科技",
     "price": 199.0, "score": 4.6, "title": "机器人 编程 实战 教程"},
    # 7
    {"city": "广州", "gender": "女", "category": "教育",
     "price": 150.0, "score": 4.3, "title": "儿童 英语 启蒙 推荐"},
]

# ── 3. 构建索引 ─────────────────────────────────────────────────────────────

print(f"Items: {len(items)} 条")
print(f"Schema: discrete={schema.discrete_fields}, "
      f"numeric={schema.numeric_fields}, text={schema.text_fields}")

t0 = time.time()
builder = IndexBuilder(schema)
model, meta = builder.build(items)
print(f"构建耗时: {time.time() - t0:.3f}s")
print(f"  Bitmaps 数量: {meta['num_bitmaps']}  形状: {model.bitmaps.shape}")
print(f"  Numeric 列数: {model.numeric_data.shape[0]}")

# ── 4. 保存产出 ─────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

meta_path = output_dir / "index_meta.json"
builder.save_meta(meta, meta_path)
print(f"\n✓ 元数据已保存: {meta_path}")

items_path = output_dir / "items.json"
with open(items_path, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)
print(f"✓ Items 已保存: {items_path}")

# ── 5. 导出 .pt2 模型 ───────────────────────────────────────────────────────

pt2_path = output_dir / "model.pt2"
print(f"\n正在导出模型到 {pt2_path} ...")
t0 = time.time()
export_model(model, meta, str(pt2_path))
print(f"✓ 模型已导出 ({time.time() - t0:.1f}s, {pt2_path.stat().st_size / 1024:.0f} KB)")

print("\n" + "=" * 60)
print("离线构建完成！产出文件:")
print(f"  {pt2_path}")
print(f"  {meta_path}")
print(f"  {items_path}")
print("=" * 60)

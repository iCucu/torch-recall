#!/usr/bin/env python3
"""
Step 2 — 定向召回: Python 在线匹配
====================================

加载离线产出的模型和元数据，用 Python 为不同用户查找匹配的定向规则。

运行方式 (需先跑 01_build_targeting.py):
    PYTHONPATH=index_model python examples/02_query_targeting.py
"""

import json
from pathlib import Path

from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder

# ── 加载数据 ──────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent / "output"

with open(output_dir / "targeting_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(output_dir / "targeting_rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

schema = Schema(
    discrete_fields=["city", "gender", "category"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)

builder = TargetingBuilder(schema)
model, _ = builder.build(rules)
model.eval()

# ── 用户列表 ──────────────────────────────────────────────────────────────

users = [
    ({"city": "北京", "gender": "男", "age": 30, "price": 50.0, "tags": "游戏 科技"},
     "北京 30岁男性, 喜欢游戏科技, 价格敏感"),

    ({"city": "上海", "gender": "女", "age": 20, "price": 80.0, "tags": "美食 旅行"},
     "上海 20岁女性, 喜欢美食旅行"),

    ({"city": "广州", "gender": "女", "age": 35, "price": 200.0, "tags": "教育"},
     "广州 35岁女性, 关注教育"),

    ({"city": "北京", "gender": "女", "age": 16, "price": 30.0, "tags": "游戏 美食"},
     "北京 16岁女性, 喜欢游戏美食"),

    ({"city": "深圳", "gender": "男", "age": 40, "price": 150.0, "tags": "科技"},
     "深圳 40岁男性, 关注科技"),
]

# ── 执行匹配 ──────────────────────────────────────────────────────────────

print("=" * 70)
print("定向召回演示 — Python 在线匹配")
print("=" * 70)

for user_attrs, desc in users:
    result = model.query(user_attrs, meta)
    matched = [i for i in range(len(rules)) if result[i].item()]

    print(f"\n【{desc}】")
    print(f"  属性: {user_attrs}")
    print(f"  命中: {len(matched)} 条规则  IDs={matched}")
    for idx in matched:
        print(f"    [{idx}] {rules[idx]}")

print("\n" + "=" * 70)
print("完成！")

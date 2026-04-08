#!/usr/bin/env python3
"""
Step 2 — Python 在线推理
========================

加载离线产出的 model 和 meta，用 Python 执行多种查询，
打印匹配的 item 详情。

运行方式 (需先跑 01_build_index.py):
    python examples/02_query_python.py
"""

import json
from pathlib import Path

import torch

from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query

# ── 加载数据 ─────────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent / "output"

with open(output_dir / "index_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

with open(output_dir / "items.json", "r", encoding="utf-8") as f:
    items = json.load(f)

schema = Schema(
    discrete_fields=meta["schema"]["discrete"],
    numeric_fields=meta["schema"]["numeric"],
    text_fields=meta["schema"]["text"],
)

# 重建模型 (从原始 items 还原，生产环境直接加载 .pt2)
builder = IndexBuilder(schema)
model, _ = builder.build(items)
model.eval()

# ── 解码工具 ─────────────────────────────────────────────────────────────────

def decode_bitmap(result: torch.Tensor, num_items: int) -> list[int]:
    ids = []
    mask64 = (1 << 64) - 1
    for w in range(result.shape[0]):
        word = result[w].item() & mask64
        if word == 0:
            continue
        base = w * 64
        while word:
            bit = (word & -word).bit_length() - 1
            if base + bit < num_items:
                ids.append(base + bit)
            word &= word - 1
    return ids


def run_query(query_str: str):
    tensors = encode_query(query_str, meta)
    with torch.no_grad():
        result = model(**tensors)
    matched = decode_bitmap(result, len(items))
    return matched

# ── 执行查询 ─────────────────────────────────────────────────────────────────

queries = [
    ('city == "北京"',
     "查找北京的所有 item"),

    ('city == "北京" AND gender == "男"',
     "北京 + 男性"),

    ('price < 100.0',
     "价格 < 100"),

    ('(city == "北京" OR city == "上海") AND price < 200.0',
     "北京或上海 + 价格 < 200"),

    ('NOT category == "美食"',
     "排除美食类"),

    ('title contains "攻略"',
     "标题包含「攻略」"),

    ('title contains "游戏" AND price < 60.0',
     "标题包含「游戏」且价格 < 60"),

    ('(title contains "美食" OR category == "教育") AND score >= 4.3',
     "标题含「美食」或教育类，且评分 ≥ 4.3"),
]

print("=" * 70)
print("Python 在线推理演示")
print("=" * 70)

for query_str, desc in queries:
    matched = run_query(query_str)
    print(f"\n【{desc}】")
    print(f"  查询: {query_str}")
    print(f"  命中: {len(matched)} 条  IDs={matched}")
    for idx in matched:
        item = items[idx]
        print(f"    [{idx}] city={item['city']} gender={item['gender']} "
              f"cat={item['category']} price={item['price']} "
              f"score={item['score']} title=\"{item['title']}\"")

print("\n" + "=" * 70)
print("完成！")

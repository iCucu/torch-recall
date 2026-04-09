#!/usr/bin/env python3
"""
Step 1 — 定向召回: 离线构建
===========================

演示 "反向召回" 场景: item 携带定向规则(布尔表达式),
给定用户标签，找出所有匹配用户的 item。

运行方式:
    cd torch-recall
    source .venv/bin/activate
    PYTHONPATH=index_model python examples/01_build_targeting.py

产出 (写入 examples/output/):
    - targeting_model.pt2      AOTInductor 编译后的定向召回模型
    - targeting_meta.json      谓词注册表 + 元信息
    - targeting_rules.json     原始规则，方便对照
"""

import json
import time
from pathlib import Path

from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.scheduler.exporter import export_recall_model

# ── 1. 定义 Schema (用户可能拥有的属性类型) ────────────────────────────────

schema = Schema(
    discrete_fields=["city", "gender", "category"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)

# ── 2. 定义 Item 定向规则 ──────────────────────────────────────────────────

rules = [
    # 0: 定向北京男性
    'city == "北京" AND gender == "男"',
    # 1: 定向上海用户
    'city == "上海"',
    # 2: 定向年龄 > 18 的用户
    "age > 18",
    # 3: 定向北京或上海、且年龄 >= 25 的用户
    '(city == "北京" OR city == "上海") AND age >= 25',
    # 4: 定向标签含 "游戏" 的用户
    'tags contains "游戏"',
    # 5: 定向价格敏感且喜欢美食的用户
    'price < 100.0 AND tags contains "美食"',
    # 6: 定向非广州用户
    'city != "广州"',
    # 7: 定向广州女性 或 年龄 > 30 的用户
    '(city == "广州" AND gender == "女") OR age > 30',
]

# ── 3. 构建定向索引 ────────────────────────────────────────────────────────

print(f"Rules: {len(rules)} 条定向规则")
print(f"Schema: discrete={schema.discrete_fields}, "
      f"numeric={schema.numeric_fields}, text={schema.text_fields}")

t0 = time.time()
builder = TargetingBuilder(schema)
model, meta = builder.build(rules)
print(f"构建耗时: {time.time() - t0:.3f}s")
print(f"  谓词数量: {meta['num_preds']}")
print(f"  Conjunction 数量: {meta['num_conjs']}")

# ── 4. 保存产出 ────────────────────────────────────────────────────────────

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(exist_ok=True)

meta_path = output_dir / "targeting_meta.json"
builder.save_meta(meta, meta_path)
print(f"\n✓ 元数据已保存: {meta_path}")

rules_path = output_dir / "targeting_rules.json"
with open(rules_path, "w", encoding="utf-8") as f:
    json.dump(rules, f, ensure_ascii=False, indent=2)
print(f"✓ 规则已保存: {rules_path}")

# ── 5. 导出 .pt2 模型 ──────────────────────────────────────────────────────

pt2_path = output_dir / "targeting_model.pt2"
print(f"\n正在导出模型到 {pt2_path} ...")
t0 = time.time()
export_recall_model(model, str(pt2_path))
print(f"✓ 模型已导出 ({time.time() - t0:.1f}s, {pt2_path.stat().st_size / 1024:.0f} KB)")

print("\n" + "=" * 60)
print("定向索引构建完成！产出文件:")
print(f"  {pt2_path}")
print(f"  {meta_path}")
print(f"  {rules_path}")
print("=" * 60)

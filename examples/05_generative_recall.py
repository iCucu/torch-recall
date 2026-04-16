#!/usr/bin/env python3
"""
生成式召回示例 — Trie 约束 Beam Search
======================================

演示 Generative Recall:
  - 每个 item 有一条 sid_path (语义 ID 路径, 长度 5, 每个 sid ∈ [0, 2048))
  - 离线: 从 sid_path 构建混合 Dense/Sparse Trie
  - 在线: 根据 targeting 过滤 → Trie 有效性传播 → beam search 选出 top-K 路径 → 映射回 item

本例使用 MockDecoder (随机权重) 演示完整流程.

运行方式:
    cd torch-recall
    PYTHONPATH=index python examples/05_generative_recall.py
"""

import time
from pathlib import Path

import torch

from torch_recall.schema import Schema, Item
from torch_recall.scheduler import (
    Generative,
    PipelineBuilder,
    encode_pipeline_inputs,
)
from torch_recall.recall_method.generative.decoder import MockDecoder

# ── 1. 定义 Schema ──────────────────────────────────────────────────────────

schema = Schema(
    discrete_fields=["city"],
    numeric_fields=["age"],
)

# ── 2. 准备 Item (同时携带定向规则和 sid_path) ──────────────────────────────

items = [
    Item(id="北京烤鸭",    targeting_rule='city == "北京"',          sid_path=[0, 1, 2, 3, 4]),
    Item(id="全聚德",       targeting_rule='city == "北京"',          sid_path=[0, 1, 2, 3, 4]),
    Item(id="上海小笼包",   targeting_rule='city == "上海"',          sid_path=[0, 1, 5, 6, 7]),
    Item(id="连锁火锅",     targeting_rule="age > 18",               sid_path=[0, 2, 8, 9, 10]),
    Item(id="学生套餐",     targeting_rule="age > 10",               sid_path=[0, 2, 8, 9, 11]),
    Item(id="北京SPA",      targeting_rule='city == "北京" AND age >= 25', sid_path=[1, 3, 12, 13, 14]),
    Item(id="广州早茶",     targeting_rule='city == "广州"',          sid_path=[1, 3, 12, 15, 16]),
    Item(id="深圳科技餐",   targeting_rule='city == "深圳"',          sid_path=[1, 4, 17, 18, 19]),
]

# ── 3. 创建 MockDecoder ─────────────────────────────────────────────────────

USER_DIM = 32
BEAM_WIDTH = 4
K = 5

decoder = MockDecoder(user_dim=USER_DIM, vocab_size=2048)
torch.manual_seed(42)
decoder.eval()

# ── 4. 声明 Generative 召回 ────────────────────────────────────────────────

spec = Generative(
    schema,
    decoder=decoder,
    beam_width=BEAM_WIDTH,
    dense_levels=2,
)

# ── 5. 构建 Pipeline ────────────────────────────────────────────────────────

print("=" * 60)
print("生成式召回示例: Trie-Constrained Beam Search")
print("=" * 60)

t0 = time.time()
builder = PipelineBuilder(spec, k=K)
pipeline, meta = builder.build(items)
pipeline.eval()
print(f"\n构建耗时: {time.time() - t0:.3f}s")
print(f"  item 数量: {meta['num_items']}")
print(f"  谓词数量: {meta['num_preds']}")
print(f"  beam 宽度: {meta['generative_leaves'][0]['beam_width']}")
print(f"  Trie 各层节点数: {meta['generative_leaves'][0]['nodes_per_level']}")

# ── 6. 在线查询 ─────────────────────────────────────────────────────────────

gen_meta = meta["generative_leaves"][0]
item_ids = meta["item_ids"] or [f"item_{i}" for i in range(len(items))]

users = [
    ({"city": "北京", "age": 30}, "北京 30岁"),
    ({"city": "上海", "age": 22}, "上海 22岁"),
    ({"city": "广州", "age": 35}, "广州 35岁"),
    ({"city": "深圳", "age": 15}, "深圳 15岁"),
]

print("\n" + "-" * 60)
for user_attrs, desc in users:
    from torch_recall.recall_method.targeting.encoder import encode_user
    pred = encode_user(user_attrs, gen_meta["targeting"]).unsqueeze(0)
    query = torch.randn(1, meta["total_query_dim"])

    with torch.no_grad():
        top_scores, top_indices = pipeline(pred, query)

    print(f"\n【{desc}】 属性={user_attrs}")
    print(f"  召回 top-{K}:")
    for rank in range(K):
        idx = top_indices[0, rank].item()
        score = top_scores[0, rank].item()
        if score > float("-inf"):
            name = item_ids[idx]
            print(f"    #{rank+1}  {name:<12s}  score={score:.4f}  (item[{idx}])")
        else:
            print(f"    #{rank+1}  (无匹配)  score=-inf")

# ── 7. Trie 结构概览 ────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("Trie 结构概览 (Dense/CSR hybrid):")
trie = pipeline.root.trie
print(f"  总状态数: {trie.num_states}")
print(f"  Dense 层数: {trie.d_dense}")
print(f"  CSR 边数: {trie.packed_csr.shape[0] - trie.vocab_size} (+ {trie.vocab_size} padding)")
for d in range(trie.path_length):
    s, c = trie._level_start[d], trie._level_count[d]
    print(f"  Depth {d}: {c} 个节点 (state {s}..{s+c-1})")
print(f"  叶子节点: {trie.num_leaves}, 最多 {trie.leaf_item_ids.shape[1]} 个 item/leaf")

print("=" * 60)

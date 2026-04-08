#!/usr/bin/env python3
"""
Head-to-head comparison: torch-recall vs tantivy (Rust/Lucene) vs pyroaring (Roaring Bitmap).

Same data, same queries, same machine. Measures build time, query latency, and hit count.
"""

import gc
import os
import random
import sys
import tempfile
import time

import pyroaring
import tantivy
import torch

from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query

# ── Config ───────────────────────────────────────────────────────────────

N = 1_000_000
WARMUP = 5
RUNS = 30

CITIES = [f"city_{i}" for i in range(50)]
GENDERS = ["男", "女"]
CATEGORIES = [f"cat_{i}" for i in range(100)]

UINT64_MASK = (1 << 64) - 1


def log(msg: str = ""):
    print(msg, flush=True)


# ── Data generation ──────────────────────────────────────────────────────

def generate_items(n: int):
    items = []
    for _ in range(n):
        items.append({
            "city": random.choice(CITIES),
            "gender": random.choice(GENDERS),
            "category": random.choice(CATEGORIES),
            "price": random.uniform(0, 1000),
            "score": random.uniform(1, 5),
        })
    return items


# ── Latency measurement ─────────────────────────────────────────────────

def measure_latency(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "avg": sum(times) / len(times),
        "p50": times[len(times) // 2],
        "p99": times[int(len(times) * 0.99)],
    }


def fmt_lat(stats: dict) -> str:
    return f"avg={stats['avg']:8.2f}ms  p50={stats['p50']:8.2f}ms  p99={stats['p99']:8.2f}ms"


# ── torch-recall ─────────────────────────────────────────────────────────

def build_torch_recall(items):
    schema = Schema(
        discrete_fields=["city", "gender", "category"],
        numeric_fields=["price", "score"],
        text_fields=[],
    )
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()
    return model, meta


def count_hits_tr(result: torch.Tensor, num_items: int) -> int:
    count = 0
    for w in range(result.shape[0]):
        word = result[w].item() & UINT64_MASK
        while word:
            bit = (word & -word).bit_length() - 1
            if w * 64 + bit < num_items:
                count += 1
            word &= word - 1
    return count


# ── tantivy ──────────────────────────────────────────────────────────────

def build_tantivy(items):
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("city", stored=False, tokenizer_name="raw", index_option="basic")
    sb.add_text_field("gender", stored=False, tokenizer_name="raw", index_option="basic")
    sb.add_text_field("category", stored=False, tokenizer_name="raw", index_option="basic")
    sb.add_float_field("price", stored=False, indexed=True, fast=True)
    sb.add_float_field("score", stored=False, indexed=True, fast=True)
    schema = sb.build()

    idx = tantivy.Index(schema)
    writer = idx.writer(heap_size=256_000_000, num_threads=1)
    for item in items:
        writer.add_document(tantivy.Document(
            city=item["city"],
            gender=item["gender"],
            category=item["category"],
            price=item["price"],
            score=item["score"],
        ))
    writer.commit()
    idx.reload()
    return idx


# ── pyroaring ────────────────────────────────────────────────────────────

def build_roaring(items):
    discrete_bitmaps = {}
    prices = []
    scores = []

    for idx, item in enumerate(items):
        for field in ("city", "gender", "category"):
            key = f"{field}={item[field]}"
            if key not in discrete_bitmaps:
                discrete_bitmaps[key] = pyroaring.BitMap()
            discrete_bitmaps[key].add(idx)
        prices.append(item["price"])
        scores.append(item["score"])

    all_ids = pyroaring.BitMap(range(len(items)))
    return discrete_bitmaps, prices, scores, all_ids


# ── Queries ──────────────────────────────────────────────────────────────

def define_queries():
    """Return query definitions that can be executed on all three systems."""
    return {
        "单条件 (city)": {
            "tr": 'city == "city_0"',
            "tantivy": "city:city_0",
            "roaring": lambda bm, prices, scores, all_ids: bm.get("city=city_0", pyroaring.BitMap()),
        },
        "双条件 AND": {
            "tr": 'city == "city_0" AND gender == "男"',
            "tantivy": "city:city_0 AND gender:男",
            "roaring": lambda bm, prices, scores, all_ids: (
                bm.get("city=city_0", pyroaring.BitMap()) & bm.get("gender=男", pyroaring.BitMap())
            ),
        },
        "三条件 AND (含数值)": {
            "tr": 'city == "city_0" AND gender == "男" AND price < 500.0',
            "tantivy": "city:city_0 AND gender:男 AND price:[0 TO 500}",
            "roaring": lambda bm, prices, scores, all_ids: (
                bm.get("city=city_0", pyroaring.BitMap())
                & bm.get("gender=男", pyroaring.BitMap())
                & pyroaring.BitMap([i for i, p in enumerate(prices) if p < 500.0])
            ),
        },
        "OR (2 city)": {
            "tr": 'city == "city_0" OR city == "city_1"',
            "tantivy": "city:city_0 OR city:city_1",
            "roaring": lambda bm, prices, scores, all_ids: (
                bm.get("city=city_0", pyroaring.BitMap()) | bm.get("city=city_1", pyroaring.BitMap())
            ),
        },
        "NOT": {
            "tr": 'NOT category == "cat_0"',
            "tantivy": None,  # tantivy 不支持纯 NOT 查询
            "roaring": lambda bm, prices, scores, all_ids: (
                all_ids - bm.get("category=cat_0", pyroaring.BitMap())
            ),
        },
        "复杂 (OR+AND+数值)": {
            "tr": '(city == "city_0" OR city == "city_1") AND price < 300.0 AND score > 2.0',
            "tantivy": "(city:city_0 OR city:city_1) AND price:[0 TO 300} AND score:{2 TO *}",
            "roaring": lambda bm, prices, scores, all_ids: (
                (bm.get("city=city_0", pyroaring.BitMap()) | bm.get("city=city_1", pyroaring.BitMap()))
                & pyroaring.BitMap([i for i, p in enumerate(prices) if p < 300.0])
                & pyroaring.BitMap([i for i, s in enumerate(scores) if s > 2.0])
            ),
        },
    }


# pyroaring 数值条件需要预建 bitmap 才公平 (避免每次查询重建)
def prebuild_numeric_bitmaps(prices, scores):
    return {
        "price<500": pyroaring.BitMap([i for i, p in enumerate(prices) if p < 500.0]),
        "price<300": pyroaring.BitMap([i for i, p in enumerate(prices) if p < 300.0]),
        "score>2": pyroaring.BitMap([i for i, s in enumerate(scores) if s > 2.0]),
    }


def define_queries_prebuilt(num_bm):
    """Roaring queries with prebuilt numeric bitmaps for fairer comparison."""
    return {
        "三条件 AND (含数值)": lambda bm, all_ids: (
            bm.get("city=city_0", pyroaring.BitMap())
            & bm.get("gender=男", pyroaring.BitMap())
            & num_bm["price<500"]
        ),
        "复杂 (OR+AND+数值)": lambda bm, all_ids: (
            (bm.get("city=city_0", pyroaring.BitMap()) | bm.get("city=city_1", pyroaring.BitMap()))
            & num_bm["price<300"]
            & num_bm["score>2"]
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    torch.manual_seed(42)

    log("=" * 78)
    log("  torch-recall vs tantivy vs pyroaring — 同数据同查询实测对比")
    log(f"  N={N:,}  warmup={WARMUP}  runs={RUNS}  CPU cores={os.cpu_count()}")
    log("=" * 78)

    # ── Generate data ────────────────────────────────────────────────────
    log("\n生成数据...")
    t0 = time.time()
    items = generate_items(N)
    log(f"  数据生成: {time.time() - t0:.1f}s")

    # ── Build indexes ────────────────────────────────────────────────────
    log(f"\n{'─' * 78}")
    log("  1. 索引构建时间")
    log(f"{'─' * 78}")

    t0 = time.time()
    tr_model, tr_meta = build_torch_recall(items)
    tr_build = time.time() - t0
    log(f"  torch-recall     : {tr_build:8.1f}s")

    t0 = time.time()
    tv_index = build_tantivy(items)
    tv_build = time.time() - t0
    log(f"  tantivy          : {tv_build:8.1f}s")

    t0 = time.time()
    ro_bitmaps, ro_prices, ro_scores, ro_all = build_roaring(items)
    ro_build = time.time() - t0
    log(f"  pyroaring        : {ro_build:8.1f}s")

    num_bm = prebuild_numeric_bitmaps(ro_prices, ro_scores)
    roaring_prebuilt = define_queries_prebuilt(num_bm)

    # ── torch.compile ────────────────────────────────────────────────────
    log("\n编译 torch-recall (torch.compile)...")
    t0 = time.time()
    tr_compiled = torch.compile(tr_model)
    warmup_t = encode_query('city == "city_0"', tr_meta)
    with torch.no_grad():
        _ = tr_compiled(**warmup_t)
        _ = tr_compiled(**warmup_t)
    log(f"  编译完成: {time.time() - t0:.1f}s")

    # ── Query latency ────────────────────────────────────────────────────
    queries = define_queries()
    tv_searcher = tv_index.searcher()

    log(f"\n{'─' * 78}")
    log("  2. 查询延迟对比 (1M items)")
    log(f"{'─' * 78}")

    for qname, qdef in queries.items():
        log(f"\n  ▸ {qname}")

        # --- torch-recall (compiled) ---
        tensors = encode_query(qdef["tr"], tr_meta)
        def tr_fn(t=tensors):
            with torch.no_grad():
                return tr_compiled(**t)
        stats = measure_latency(tr_fn)
        with torch.no_grad():
            r = tr_compiled(**tensors)
        hits = count_hits_tr(r, tr_meta["num_items"])
        log(f"    torch-recall (compiled)  {fmt_lat(stats)}  hits={hits}")

        # --- torch-recall (eager) ---
        def tr_eager_fn(t=tensors):
            with torch.no_grad():
                return tr_model(**t)
        stats_eager = measure_latency(tr_eager_fn)
        log(f"    torch-recall (eager)     {fmt_lat(stats_eager)}  hits={hits}")

        # --- tantivy ---
        if qdef["tantivy"] is not None:
            tq = tv_index.parse_query(qdef["tantivy"])
            def tv_fn(q=tq):
                s = tv_index.searcher()
                return s.search(q, N)
            stats_tv = measure_latency(tv_fn)
            tv_hits = tv_fn().count
            log(f"    tantivy                 {fmt_lat(stats_tv)}  hits={tv_hits}")
        else:
            log(f"    tantivy                 (不支持纯 NOT 查询)")

        # --- pyroaring ---
        ro_fn_raw = qdef["roaring"]
        def ro_fn(f=ro_fn_raw):
            return f(ro_bitmaps, ro_prices, ro_scores, ro_all)
        stats_ro = measure_latency(ro_fn)
        ro_result = ro_fn()
        ro_hits = len(ro_result)

        if qname in roaring_prebuilt:
            prebuilt_fn_raw = roaring_prebuilt[qname]
            def ro_pre_fn(f=prebuilt_fn_raw):
                return f(ro_bitmaps, ro_all)
            stats_ro_pre = measure_latency(ro_pre_fn)
            log(f"    pyroaring (含数值构建)   {fmt_lat(stats_ro)}  hits={ro_hits}")
            log(f"    pyroaring (预建数值bm)  {fmt_lat(stats_ro_pre)}  hits={len(ro_pre_fn())}")
        else:
            log(f"    pyroaring               {fmt_lat(stats_ro)}  hits={ro_hits}")

    # ── Summary ──────────────────────────────────────────────────────────
    log(f"\n{'─' * 78}")
    log("  3. 命中数一致性检查")
    log(f"{'─' * 78}")

    all_match = True
    for qname, qdef in queries.items():
        tensors = encode_query(qdef["tr"], tr_meta)
        with torch.no_grad():
            r = tr_compiled(**tensors)
        tr_hits = count_hits_tr(r, tr_meta["num_items"])

        ro_fn_raw = qdef["roaring"]
        ro_hits = len(ro_fn_raw(ro_bitmaps, ro_prices, ro_scores, ro_all))

        if qdef["tantivy"] is not None:
            tq = tv_index.parse_query(qdef["tantivy"])
            tv_hits = tv_index.searcher().search(tq, N).count
            match = (tr_hits == ro_hits == tv_hits)
            status = "✓" if match else "✗"
            log(f"  {status} {qname:<28s}  tr={tr_hits:>8,}  tantivy={tv_hits:>8,}  roaring={ro_hits:>8,}")
        else:
            match = (tr_hits == ro_hits)
            status = "✓" if match else "✗"
            log(f"  {status} {qname:<28s}  tr={tr_hits:>8,}  tantivy={'N/A':>8s}  roaring={ro_hits:>8,}")

        if not match:
            all_match = False

    if all_match:
        log("\n  所有系统命中数一致 ✓")
    else:
        log("\n  ⚠ 存在命中数不一致，请检查查询语义是否等价")

    log(f"\n{'=' * 78}")
    log("  对比测试完成")
    log(f"{'=' * 78}")


if __name__ == "__main__":
    main()

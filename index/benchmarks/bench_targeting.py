#!/usr/bin/env python3
"""
Targeting Recall 性能测试
=========================

测量不同规模下的:
  - 索引构建时间
  - 内存占用
  - 用户编码延迟
  - forward() 延迟 (eager / torch.compile)
  - 端到端延迟 (encode + forward)

运行:
    cd torch-recall
    PYTHONPATH=index python index/benchmarks/bench_targeting.py
"""

import gc
import random
import statistics
import sys
import time

import torch

from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.targeting.encoder import encode_user

# ── 合成数据生成 ────────────────────────────────────────────────────────────

CITIES = [f"city_{i}" for i in range(50)]
GENDERS = ["男", "女"]
CATEGORIES = [f"cat_{i}" for i in range(100)]
TAGS = [f"tag_{i}" for i in range(200)]
AGE_THRESHOLDS = list(range(10, 70, 5))
PRICE_THRESHOLDS = [10, 20, 50, 100, 200, 500, 1000]

SCHEMA = Schema(
    discrete_fields=["city", "gender", "category"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)


def _random_rule(rng: random.Random) -> str:
    """Generate a random targeting rule with 1-4 predicates, possibly with OR."""
    num_conj = rng.choices([1, 2], weights=[0.7, 0.3])[0]
    conjs = []
    for _ in range(num_conj):
        n_preds = rng.randint(1, 4)
        preds = []
        used_fields: set[str] = set()
        for _ in range(n_preds):
            kind = rng.choices(["discrete", "numeric", "text"], weights=[0.5, 0.3, 0.2])[0]
            if kind == "discrete":
                field = rng.choice(["city", "gender", "category"])
                if field in used_fields:
                    continue
                used_fields.add(field)
                if field == "city":
                    val = rng.choice(CITIES)
                elif field == "gender":
                    val = rng.choice(GENDERS)
                else:
                    val = rng.choice(CATEGORIES)
                op = rng.choice(["==", "!="])
                preds.append(f'{field} {op} "{val}"')
            elif kind == "numeric":
                field = rng.choice(["age", "price"])
                if field in used_fields:
                    continue
                used_fields.add(field)
                if field == "age":
                    val = rng.choice(AGE_THRESHOLDS)
                else:
                    val = rng.choice(PRICE_THRESHOLDS)
                op = rng.choice([">", ">=", "<", "<="])
                preds.append(f"{field} {op} {val}")
            else:
                if "tags" in used_fields:
                    continue
                used_fields.add("tags")
                tag = rng.choice(TAGS)
                preds.append(f'tags contains "{tag}"')
        if not preds:
            preds.append(f'city == "{rng.choice(CITIES)}"')
        conjs.append(" AND ".join(preds))
    if len(conjs) == 1:
        return conjs[0]
    return "(" + ") OR (".join(conjs) + ")"


def _random_user(rng: random.Random) -> dict:
    return {
        "city": rng.choice(CITIES),
        "gender": rng.choice(GENDERS),
        "category": rng.choice(CATEGORIES),
        "age": rng.randint(10, 70),
        "price": rng.uniform(5, 1500),
        "tags": " ".join(rng.sample(TAGS, k=rng.randint(1, 5))),
    }


def _tensor_mem_mb(model: torch.nn.Module) -> float:
    total = 0
    for buf in model.buffers():
        total += buf.nelement() * buf.element_size()
    return total / (1024 * 1024)


# ── Benchmark 函数 ──────────────────────────────────────────────────────────

def benchmark_build(n_items: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    rules = [_random_rule(rng) for _ in range(n_items)]

    gc.collect()
    t0 = time.perf_counter()
    builder = TargetingBuilder(SCHEMA)
    model, meta = builder.build(rules)
    build_time = time.perf_counter() - t0

    mem_mb = _tensor_mem_mb(model)
    return {
        "n_items": n_items,
        "build_time_s": build_time,
        "memory_mb": mem_mb,
        "num_preds": meta["num_preds"],
        "num_conjs": meta["num_conjs"],
        "model": model,
        "meta": meta,
    }


def benchmark_latency(
    model: torch.nn.Module,
    meta: dict,
    n_users: int = 200,
    warmup: int = 20,
    seed: int = 123,
) -> dict:
    rng = random.Random(seed)
    users = [_random_user(rng) for _ in range(n_users + warmup)]

    model.eval()

    # ── encode latency ──
    encode_times = []
    for u in users:
        t0 = time.perf_counter()
        encode_user(u, meta)
        encode_times.append(time.perf_counter() - t0)
    encode_times = encode_times[warmup:]

    # pre-encode all users
    pred_tensors = [encode_user(u, meta) for u in users]

    # ── eager forward latency ──
    eager_times = []
    with torch.no_grad():
        for pt in pred_tensors:
            t0 = time.perf_counter()
            model(pt)
            eager_times.append(time.perf_counter() - t0)
    eager_times = eager_times[warmup:]

    # ── torch.compile forward latency ──
    compiled_model = torch.compile(model)
    compile_times = []
    with torch.no_grad():
        for pt in pred_tensors:
            t0 = time.perf_counter()
            compiled_model(pt)
            compile_times.append(time.perf_counter() - t0)
    compile_times = compile_times[warmup:]

    # ── e2e latency (encode + eager forward) ──
    e2e_times = []
    with torch.no_grad():
        for u in users:
            t0 = time.perf_counter()
            ps = encode_user(u, meta)
            model(ps)
            e2e_times.append(time.perf_counter() - t0)
    e2e_times = e2e_times[warmup:]

    # count matches for sanity
    with torch.no_grad():
        sample_result = model(pred_tensors[warmup])
    match_count = sample_result.sum().item()

    def _stats(times: list[float]) -> dict:
        times_ms = [t * 1000 for t in times]
        return {
            "avg_ms": statistics.mean(times_ms),
            "p50_ms": statistics.median(times_ms),
            "p99_ms": sorted(times_ms)[int(len(times_ms) * 0.99)],
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
        }

    return {
        "encode": _stats(encode_times),
        "eager": _stats(eager_times),
        "compiled": _stats(compile_times),
        "e2e_eager": _stats(e2e_times),
        "sample_matches": match_count,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def _fmt_stats(s: dict) -> str:
    return (
        f"avg={s['avg_ms']:.3f}ms  "
        f"p50={s['p50_ms']:.3f}ms  "
        f"p99={s['p99_ms']:.3f}ms"
    )


def main():
    scales = [1_000, 10_000, 100_000]
    if "--large" in sys.argv:
        scales.append(1_000_000)

    print("=" * 75)
    print("Targeting Recall 性能测试")
    print("=" * 75)
    print(f"Schema: discrete={SCHEMA.discrete_fields}, "
          f"numeric={SCHEMA.numeric_fields}, text={SCHEMA.text_fields}")
    print(f"离散值空间: city={len(CITIES)}, gender={len(GENDERS)}, "
          f"category={len(CATEGORIES)}")
    print(f"文本词表: tags={len(TAGS)}")
    print()

    for n in scales:
        print(f"{'─' * 75}")
        print(f"规模: {n:,} items")
        print(f"{'─' * 75}")

        info = benchmark_build(n)
        print(f"  构建时间:    {info['build_time_s']:.3f}s")
        print(f"  内存占用:    {info['memory_mb']:.1f} MB")
        print(f"  谓词数:      {info['num_preds']}")
        print(f"  Conjunction: {info['num_conjs']}")

        lat = benchmark_latency(info["model"], info["meta"])
        print(f"  样本命中:    {lat['sample_matches']} / {n}")
        print()
        print(f"  encode_user:          {_fmt_stats(lat['encode'])}")
        print(f"  forward (eager):      {_fmt_stats(lat['eager'])}")
        print(f"  forward (compiled):   {_fmt_stats(lat['compiled'])}")
        print(f"  E2E (encode+eager):   {_fmt_stats(lat['e2e_eager'])}")
        print()

    print("=" * 75)
    print("提示: 加 --large 参数运行 1M 规模测试 (需更多内存和时间)")
    print("=" * 75)


if __name__ == "__main__":
    main()

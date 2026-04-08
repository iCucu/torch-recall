#!/usr/bin/env python3
"""
Comprehensive benchmark for torch-recall.

Evaluates:
  1. Index build time (10K / 100K / 1M)
  2. Memory footprint breakdown
  3. Query latency: eager vs torch.compile, 6 query patterns
  4. Selectivity impact
  5. Export time (torch.export + AOTInductor)
"""

import gc
import os
import random
import sys
import tempfile
import time

import torch

from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query
from torch_recall.exporter import export_model

SECTION_TIMEOUT = 300

SCHEMA = Schema(
    discrete_fields=["city", "gender", "category"],
    numeric_fields=["price", "score"],
    text_fields=["title"],
)

CITIES = [f"city_{i}" for i in range(50)]
GENDERS = ["男", "女"]
CATEGORIES = [f"cat_{i}" for i in range(100)]

UINT64_MASK = (1 << 64) - 1


def log(msg: str = ""):
    print(msg, flush=True)


def generate_items(n: int) -> list[dict]:
    items = []
    for _ in range(n):
        items.append({
            "city": random.choice(CITIES),
            "gender": random.choice(GENDERS),
            "category": random.choice(CATEGORIES),
            "price": random.uniform(0, 1000),
            "score": random.uniform(1, 5),
            "title": " ".join(f"term_{random.randint(0, 999)}" for _ in range(5)),
        })
    return items


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def tensor_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


def measure_latency(fn, warmup=3, runs=20):
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
        "min": times[0],
        "max": times[-1],
    }


def count_hits(result: torch.Tensor, num_items: int) -> int:
    """Count set bits in packed int64 bitmap. Masks to 64 bits to avoid
    infinite loops on Python's unlimited-precision negative ints."""
    count = 0
    for w in range(result.shape[0]):
        word = result[w].item() & UINT64_MASK
        base = w * 64
        while word:
            bit = (word & -word).bit_length() - 1
            if base + bit < num_items:
                count += 1
            word &= word - 1
    return count


def print_lat(label: str, stats: dict, hits: int | None = None):
    line = f"  {label:<40s}  avg={stats['avg']:8.3f}ms  p50={stats['p50']:8.3f}ms  p99={stats['p99']:8.3f}ms"
    if hits is not None:
        line += f"  hits={hits}"
    log(line)


QUERIES = {
    "单条件 (离散)":           'city == "city_0"',
    "双条件 AND (离散)":       'city == "city_0" AND gender == "男"',
    "三条件 AND (离散+数值)":   'city == "city_0" AND gender == "男" AND price < 500.0',
    "OR (2 个离散值)":         'city == "city_0" OR city == "city_1"',
    "NOT":                    'NOT category == "cat_0"',
    "复杂 (OR+AND+数值)":     '(city == "city_0" OR city == "city_1") AND price < 300.0 AND score > 2.0',
}


def bench_latency(model, meta, N, mode_label, run_fn_factory):
    """Run latency benchmark for all query patterns."""
    for label, qstr in QUERIES.items():
        tensors = encode_query(qstr, meta)
        run_fn = run_fn_factory(tensors)
        t0 = time.time()
        stats = measure_latency(run_fn)
        if time.time() - t0 > SECTION_TIMEOUT:
            log(f"  ⚠ TIMEOUT on {label}")
            return
        with torch.no_grad():
            r = run_fn()
        hits = count_hits(r, N)
        print_lat(label, stats, hits)


def main():
    random.seed(42)
    torch.manual_seed(42)
    wall_start = time.time()

    log("=" * 72)
    log(f"  torch-recall comprehensive benchmark")
    log(f"  PyTorch {torch.__version__}  |  CPU cores: {os.cpu_count()}")
    log("=" * 72)

    # ── 1. Build Time ────────────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  1. Index Build Time")
    log(f"{'─' * 72}")
    log(f"  {'Scale':<8s}  {'Build (s)':>10s}  {'Bitmaps':>14s}  {'Mem(bm)':>10s}  {'Mem(num)':>10s}")

    model_1m = None
    meta_1m = None

    for n in [10_000, 100_000, 1_000_000]:
        label = f"{n // 1000}K" if n < 1_000_000 else "1M"

        t0 = time.time()
        items = generate_items(n)
        gen_s = time.time() - t0

        t0 = time.time()
        builder = IndexBuilder(SCHEMA)
        model, meta = builder.build(items)
        build_s = time.time() - t0

        bm_mem = tensor_bytes(model.bitmaps)
        nm_mem = tensor_bytes(model.numeric_data)
        shape = f"{model.bitmaps.shape[0]}×{model.bitmaps.shape[1]}"

        log(f"  {label:<8s}  {build_s:>10.1f}  {shape:>14s}  {fmt_bytes(bm_mem):>10s}  {fmt_bytes(nm_mem):>10s}  (gen={gen_s:.1f}s)")

        if n == 1_000_000:
            model_1m = model
            meta_1m = meta
        del items
        gc.collect()

    model = model_1m
    meta = meta_1m
    N = meta["num_items"]
    model.eval()

    # ── 2. Memory Breakdown ──────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  2. Memory Breakdown (1M items)")
    log(f"{'─' * 72}")

    components = [
        ("bitmaps",      tensor_bytes(model.bitmaps)),
        ("numeric_data", tensor_bytes(model.numeric_data)),
        ("item_ids",     tensor_bytes(model.item_ids)),
        ("all_ones",     tensor_bytes(model.all_ones)),
        ("valid_mask",   tensor_bytes(model.valid_mask)),
        ("pack_powers",  tensor_bytes(model.pack_powers)),
    ]
    total = sum(s for _, s in components)
    for name, size in components:
        log(f"  {name:<20s}  {fmt_bytes(size):>12s}  ({size / total * 100:5.1f}%)")
    log(f"  {'TOTAL':<20s}  {fmt_bytes(total):>12s}")
    log(f"  Bitmap entries: {meta['num_bitmaps']:,d}  |  "
        f"Bitmap length: {meta['bitmap_len']:,d} words ({meta['bitmap_len'] * 8 / 1024:.0f} KB/entry)")

    # ── 3. Eager ─────────────────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  3. Query Latency — Eager Mode (1M items, 20 runs)")
    log(f"{'─' * 72}")

    def eager_factory(tensors):
        def run():
            with torch.no_grad():
                return model(**tensors)
        return run

    bench_latency(model, meta, N, "eager", eager_factory)

    # ── 4. Compiled ──────────────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  4. Query Latency — torch.compile (1M items, 20 runs)")
    log(f"{'─' * 72}")
    log("  Compiling...")

    t0 = time.time()
    compiled = torch.compile(model)
    warmup_t = encode_query('city == "city_0"', meta)
    with torch.no_grad():
        _ = compiled(**warmup_t)
        _ = compiled(**warmup_t)
    log(f"  Compilation: {time.time() - t0:.1f}s")

    def compiled_factory(tensors):
        def run():
            with torch.no_grad():
                return compiled(**tensors)
        return run

    bench_latency(model, meta, N, "compiled", compiled_factory)

    # ── 5. Selectivity ───────────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  5. Selectivity Impact — torch.compile (1M items)")
    log(f"{'─' * 72}")

    sel_queries = {
        "极高 (~1%)":  'city == "city_0" AND gender == "男" AND price < 500.0',
        "高 (~2%)":    'city == "city_0" AND gender == "男"',
        "中 (~10%)":   'price < 100.0',
        "低 (~50%)":   'gender == "男"',
        "极低 (~99%)": 'NOT category == "cat_0"',
    }
    for label, qstr in sel_queries.items():
        tensors = encode_query(qstr, meta)

        def run(t=tensors):
            with torch.no_grad():
                return compiled(**t)

        stats = measure_latency(run)
        with torch.no_grad():
            r = compiled(**tensors)
        hits = count_hits(r, N)
        pct = hits / N * 100
        print_lat(f"{label}  actual={pct:.1f}%", stats, hits)

    # ── 6. Export ────────────────────────────────────────────────────────
    log(f"\n{'─' * 72}")
    log("  6. Export Time (torch.export + AOTInductor)")
    log(f"{'─' * 72}")

    with tempfile.TemporaryDirectory() as tmpdir:
        pt2_path = os.path.join(tmpdir, "model.pt2")
        t0 = time.time()
        export_model(model, meta, pt2_path)
        export_s = time.time() - t0
        pt2_size = os.path.getsize(pt2_path)
        log(f"  Export time: {export_s:.1f}s")
        log(f"  .pt2 size:   {fmt_bytes(pt2_size)}")

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    log(f"\n{'=' * 72}")
    log(f"  All benchmarks completed in {elapsed:.0f}s")
    log(f"{'=' * 72}")


if __name__ == "__main__":
    main()

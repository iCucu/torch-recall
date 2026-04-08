"""Benchmark: build index and query 1M items."""
import time
import random

import torch
from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query


def generate_items(n: int, schema: Schema) -> list[dict]:
    cities = [f"city_{i}" for i in range(50)]
    genders = ["男", "女"]
    categories = [f"cat_{i}" for i in range(100)]
    items = []
    for _ in range(n):
        item = {}
        for f in schema.discrete_fields:
            if f == "city":
                item[f] = random.choice(cities)
            elif f == "gender":
                item[f] = random.choice(genders)
            elif f == "category":
                item[f] = random.choice(categories)
            else:
                item[f] = f"val_{random.randint(0, 99)}"
        for f in schema.numeric_fields:
            item[f] = random.uniform(0, 1000)
        for f in schema.text_fields:
            words = [f"term_{random.randint(0, 999)}" for _ in range(5)]
            item[f] = " ".join(words)
        items.append(item)
    return items


def main():
    N = 1_000_000
    schema = Schema(
        discrete_fields=["city", "gender", "category"],
        numeric_fields=["price", "score"],
        text_fields=["title"],
    )

    print(f"Generating {N} items...")
    t0 = time.time()
    items = generate_items(N, schema)
    print(f"  Generated in {time.time() - t0:.1f}s")

    print("Building index...")
    t0 = time.time()
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.1f}s")
    print(f"  Bitmaps: {model.bitmaps.shape}")
    print(f"  Numeric: {model.numeric_data.shape}")

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")

    query_str = 'city == "city_0" AND gender == "男" AND price < 500.0'
    tensors = encode_query(query_str, meta)
    tensors = {k: v.to(device) for k, v in tensors.items()}

    print("\n--- Eager mode ---")
    with torch.no_grad():
        _ = model(**tensors)
    if device == "cuda":
        torch.cuda.synchronize()

    num_queries = 100
    times = []
    for _ in range(num_queries):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model(**tensors)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"  Avg: {avg_ms:.3f}ms  P50: {p50:.3f}ms  P99: {p99:.3f}ms")

    print("\n--- Compiled mode (torch.compile) ---")
    compiled_model = torch.compile(model)
    with torch.no_grad():
        _ = compiled_model(**tensors)
        _ = compiled_model(**tensors)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_queries):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            result = compiled_model(**tensors)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    print(f"\nQuery benchmark ({num_queries} runs):")
    print(f"  Avg: {avg_ms:.3f}ms")
    print(f"  P50: {p50:.3f}ms")
    print(f"  P99: {p99:.3f}ms")

    bits = []
    packed = result.cpu()
    for word_idx in range(packed.shape[0]):
        val = packed[word_idx].item()
        for bit in range(64):
            if val & (1 << bit):
                bits.append(word_idx * 64 + bit)
    matching = [b for b in bits if b < N]
    print(f"  Matching items: {len(matching)}")


if __name__ == "__main__":
    main()

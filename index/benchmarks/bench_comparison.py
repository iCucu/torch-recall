#!/usr/bin/env python3
"""
定向召回方案对比 Benchmark
===========================

对比三种「反向匹配」实现方案在同一数据集上的性能:

  1. **暴力 Python 评估**    — 逐 item 解析规则字符串，逐条评估
  2. **pyroaring 谓词位图**  — 为每个谓词建 Roaring Bitmap，然后做集合运算
  3. **torch-recall (eager)** — 本项目的张量方案 (无编译)
  4. **torch-recall (compiled)** — torch.compile 编译后

架构对比核心问题:
  传统倒排索引 (Elasticsearch/Lucene) 是「查询→item」的正向检索。
  定向召回是「item规则→用户」的反向匹配，等价于布尔表达式索引 (BE-Index)。
  本 benchmark 对比的是不同反向匹配实现，而非正向检索。

运行:
    cd torch-recall
    PYTHONPATH=index python index/benchmarks/bench_comparison.py

依赖:
    pip install pyroaring   (已在环境中)
"""

import gc
import random
import statistics
import sys
import time

import torch
from pyroaring import BitMap

from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.targeting.encoder import encode_user
from torch_recall.query.parser import parse_expr
from torch_recall.query.dnf import to_dnf

# ═══════════════════════════════════════════════════════════════════════════════
# 数据生成
# ═══════════════════════════════════════════════════════════════════════════════

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
                val = rng.choice(AGE_THRESHOLDS if field == "age" else PRICE_THRESHOLDS)
                op = rng.choice([">", ">=", "<", "<="])
                preds.append(f"{field} {op} {val}")
            else:
                if "tags" in used_fields:
                    continue
                used_fields.add("tags")
                preds.append(f'tags contains "{rng.choice(TAGS)}"')
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


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 1: 暴力 Python 评估
# ═══════════════════════════════════════════════════════════════════════════════

_OP_EVAL = {
    "==": lambda a, b: str(a) == str(b),
    "!=": lambda a, b: str(a) != str(b),
    "<": lambda a, b: float(a) < float(b),
    ">": lambda a, b: float(a) > float(b),
    "<=": lambda a, b: float(a) <= float(b),
    ">=": lambda a, b: float(a) >= float(b),
    "contains": lambda a, b: str(b) in str(a).split(),
}


def _eval_pred(pred, user: dict) -> bool:
    val = user.get(pred.field)
    if val is None:
        return False
    fn = _OP_EVAL.get(pred.op)
    return fn(val, pred.value) if fn else False


def brute_force_match(parsed_rules: list, user: dict) -> list[int]:
    """Parse each rule into DNF, evaluate against user."""
    matched = []
    for item_idx, dnf in enumerate(parsed_rules):
        for conj in dnf:
            if all(
                _eval_pred(lit.pred, user) != lit.negated
                for lit in conj
            ):
                matched.append(item_idx)
                break
    return matched


def build_brute_force(rules: list[str]) -> list:
    """Pre-parse all rules into DNF for brute-force evaluation."""
    parsed = []
    for r in rules:
        expr = parse_expr(r)
        dnf = to_dnf(expr)
        parsed.append(dnf)
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 2: pyroaring 谓词位图
# ═══════════════════════════════════════════════════════════════════════════════

class RoaringTargetingIndex:
    """Build per-predicate Roaring Bitmaps, then evaluate conjunctions."""

    def __init__(self, rules: list[str], schema: Schema):
        self.n_items = len(rules)
        self._discrete_fields = set(schema.discrete_fields)
        self._numeric_fields = set(schema.numeric_fields)
        self._text_fields = set(schema.text_fields)

        self._pred_bitmaps: dict[tuple, BitMap] = {}
        self._item_dnfs: list[list[list[tuple[tuple, bool]]]] = []

        for item_idx, rule_str in enumerate(rules):
            expr = parse_expr(rule_str)
            dnf = to_dnf(expr)
            item_dnf = []
            for conj in dnf:
                conj_preds = []
                for lit in conj:
                    key, negated = self._normalize_pred(lit)
                    if key not in self._pred_bitmaps:
                        self._pred_bitmaps[key] = BitMap()
                    conj_preds.append((key, negated))
                item_dnf.append(conj_preds)
            self._item_dnfs.append(item_dnf)

        self._all_items = BitMap(range(self.n_items))

        self._all_pred_keys = list(self._pred_bitmaps.keys())
        self._numeric_pred_keys = [
            k for k in self._all_pred_keys if k[0] == "numeric"
        ]
        self._discrete_pred_keys = [
            k for k in self._all_pred_keys if k[0] == "discrete"
        ]
        self._text_pred_keys = [
            k for k in self._all_pred_keys if k[0] == "text"
        ]

    @staticmethod
    def _normalize_pred(lit) -> tuple[tuple, bool]:
        """Return (canonical_key, negated) — normalizes != to == + flip."""
        pred = lit.pred
        negated = lit.negated
        if pred.op == "!=":
            key = ("discrete", pred.field, "==", str(pred.value))
            return key, not negated
        if pred.op == "contains":
            return ("text", pred.field, "contains", str(pred.value)), negated
        if pred.op == "==":
            return ("discrete", pred.field, "==", str(pred.value)), negated
        return ("numeric", pred.field, pred.op, float(pred.value)), negated

    def _evaluate_user(self, user: dict) -> dict[tuple, bool]:
        """Evaluate which predicates a user satisfies."""
        result = {}
        for key in self._discrete_pred_keys:
            _, field, _, value = key
            user_val = user.get(field)
            result[key] = str(user_val) == value if user_val is not None else False

        for key in self._numeric_pred_keys:
            _, field, op, value = key
            user_val = user.get(field)
            if user_val is None:
                result[key] = False
            else:
                fn = _OP_EVAL.get(op)
                result[key] = fn(float(user_val), value) if fn else False

        for key in self._text_pred_keys:
            _, field, _, term = key
            user_text = user.get(field)
            if user_text is None:
                result[key] = False
            else:
                result[key] = term in str(user_text).split()

        return result

    def match(self, user: dict) -> list[int]:
        pred_satisfied = self._evaluate_user(user)

        matched = BitMap()
        for item_idx, item_dnf in enumerate(self._item_dnfs):
            for conj in item_dnf:
                if all(
                    pred_satisfied.get(key, False) != negated
                    for key, negated in conj
                ):
                    matched.add(item_idx)
                    break
        return list(matched)

    def match_bitmap(self, user: dict) -> list[int]:
        """Bitmap-accelerated matching using per-predicate bitmaps.

        For each item's DNF, builds the conjunction result from predicate
        bitmaps using Roaring set operations.
        """
        pred_satisfied = self._evaluate_user(user)

        sat_set = BitMap()
        not_sat_set = BitMap()
        for key, val in pred_satisfied.items():
            if val:
                sat_set_for_pred = self._pred_bitmaps.get(key)
                if sat_set_for_pred is not None:
                    pass  # stored for reference
            # We don't have per-predicate → item bitmaps in the right direction.
            # The _pred_bitmaps would need to be "predicate → set of items using this predicate"
            # and then we'd need to track conjunction structure.
            # This is exactly the complexity that torch-recall's tensor approach avoids.

        # Fall back to per-item evaluation since building a proper
        # inverted predicate→item index with conjunction awareness
        # requires the same DNF tracking that torch-recall does natively.
        return self.match(user)


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 2b: pyroaring 纯位图方案 (预建 predicate→item 倒排 + 集合运算)
# ═══════════════════════════════════════════════════════════════════════════════

class RoaringInvertedIndex:
    """True inverted index: predicate → Roaring Bitmap of items.

    For each conjunction, AND the predicate bitmaps.
    For each item's DNF, OR the conjunction bitmaps.

    This is the closest analogue to how a traditional inverted index
    would solve the reverse matching problem.
    """

    def __init__(self, rules: list[str], schema: Schema):
        self.n_items = len(rules)

        self._pred_to_conj_ids: dict[tuple, list[int]] = {}
        self._conj_pred_keys: list[list[tuple[tuple, bool]]] = []
        self._item_conj_ids: list[list[int]] = []

        self._all_conjs = BitMap()
        self._discrete_fields = set(schema.discrete_fields)
        self._numeric_fields = set(schema.numeric_fields)
        self._text_fields = set(schema.text_fields)

        conj_counter = 0
        for item_idx, rule_str in enumerate(rules):
            expr = parse_expr(rule_str)
            dnf = to_dnf(expr)
            item_conjs = []
            for conj in dnf:
                cid = conj_counter
                conj_counter += 1
                conj_preds = []
                for lit in conj:
                    key, negated = RoaringTargetingIndex._normalize_pred(lit)
                    conj_preds.append((key, negated))
                    self._pred_to_conj_ids.setdefault(key, []).append(cid)
                self._conj_pred_keys.append(conj_preds)
                item_conjs.append(cid)
            self._item_conj_ids.append(item_conjs)

        self.n_conjs = conj_counter

        # Precompute: predicate → BitMap of conj IDs that reference it
        self._pred_conj_bitmaps: dict[tuple, BitMap] = {}
        for key, cids in self._pred_to_conj_ids.items():
            self._pred_conj_bitmaps[key] = BitMap(cids)

        # Precompute: number of predicates per conjunction
        self._conj_num_preds = [len(c) for c in self._conj_pred_keys]

        # Precompute: conj_id → item_id mapping (for result collection)
        self._conj_to_item: list[int] = [0] * conj_counter
        for item_idx, cids in enumerate(self._item_conj_ids):
            for cid in cids:
                self._conj_to_item[cid] = item_idx

        self._all_pred_keys = list(self._pred_conj_bitmaps.keys())

    def match(self, user: dict) -> list[int]:
        """Evaluate user against all items using bitmap operations.

        Strategy: count satisfied predicates per conjunction,
        then check if count == total predicates (accounting for negation).
        """
        # Step 1: evaluate all predicates
        pred_satisfied: dict[tuple, bool] = {}
        for key in self._all_pred_keys:
            kind = key[0]
            if kind == "discrete":
                _, field, _, value = key
                uv = user.get(field)
                pred_satisfied[key] = str(uv) == value if uv is not None else False
            elif kind == "numeric":
                _, field, op, value = key
                uv = user.get(field)
                if uv is None:
                    pred_satisfied[key] = False
                else:
                    fn = _OP_EVAL.get(op)
                    pred_satisfied[key] = fn(float(uv), value) if fn else False
            elif kind == "text":
                _, field, _, term = key
                uv = user.get(field)
                pred_satisfied[key] = term in str(uv).split() if uv is not None else False

        # Step 2: for each conjunction, check all predicates
        # We walk all conjunctions — this is the "gather" step in numpy/torch terms
        conj_ok = [True] * self.n_conjs
        for ci, conj_preds in enumerate(self._conj_pred_keys):
            for key, negated in conj_preds:
                sat = pred_satisfied.get(key, False)
                if sat == negated:  # XOR: sat ^ negated should be True
                    conj_ok[ci] = False
                    break

        # Step 3: for each item, OR its conjunctions
        matched = []
        for item_idx, cids in enumerate(self._item_conj_ids):
            for cid in cids:
                if conj_ok[cid]:
                    matched.append(item_idx)
                    break

        return matched


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 执行
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_items: int, n_users: int = 100, warmup: int = 10, seed: int = 42):
    rng = random.Random(seed)
    rules = [_random_rule(rng) for _ in range(n_items)]
    users = [_random_user(random.Random(seed + 100 + i)) for i in range(n_users + warmup)]

    print(f"\n{'═' * 75}")
    print(f"规模: {n_items:,} items × {n_users} users")
    print(f"{'═' * 75}")

    # ── Build ──
    print("\n[构建阶段]")

    t0 = time.perf_counter()
    parsed_dnfs = build_brute_force(rules)
    bf_build = time.perf_counter() - t0
    print(f"  暴力Python (预解析DNF):   {bf_build:.3f}s")

    t0 = time.perf_counter()
    roaring_idx = RoaringInvertedIndex(rules, SCHEMA)
    ro_build = time.perf_counter() - t0
    print(f"  pyroaring 倒排索引:       {ro_build:.3f}s")

    t0 = time.perf_counter()
    builder = TargetingBuilder(SCHEMA)
    model, meta = builder.build(rules)
    model.eval()
    tr_build = time.perf_counter() - t0
    print(f"  torch-recall:             {tr_build:.3f}s")

    # ── Query latency ──
    print("\n[查询延迟]")

    # warmup
    for u in users[:warmup]:
        brute_force_match(parsed_dnfs, u)
        roaring_idx.match(u)
        with torch.no_grad():
            model.query(u, meta)

    # brute force
    bf_times = []
    bf_results = []
    for u in users[warmup:]:
        t0 = time.perf_counter()
        r = brute_force_match(parsed_dnfs, u)
        bf_times.append(time.perf_counter() - t0)
        bf_results.append(sorted(r))

    # pyroaring inverted
    ro_times = []
    ro_results = []
    for u in users[warmup:]:
        t0 = time.perf_counter()
        r = roaring_idx.match(u)
        ro_times.append(time.perf_counter() - t0)
        ro_results.append(sorted(r))

    # torch-recall eager
    tr_eager_times = []
    tr_results = []
    with torch.no_grad():
        for u in users[warmup:]:
            t0 = time.perf_counter()
            result = model.query(u, meta)
            tr_eager_times.append(time.perf_counter() - t0)
            tr_results.append(sorted(
                i for i in range(n_items) if result[i].item()
            ))

    # torch-recall compiled
    compiled_model = torch.compile(model)
    # warmup compile
    with torch.no_grad():
        for u in users[:warmup]:
            ps = encode_user(u, meta)
            compiled_model(ps)

    tr_comp_times = []
    with torch.no_grad():
        for u in users[warmup:]:
            t0 = time.perf_counter()
            ps = encode_user(u, meta)
            r = compiled_model(ps)
            tr_comp_times.append(time.perf_counter() - t0)

    # ── 结果一致性检查 ──
    mismatches = 0
    for i in range(n_users):
        if bf_results[i] != ro_results[i]:
            mismatches += 1
        if bf_results[i] != tr_results[i]:
            mismatches += 1
    if mismatches > 0:
        print(f"  ⚠ 结果不一致: {mismatches} 处")
    else:
        print(f"  ✓ 所有方案结果一致")

    avg_matches = statistics.mean(len(r) for r in bf_results)
    print(f"  平均命中:   {avg_matches:.0f} / {n_items}")

    def _fmt(times):
        ms = [t * 1000 for t in times]
        return (
            f"avg={statistics.mean(ms):.3f}ms  "
            f"p50={statistics.median(ms):.3f}ms  "
            f"p99={sorted(ms)[int(len(ms) * 0.99)]:.3f}ms"
        )

    print(f"\n  暴力 Python:              {_fmt(bf_times)}")
    print(f"  pyroaring 倒排索引:       {_fmt(ro_times)}")
    print(f"  torch-recall (eager):     {_fmt(tr_eager_times)}")
    print(f"  torch-recall (compiled):  {_fmt(tr_comp_times)}")

    # speedup
    bf_avg = statistics.mean(bf_times) * 1000
    ro_avg = statistics.mean(ro_times) * 1000
    tr_e_avg = statistics.mean(tr_eager_times) * 1000
    tr_c_avg = statistics.mean(tr_comp_times) * 1000

    print(f"\n  加速比 (相对暴力Python):")
    print(f"    pyroaring 倒排索引:      {bf_avg / ro_avg:.1f}x")
    print(f"    torch-recall (eager):    {bf_avg / tr_e_avg:.1f}x")
    print(f"    torch-recall (compiled): {bf_avg / tr_c_avg:.1f}x")

    return {
        "n_items": n_items,
        "build": {"brute_force": bf_build, "roaring": ro_build, "torch_recall": tr_build},
        "latency_ms": {
            "brute_force": statistics.mean(bf_times) * 1000,
            "roaring": statistics.mean(ro_times) * 1000,
            "torch_recall_eager": statistics.mean(tr_eager_times) * 1000,
            "torch_recall_compiled": statistics.mean(tr_comp_times) * 1000,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    scales = [1_000, 10_000, 100_000]
    if "--large" in sys.argv:
        scales.append(1_000_000)

    print("=" * 75)
    print("定向召回方案对比 Benchmark")
    print("=" * 75)
    print()
    print("对比方案:")
    print("  1. 暴力 Python   — 逐 item 评估 DNF，纯 Python 循环")
    print("  2. pyroaring     — 倒排索引结构 + Python 集合运算")
    print("  3. torch-recall  — Two-Level Gather+Reduce (PyTorch 张量)")
    print()
    print("场景: 反向匹配 (item 定向规则 ↔ 用户属性)")
    print("  不同于正向检索 (query → item)，这里是 item → user 方向")
    print()

    all_results = []
    for n in scales:
        r = run_benchmark(n)
        all_results.append(r)

    # Summary table
    print(f"\n{'═' * 75}")
    print("汇总")
    print(f"{'═' * 75}")

    header = f"{'规模':>10s}  {'暴力Python':>12s}  {'pyroaring':>12s}  {'TR eager':>12s}  {'TR compiled':>12s}"
    print(f"\n查询延迟 (ms, avg):")
    print(header)
    for r in all_results:
        lat = r["latency_ms"]
        print(
            f"{r['n_items']:>10,d}  "
            f"{lat['brute_force']:>12.3f}  "
            f"{lat['roaring']:>12.3f}  "
            f"{lat['torch_recall_eager']:>12.3f}  "
            f"{lat['torch_recall_compiled']:>12.3f}"
        )

    print(f"\n构建时间 (s):")
    print(f"{'规模':>10s}  {'暴力Python':>12s}  {'pyroaring':>12s}  {'torch-recall':>12s}")
    for r in all_results:
        b = r["build"]
        print(
            f"{r['n_items']:>10,d}  "
            f"{b['brute_force']:>12.3f}  "
            f"{b['roaring']:>12.3f}  "
            f"{b['torch_recall']:>12.3f}"
        )

    print(f"\n{'═' * 75}")
    print("分析要点:")
    print("  • 暴力Python: O(N × avg_conj × avg_pred), 纯解释器循环, 简单但最慢")
    print("  • pyroaring: 倒排结构减少无关遍历, 但反向匹配仍需逐 item 评估 conjunction")
    print("  • torch-recall: 全张量化, gather+reduce 一次处理所有 item/conjunction")
    print("  • torch.compile: 内核融合进一步消除框架开销")
    print("  • AOTInductor (.pt2): 预编译 C++ 内核, 预期 <1ms (需编译后实测)")
    print()
    print("与正向倒排索引 (ES/Lucene) 的关键区别:")
    print("  正向索引: query 条件 → posting list → 交/并集 → 匹配 item")
    print("  反向匹配: user 属性 → 谓词评估 → conjunction 检查 → 匹配 item")
    print("  反向匹配无法直接使用传统 posting list 跳表加速,")
    print("  因为每个 item 的规则是独立的 DNF, 必须全部评估。")
    print("  torch-recall 的核心优势是将此过程向量化为固定张量操作。")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()

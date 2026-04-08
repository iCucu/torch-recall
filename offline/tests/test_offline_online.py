"""
Full offline → online integration test.

Workflow:
  1. (Offline / Python)  Build index from sample items
  2. (Offline / Python)  Export model to .pt2 + save metadata JSON
  3. (Offline / Python)  Query via Python model for reference answers
  4. (Online  / C++)     Invoke torch_recall_cli binary with the same queries
  5.                     Assert C++ results match Python results exactly
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from torch_recall.builder import IndexBuilder
from torch_recall.exporter import export_model
from torch_recall.query import encode_query
from torch_recall.schema import Schema

CPP_CLI = Path(__file__).resolve().parents[2] / "online" / "build" / "torch_recall_cli"


def _decode_bitmap(result: torch.Tensor, num_items: int) -> list[int]:
    ids = []
    for w in range(result.shape[0]):
        word = result[w].item()
        if word == 0:
            continue
        base = w * 64
        while word:
            bit = (word & -word).bit_length() - 1
            idx = base + bit
            if idx < num_items:
                ids.append(idx)
            word &= word - 1
    return sorted(ids)


def _run_cpp_query(model_path: str, meta_path: str, query: str) -> list[int]:
    """Run the C++ CLI and parse the matching item IDs from stdout."""
    result = subprocess.run(
        [str(CPP_CLI), model_path, meta_path, query],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ CLI failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    stdout = result.stdout

    count_match = re.search(r"Matching items:\s*(\d+)", stdout)
    assert count_match, f"Cannot parse matching count from:\n{stdout}"
    count = int(count_match.group(1))
    if count == 0:
        return []

    ids_match = re.search(r"IDs:([\d\s]+)", stdout)
    first_ids_match = re.search(r"First 20 IDs:([\d\s.]+)", stdout)
    if ids_match:
        return sorted(int(x) for x in ids_match.group(1).split())
    elif first_ids_match:
        nums = [x for x in first_ids_match.group(1).split() if x != "..."]
        return sorted(int(x) for x in nums)
    else:
        raise RuntimeError(f"Cannot parse IDs from:\n{stdout}")


ITEMS = [
    {"city": "北京", "gender": "男", "price": 10.0, "title": "游戏 攻略 新手"},
    {"city": "上海", "gender": "女", "price": 20.0, "title": "美食 推荐 火锅"},
    {"city": "北京", "gender": "女", "price": 30.0, "title": "游戏 推荐 策略"},
    {"city": "上海", "gender": "男", "price": 40.0, "title": "美食 攻略 甜点"},
    {"city": "广州", "gender": "男", "price": 15.0, "title": "旅行 攻略 广州"},
    {"city": "北京", "gender": "男", "price": 50.0, "title": "科技 新闻 AI"},
    {"city": "上海", "gender": "女", "price": 5.0,  "title": "游戏 攻略 策略"},
    {"city": "广州", "gender": "女", "price": 25.0, "title": "美食 推荐 早茶"},
]

SCHEMA = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["price"],
    text_fields=["title"],
)

QUERIES = {
    # discrete equality
    'city == "北京"':                           [0, 2, 5],
    # discrete AND
    'city == "北京" AND gender == "男"':         [0, 5],
    # numeric range
    "price < 20.0":                              [0, 4, 6],
    # numeric + discrete
    'city == "上海" AND price <= 20.0':          [1, 6],
    # OR
    'city == "北京" OR city == "广州"':           [0, 2, 4, 5, 7],
    # complex: (OR) AND numeric
    '(city == "北京" OR city == "上海") AND price < 25.0': [0, 1, 6],
    # NOT
    'NOT gender == "男"':                        [1, 2, 6, 7],
    # text contains
    'title contains "游戏"':                     [0, 2, 6],
    # text + discrete (item 5 title is "科技 新闻 AI", no "攻略")
    'title contains "攻略" AND city == "北京"':  [0],
    # complex: text OR discrete, intersected with numeric (item 7: 美食+广州, price=25)
    '(title contains "美食" OR city == "广州") AND price < 30.0': [1, 4, 7],
}


@pytest.fixture(scope="module")
def exported_artifacts():
    """Build, export, and return (tmpdir, model, meta) once for the module."""
    builder = IndexBuilder(SCHEMA)
    model, meta = builder.build(ITEMS)
    model.eval()

    tmpdir = tempfile.mkdtemp(prefix="torch_recall_e2e_")
    meta_path = os.path.join(tmpdir, "index_meta.json")
    pt2_path = os.path.join(tmpdir, "model.pt2")

    builder.save_meta(meta, meta_path)
    export_model(model, meta, pt2_path)

    yield tmpdir, model, meta, meta_path, pt2_path

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("query,expected", list(QUERIES.items()), ids=list(QUERIES.keys()))
def test_python_query(exported_artifacts, query, expected):
    """Verify the Python model returns the expected item IDs."""
    _, model, meta, _, _ = exported_artifacts
    tensors = encode_query(query, meta)
    with torch.no_grad():
        result = model(**tensors)
    assert _decode_bitmap(result, len(ITEMS)) == expected


@pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ CLI not built")
@pytest.mark.parametrize("query,expected", list(QUERIES.items()), ids=list(QUERIES.keys()))
def test_cpp_query(exported_artifacts, query, expected):
    """Invoke the C++ CLI and verify results match expected item IDs."""
    tmpdir, _, _, meta_path, pt2_path = exported_artifacts
    cpp_ids = _run_cpp_query(pt2_path, meta_path, query)
    assert cpp_ids == expected


@pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ CLI not built")
@pytest.mark.parametrize("query", list(QUERIES.keys()), ids=list(QUERIES.keys()))
def test_python_cpp_agreement(exported_artifacts, query):
    """Cross-check: Python and C++ must return identical results."""
    tmpdir, model, meta, meta_path, pt2_path = exported_artifacts

    tensors = encode_query(query, meta)
    with torch.no_grad():
        result = model(**tensors)
    py_ids = _decode_bitmap(result, len(ITEMS))

    cpp_ids = _run_cpp_query(pt2_path, meta_path, query)

    assert py_ids == cpp_ids, (
        f"Mismatch for query: {query}\n  Python: {py_ids}\n  C++: {cpp_ids}"
    )

import tempfile
from pathlib import Path

import torch
from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder
from torch_recall.query import encode_query
from torch_recall.exporter import export_model


def _decode_bitmap(result, num_items):
    bits = []
    for word_idx in range(result.shape[0]):
        val = result[word_idx].item()
        for bit in range(64):
            if val & (1 << bit):
                bits.append(word_idx * 64 + bit)
    return [b for b in bits if b < num_items]


def test_e2e_query():
    schema = Schema(
        discrete_fields=["city", "gender"],
        numeric_fields=["price"],
        text_fields=["title"],
    )
    items = [
        {"city": "北京", "gender": "男", "price": 10.0, "title": "游戏 攻略"},
        {"city": "上海", "gender": "女", "price": 20.0, "title": "美食 推荐"},
        {"city": "北京", "gender": "女", "price": 30.0, "title": "游戏 推荐"},
        {"city": "上海", "gender": "男", "price": 40.0, "title": "美食 攻略"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()

    tensors = encode_query('city == "北京"', meta)
    with torch.no_grad():
        result = model(**tensors)

    assert sorted(_decode_bitmap(result, len(items))) == [0, 2]


def test_e2e_complex_query():
    schema = Schema(
        discrete_fields=["city", "gender"],
        numeric_fields=["price"],
    )
    items = [
        {"city": "北京", "gender": "男", "price": 10.0},
        {"city": "上海", "gender": "女", "price": 20.0},
        {"city": "北京", "gender": "女", "price": 30.0},
        {"city": "上海", "gender": "男", "price": 40.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()

    tensors = encode_query(
        '(city == "北京" OR city == "上海") AND price < 25.0', meta
    )
    with torch.no_grad():
        result = model(**tensors)

    assert sorted(_decode_bitmap(result, len(items))) == [0, 1]


def test_e2e_text_contains():
    schema = Schema(text_fields=["title"])
    items = [
        {"title": "游戏 攻略"},
        {"title": "美食 推荐"},
        {"title": "游戏 推荐"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)
    model.eval()

    tensors = encode_query('title contains "游戏"', meta)
    with torch.no_grad():
        result = model(**tensors)

    assert sorted(_decode_bitmap(result, len(items))) == [0, 2]


def test_export_and_reload():
    schema = Schema(discrete_fields=["city"], numeric_fields=["price"])
    items = [
        {"city": "北京", "price": 10.0},
        {"city": "上海", "price": 20.0},
        {"city": "北京", "price": 30.0},
        {"city": "上海", "price": 40.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "index_meta.json"
        builder.save_meta(meta, meta_path)

        pt2_path = Path(tmpdir) / "model.pt2"
        export_model(model, meta, str(pt2_path))

        assert pt2_path.exists()

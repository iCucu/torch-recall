import torch
from torch_recall.schema import MAX_BP
from torch_recall.query import parse_expr, to_dnf, encode_query, And, Or, Not, Predicate


def test_parse_simple_eq():
    expr = parse_expr('city == "北京"')
    assert isinstance(expr, Predicate)
    assert expr.field == "city"
    assert expr.op == "=="
    assert expr.value == "北京"


def test_parse_and():
    expr = parse_expr('city == "北京" AND gender == "男"')
    assert isinstance(expr, And)
    assert len(expr.children) == 2


def test_parse_or():
    expr = parse_expr('city == "北京" OR city == "上海"')
    assert isinstance(expr, Or)


def test_parse_not():
    expr = parse_expr('NOT category == "体育"')
    assert isinstance(expr, Not)


def test_parse_nested():
    expr = parse_expr('(city == "北京" OR city == "上海") AND gender == "男"')
    assert isinstance(expr, And)


def test_parse_numeric():
    expr = parse_expr("price < 100.0")
    assert isinstance(expr, Predicate)
    assert expr.op == "<"
    assert expr.value == 100.0


def test_parse_contains():
    expr = parse_expr('title contains "游戏"')
    assert isinstance(expr, Predicate)
    assert expr.op == "contains"


def test_dnf_simple_and():
    expr = parse_expr('city == "北京" AND gender == "男"')
    dnf = to_dnf(expr)
    assert len(dnf) == 1
    assert len(dnf[0]) == 2


def test_dnf_or_distributes():
    expr = parse_expr('(city == "北京" OR city == "上海") AND gender == "男"')
    dnf = to_dnf(expr)
    assert len(dnf) == 2


def test_dnf_not_pushdown():
    expr = parse_expr('NOT (city == "北京" AND gender == "男")')
    dnf = to_dnf(expr)
    assert len(dnf) == 2


def test_encode_query():
    meta = {
        "schema": {"discrete": ["city", "gender"], "numeric": ["price"], "text": ["title"]},
        "discrete_dicts": {
            "city": {"北京": 0, "上海": 1},
            "gender": {"男": 0, "女": 1},
        },
        "text_dicts": {"title": {"游戏": 0, "美食": 1}},
        "bitmap_lookup": {
            "d:city": {"0": 0, "1": 1},
            "d:gender": {"0": 2, "1": 3},
            "t:title": {"0": 4, "1": 5},
        },
        "num_items": 100,
        "bitmap_len": 2,
        "num_bitmaps": 6,
    }
    tensors = encode_query('city == "北京" AND price < 50.0', meta)
    assert tensors["bitmap_indices"].shape[0] == MAX_BP
    assert tensors["bitmap_valid"][0].item() is True
    assert tensors["numeric_valid"][0].item() is True
    assert tensors["conj_valid"][0].item() is True

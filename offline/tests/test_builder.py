from torch_recall.schema import Schema
from torch_recall.builder import IndexBuilder


def test_build_discrete_only():
    schema = Schema(discrete_fields=["city", "gender"])
    items = [
        {"city": "北京", "gender": "男"},
        {"city": "上海", "gender": "女"},
        {"city": "北京", "gender": "女"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert model.bitmaps.shape[0] > 0
    assert model.num_items == 3
    assert "city" in meta["discrete_dicts"]
    assert "北京" in meta["discrete_dicts"]["city"]


def test_build_with_numeric():
    schema = Schema(discrete_fields=["city"], numeric_fields=["price"])
    items = [
        {"city": "北京", "price": 10.0},
        {"city": "上海", "price": 20.0},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert model.numeric_data.shape == (1, 2)
    assert model.numeric_data[0, 0].item() == 10.0


def test_build_with_text():
    schema = Schema(text_fields=["title"])
    items = [
        {"title": "游戏 攻略"},
        {"title": "美食 推荐"},
        {"title": "游戏 推荐"},
    ]
    builder = IndexBuilder(schema)
    model, meta = builder.build(items)

    assert "title" in meta["text_dicts"]
    term_dict = meta["text_dicts"]["title"]
    assert "游戏" in term_dict
    assert "推荐" in term_dict

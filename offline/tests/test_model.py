import pytest
import torch
from torch_recall.schema import Schema
from torch_recall.model import InvertedIndexModel


def test_schema_validation_ok():
    s = Schema(discrete_fields=["city"], numeric_fields=["price"], text_fields=["title"])
    s.validate()


def test_schema_empty_raises():
    s = Schema()
    with pytest.raises(ValueError, match="at least one field"):
        s.validate()


def test_schema_duplicate_raises():
    s = Schema(discrete_fields=["city"], numeric_fields=["city"])
    with pytest.raises(ValueError, match="Duplicate"):
        s.validate()


def test_field_index():
    s = Schema(discrete_fields=["city", "gender"], numeric_fields=["price"])
    assert s.discrete_field_index == {"city": 0, "gender": 1}
    assert s.numeric_field_index == {"price": 0}


def _make_small_model():
    N = 4
    L = 1
    bitmaps = torch.zeros(3, L, dtype=torch.int64)
    bitmaps[0, 0] = 5   # items {0, 2}
    bitmaps[1, 0] = 10  # items {1, 3}
    bitmaps[2, 0] = 7   # items {0, 1, 2}

    numeric_data = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    item_ids = torch.arange(4, dtype=torch.int64)

    return InvertedIndexModel(
        bitmaps=bitmaps,
        numeric_data=numeric_data,
        item_ids=item_ids,
        num_items=N,
    )


def _make_query_tensors(**overrides):
    tensors = {
        "bitmap_indices": torch.zeros(32, dtype=torch.int64),
        "bitmap_valid": torch.zeros(32, dtype=torch.bool),
        "numeric_fields": torch.zeros(16, dtype=torch.int64),
        "numeric_ops": torch.zeros(16, dtype=torch.int64),
        "numeric_values": torch.zeros(16, dtype=torch.float32),
        "numeric_valid": torch.zeros(16, dtype=torch.bool),
        "negation_mask": torch.zeros(48, dtype=torch.bool),
        "conj_matrix": torch.zeros(16, 48, dtype=torch.bool),
        "conj_valid": torch.zeros(16, dtype=torch.bool),
    }
    for k, v in overrides.items():
        tensors[k] = v
    return tensors


def test_single_bitmap_predicate():
    model = _make_small_model()
    t = _make_query_tensors()
    t["bitmap_indices"][0] = 0
    t["bitmap_valid"][0] = True
    t["conj_matrix"][0, 0] = True
    t["conj_valid"][0] = True

    result = model(**t)
    assert result[0].item() == 5


def test_and_two_bitmaps():
    model = _make_small_model()
    t = _make_query_tensors()
    t["bitmap_indices"][0] = 0
    t["bitmap_indices"][1] = 2
    t["bitmap_valid"][0] = True
    t["bitmap_valid"][1] = True
    t["conj_matrix"][0, 0] = True
    t["conj_matrix"][0, 1] = True
    t["conj_valid"][0] = True

    result = model(**t)
    assert result[0].item() == 5


def test_or_two_bitmaps():
    model = _make_small_model()
    t = _make_query_tensors()
    t["bitmap_indices"][0] = 0
    t["bitmap_indices"][1] = 1
    t["bitmap_valid"][0] = True
    t["bitmap_valid"][1] = True
    t["conj_matrix"][0, 0] = True
    t["conj_matrix"][1, 1] = True
    t["conj_valid"][0] = True
    t["conj_valid"][1] = True

    result = model(**t)
    assert result[0].item() == 15


def test_not_bitmap():
    model = _make_small_model()
    t = _make_query_tensors()
    t["bitmap_indices"][0] = 0
    t["bitmap_valid"][0] = True
    t["negation_mask"][0] = True
    t["conj_matrix"][0, 0] = True
    t["conj_valid"][0] = True

    result = model(**t)
    assert result[0].item() == 10


def test_numeric_less_than():
    model = _make_small_model()
    t = _make_query_tensors()
    t["numeric_fields"][0] = 0
    t["numeric_ops"][0] = 1  # LT
    t["numeric_values"][0] = 25.0
    t["numeric_valid"][0] = True
    t["conj_matrix"][0, 32] = True
    t["conj_valid"][0] = True

    result = model(**t)
    assert result[0].item() == 3

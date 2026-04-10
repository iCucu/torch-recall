"""Tests for the targeting (reverse) recall module."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from torch_recall.schema import Item, Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.targeting.encoder import encode_user
from torch_recall.scheduler.exporter import export_recall_model

CPP_CLI = Path(__file__).resolve().parents[2] / "inference" / "build" / "torch_recall_cli"


# ── Test data ────────────────────────────────────────────────────────────────

SCHEMA = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)

ITEMS = [
    # 0: targets 北京 males
    Item(id="item-0", targeting_rule='city == "北京" AND gender == "男"'),
    # 1: targets anyone in 上海
    Item(id="item-1", targeting_rule='city == "上海"'),
    # 2: targets age > 18
    Item(id="item-2", targeting_rule="age > 18"),
    # 3: targets 北京 OR 上海 with age >= 25
    Item(id="item-3", targeting_rule='(city == "北京" OR city == "上海") AND age >= 25'),
    # 4: targets users whose tags contain 游戏
    Item(id="item-4", targeting_rule='tags contains "游戏"'),
    # 5: targets price < 100 AND tags contain 美食
    Item(id="item-5", targeting_rule='price < 100.0 AND tags contains "美食"'),
    # 6: targets NOT 广州 (i.e. city != 广州)
    Item(id="item-6", targeting_rule='city != "广州"'),
    # 7: targets 广州 females OR anyone with age > 30
    Item(id="item-7", targeting_rule='(city == "广州" AND gender == "女") OR age > 30'),
]

USERS_AND_EXPECTED = [
    (
        {"city": "北京", "gender": "男", "age": 30, "price": 50.0, "tags": "游戏 科技"},
        [0, 2, 3, 4, 6],
    ),
    (
        {"city": "上海", "gender": "女", "age": 20, "price": 80.0, "tags": "美食 旅行"},
        [1, 2, 5, 6],
    ),
    (
        {"city": "广州", "gender": "女", "age": 35, "price": 200.0, "tags": "教育"},
        [2, 7],
    ),
    (
        {"city": "北京", "gender": "女", "age": 16, "price": 30.0, "tags": "游戏 美食"},
        [4, 5, 6],
    ),
    (
        {"city": "深圳", "gender": "男", "age": 40, "price": 150.0, "tags": "科技"},
        [2, 6, 7],
    ),
]


# ── Unit tests: builder ──────────────────────────────────────────────────────

class TestBuilder:
    def test_build_returns_model_and_meta(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)
        assert isinstance(model, torch.nn.Module)
        assert meta["num_items"] == len(ITEMS)
        assert meta["num_preds"] > 0
        assert meta["num_conjs"] > 0

    def test_predicate_registry_has_all_types(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(ITEMS)
        reg = meta["predicate_registry"]
        assert "city" in reg["discrete"]
        assert len(reg["numeric"]) > 0
        assert "tags" in reg["text"]

    def test_model_buffers_shapes(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)
        C = meta["num_conjs"]
        N = meta["num_items"]
        assert model.conj_pred_ids.shape[0] == C
        assert model.item_conj_ids.shape[0] == N

    def test_adaptive_k_fits_actual_data(self):
        schema = Schema(discrete_fields=[f"f{i}" for i in range(12)])
        rule = " AND ".join(f'f{i} == "v"' for i in range(12))
        builder = TargetingBuilder(schema)
        model, meta = builder.build([Item(targeting_rule=rule)])
        assert meta["max_preds_per_conj"] == 12
        assert model.conj_pred_ids.shape[1] == 12

    def test_adaptive_j_fits_actual_data(self):
        """OR of 5 predicates → 5 conjunctions (each single-predicate)."""
        schema = Schema(discrete_fields=["city"])
        rule = " OR ".join(f'city == "c{i}"' for i in range(5))
        builder = TargetingBuilder(schema)
        model, meta = builder.build([Item(targeting_rule=rule)])
        assert meta["max_conj_per_item"] == 5
        assert model.item_conj_ids.shape[1] == 5

    def test_upper_bound_k_enforced(self):
        schema = Schema(discrete_fields=[f"f{i}" for i in range(10)])
        rule = " AND ".join(f'f{i} == "v"' for i in range(10))
        builder = TargetingBuilder(schema, max_preds_per_conj=8)
        with pytest.raises(ValueError, match="max_preds_per_conj"):
            builder.build([Item(targeting_rule=rule)])

    def test_upper_bound_j_enforced(self):
        schema = Schema(discrete_fields=["city"])
        rule = " OR ".join(f'city == "c{i}"' for i in range(20))
        builder = TargetingBuilder(schema, max_conj_per_item=10)
        with pytest.raises(ValueError, match="max_conj_per_item"):
            builder.build([Item(targeting_rule=rule)])

    def test_meta_contains_k_j(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(ITEMS)
        assert "max_preds_per_conj" in meta
        assert "max_conj_per_item" in meta
        assert meta["max_preds_per_conj"] >= 1
        assert meta["max_conj_per_item"] >= 1

    def test_missing_targeting_rule_raises(self):
        builder = TargetingBuilder(SCHEMA)
        with pytest.raises(ValueError, match="targeting_rule is None"):
            builder.build([Item(embedding=[1.0, 2.0])])

    def test_meta_item_ids(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(ITEMS)
        assert meta["item_ids"] is not None
        assert meta["item_ids"][0] == "item-0"


# ── Unit tests: encoder ──────────────────────────────────────────────────────

class TestEncoder:
    @pytest.fixture()
    def meta(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(ITEMS)
        return meta

    def test_encode_returns_correct_size(self, meta):
        user = {"city": "北京", "gender": "男", "age": 25}
        result = encode_user(user, meta)
        assert result.shape == (meta["num_preds"],)
        assert result.dtype == torch.bool

    def test_discrete_match(self, meta):
        user = {"city": "北京"}
        result = encode_user(user, meta)
        pred_id = meta["predicate_registry"]["discrete"]["city"]["北京"]
        assert result[pred_id].item() is True

    def test_discrete_no_match(self, meta):
        user = {"city": "深圳"}
        result = encode_user(user, meta)
        for val, pid in meta["predicate_registry"]["discrete"]["city"].items():
            if val != "深圳":
                assert result[pid].item() is False

    def test_numeric_match(self, meta):
        user = {"age": 25}
        result = encode_user(user, meta)
        for entry in meta["predicate_registry"]["numeric"]:
            if entry["field"] == "age" and entry["op"] == ">" and entry["value"] == 18.0:
                assert result[entry["pred_id"]].item() is True

    def test_text_match(self, meta):
        user = {"tags": "游戏 科技"}
        result = encode_user(user, meta)
        pid = meta["predicate_registry"]["text"]["tags"]["游戏"]
        assert result[pid].item() is True

    def test_missing_field_stays_false(self, meta):
        result = encode_user({}, meta)
        assert result.sum().item() == 0


# ── E2E tests: Python forward ────────────────────────────────────────────────

class TestE2E:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    @pytest.mark.parametrize(
        "user,expected",
        USERS_AND_EXPECTED,
        ids=[u["city"] + "_" + u["gender"] for u, _ in USERS_AND_EXPECTED],
    )
    def test_query(self, model_and_meta, user, expected):
        model, meta = model_and_meta
        result = model.query(user, meta)
        matched = sorted(i for i in range(len(ITEMS)) if result[i].item())
        assert matched == expected

    @pytest.mark.parametrize(
        "user,expected",
        USERS_AND_EXPECTED,
        ids=[u["city"] + "_" + u["gender"] for u, _ in USERS_AND_EXPECTED],
    )
    def test_forward_matches_query(self, model_and_meta, user, expected):
        model, meta = model_and_meta
        pred_satisfied = encode_user(user, meta).unsqueeze(0)  # [1, P]
        dummy_query = torch.zeros(1, 1)
        with torch.no_grad():
            scores = model(pred_satisfied, dummy_query)  # [1, N] float
        matched = sorted(
            i for i in range(len(ITEMS)) if scores[0, i].item() > float("-inf")
        )
        assert matched == expected


# ── Batch tests ──────────────────────────────────────────────────────────────

class TestBatch:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_batch_forward(self, model_and_meta):
        model, meta = model_and_meta
        users = [u for u, _ in USERS_AND_EXPECTED]
        expected_all = [e for _, e in USERS_AND_EXPECTED]
        B = len(users)

        batch_pred = torch.stack(
            [encode_user(u, meta) for u in users]
        )  # [B, P]
        dummy_query = torch.zeros(B, 1)
        with torch.no_grad():
            scores = model(batch_pred, dummy_query)  # [B, N]
        assert scores.shape == (B, len(ITEMS))

        for b in range(B):
            matched = sorted(
                i for i in range(len(ITEMS)) if scores[b, i].item() > float("-inf")
            )
            assert matched == expected_all[b]

    def test_batch_single_equals_unbatched(self, model_and_meta):
        model, meta = model_and_meta
        user = USERS_AND_EXPECTED[0][0]
        pred = encode_user(user, meta)  # [P]
        single_batch = pred.unsqueeze(0)  # [1, P]
        dummy_q = torch.zeros(1, 1)
        with torch.no_grad():
            batch_scores = model(single_batch, dummy_q)  # [1, N]
        assert batch_scores.shape == (1, len(ITEMS))


# ── Export test ───────────────────────────────────────────────────────────────

class TestExport:
    def test_export_and_reload(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)

        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "model.pt2")
            export_recall_model(model, pt2_path)
            assert os.path.exists(pt2_path)


# ── C++ integration test ─────────────────────────────────────────────────────

@pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ CLI not built")
class TestCppIntegration:
    @pytest.fixture(scope="class")
    def exported_artifacts(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(ITEMS)
        model.eval()

        tmpdir = tempfile.mkdtemp(prefix="torch_recall_targeting_")
        meta_path = os.path.join(tmpdir, "targeting_meta.json")
        pt2_path = os.path.join(tmpdir, "model.pt2")

        builder.save_meta(meta, meta_path)
        export_recall_model(model, pt2_path)

        yield tmpdir, model, meta, meta_path, pt2_path

        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.parametrize(
        "user,expected",
        USERS_AND_EXPECTED,
        ids=[u["city"] + "_" + u["gender"] for u, _ in USERS_AND_EXPECTED],
    )
    def test_cpp_targeting(self, exported_artifacts, user, expected):
        tmpdir, model, meta, _, pt2_path = exported_artifacts

        tensors_path = os.path.join(tmpdir, "tensors.pt")
        model.save_user_tensors(user, meta, tensors_path)

        result = subprocess.run(
            [str(CPP_CLI), pt2_path, tensors_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"C++ CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @pytest.mark.parametrize(
        "user,expected",
        USERS_AND_EXPECTED,
        ids=[u["city"] + "_" + u["gender"] for u, _ in USERS_AND_EXPECTED],
    )
    def test_python_cpp_agreement(self, exported_artifacts, user, expected):
        tmpdir, model, meta, _, pt2_path = exported_artifacts

        py_result = model.query(user, meta)
        py_matched = sorted(i for i in range(len(ITEMS)) if py_result[i].item())

        tensors_path = os.path.join(tmpdir, "tensors.pt")
        model.save_user_tensors(user, meta, tensors_path)

        result = subprocess.run(
            [str(CPP_CLI), pt2_path, tensors_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0

        assert py_matched == expected

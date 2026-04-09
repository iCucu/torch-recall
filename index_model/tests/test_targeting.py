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

from torch_recall.schema import Schema
from torch_recall.recall_method.targeting.builder import TargetingBuilder
from torch_recall.recall_method.targeting.encoder import encode_user
from torch_recall.scheduler.exporter import export_recall_model

CPP_CLI = Path(__file__).resolve().parents[2] / "inference_engine" / "build" / "torch_recall_cli"


# ── Test data ────────────────────────────────────────────────────────────────

SCHEMA = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age", "price"],
    text_fields=["tags"],
)

RULES = [
    # 0: targets 北京 males
    'city == "北京" AND gender == "男"',
    # 1: targets anyone in 上海
    'city == "上海"',
    # 2: targets age > 18
    "age > 18",
    # 3: targets 北京 OR 上海 with age >= 25
    '(city == "北京" OR city == "上海") AND age >= 25',
    # 4: targets users whose tags contain 游戏
    'tags contains "游戏"',
    # 5: targets price < 100 AND tags contain 美食
    'price < 100.0 AND tags contains "美食"',
    # 6: targets NOT 广州 (i.e. city != 广州)
    'city != "广州"',
    # 7: targets 广州 females OR anyone with age > 30
    '(city == "广州" AND gender == "女") OR age > 30',
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
        model, meta = builder.build(RULES)
        assert isinstance(model, torch.nn.Module)
        assert meta["num_items"] == len(RULES)
        assert meta["num_preds"] > 0
        assert meta["num_conjs"] > 0

    def test_predicate_registry_has_all_types(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(RULES)
        reg = meta["predicate_registry"]
        assert "city" in reg["discrete"]
        assert len(reg["numeric"]) > 0
        assert "tags" in reg["text"]

    def test_model_buffers_shapes(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(RULES)
        C = meta["num_conjs"]
        N = meta["num_items"]
        assert model.conj_pred_ids.shape[0] == C
        assert model.item_conj_ids.shape[0] == N

    def test_too_many_preds_per_conj_raises(self):
        schema = Schema(discrete_fields=[f"f{i}" for i in range(10)])
        rule = " AND ".join(f'f{i} == "v"' for i in range(10))
        builder = TargetingBuilder(schema)
        with pytest.raises(ValueError, match="max"):
            builder.build([rule])


# ── Unit tests: encoder ──────────────────────────────────────────────────────

class TestEncoder:
    @pytest.fixture()
    def meta(self):
        builder = TargetingBuilder(SCHEMA)
        _, meta = builder.build(RULES)
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
        model, meta = builder.build(RULES)
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
        matched = sorted(i for i in range(len(RULES)) if result[i].item())
        assert matched == expected

    @pytest.mark.parametrize(
        "user,expected",
        USERS_AND_EXPECTED,
        ids=[u["city"] + "_" + u["gender"] for u, _ in USERS_AND_EXPECTED],
    )
    def test_forward_matches_query(self, model_and_meta, user, expected):
        model, meta = model_and_meta
        pred_satisfied = encode_user(user, meta)
        with torch.no_grad():
            result = model(pred_satisfied)
        matched = sorted(i for i in range(len(RULES)) if result[i].item())
        assert matched == expected


# ── Export test ───────────────────────────────────────────────────────────────

class TestExport:
    def test_export_and_reload(self):
        builder = TargetingBuilder(SCHEMA)
        model, meta = builder.build(RULES)

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
        model, meta = builder.build(RULES)
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
        py_matched = sorted(i for i in range(len(RULES)) if py_result[i].item())

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

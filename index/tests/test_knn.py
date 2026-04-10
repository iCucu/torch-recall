"""Tests for the KNN (K-Nearest Neighbor) recall module."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from torch_recall.schema import Item
from torch_recall.recall_method.knn.builder import KNNBuilder
from torch_recall.recall_method.knn.encoder import encode_query
from torch_recall.recall_method.knn.recall import KNNRecall
from torch_recall.scheduler.exporter import export_recall_model

CPP_CLI = (
    Path(__file__).resolve().parents[2]
    / "Inference"
    / "build"
    / "torch_recall_cli"
)


# ── Test data ────────────────────────────────────────────────────────────────

D = 4
K = 3

ITEMS = [
    Item(id="i0", embedding=[1.0, 0.0, 0.0, 0.0]),  # unit x
    Item(id="i1", embedding=[0.0, 1.0, 0.0, 0.0]),  # unit y
    Item(id="i2", embedding=[0.0, 0.0, 1.0, 0.0]),  # unit z
    Item(id="i3", embedding=[0.0, 0.0, 0.0, 1.0]),  # unit w
    Item(id="i4", embedding=[1.0, 1.0, 0.0, 0.0]),  # x+y (not normalised)
]

EMBEDDINGS_RAW = torch.tensor(
    [item.embedding for item in ITEMS], dtype=torch.float32
)

QUERY = torch.tensor([1.0, 0.5, 0.0, 0.0])  # close to item 0 & 4

DUMMY_PRED = torch.zeros(1, 1, dtype=torch.bool)


# ── Unit tests: builder ──────────────────────────────────────────────────────


class TestBuilder:
    @pytest.mark.parametrize("metric", ["inner_product", "cosine", "l2"])
    def test_build_returns_model_and_meta(self, metric):
        builder = KNNBuilder(k=K, metric=metric)
        model, meta = builder.build(ITEMS)
        assert isinstance(model, KNNRecall)
        assert meta["num_items"] == len(ITEMS)
        assert meta["embedding_dim"] == D
        assert meta["k"] == K
        assert meta["metric"] == metric

    def test_meta_item_ids(self):
        builder = KNNBuilder(k=K)
        _, meta = builder.build(ITEMS)
        assert meta["item_ids"] == [item.id for item in ITEMS]

    def test_meta_item_ids_none(self):
        items_no_id = [Item(embedding=item.embedding) for item in ITEMS]
        builder = KNNBuilder(k=K)
        _, meta = builder.build(items_no_id)
        assert meta["item_ids"] is None

    def test_buffer_shapes(self):
        builder = KNNBuilder(k=K, metric="cosine")
        model, _ = builder.build(ITEMS)
        N = len(ITEMS)
        assert model.embeddings.shape == (N, D)
        assert model.embedding_norms.shape == (N,)

    def test_cosine_pre_normalises_embeddings(self):
        builder = KNNBuilder(k=K, metric="cosine")
        model, _ = builder.build(ITEMS)
        norms = model.embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-6)

    def test_l2_pre_computes_norms(self):
        builder = KNNBuilder(k=K, metric="l2")
        model, _ = builder.build(ITEMS)
        expected = (EMBEDDINGS_RAW * EMBEDDINGS_RAW).sum(dim=1)
        assert torch.allclose(model.embedding_norms, expected)

    def test_k_exceeds_n_raises(self):
        builder = KNNBuilder(k=100)
        with pytest.raises(ValueError, match="k=100 exceeds"):
            builder.build(ITEMS)

    def test_missing_embedding_raises(self):
        builder = KNNBuilder(k=1)
        with pytest.raises(ValueError, match="embedding is None"):
            builder.build([Item(targeting_rule='city == "北京"')])

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            KNNBuilder(k=K, metric="hamming")

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            KNNBuilder(k=0)

    def test_save_meta(self):
        builder = KNNBuilder(k=K)
        _, meta = builder.build(ITEMS)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            builder.save_meta(meta, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["num_items"] == meta["num_items"]
            assert loaded["k"] == meta["k"]
        finally:
            os.unlink(path)


# ── Unit tests: encoder ──────────────────────────────────────────────────────


class TestEncoder:
    @pytest.fixture()
    def meta(self):
        builder = KNNBuilder(k=K)
        _, meta = builder.build(ITEMS)
        return meta

    def test_encode_from_list(self, meta):
        q = encode_query([1.0, 0.0, 0.0, 0.0], meta)
        assert q.shape == (1, D)
        assert q.dtype == torch.float32

    def test_encode_from_1d_tensor(self, meta):
        q = encode_query(torch.randn(D), meta)
        assert q.shape == (1, D)

    def test_encode_from_2d_tensor(self, meta):
        q = encode_query(torch.randn(1, D), meta)
        assert q.shape == (1, D)

    def test_dim_mismatch_raises(self, meta):
        with pytest.raises(ValueError, match="embedding_dim"):
            encode_query([1.0, 2.0], meta)


# ── E2E tests: Python forward ────────────────────────────────────────────────


def _brute_force_scores_ip(embeddings, query):
    """Full [1, N] similarity scores — inner product."""
    return query @ embeddings.T


def _brute_force_scores_cosine(embeddings, query):
    """Full [1, N] similarity scores — cosine."""
    e_norm = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q_norm = query / query.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return q_norm @ e_norm.T


def _brute_force_scores_l2(embeddings, query):
    """Full [1, N] similarity scores — negative L2 distance."""
    diff = query.unsqueeze(1) - embeddings.unsqueeze(0)
    dists = (diff * diff).sum(dim=2)
    return -dists


class TestE2EInnerProduct:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = KNNBuilder(k=K, metric="inner_product")
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_forward_matches_brute_force(self, model_and_meta):
        model, _ = model_and_meta
        q = QUERY.unsqueeze(0)
        scores = model(DUMMY_PRED, q)  # [1, N]
        expected = _brute_force_scores_ip(EMBEDDINGS_RAW, q)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_query_api(self, model_and_meta):
        model, meta = model_and_meta
        scores, indices = model.query(QUERY, meta)
        assert scores.shape == (K,)
        assert indices.shape == (K,)


class TestE2ECosine:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = KNNBuilder(k=K, metric="cosine")
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_forward_matches_brute_force(self, model_and_meta):
        model, _ = model_and_meta
        q = QUERY.unsqueeze(0)
        scores = model(DUMMY_PRED, q)  # [1, N]
        expected = _brute_force_scores_cosine(EMBEDDINGS_RAW, q)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_query_api(self, model_and_meta):
        model, meta = model_and_meta
        scores, indices = model.query(QUERY, meta)
        assert scores.shape == (K,)
        assert indices.shape == (K,)


class TestE2EL2:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = KNNBuilder(k=K, metric="l2")
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_forward_matches_brute_force(self, model_and_meta):
        model, _ = model_and_meta
        q = QUERY.unsqueeze(0)
        scores = model(DUMMY_PRED, q)  # [1, N]
        expected = _brute_force_scores_l2(EMBEDDINGS_RAW, q)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_query_api_returns_distances(self, model_and_meta):
        model, meta = model_and_meta
        scores, indices = model.query(QUERY, meta)
        assert scores.shape == (K,)
        assert (scores >= 0).all(), "L2 distances should be non-negative"

    def test_nearest_is_closest(self, model_and_meta):
        model, meta = model_and_meta
        scores, indices = model.query(QUERY, meta)
        assert scores[0] <= scores[-1], "First result should have smallest distance"


# ── Batch tests ──────────────────────────────────────────────────────────────


class TestBatch:
    @pytest.fixture(scope="class")
    def model(self):
        builder = KNNBuilder(k=K, metric="inner_product")
        model, _ = builder.build(ITEMS)
        model.eval()
        return model

    def test_batch_forward(self, model):
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        batch_q = torch.cat([q1, q2], dim=0)  # [2, D]
        batch_pred = torch.zeros(2, 1, dtype=torch.bool)
        scores = model(batch_pred, batch_q)  # [2, N]
        assert scores.shape == (2, len(ITEMS))
        expected_0 = _brute_force_scores_ip(EMBEDDINGS_RAW, q1)  # [1, N]
        expected_1 = _brute_force_scores_ip(EMBEDDINGS_RAW, q2)  # [1, N]
        assert torch.allclose(scores[0:1], expected_0, atol=1e-5)
        assert torch.allclose(scores[1:2], expected_1, atol=1e-5)

    def test_batch_single_equals_single(self, model):
        q = QUERY.unsqueeze(0)  # [1, D]
        single = model(DUMMY_PRED, q)
        assert single.shape == (1, len(ITEMS))


# ── Export test ───────────────────────────────────────────────────────────────


class TestExport:
    @pytest.mark.parametrize("metric", ["inner_product", "cosine", "l2"])
    def test_export_and_reload(self, metric):
        builder = KNNBuilder(k=K, metric=metric)
        model, _ = builder.build(ITEMS)

        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "model.pt2")
            export_recall_model(model, pt2_path)
            assert os.path.exists(pt2_path)


# ── C++ integration test ─────────────────────────────────────────────────────


@pytest.mark.skipif(not CPP_CLI.exists(), reason="C++ CLI not built")
class TestCppIntegration:
    @pytest.fixture(scope="class")
    def exported_artifacts(self):
        builder = KNNBuilder(k=K, metric="cosine")
        model, meta = builder.build(ITEMS)
        model.eval()

        tmpdir = tempfile.mkdtemp(prefix="torch_recall_knn_")
        meta_path = os.path.join(tmpdir, "knn_meta.json")
        pt2_path = os.path.join(tmpdir, "model.pt2")

        builder.save_meta(meta, meta_path)
        export_recall_model(model, pt2_path)

        yield tmpdir, model, meta, meta_path, pt2_path

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cpp_knn(self, exported_artifacts):
        tmpdir, model, meta, _, pt2_path = exported_artifacts

        tensors_path = os.path.join(tmpdir, "tensors.pt")
        model.save_query_tensors(QUERY, meta, tensors_path)

        result = subprocess.run(
            [str(CPP_CLI), pt2_path, tensors_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"C++ CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

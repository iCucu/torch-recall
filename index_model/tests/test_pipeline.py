"""Tests for the declarative recall pipeline (scheduler)."""

import os
import tempfile

import pytest
import torch

from torch_recall.schema import Item, Schema
from torch_recall.scheduler.spec import (
    RecallSpec,
    Targeting,
    KNN,
    And,
    Or,
)
from torch_recall.scheduler.pipeline import (
    AndModule,
    OrModule,
    RecallPipeline,
)
from torch_recall.scheduler.pipeline_builder import PipelineBuilder, _collect_leaves
from torch_recall.scheduler.encoder import encode_pipeline_inputs
from torch_recall.scheduler.exporter import export_recall_model


# ── shared test data ─────────────────────────────────────────────────────────

SCHEMA = Schema(
    discrete_fields=["city", "gender"],
    numeric_fields=["age"],
    text_fields=[],
)

D = 4
K = 3

ITEMS = [
    Item(
        id="item-0",
        targeting_rule='city == "北京" AND gender == "男"',
        embedding=[1.0, 0.0, 0.0, 0.0],
    ),
    Item(
        id="item-1",
        targeting_rule='city == "上海"',
        embedding=[0.0, 1.0, 0.0, 0.0],
    ),
    Item(
        id="item-2",
        targeting_rule="age > 18",
        embedding=[0.0, 0.0, 1.0, 0.0],
    ),
    Item(
        id="item-3",
        targeting_rule='(city == "北京" OR city == "上海") AND age >= 25',
        embedding=[0.0, 0.0, 0.0, 1.0],
    ),
    Item(
        id="item-4",
        targeting_rule='city == "北京"',
        embedding=[1.0, 1.0, 0.0, 0.0],
    ),
]


# ── TestSpec ─────────────────────────────────────────────────────────────────

class TestSpec:
    def test_targeting_leaf(self):
        t = Targeting(SCHEMA)
        assert isinstance(t, RecallSpec)
        assert t.schema is SCHEMA

    def test_knn_leaf(self):
        k = KNN(metric="l2", weight=0.5)
        assert isinstance(k, RecallSpec)
        assert k.metric == "l2"
        assert k.weight == 0.5

    def test_and_requires_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            And(Targeting(SCHEMA))

    def test_or_requires_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            Or(KNN())

    def test_collect_leaves(self):
        spec = And(Targeting(SCHEMA), Or(KNN("cosine"), KNN("l2")))
        targeting_leaves: list = []
        knn_leaves: list = []
        _collect_leaves(spec, targeting_leaves, knn_leaves)
        assert len(targeting_leaves) == 1
        assert len(knn_leaves) == 2
        assert knn_leaves[0].metric == "cosine"
        assert knn_leaves[1].metric == "l2"

    def test_collect_leaves_order(self):
        spec = Or(
            And(Targeting(SCHEMA), KNN("cosine")),
            KNN("l2"),
        )
        targeting_leaves: list = []
        knn_leaves: list = []
        _collect_leaves(spec, targeting_leaves, knn_leaves)
        assert len(knn_leaves) == 2
        assert knn_leaves[0].metric == "cosine"
        assert knn_leaves[1].metric == "l2"


# ── TestTargetingOnly ────────────────────────────────────────────────────────

class TestTargetingOnly:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = Targeting(SCHEMA)
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_structure(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert meta["k"] == K
        assert meta["num_items"] == len(ITEMS)
        assert meta["num_preds"] > 0
        assert "targeting" in meta
        assert "knn_leaves" not in meta

    def test_matching_user_gets_finite_scores(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        pred_satisfied, query = encode_pipeline_inputs(
            {"city": "北京", "gender": "男", "age": 30},
            None,
            meta,
        )
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)

        assert top_vals.shape == (1, K)
        finite_mask = top_vals[0] > float("-inf")
        assert finite_mask.any(), "At least one item should match"

    def test_no_match_gets_neg_inf(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        pred_satisfied, query = encode_pipeline_inputs(
            {"city": "广州", "gender": "女", "age": 10},
            None,
            meta,
        )
        with torch.no_grad():
            top_vals, _ = pipeline(pred_satisfied, query)
        assert (top_vals == 0.0).sum().item() == 0, (
            "No items should match this user"
        )


# ── TestKNNOnly ──────────────────────────────────────────────────────────────

class TestKNNOnly:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = KNN(metric="cosine")
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_structure(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert meta["k"] == K
        assert meta["num_preds"] == 1
        assert "knn_leaves" in meta
        assert len(meta["knn_leaves"]) == 1
        assert meta["knn_leaves"][0]["metric"] == "cosine"
        assert "targeting" not in meta

    def test_returns_top_k(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        q_vec = [1.0, 0.5, 0.0, 0.0]
        pred_satisfied, query = encode_pipeline_inputs(None, [q_vec], meta)
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)
        assert top_vals.shape == (1, K)
        assert top_idx.shape == (1, K)
        assert (top_vals[0] > float("-inf")).all()

    def test_scores_are_descending(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        q_vec = [1.0, 0.5, 0.0, 0.0]
        pred_satisfied, query = encode_pipeline_inputs(None, [q_vec], meta)
        with torch.no_grad():
            top_vals, _ = pipeline(pred_satisfied, query)
        vals = top_vals[0]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]


# ── TestAndTargetingKNN ──────────────────────────────────────────────────────

class TestAndTargetingKNN:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = And(Targeting(SCHEMA), KNN(metric="inner_product"))
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_has_both(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert "targeting" in meta
        assert "knn_leaves" in meta

    def test_filtered_items_excluded(self, pipeline_and_meta):
        """Items not matching targeting should get -inf and never appear in topk."""
        pipeline, meta = pipeline_and_meta
        user = {"city": "上海", "age": 10}
        q_vec = [0.0, 1.0, 0.0, 0.0]
        pred_satisfied, query = encode_pipeline_inputs(user, [q_vec], meta)
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)
        finite_indices = top_idx[0][top_vals[0] > float("-inf")].tolist()
        assert 1 in finite_indices, "item-1 (上海) should match"
        assert 0 not in finite_indices, "item-0 (北京+男) should not match"

    def test_knn_ranks_within_matched(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        user = {"city": "北京", "age": 30}
        q_vec = [1.0, 0.0, 0.0, 0.0]
        pred_satisfied, query = encode_pipeline_inputs(user, [q_vec], meta)
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)
        finite_mask = top_vals[0] > float("-inf")
        finite_vals = top_vals[0][finite_mask]
        for i in range(len(finite_vals) - 1):
            assert finite_vals[i] >= finite_vals[i + 1]


# ── TestOrMultiKNN ───────────────────────────────────────────────────────────

class TestOrMultiKNN:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = Or(KNN(metric="cosine"), KNN(metric="inner_product"))
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_has_two_knn_leaves(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert len(meta["knn_leaves"]) == 2
        assert meta["knn_leaves"][0]["metric"] == "cosine"
        assert meta["knn_leaves"][1]["metric"] == "inner_product"

    def test_query_dim_is_sum(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert meta["total_query_dim"] == D + D

    def test_returns_valid_results(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        q1 = [1.0, 0.5, 0.0, 0.0]
        q2 = [0.0, 0.0, 1.0, 0.5]
        pred_satisfied, query = encode_pipeline_inputs(None, [q1, q2], meta)
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)
        assert top_vals.shape == (1, K)
        assert (top_vals[0] > float("-inf")).all()


# ── TestComplex ──────────────────────────────────────────────────────────────

class TestComplex:
    """Or(And(Targeting, KNN("cosine")), KNN("inner_product")) — nested tree."""

    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = Or(
            And(Targeting(SCHEMA), KNN(metric="cosine")),
            KNN(metric="inner_product"),
        )
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_structure(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert "targeting" in meta
        assert len(meta["knn_leaves"]) == 2

    def test_or_rescues_filtered_items(self, pipeline_and_meta):
        """Even if targeting filters an item, the KNN branch can still include it."""
        pipeline, meta = pipeline_and_meta
        user = {"city": "上海", "age": 10}
        q_vec = [1.0, 0.0, 0.0, 0.0]
        pred_satisfied, query = encode_pipeline_inputs(
            user, [q_vec, q_vec], meta
        )
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred_satisfied, query)
        finite_mask = top_vals[0] > float("-inf")
        finite_indices = top_idx[0][finite_mask].tolist()
        assert 0 in finite_indices, (
            "item-0 should appear via the standalone KNN branch"
        )


# ── TestWeights ──────────────────────────────────────────────────────────────

class TestWeights:
    def test_weight_affects_score(self):
        spec_w1 = KNN(metric="cosine", weight=1.0)
        spec_w2 = KNN(metric="cosine", weight=2.0)

        builder1 = PipelineBuilder(spec_w1, k=K)
        p1, m1 = builder1.build(ITEMS)
        p1.eval()

        builder2 = PipelineBuilder(spec_w2, k=K)
        p2, m2 = builder2.build(ITEMS)
        p2.eval()

        q_vec = [1.0, 0.5, 0.0, 0.0]
        pred1, q1 = encode_pipeline_inputs(None, [q_vec], m1)
        pred2, q2 = encode_pipeline_inputs(None, [q_vec], m2)

        with torch.no_grad():
            v1, _ = p1(pred1, q1)
            v2, _ = p2(pred2, q2)

        assert torch.allclose(v2, v1 * 2, atol=1e-5)


# ── TestBatch ────────────────────────────────────────────────────────────────

class TestBatch:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        spec = And(Targeting(SCHEMA), KNN(metric="inner_product"))
        builder = PipelineBuilder(spec, k=K)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_batch_pipeline(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta

        user1 = {"city": "北京", "gender": "男", "age": 30}
        user2 = {"city": "上海", "age": 10}
        q_vec = [1.0, 0.0, 0.0, 0.0]

        p1, q1 = encode_pipeline_inputs(user1, [q_vec], meta)
        p2, q2 = encode_pipeline_inputs(user2, [q_vec], meta)
        batch_pred = torch.cat([p1, p2], dim=0)   # [2, P]
        batch_query = torch.cat([q1, q2], dim=0)   # [2, D]

        with torch.no_grad():
            top_vals, top_idx = pipeline(batch_pred, batch_query)
        assert top_vals.shape == (2, K)
        assert top_idx.shape == (2, K)

    def test_batch_equals_individual(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta

        user1 = {"city": "北京", "gender": "男", "age": 30}
        user2 = {"city": "上海", "age": 10}
        q_vec = [0.0, 1.0, 0.0, 0.0]

        p1, q1 = encode_pipeline_inputs(user1, [q_vec], meta)
        p2, q2 = encode_pipeline_inputs(user2, [q_vec], meta)

        with torch.no_grad():
            v1, i1 = pipeline(p1, q1)
            v2, i2 = pipeline(p2, q2)

        batch_pred = torch.cat([p1, p2], dim=0)
        batch_query = torch.cat([q1, q2], dim=0)
        with torch.no_grad():
            bv, bi = pipeline(batch_pred, batch_query)

        assert torch.allclose(bv[0:1], v1, atol=1e-6)
        assert torch.allclose(bv[1:2], v2, atol=1e-6)
        assert torch.equal(bi[0:1], i1)
        assert torch.equal(bi[1:2], i2)


# ── TestExport ───────────────────────────────────────────────────────────────

class TestExport:
    @pytest.mark.parametrize(
        "spec",
        [
            Targeting(SCHEMA),
            KNN(metric="cosine"),
            And(Targeting(SCHEMA), KNN(metric="inner_product")),
            Or(KNN(metric="cosine"), KNN(metric="l2")),
            Or(And(Targeting(SCHEMA), KNN("cosine")), KNN("l2")),
        ],
        ids=[
            "targeting_only",
            "knn_only",
            "and_targeting_knn",
            "or_multi_knn",
            "complex_nested",
        ],
    )
    def test_export_pipeline(self, spec):
        builder = PipelineBuilder(spec, k=K)
        pipeline, _ = builder.build(ITEMS)
        with tempfile.TemporaryDirectory() as tmpdir:
            pt2_path = os.path.join(tmpdir, "pipeline.pt2")
            export_recall_model(pipeline, pt2_path)
            assert os.path.exists(pt2_path)


# ── TestEncoder ──────────────────────────────────────────────────────────────

class TestEncoder:
    def test_targeting_only_shapes(self):
        spec = Targeting(SCHEMA)
        builder = PipelineBuilder(spec, k=K)
        _, meta = builder.build(ITEMS)
        pred, query = encode_pipeline_inputs(
            {"city": "北京"}, None, meta
        )
        assert pred.dtype == torch.bool
        assert pred.shape == (1, meta["num_preds"])
        assert query.shape == (1, meta["total_query_dim"])

    def test_knn_only_shapes(self):
        spec = KNN(metric="cosine")
        builder = PipelineBuilder(spec, k=K)
        _, meta = builder.build(ITEMS)
        pred, query = encode_pipeline_inputs(None, [[1.0, 0.0, 0.0, 0.0]], meta)
        assert pred.shape == (1, 1)
        assert query.shape == (1, D)

    def test_both_shapes(self):
        spec = And(Targeting(SCHEMA), KNN(metric="cosine"))
        builder = PipelineBuilder(spec, k=K)
        _, meta = builder.build(ITEMS)
        pred, query = encode_pipeline_inputs(
            {"city": "北京"}, [[1.0, 0.0, 0.0, 0.0]], meta
        )
        assert pred.shape == (1, meta["num_preds"])
        assert query.shape == (1, D)

    def test_wrong_num_query_vectors_raises(self):
        spec = Or(KNN("cosine"), KNN("l2"))
        builder = PipelineBuilder(spec, k=K)
        _, meta = builder.build(ITEMS)
        with pytest.raises(ValueError, match="Expected 2"):
            encode_pipeline_inputs(None, [[1.0, 0.0, 0.0, 0.0]], meta)

    def test_no_targeting_no_knn_defaults(self):
        spec = Targeting(SCHEMA)
        builder = PipelineBuilder(spec, k=K)
        _, meta = builder.build(ITEMS)
        pred, query = encode_pipeline_inputs(None, None, meta)
        assert pred.shape == (1, meta["num_preds"])
        assert pred.sum().item() == 0
        assert query.shape == (1, 1)

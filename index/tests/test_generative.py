"""Tests for the generative (trie-constrained beam search) recall module."""

import pytest
import torch

from torch_recall.schema import Item, Schema
from torch_recall.recall_method.generative.builder import GenerativeBuilder
from torch_recall.recall_method.generative.decoder import MockDecoder
from torch_recall.recall_method.generative.trie import Trie
from torch_recall.recall_method.generative.recall import GenerativeRecall
from torch_recall.recall_method.targeting.encoder import encode_user
from torch_recall.scheduler.spec import Generative, KNN, Or
from torch_recall.scheduler.pipeline_builder import PipelineBuilder


# ── Test data ────────────────────────────────────────────────────────────────

SCHEMA = Schema(discrete_fields=["city"], numeric_fields=["age"])

ITEMS = [
    Item(id="a", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="b", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="c", targeting_rule='city == "上海"', sid_path=[0, 1, 5, 6, 7]),
    Item(id="d", targeting_rule="age > 18",       sid_path=[0, 2, 8, 9, 10]),
    Item(id="e", targeting_rule='city == "北京"', sid_path=[0, 2, 8, 9, 11]),
]

USER_DIM = 16
BEAM_WIDTH = 3


def _make_decoder():
    return MockDecoder(user_dim=USER_DIM)


# ── TestBuilder ──────────────────────────────────────────────────────────────

class TestBuilder:
    def test_build_returns_model_and_meta(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        model, meta = builder.build(ITEMS)
        assert isinstance(model, GenerativeRecall)
        assert meta["num_items"] == len(ITEMS)
        assert meta["beam_width"] == BEAM_WIDTH
        assert meta["path_length"] == 5

    def test_meta_has_trie_info(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        _, meta = builder.build(ITEMS)
        assert "nodes_per_level" in meta
        assert meta["num_leaves"] > 0
        assert meta["num_states"] > 0
        assert "max_branch_factors" in meta

    def test_missing_sid_path_raises(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder())
        with pytest.raises(ValueError, match="sid_path is None"):
            builder.build([Item(targeting_rule='city == "北京"')])

    def test_wrong_path_length_raises(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder())
        with pytest.raises(ValueError, match="sid_path length"):
            builder.build([
                Item(targeting_rule='city == "北京"', sid_path=[0, 1, 2]),
            ])

    def test_sid_out_of_range_raises(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder())
        with pytest.raises(ValueError, match="not in"):
            builder.build([
                Item(targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 9999]),
            ])

    def test_item_ids_in_meta(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        _, meta = builder.build(ITEMS)
        assert meta["item_ids"] == ["a", "b", "c", "d", "e"]

    def test_shared_path_leaf_mapping(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        _, meta = builder.build(ITEMS)
        assert meta["max_items_per_leaf"] >= 2


# ── TestTrie ─────────────────────────────────────────────────────────────────

class TestTrie:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_trie_is_static(self, model_and_meta):
        model, _ = model_and_meta
        assert isinstance(model.trie, Trie)
        assert model.trie.d_dense == 2
        assert model.trie.path_length == 5
        assert model.trie.vocab_size == 2048

    def test_csr_indptr_shape(self, model_and_meta):
        model, meta = model_and_meta
        assert model.trie.csr_indptr.shape[0] == meta["num_states"] + 2

    def test_propagate_all_valid(self, model_and_meta):
        model, _ = model_and_meta
        N = len(ITEMS)
        item_mask = torch.ones(1, N, dtype=torch.bool)
        nv = model.trie.propagate_validity(item_mask)
        assert nv.shape == (1, model.trie.num_states)
        assert nv[:, 1].item() is True

    def test_propagate_none_valid(self, model_and_meta):
        model, _ = model_and_meta
        N = len(ITEMS)
        item_mask = torch.zeros(1, N, dtype=torch.bool)
        nv = model.trie.propagate_validity(item_mask)
        assert not nv.any().item()

    def test_propagate_partial(self, model_and_meta):
        model, _ = model_and_meta
        N = len(ITEMS)
        item_mask = torch.zeros(1, N, dtype=torch.bool)
        item_mask[0, 0] = True
        nv = model.trie.propagate_validity(item_mask)
        assert nv.any().item()


# ── TestForward ──────────────────────────────────────────────────────────────

class TestForward:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_output_shape(self, model_and_meta):
        model, meta = model_and_meta
        pred = encode_user({"city": "北京", "age": 25}, meta["targeting"])
        pred = pred.unsqueeze(0)
        query = torch.randn(1, USER_DIM)
        with torch.no_grad():
            scores = model(pred, query)
        assert scores.shape == (1, len(ITEMS))

    def test_filtered_items_get_neg_inf(self, model_and_meta):
        model, meta = model_and_meta
        pred = encode_user({"city": "广州", "age": 10}, meta["targeting"])
        pred = pred.unsqueeze(0)
        query = torch.randn(1, USER_DIM)
        with torch.no_grad():
            scores = model(pred, query)
        assert (scores == float("-inf")).all(), (
            "No items should match city=广州,age=10"
        )

    def test_some_items_have_finite_scores(self, model_and_meta):
        model, meta = model_and_meta
        pred = encode_user({"city": "北京", "age": 25}, meta["targeting"])
        pred = pred.unsqueeze(0)
        query = torch.randn(1, USER_DIM)
        with torch.no_grad():
            scores = model(pred, query)
        finite_mask = scores[0] > float("-inf")
        assert finite_mask.any(), "At least one item should have a finite score"


# ── TestBatch ────────────────────────────────────────────────────────────────

class TestBatch:
    @pytest.fixture(scope="class")
    def model_and_meta(self):
        builder = GenerativeBuilder(SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH)
        model, meta = builder.build(ITEMS)
        model.eval()
        return model, meta

    def test_batch_shape(self, model_and_meta):
        model, meta = model_and_meta
        B = 3
        pred1 = encode_user({"city": "北京"}, meta["targeting"])
        pred2 = encode_user({"city": "上海"}, meta["targeting"])
        pred3 = encode_user({"age": 25}, meta["targeting"])
        batch_pred = torch.stack([pred1, pred2, pred3])
        batch_query = torch.randn(B, USER_DIM)
        with torch.no_grad():
            scores = model(batch_pred, batch_query)
        assert scores.shape == (B, len(ITEMS))

    def test_batch_equals_individual(self, model_and_meta):
        model, meta = model_and_meta
        torch.manual_seed(42)
        pred1 = encode_user({"city": "北京"}, meta["targeting"]).unsqueeze(0)
        pred2 = encode_user({"city": "上海"}, meta["targeting"]).unsqueeze(0)
        q1 = torch.randn(1, USER_DIM)
        q2 = torch.randn(1, USER_DIM)
        with torch.no_grad():
            s1 = model(pred1, q1)
            s2 = model(pred2, q2)

        batch_pred = torch.cat([pred1, pred2], dim=0)
        batch_q = torch.cat([q1, q2], dim=0)
        with torch.no_grad():
            sb = model(batch_pred, batch_q)

        assert torch.equal(sb[0], s1[0])
        assert torch.equal(sb[1], s2[0])


# ── TestPipelineIntegration ──────────────────────────────────────────────────

ITEMS_WITH_EMB = [
    Item(id="a", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4],
         embedding=[1.0, 0.0, 0.0, 0.0]),
    Item(id="b", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4],
         embedding=[0.0, 1.0, 0.0, 0.0]),
    Item(id="c", targeting_rule='city == "上海"', sid_path=[0, 1, 5, 6, 7],
         embedding=[0.0, 0.0, 1.0, 0.0]),
    Item(id="d", targeting_rule="age > 18",       sid_path=[0, 2, 8, 9, 10],
         embedding=[0.0, 0.0, 0.0, 1.0]),
    Item(id="e", targeting_rule='city == "北京"', sid_path=[0, 2, 8, 9, 11],
         embedding=[1.0, 1.0, 0.0, 0.0]),
]


class TestPipelineGenerativeOnly:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        decoder = _make_decoder()
        spec = Generative(SCHEMA, decoder, beam_width=BEAM_WIDTH)
        builder = PipelineBuilder(spec, k=3)
        pipeline, meta = builder.build(ITEMS)
        pipeline.eval()
        return pipeline, meta

    def test_meta_structure(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert meta["k"] == 3
        assert meta["num_items"] == len(ITEMS)
        assert "generative_leaves" in meta

    def test_returns_top_k(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        gen_meta = meta["generative_leaves"][0]
        pred = encode_user(
            {"city": "北京", "age": 25}, gen_meta["targeting"]
        ).unsqueeze(0)
        query = torch.randn(1, meta["total_query_dim"])
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred, query)
        assert top_vals.shape == (1, 3)
        assert top_idx.shape == (1, 3)


class TestPipelineOrGenerativeKNN:
    @pytest.fixture(scope="class")
    def pipeline_and_meta(self):
        decoder = _make_decoder()
        spec = Or(
            Generative(SCHEMA, decoder, beam_width=BEAM_WIDTH),
            KNN(metric="cosine"),
        )
        builder = PipelineBuilder(spec, k=3)
        pipeline, meta = builder.build(ITEMS_WITH_EMB)
        pipeline.eval()
        return pipeline, meta

    def test_meta_has_both(self, pipeline_and_meta):
        _, meta = pipeline_and_meta
        assert "generative_leaves" in meta
        assert "knn_leaves" in meta

    def test_returns_valid_results(self, pipeline_and_meta):
        pipeline, meta = pipeline_and_meta
        gen_meta = meta["generative_leaves"][0]
        pred = encode_user({"city": "北京"}, gen_meta["targeting"]).unsqueeze(0)
        total_dim = meta["total_query_dim"]
        query = torch.randn(1, total_dim)
        with torch.no_grad():
            top_vals, top_idx = pipeline(pred, query)
        assert top_vals.shape == (1, 3)
        assert (top_vals[0] > float("-inf")).any()

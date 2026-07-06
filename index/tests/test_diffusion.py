"""Tests for the diffusion recall module (SidPathFilter + DiffusionRecall)."""

import pytest
import torch

from torch_recall.schema import Item, Schema
from torch_recall.recall_method.diffusion.path_filter import SidPathFilter
from torch_recall.recall_method.diffusion.model import MockDiffusionDecoder
from torch_recall.recall_method.diffusion.recall import DiffusionRecall
from torch_recall.recall_method.diffusion.builder import DiffusionBuilder


# ── Shared test data ──────────────────────────────────────────────────────────

SCHEMA = Schema(discrete_fields=["city"], numeric_fields=["age"])

# Same 5 items used in test_generative.py; items a & b share the same SID path.
ITEMS = [
    Item(id="a", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="b", targeting_rule='city == "北京"', sid_path=[0, 1, 2, 3, 4]),
    Item(id="c", targeting_rule='city == "上海"', sid_path=[0, 1, 5, 6, 7]),
    Item(id="d", targeting_rule="age > 18",       sid_path=[0, 2, 8, 9, 10]),
    Item(id="e", targeting_rule='city == "北京"', sid_path=[0, 2, 8, 9, 11]),
]

USER_DIM = 16
BEAM_WIDTH = 3
VOCAB_SIZE = 32   # small vocab for tests
PATH_LENGTH = 5


def _make_decoder() -> MockDiffusionDecoder:
    return MockDiffusionDecoder(
        user_dim=USER_DIM,
        path_length=PATH_LENGTH,
        vocab_size=VOCAB_SIZE,
    )


def _make_path_vectors() -> torch.Tensor:
    return torch.tensor([item.sid_path for item in ITEMS], dtype=torch.long)


# ── TestSidPathFilter ─────────────────────────────────────────────────────────


class TestSidPathFilter:
    def _make_filter(self) -> SidPathFilter:
        return SidPathFilter(_make_path_vectors(), vocab_size=VOCAB_SIZE)

    def test_init_beam_mask_shape(self):
        spf = self._make_filter()
        item_mask = torch.ones(2, len(ITEMS), dtype=torch.bool)
        mask = spf.init_beam_mask(item_mask, beam_width=3)
        assert mask.shape == (2, 3, len(ITEMS))

    def test_init_beam_mask_broadcasts_item_mask(self):
        spf = self._make_filter()
        # Only item 0 active
        item_mask = torch.zeros(1, len(ITEMS), dtype=torch.bool)
        item_mask[0, 0] = True
        mask = spf.init_beam_mask(item_mask, beam_width=4)
        assert mask.shape == (1, 4, len(ITEMS))
        # All beams should mirror the same item_mask row
        assert mask[0, 0, 0].item() is True
        assert mask[0, 3, 0].item() is True
        assert not mask[0, 0, 1].item()

    def test_collect_valid_tokens_all_active(self):
        spf = self._make_filter()
        N = len(ITEMS)
        beam_mask = torch.ones(1, 1, N, dtype=torch.bool)
        # Position 0: all items have token 0
        valid = spf.collect_valid_tokens(beam_mask, position=0)
        assert valid.shape == (1, 1, VOCAB_SIZE)
        assert valid[0, 0, 0].item()   # token 0 is reachable
        assert not valid[0, 0, 1].item()  # token 1 is not

    def test_collect_valid_tokens_partial_active(self):
        spf = self._make_filter()
        N = len(ITEMS)
        # Only item 2 active (sid=[0,1,5,6,7]), depth 2 → token 5
        beam_mask = torch.zeros(1, 1, N, dtype=torch.bool)
        beam_mask[0, 0, 2] = True
        valid = spf.collect_valid_tokens(beam_mask, position=2)
        assert valid[0, 0, 5].item()
        assert not valid[0, 0, 2].item()

    def test_filter_beam_paths_keeps_consistent(self):
        spf = self._make_filter()
        N = len(ITEMS)
        beam_mask = torch.ones(1, 2, N, dtype=torch.bool)
        # Beam 0 commits token=1 at position=1, beam 1 commits token=2 at position=1
        # Items with depth-1 token:
        #   a/b → 1, c → 1, d → 2, e → 2
        committed = torch.tensor([[1, 2]], dtype=torch.long)  # [B=1, beam=2]
        filtered = spf.filter_beam_paths(beam_mask, position=1, committed_tokens=committed)
        # Beam 0 (token=1): items a(0), b(1), c(2) have depth-1=1 → kept
        assert filtered[0, 0, 0].item()   # a
        assert filtered[0, 0, 1].item()   # b
        assert filtered[0, 0, 2].item()   # c
        assert not filtered[0, 0, 3].item()  # d (depth-1=2)
        assert not filtered[0, 0, 4].item()  # e (depth-1=2)
        # Beam 1 (token=2): items d(3), e(4) → kept
        assert not filtered[0, 1, 0].item()
        assert filtered[0, 1, 3].item()   # d
        assert filtered[0, 1, 4].item()   # e

    def test_filter_is_monotone(self):
        """beam_mask can only shrink, never grow."""
        spf = self._make_filter()
        N = len(ITEMS)
        beam_mask = torch.ones(1, 1, N, dtype=torch.bool)
        before_count = beam_mask.sum().item()
        committed = torch.tensor([[0]], dtype=torch.long)  # token=0 at pos=0
        filtered = spf.filter_beam_paths(beam_mask, position=0, committed_tokens=committed)
        # All items have token=0 at depth 0, so nothing removed
        assert filtered.sum().item() == before_count

        # Now filter at position 1 with token=1: only items a/b/c qualify
        committed2 = torch.tensor([[1]], dtype=torch.long)
        filtered2 = spf.filter_beam_paths(filtered, position=1, committed_tokens=committed2)
        assert filtered2.sum().item() <= filtered.sum().item()


# ── TestMockDiffusionDecoder ──────────────────────────────────────────────────


class TestMockDiffusionDecoder:
    def test_output_shape(self):
        dec = _make_decoder()
        user_repr = torch.randn(4, USER_DIM)
        partial_paths = torch.full((4, PATH_LENGTH), MockDiffusionDecoder.MASK_ID)
        out = dec(user_repr, partial_paths)
        assert out.shape == (4, PATH_LENGTH, VOCAB_SIZE)

    def test_all_positions_produced(self):
        dec = _make_decoder()
        user_repr = torch.randn(1, USER_DIM)
        partial = torch.zeros(1, PATH_LENGTH, dtype=torch.long)
        out = dec(user_repr, partial)
        assert out.shape == (1, PATH_LENGTH, VOCAB_SIZE)

    def test_different_users_different_logits(self):
        dec = _make_decoder()
        u1 = torch.randn(1, USER_DIM)
        u2 = torch.randn(1, USER_DIM)
        partial = torch.zeros(1, PATH_LENGTH, dtype=torch.long)
        out1 = dec(u1, partial)
        out2 = dec(u2, partial)
        assert not torch.allclose(out1, out2)


# ── TestDiffusionBuilder ──────────────────────────────────────────────────────


class TestDiffusionBuilder:
    def test_build_returns_model_and_meta(self):
        builder = DiffusionBuilder(
            SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH,
            sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH,
        )
        model, meta = builder.build(ITEMS)
        assert isinstance(model, DiffusionRecall)
        assert meta["num_items"] == len(ITEMS)
        assert meta["beam_width"] == BEAM_WIDTH
        assert meta["path_length"] == PATH_LENGTH

    def test_missing_sid_path_raises(self):
        builder = DiffusionBuilder(SCHEMA, _make_decoder(),
                                   sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH)
        with pytest.raises(ValueError, match="sid_path is None"):
            builder.build([Item(targeting_rule='city == "北京"')])

    def test_wrong_path_length_raises(self):
        builder = DiffusionBuilder(SCHEMA, _make_decoder(),
                                   sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH)
        with pytest.raises(ValueError, match="length"):
            builder.build([Item(targeting_rule='city == "北京"', sid_path=[0, 1])])

    def test_sid_out_of_range_raises(self):
        builder = DiffusionBuilder(SCHEMA, _make_decoder(),
                                   sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH)
        bad_sid = [VOCAB_SIZE, 1, 2, 3, 4]  # first token out of range
        with pytest.raises(ValueError, match="not in"):
            builder.build([Item(targeting_rule='city == "北京"', sid_path=bad_sid)])


# ── TestDiffusionRecallForward ────────────────────────────────────────────────


def _build_model(beam_width: int = BEAM_WIDTH) -> DiffusionRecall:
    builder = DiffusionBuilder(
        SCHEMA, _make_decoder(), beam_width=beam_width,
        sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH,
    )
    model, _ = builder.build(ITEMS)
    return model


def _encode_user(model: DiffusionRecall, meta: dict,
                 city: str | None = None, age: float | None = None) -> torch.Tensor:
    from torch_recall.recall_method.targeting.encoder import encode_user
    attrs: dict = {}
    if city is not None:
        attrs["city"] = city
    if age is not None:
        attrs["age"] = age
    pred = encode_user(attrs, meta["targeting"])
    return pred.unsqueeze(0)


class TestDiffusionRecallForward:
    def test_output_shape(self):
        model = _build_model()
        pred = torch.zeros(1, model.num_preds, dtype=torch.bool)
        query = torch.randn(1, model.user_dim)
        out = model(pred, query)
        assert out.shape == (1, len(ITEMS))

    def test_filtered_items_get_neg_inf(self):
        """Items that fail targeting should receive −inf."""
        builder = DiffusionBuilder(
            SCHEMA, _make_decoder(), beam_width=BEAM_WIDTH,
            sid_vocab_size=VOCAB_SIZE, path_length=PATH_LENGTH,
        )
        model, meta = builder.build(ITEMS)
        # User matches only 北京 items (a, b, e); 上海 item c should be −inf
        pred = _encode_user(model, meta, city="北京")
        query = torch.randn(1, model.user_dim)
        out = model(pred, query)
        # Item c (index 2) is 上海-only — must be −inf
        assert out[0, 2].item() == float("-inf")

    def test_some_items_have_finite_scores(self):
        """At least one item should have a finite score."""
        model = _build_model()
        pred = torch.ones(1, model.num_preds, dtype=torch.bool)  # all preds satisfied
        query = torch.randn(1, model.user_dim)
        out = model(pred, query)
        assert (out > float("-inf")).any()

    def test_batch_shape(self):
        model = _build_model()
        B = 4
        pred = torch.zeros(B, model.num_preds, dtype=torch.bool)
        query = torch.randn(B, model.user_dim)
        out = model(pred, query)
        assert out.shape == (B, len(ITEMS))

    def test_batch_equals_individual(self):
        """Batched forward must equal individual forwards."""
        torch.manual_seed(42)
        model = _build_model()
        model.eval()
        B = 3
        preds = torch.zeros(B, model.num_preds, dtype=torch.bool)
        queries = torch.randn(B, model.user_dim)

        out_batch = model(preds, queries)
        for b in range(B):
            out_single = model(preds[b : b + 1], queries[b : b + 1])
            assert torch.allclose(out_batch[b], out_single[0], equal_nan=True), (
                f"Batch item {b} differs from individual result"
            )

    def test_no_active_items_all_neg_inf(self):
        """When targeting excludes all items the output is all −inf."""
        model = _build_model()
        # No predicate satisfied → targeting gives −inf for every item
        pred = torch.zeros(1, model.num_preds, dtype=torch.bool)
        query = torch.randn(1, model.user_dim)
        out = model(pred, query)
        assert (out == float("-inf")).all()

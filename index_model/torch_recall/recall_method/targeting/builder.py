from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from torch_recall.schema import (
    Schema,
    MAX_PREDS_PER_CONJ,
    MAX_CONJ_PER_ITEM,
    MAX_CONJ,
)
from torch_recall.query.parser import Predicate, parse_expr
from torch_recall.query.dnf import to_dnf, LiteralPred
from torch_recall.recall_method.targeting.recall import TargetingRecall


NUMERIC_OPS = frozenset({"==", "<", ">", "<=", ">="})


class TargetingBuilder:
    """Build a TargetingRecall model from item targeting rules.

    Each item is a boolean expression string using the standard query syntax.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self._discrete_fields = set(schema.discrete_fields)
        self._numeric_fields = set(schema.numeric_fields)
        self._text_fields = set(schema.text_fields)

    def build(self, rules: list[str]) -> tuple[TargetingRecall, dict]:
        """Parse item targeting rules and build the model + meta.

        Args:
            rules: one boolean expression string per item.
        Returns:
            (model, meta) tuple.
        """
        N = len(rules)

        # --- Phase 1: parse rules → DNFs, build predicate registry ---
        pred_key_to_id: dict[tuple, int] = {}
        pred_list: list[dict] = []

        # per-item: list of conjunctions, each conjunction is list of (pred_id, negated)
        item_conjs: list[list[list[tuple[int, bool]]]] = []

        for item_idx, rule_str in enumerate(rules):
            expr = parse_expr(rule_str)
            dnf = to_dnf(expr)
            if len(dnf) > MAX_CONJ:
                raise ValueError(
                    f"Item {item_idx}: DNF has {len(dnf)} conjunctions "
                    f"(max {MAX_CONJ})"
                )

            conjs_for_item: list[list[tuple[int, bool]]] = []
            for conj in dnf:
                preds_in_conj: list[tuple[int, bool]] = []
                for lit in conj:
                    pred_id, negated = self._register_predicate(
                        lit, pred_key_to_id, pred_list
                    )
                    preds_in_conj.append((pred_id, negated))

                if len(preds_in_conj) > MAX_PREDS_PER_CONJ:
                    raise ValueError(
                        f"Item {item_idx}: conjunction has "
                        f"{len(preds_in_conj)} predicates "
                        f"(max {MAX_PREDS_PER_CONJ})"
                    )
                conjs_for_item.append(preds_in_conj)

            if len(conjs_for_item) > MAX_CONJ_PER_ITEM:
                raise ValueError(
                    f"Item {item_idx}: {len(conjs_for_item)} conjunctions "
                    f"(max {MAX_CONJ_PER_ITEM})"
                )
            item_conjs.append(conjs_for_item)

        P = len(pred_list)
        K = MAX_PREDS_PER_CONJ
        J = MAX_CONJ_PER_ITEM

        # --- Phase 2: build conjunction tensors ---
        all_conj_preds: list[list[tuple[int, bool]]] = []
        item_conj_offsets: list[list[int]] = []

        for conjs in item_conjs:
            offsets: list[int] = []
            for conj in conjs:
                offsets.append(len(all_conj_preds))
                all_conj_preds.append(conj)
            item_conj_offsets.append(offsets)

        C = len(all_conj_preds)

        conj_pred_ids = torch.zeros(C, K, dtype=torch.int64)
        conj_pred_negated = torch.zeros(C, K, dtype=torch.bool)
        conj_pred_valid = torch.zeros(C, K, dtype=torch.bool)

        for ci, preds in enumerate(all_conj_preds):
            for pi, (pred_id, negated) in enumerate(preds):
                conj_pred_ids[ci, pi] = pred_id
                conj_pred_negated[ci, pi] = negated
                conj_pred_valid[ci, pi] = True

        # --- Phase 3: build item-conjunction tensors ---
        item_conj_ids = torch.zeros(N, J, dtype=torch.int64)
        item_conj_valid = torch.zeros(N, J, dtype=torch.bool)

        for item_idx, offsets in enumerate(item_conj_offsets):
            for ji, conj_idx in enumerate(offsets):
                item_conj_ids[item_idx, ji] = conj_idx
                item_conj_valid[item_idx, ji] = True

        # --- Phase 4: build meta ---
        discrete_registry: dict[str, dict[str, int]] = {}
        numeric_registry: list[dict] = []
        text_registry: dict[str, dict[str, int]] = {}

        for pred in pred_list:
            if pred["type"] == "discrete":
                discrete_registry.setdefault(pred["field"], {})[
                    str(pred["value"])
                ] = pred["pred_id"]
            elif pred["type"] == "numeric":
                numeric_registry.append(pred)
            elif pred["type"] == "text":
                text_registry.setdefault(pred["field"], {})[
                    str(pred["value"])
                ] = pred["pred_id"]

        meta = {
            "num_items": N,
            "num_preds": P,
            "num_conjs": C,
            "predicate_registry": {
                "discrete": discrete_registry,
                "numeric": numeric_registry,
                "text": text_registry,
            },
        }

        model = TargetingRecall(
            conj_pred_ids=conj_pred_ids,
            conj_pred_negated=conj_pred_negated,
            conj_pred_valid=conj_pred_valid,
            item_conj_ids=item_conj_ids,
            item_conj_valid=item_conj_valid,
            num_items=N,
            num_preds=P,
        )
        return model, meta

    def _register_predicate(
        self,
        lit: LiteralPred,
        key_map: dict[tuple, int],
        pred_list: list[dict],
    ) -> tuple[int, bool]:
        """Register a predicate and return (pred_id, negated)."""
        pred = lit.pred
        negated = lit.negated

        if pred.op == "!=":
            pred = Predicate(pred.field, "==", pred.value)
            negated = not negated

        if pred.field in self._discrete_fields and pred.op == "==":
            key = ("discrete", pred.field, str(pred.value))
        elif pred.field in self._numeric_fields and pred.op in NUMERIC_OPS:
            key = ("numeric", pred.field, pred.op, float(pred.value))
        elif pred.field in self._text_fields and pred.op == "contains":
            key = ("text", pred.field, str(pred.value))
        else:
            raise ValueError(
                f"Cannot classify predicate: {pred.field} {pred.op} {pred.value}"
            )

        if key not in key_map:
            pred_id = len(pred_list)
            key_map[key] = pred_id
            entry: dict = {"pred_id": pred_id, "type": key[0], "field": pred.field}
            if key[0] == "discrete":
                entry["value"] = str(pred.value)
            elif key[0] == "numeric":
                entry["op"] = pred.op
                entry["value"] = float(pred.value)
            elif key[0] == "text":
                entry["value"] = str(pred.value)
            pred_list.append(entry)

        return key_map[key], negated

    @staticmethod
    def save_meta(meta: dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

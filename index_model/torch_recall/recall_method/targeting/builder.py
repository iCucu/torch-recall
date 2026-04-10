from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from torch_recall.schema import Item, Schema, MAX_CONJ
from torch_recall.query.parser import Predicate, parse_expr
from torch_recall.query.dnf import to_dnf, LiteralPred
from torch_recall.recall_method.targeting.recall import TargetingRecall


NUMERIC_OPS = frozenset({"==", "<", ">", "<=", ">="})


class TargetingBuilder:
    """Build a TargetingRecall model from item targeting rules.

    Each item is a boolean expression string using the standard query syntax.
    """

    def __init__(
        self,
        schema: Schema,
        max_preds_per_conj: int | None = None,
        max_conj_per_item: int | None = None,
    ):
        """
        Args:
            schema: field type definitions.
            max_preds_per_conj: optional upper bound for K. If None, K is
                determined adaptively from the actual rules.
            max_conj_per_item: optional upper bound for J. If None, J is
                determined adaptively from the actual rules.
        """
        self.schema = schema
        self._discrete_fields = set(schema.discrete_fields)
        self._numeric_fields = set(schema.numeric_fields)
        self._text_fields = set(schema.text_fields)
        self._max_k = max_preds_per_conj
        self._max_j = max_conj_per_item

    def build(self, items: list[Item]) -> tuple[TargetingRecall, dict]:
        """Parse item targeting rules and build the model + meta.

        Args:
            items: list of Item objects, each must have targeting_rule set.
        Returns:
            (model, meta) tuple.
        """
        N = len(items)

        # --- Phase 1: parse rules → DNFs, build predicate registry ---
        pred_key_to_id: dict[tuple, int] = {}
        pred_list: list[dict] = []

        # per-item: list of conjunctions, each conjunction is list of (pred_id, negated)
        item_conjs: list[list[list[tuple[int, bool]]]] = []

        for item_idx, item in enumerate(items):
            rule_str = item.targeting_rule
            if rule_str is None:
                raise ValueError(
                    f"Item {item_idx}: targeting_rule is None"
                )
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
                conjs_for_item.append(preds_in_conj)
            item_conjs.append(conjs_for_item)

        P = len(pred_list)

        # Adaptive K/J: scan actual data, apply optional upper bounds
        actual_k = max(
            (len(conj) for conjs in item_conjs for conj in conjs), default=1
        )
        actual_j = max((len(conjs) for conjs in item_conjs), default=1)

        K = actual_k if self._max_k is None else self._max_k
        J = actual_j if self._max_j is None else self._max_j

        if actual_k > K:
            raise ValueError(
                f"Conjunction has {actual_k} predicates "
                f"(max_preds_per_conj={K})"
            )
        if actual_j > J:
            raise ValueError(
                f"Item has {actual_j} conjunctions "
                f"(max_conj_per_item={J})"
            )

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

        raw_ids = [item.id for item in items]
        item_ids = raw_ids if any(i is not None for i in raw_ids) else None

        meta = {
            "num_items": N,
            "num_preds": P,
            "num_conjs": C,
            "max_preds_per_conj": K,
            "max_conj_per_item": J,
            "item_ids": item_ids,
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

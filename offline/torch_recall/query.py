from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

import torch

from torch_recall.schema import (
    MAX_BP, MAX_NP, MAX_CONJ, P_TOTAL,
    CONJ_PER_PASS,
)


# --- AST Nodes ---


@dataclass
class Predicate:
    field: str
    op: str
    value: object


@dataclass
class And:
    children: list[Expr]


@dataclass
class Or:
    children: list[Expr]


@dataclass
class Not:
    child: Expr


Expr = Union[Predicate, And, Or, Not]


@dataclass
class LiteralPred:
    pred: Predicate
    negated: bool = False


# --- Tokenizer for expression strings ---

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (AND|OR|NOT|contains)          # keywords
        |([<>!=]=?|==)                 # operators
        |"([^"]*)"                     # quoted string
        |'([^']*)'                     # single-quoted string
        |(\()                          # lparen
        |(\))                          # rparen
        |([^\s()'"<>!=]+)              # bare word (field name or number)
    )\s*
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        if m.group(1):
            tokens.append(("KW", m.group(1)))
        elif m.group(2):
            tokens.append(("OP", m.group(2)))
        elif m.group(3) is not None:
            tokens.append(("STR", m.group(3)))
        elif m.group(4) is not None:
            tokens.append(("STR", m.group(4)))
        elif m.group(5):
            tokens.append(("LPAREN", "("))
        elif m.group(6):
            tokens.append(("RPAREN", ")"))
        elif m.group(7):
            tokens.append(("WORD", m.group(7)))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok}")
        self.pos += 1
        return tok

    def parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        children = [left]
        while self.peek() and self.peek() == ("KW", "OR"):
            self.consume()
            children.append(self._parse_and())
        return children[0] if len(children) == 1 else Or(children)

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        children = [left]
        while self.peek() and self.peek() == ("KW", "AND"):
            self.consume()
            children.append(self._parse_not())
        return children[0] if len(children) == 1 else And(children)

    def _parse_not(self) -> Expr:
        if self.peek() and self.peek() == ("KW", "NOT"):
            self.consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        if self.peek() and self.peek()[0] == "LPAREN":
            self.consume()
            expr = self.parse_expr()
            self.consume("RPAREN")
            return expr
        return self._parse_predicate()

    def _parse_predicate(self) -> Predicate:
        field_tok = self.consume("WORD")
        field_name = field_tok[1]

        next_tok = self.peek()
        if next_tok and next_tok == ("KW", "contains"):
            self.consume()
            val_tok = self.consume("STR")
            return Predicate(field_name, "contains", val_tok[1])

        op_tok = self.consume("OP")
        val_tok = self.consume()
        if val_tok[0] == "STR":
            value = val_tok[1]
        elif val_tok[0] == "WORD":
            try:
                value = float(val_tok[1])
            except ValueError:
                value = val_tok[1]
        else:
            raise ValueError(f"Unexpected value token: {val_tok}")

        return Predicate(field_name, op_tok[1], value)


def parse_expr(expr_str: str) -> Expr:
    tokens = _tokenize(expr_str)
    parser = _Parser(tokens)
    result = parser.parse_expr()
    if parser.pos != len(tokens):
        raise ValueError(f"Unexpected tokens after position {parser.pos}")
    return result


# --- DNF Converter ---

Conjunction = list[LiteralPred]
DNF = list[Conjunction]


def to_dnf(expr: Expr) -> DNF:
    return _to_dnf(expr)


def _to_dnf(expr: Expr) -> DNF:
    if isinstance(expr, Predicate):
        return [[LiteralPred(expr, negated=False)]]
    elif isinstance(expr, Not):
        return _negate_dnf(expr.child)
    elif isinstance(expr, And):
        result: DNF = [[]]
        for child in expr.children:
            child_dnf = _to_dnf(child)
            new_result: DNF = []
            for existing_conj in result:
                for child_conj in child_dnf:
                    new_result.append(existing_conj + child_conj)
            result = new_result
            if len(result) > MAX_CONJ:
                raise ValueError(
                    f"DNF exceeds {MAX_CONJ} conjunctions. Simplify the query."
                )
        return result
    elif isinstance(expr, Or):
        result = []
        for child in expr.children:
            result.extend(_to_dnf(child))
            if len(result) > MAX_CONJ:
                raise ValueError(
                    f"DNF exceeds {MAX_CONJ} conjunctions. Simplify the query."
                )
        return result
    else:
        raise TypeError(f"Unknown expr type: {type(expr)}")


def _negate_dnf(expr: Expr) -> DNF:
    if isinstance(expr, Predicate):
        return [[LiteralPred(expr, negated=True)]]
    elif isinstance(expr, Not):
        return _to_dnf(expr.child)
    elif isinstance(expr, And):
        negated_or = Or([Not(c) for c in expr.children])
        return _to_dnf(negated_or)
    elif isinstance(expr, Or):
        negated_and = And([Not(c) for c in expr.children])
        return _to_dnf(negated_and)
    else:
        raise TypeError(f"Unknown expr type: {type(expr)}")


# --- Tensor Encoder ---


def encode_query(expr_str: str, meta: dict) -> dict[str, torch.Tensor]:
    """Encode a query string into tensor inputs for InvertedIndexModel.

    If the DNF has ≤ CONJ_PER_PASS conjunctions, returns a single dict of
    tensors suitable for ``model(**tensors)``.

    If the DNF exceeds CONJ_PER_PASS, returns a list of dicts (one per pass).
    The caller should OR the results across passes.

    For backward compatibility the function always returns the "simple" dict
    when possible.
    """
    expr = parse_expr(expr_str)
    dnf = to_dnf(expr)

    if len(dnf) <= CONJ_PER_PASS:
        return _encode_dnf_single(dnf, meta)
    return _encode_dnf_multi(dnf, meta)


def encode_query_batched(expr_str: str, meta: dict) -> list[dict[str, torch.Tensor]]:
    """Always returns a list of tensor dicts (one per pass)."""
    expr = parse_expr(expr_str)
    dnf = to_dnf(expr)

    if len(dnf) <= CONJ_PER_PASS:
        return [_encode_dnf_single(dnf, meta)]

    return _encode_dnf_multi(dnf, meta)


# ── internal helpers ─────────────────────────────────────────────────────────

def _collect_predicates(dnf: DNF, meta: dict):
    """First pass: allocate global predicate slots and fill validity/negation."""
    schema = meta["schema"]
    discrete_fields = set(schema["discrete"])
    numeric_fields_list = schema["numeric"]
    text_fields = set(schema["text"])
    numeric_field_idx = {name: i for i, name in enumerate(numeric_fields_list)}

    OP_MAP = {"==": 0, "<": 1, ">": 2, "<=": 3, ">=": 4}

    bitmap_indices = torch.zeros(MAX_BP, dtype=torch.int64)
    bitmap_valid = torch.zeros(MAX_BP, dtype=torch.bool)
    numeric_fields_t = torch.zeros(MAX_NP, dtype=torch.int64)
    numeric_ops = torch.zeros(MAX_NP, dtype=torch.int64)
    numeric_values = torch.zeros(MAX_NP, dtype=torch.float32)
    numeric_valid = torch.zeros(MAX_NP, dtype=torch.bool)
    negation_mask = torch.zeros(P_TOTAL, dtype=torch.bool)

    pred_key_to_bp_idx: dict[tuple, int] = {}
    pred_key_to_np_idx: dict[tuple, int] = {}
    bp_count = 0
    np_count = 0

    for conj in dnf:
        for lit in conj:
            pred = lit.pred
            neg = lit.negated

            if pred.op == "!=":
                pred = Predicate(pred.field, "==", pred.value)
                neg = not neg

            if pred.field in discrete_fields:
                val_id = meta["discrete_dicts"][pred.field].get(str(pred.value))
                if val_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"d:{pred.field}"].get(str(val_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key not in pred_key_to_bp_idx:
                    if bp_count >= MAX_BP:
                        raise ValueError(f"Exceeds MAX_BP={MAX_BP}")
                    idx = bp_count
                    bitmap_indices[idx] = int(bitmap_key)
                    bitmap_valid[idx] = True
                    negation_mask[idx] = neg
                    pred_key_to_bp_idx[key] = idx
                    bp_count += 1

            elif pred.field in text_fields and pred.op == "contains":
                term_id = meta["text_dicts"].get(pred.field, {}).get(str(pred.value))
                if term_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"t:{pred.field}"].get(str(term_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key not in pred_key_to_bp_idx:
                    if bp_count >= MAX_BP:
                        raise ValueError(f"Exceeds MAX_BP={MAX_BP}")
                    idx = bp_count
                    bitmap_indices[idx] = int(bitmap_key)
                    bitmap_valid[idx] = True
                    negation_mask[idx] = neg
                    pred_key_to_bp_idx[key] = idx
                    bp_count += 1

            elif pred.field in numeric_field_idx and pred.op in OP_MAP:
                key = ("numeric", pred.field, pred.op, float(pred.value), neg)
                if key not in pred_key_to_np_idx:
                    if np_count >= MAX_NP:
                        raise ValueError(f"Exceeds MAX_NP={MAX_NP}")
                    idx = np_count
                    numeric_fields_t[idx] = numeric_field_idx[pred.field]
                    numeric_ops[idx] = OP_MAP[pred.op]
                    numeric_values[idx] = float(pred.value)
                    numeric_valid[idx] = True
                    negation_mask[MAX_BP + idx] = neg
                    pred_key_to_np_idx[key] = idx
                    np_count += 1
            else:
                raise ValueError(f"Unknown field or op: {pred.field} {pred.op}")

    shared = {
        "bitmap_indices": bitmap_indices,
        "bitmap_valid": bitmap_valid,
        "numeric_fields": numeric_fields_t,
        "numeric_ops": numeric_ops,
        "numeric_values": numeric_values,
        "numeric_valid": numeric_valid,
        "negation_mask": negation_mask,
    }
    return shared, pred_key_to_bp_idx, pred_key_to_np_idx


def _fill_conj_matrix(
    dnf_slice: list[Conjunction],
    meta: dict,
    pred_key_to_bp_idx: dict[tuple, int],
    pred_key_to_np_idx: dict[tuple, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build conj_matrix + conj_valid for a batch of conjunctions."""
    schema = meta["schema"]
    discrete_fields = set(schema["discrete"])
    text_fields = set(schema["text"])
    numeric_field_idx = {name: i for i, name in enumerate(schema["numeric"])}
    OP_MAP = {"==": 0, "<": 1, ">": 2, "<=": 3, ">=": 4}

    conj_matrix = torch.zeros(CONJ_PER_PASS, P_TOTAL, dtype=torch.bool)
    conj_valid = torch.zeros(CONJ_PER_PASS, dtype=torch.bool)

    for ci, conj in enumerate(dnf_slice):
        conj_valid[ci] = True
        for lit in conj:
            pred = lit.pred
            neg = lit.negated

            if pred.op == "!=":
                pred = Predicate(pred.field, "==", pred.value)
                neg = not neg

            if pred.field in discrete_fields:
                val_id = meta["discrete_dicts"][pred.field].get(str(pred.value))
                if val_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"d:{pred.field}"].get(str(val_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key in pred_key_to_bp_idx:
                    conj_matrix[ci, pred_key_to_bp_idx[key]] = True

            elif pred.field in text_fields and pred.op == "contains":
                term_id = meta["text_dicts"].get(pred.field, {}).get(str(pred.value))
                if term_id is None:
                    continue
                bitmap_key = meta["bitmap_lookup"][f"t:{pred.field}"].get(str(term_id))
                if bitmap_key is None:
                    continue
                key = ("bitmap", int(bitmap_key), neg)
                if key in pred_key_to_bp_idx:
                    conj_matrix[ci, pred_key_to_bp_idx[key]] = True

            elif pred.field in numeric_field_idx and pred.op in OP_MAP:
                key = ("numeric", pred.field, pred.op, float(pred.value), neg)
                if key in pred_key_to_np_idx:
                    conj_matrix[ci, MAX_BP + pred_key_to_np_idx[key]] = True

    return conj_matrix, conj_valid


def _encode_dnf_single(dnf: DNF, meta: dict) -> dict[str, torch.Tensor]:
    """Encode a DNF with ≤ CONJ_PER_PASS conjunctions into one tensor dict."""
    shared, bp_map, np_map = _collect_predicates(dnf, meta)
    conj_matrix, conj_valid = _fill_conj_matrix(dnf, meta, bp_map, np_map)

    return {
        **shared,
        "conj_matrix": conj_matrix,
        "conj_valid": conj_valid,
    }


def _encode_dnf_multi(dnf: DNF, meta: dict) -> list[dict[str, torch.Tensor]]:
    """Encode a large DNF into multiple tensor dicts (one per pass)."""
    shared, bp_map, np_map = _collect_predicates(dnf, meta)

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(dnf), CONJ_PER_PASS):
        chunk = dnf[start:start + CONJ_PER_PASS]
        conj_matrix, conj_valid = _fill_conj_matrix(chunk, meta, bp_map, np_map)
        batches.append({
            **shared,
            "conj_matrix": conj_matrix,
            "conj_valid": conj_valid,
        })
    return batches

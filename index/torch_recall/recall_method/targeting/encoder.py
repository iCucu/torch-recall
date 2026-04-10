from __future__ import annotations

import torch

from torch_recall.tokenizer import WhitespaceTokenizer, Tokenizer


_OP_EVAL = {
    "==": lambda u, v: u == v,
    "<": lambda u, v: u < v,
    ">": lambda u, v: u > v,
    "<=": lambda u, v: u <= v,
    ">=": lambda u, v: u >= v,
}


def encode_user(
    user_attrs: dict,
    meta: dict,
    *,
    tokenizer: Tokenizer | None = None,
) -> torch.Tensor:
    """Evaluate all registered predicates against user attributes.

    Returns a [P] bool tensor where P = meta["num_preds"].
    """
    if tokenizer is None:
        tokenizer = WhitespaceTokenizer()

    P = meta["num_preds"]
    pred_satisfied = torch.zeros(P, dtype=torch.bool)
    registry = meta["predicate_registry"]

    # Discrete predicates
    for field, value_map in registry.get("discrete", {}).items():
        user_val = user_attrs.get(field)
        if user_val is None:
            continue
        pred_id = value_map.get(str(user_val))
        if pred_id is not None:
            pred_satisfied[pred_id] = True

    # Numeric predicates
    for entry in registry.get("numeric", []):
        field = entry["field"]
        user_val = user_attrs.get(field)
        if user_val is None:
            continue
        op_fn = _OP_EVAL.get(entry["op"])
        if op_fn is not None and op_fn(float(user_val), float(entry["value"])):
            pred_satisfied[entry["pred_id"]] = True

    # Text predicates
    for field, term_map in registry.get("text", {}).items():
        user_text = user_attrs.get(field)
        if user_text is None:
            continue
        tokens = set(tokenizer.tokenize(str(user_text)))
        for term, pred_id in term_map.items():
            if term in tokens:
                pred_satisfied[pred_id] = True

    return pred_satisfied


def save_user_tensors(
    user_attrs: dict,
    meta: dict,
    path: str,
    *,
    tokenizer: Tokenizer | None = None,
) -> None:
    """Encode user attributes and save for C++ inference.

    File format: list[list[Tensor]] with a single batch containing
    one tensor (pred_satisfied).
    """
    pred_satisfied = encode_user(user_attrs, meta, tokenizer=tokenizer)
    torch.save([[pred_satisfied]], path)

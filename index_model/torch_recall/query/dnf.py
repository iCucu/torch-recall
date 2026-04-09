from __future__ import annotations

from torch_recall.query.parser import Predicate, And, Or, Not, Expr, LiteralPred
from torch_recall.schema import MAX_CONJ

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

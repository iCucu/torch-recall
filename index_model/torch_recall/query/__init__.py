from torch_recall.query.parser import (
    Predicate, And, Or, Not, Expr, LiteralPred, parse_expr,
)
from torch_recall.query.dnf import to_dnf, Conjunction, DNF

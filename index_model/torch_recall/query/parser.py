from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


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

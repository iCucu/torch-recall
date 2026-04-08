from dataclasses import dataclass, field
from enum import IntEnum


class FieldType(IntEnum):
    DISCRETE = 0
    NUMERIC = 1
    TEXT = 2


class NumericOp(IntEnum):
    EQ = 0
    LT = 1
    GT = 2
    LE = 3
    GE = 4


@dataclass
class Schema:
    discrete_fields: list[str] = field(default_factory=list)
    numeric_fields: list[str] = field(default_factory=list)
    text_fields: list[str] = field(default_factory=list)

    def validate(self) -> None:
        all_names = self.discrete_fields + self.numeric_fields + self.text_fields
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate field names across types")
        if not all_names:
            raise ValueError("Schema must have at least one field")

    @property
    def discrete_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.discrete_fields)}

    @property
    def numeric_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.numeric_fields)}

    @property
    def text_field_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.text_fields)}


MAX_BP = 16
MAX_NP = 8
MAX_CONJ = 1024   # system-level limit: max conjunctions per query
P_TOTAL = MAX_BP + MAX_NP

# Model-level constants (used in forward / .pt2 export).
# Queries with >CONJ_PER_PASS conjunctions are split into multiple passes.
CONJ_PER_PASS = 8
CONJ_PASS_LEVELS = 3  # log2(CONJ_PER_PASS)

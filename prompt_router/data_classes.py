from dataclasses import dataclass
from enum import Enum
from typing import Any


class Intent(str, Enum):
    FACT = "fact"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    DEFINITION = "definition"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    ENTITY = "entity"
    AGGREGATION = "aggregation"
    GENERIC = "generic"
    LIST = "list"

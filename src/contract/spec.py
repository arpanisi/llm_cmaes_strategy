from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


class StrategyError(Exception):
    """Base class for contract / validation failures."""


@dataclass
class StrategySpec:
    name: str
    variables: list[str]
    lower: list[float]
    upper: list[float]
    simulate: Callable
    source: str
    description: str = ""

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        return (self.lower, self.upper)

    @property
    def n_variables(self) -> int:
        return len(self.variables)


def _is_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _namespace() -> dict:
    import builtins

    from .. import indicators as ind

    ns = {"np": np, "pd": pd}
    for name in ind.__all__:
        ns[name] = getattr(ind, name)
    ns["round_int"] = ind.round_int
    ns["__builtins__"] = builtins
    return ns


def load_module_source(source: str) -> dict:
    ns = _namespace()
    exec(compile(source, "<strategy>", "exec"), ns)
    return ns

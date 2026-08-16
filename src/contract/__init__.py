from __future__ import annotations

import numpy as np

from .ast_checks import validate_static
from .crash import CrashRecord, RunResult, run_simulate
from .lookahead import check_lookahead
from .spec import StrategyError, StrategySpec, load_module_source

__all__ = [
    "StrategyError",
    "StrategySpec",
    "validate_candidate",
    "load_strategy",
    "run_simulate",
    "CrashRecord",
    "RunResult",
]


def load_strategy(source: str, run_lookahead: bool = True) -> StrategySpec:
    """Static ast validation (checks 1-4), then load the module and return a StrategySpec."""
    declared = validate_static(source)
    ns = load_module_source(source)
    get_strategy = ns.get("get_strategy")
    if get_strategy is None:
        raise StrategyError("validation 1: `get_strategy` not callable after load.")
    try:
        obj = get_strategy()
    except Exception as exc:
        raise StrategyError(
            f"get_strategy() raised at load time: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise StrategyError("validation 2: get_strategy() must return a dict.")
    if "name" not in obj or "variables" not in obj or "bounds" not in obj or "simulate" not in obj:
        raise StrategyError(
            "validation 2: strategy dict must contain keys {name, variables, bounds, simulate}."
        )
    simulate = obj["simulate"]
    if not callable(simulate):
        raise StrategyError("validation 1: `simulate` must be callable.")
    bounds = obj["bounds"]
    if not (isinstance(bounds, tuple) and len(bounds) == 2):
        raise StrategyError("validation 2: `bounds` must be a 2-tuple (lower, upper).")
    lower, upper = bounds[0], bounds[1]
    variables = list(obj["variables"])

    if not (4 <= len(variables) <= 15):
        raise StrategyError(
            f"validation 2: expected 4..15 variables, got {len(variables)}."
        )
    if len(variables) != len(lower) or len(variables) != len(upper):
        raise StrategyError(
            "validation 2: bounds lists must match variables length "
            f"({len(variables)}), got {len(lower)}/{len(upper)}."
        )
    spec = StrategySpec(
        name=obj["name"],
        variables=variables,
        lower=[float(v) for v in lower],
        upper=[float(v) for v in upper],
        simulate=simulate,
        source=source,
        description=str(obj.get("description", "")),
    )
    if run_lookahead:
        check_lookahead(spec)
    return spec


def validate_candidate(source: str) -> StrategySpec:
    """Full Step 2 pipeline: ast checks 1-4, load, prefix-invariance check 5."""
    return load_strategy(source, run_lookahead=True)


def mid_point(spec: StrategySpec) -> np.ndarray:
    return np.array([0.5 * (lo + hi) for lo, hi in zip(spec.lower, spec.upper)])

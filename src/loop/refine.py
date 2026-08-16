from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config.settings import MAX_REFINEMENT_ATTEMPTS
from ..contract import StrategyError, StrategySpec, validate_candidate
from ..core import StrategyCrash
from ..walkforward import CandidateResult
from .context import parse_description


@dataclass
class RoundOutcome:
    ok: bool
    result: CandidateResult | None
    error: str | None = None
    attempts: int = 0


def run_round(
    propose: Callable[[str | None], str],
    evaluate: Callable[[StrategySpec], CandidateResult],
    max_attempts: int = MAX_REFINEMENT_ATTEMPTS,
) -> RoundOutcome:
    """Propose -> validate -> evaluate, feeding errors back up to max_attempts times."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        source = propose(last_error)
        try:
            spec = validate_candidate(source)
        except StrategyError as exc:
            last_error = f"validation error: {exc}"
            continue
        try:
            result = evaluate(spec)
        except StrategyCrash as exc:
            last_error = f"runtime crash: {exc.record.formatted()}"
            continue
        return RoundOutcome(ok=True, result=result, attempts=attempt)
    return RoundOutcome(ok=False, result=None, error=last_error, attempts=max_attempts)


def describe_round(source: str) -> str:
    return parse_description(source) or "(no change comment)"

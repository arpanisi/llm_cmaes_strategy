from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable


@dataclass
class RunResult:
    positions: "np.ndarray"
    num_trades: int


@dataclass
class CrashRecord:
    exception_type: str
    message: str
    traceback: str

    def formatted(self) -> str:
        return f"{self.exception_type}: {self.message}\n{self.traceback}"


def run_simulate(simulate: Callable, close, high, low, volume, macro, x) -> RunResult:
    """Run simulate capturing any runtime exception so it can be fed back to the LLM
    verbatim (Step 2 crash handling), rather than aborting the run."""
    import numpy as np

    try:
        positions, num_trades = simulate(close, high, low, volume, macro, x)
        positions = np.asarray(positions, dtype=float)
        if positions.shape != np.asarray(close).shape:
            raise ValueError(
                f"simulate returned positions of shape {positions.shape}, expected {np.asarray(close).shape}"
            )
    except Exception as exc:
        return CrashRecord(
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )
    return RunResult(positions=positions, num_trades=int(num_trades))

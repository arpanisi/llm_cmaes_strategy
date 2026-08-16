from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_sharpe: float
    max_drawdown: float
    mean_turnover: float


def metrics(returns: np.ndarray, turnover: np.ndarray, bars_per_year: int) -> PerformanceMetrics:
    R = np.asarray(returns, dtype=float)
    if R.size == 0:
        return PerformanceMetrics(total_return=0.0, annualized_sharpe=0.0, max_drawdown=0.0, mean_turnover=0.0)

    total_return = float(np.prod(1.0 + R) - 1.0)
    sd = float(np.std(R, ddof=1))
    sharpe = float(np.mean(R) / sd * np.sqrt(bars_per_year)) if sd > 0 else float("nan")

    equity = np.cumprod(1.0 + R)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_drawdown = float(np.min(dd))

    t = np.asarray(turnover, dtype=float)
    mean_turnover = float(np.mean(t)) if t.size else 0.0
    return PerformanceMetrics(total_return, sharpe, max_drawdown, mean_turnover)

from __future__ import annotations

import numpy as np

from ..config.settings import GROWTH_FLOOR
from ..contract import CrashRecord, StrategySpec, run_simulate
from ..core import BookWindow, StrategyCrash
from ..portfolio import build_portfolio


def objective(x: np.ndarray, spec: StrategySpec, window: BookWindow) -> float:
    """Fitness for one trial parameter vector: -log(portfolio growth factor)."""
    positions = []
    for ticker in window.tickers:
        res = run_simulate(
            spec.simulate,
            window.close[ticker],
            window.high[ticker],
            window.low[ticker],
            window.volume[ticker],
            window.macro,
            x,
        )
        if isinstance(res, CrashRecord):
            raise StrategyCrash(res)
        positions.append(res.positions)
    G = build_portfolio(positions, [window.ret[t] for t in window.tickers]).growth
    return -np.log(max(G, GROWTH_FLOOR))
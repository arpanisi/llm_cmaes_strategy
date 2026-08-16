from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract import CrashRecord, StrategySpec, run_simulate
from ..portfolio import compute_weights, strategy_returns


class PaperTrader:
    """Simulated cash/position ledger updated to match Step 4 target weights each bar.

    No real capital and no broker orders: rebalancing is bookkeeping only. Because the
    system models no transaction costs, the ledger value compounds exactly as Step 4's
    portfolio return series, which is what makes the live path provably the same logic.
    """

    def __init__(self, tickers: list[str], initial_capital: float = 1_000_000.0):
        self.tickers = tickers
        self.cash = initial_capital
        self.shares = {t: 0.0 for t in tickers}
        self.value = initial_capital
        self.weights = {t: 0.0 for t in tickers}
        self._last_weights = np.zeros(len(tickers))

    def _target_shares(self, prices: dict[str, float]) -> dict[str, float]:
        out = {}
        for t, w in zip(self.tickers, list(self.weights.values())):
            px = prices.get(t, np.nan)
            out[t] = self.value * w / px if px and px > 0 else 0.0
        return out

    def rebalance(self, target_weights: np.ndarray, prices: dict[str, float], date) -> dict:
        """Move the ledger to match target weights; returns a bar report."""
        target_value = self.value
        next_shares = self._target_shares(prices)
        next_cash = target_value * max(0.0, 1.0 - float(np.sum(target_weights)))
        self.shares = next_shares
        self.cash = next_cash
        self.weights = {t: float(w) for t, w in zip(self.tickers, target_weights)}
        turnover = 0.5 * float(np.abs(target_weights - self._last_weights).sum())
        self._last_weights = np.asarray(target_weights, dtype=float)
        return {
            "date": date,
            "value": self.value,
            "cash": self.cash,
            "weights": self.weights,
            "turnover": turnover,
        }

    def mark_to_market(self, prices: dict[str, float]) -> float:
        total = self.cash
        for t in self.tickers:
            total += self.shares[t] * prices.get(t, 0.0)
        self.value = total
        return total


def compute_bar(pf_returns: float, prev_value: float) -> float:
    return prev_value * (1.0 + pf_returns)


def simulate_positions(spec: StrategySpec, window, x_star: np.ndarray) -> dict[str, np.ndarray]:
    positions = {}
    for t in window.tickers:
        res = run_simulate(
            spec.simulate,
            window.close[t], window.high[t], window.low[t], window.volume[t],
            window.macro, x_star,
        )
        if isinstance(res, CrashRecord):
            raise RuntimeError(res.formatted())
        positions[t] = res.positions
    return positions

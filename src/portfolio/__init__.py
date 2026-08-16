from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .returns import growth_factor, portfolio_returns, strategy_returns
from .weights import compute_weights, trailing_volatility


@dataclass
class PortfolioResult:
    weights: np.ndarray
    strategy_returns: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray
    growth: float


def build_portfolio(positions: list[np.ndarray], ret: list[np.ndarray]) -> PortfolioResult:
    """Full Step 4 pipeline for a book's tickers on one window."""
    pos = np.stack(positions)
    rets = np.stack(ret)
    W, turnover = compute_weights(pos, rets)
    sr = strategy_returns(pos, rets)
    R = portfolio_returns(W, sr)
    return PortfolioResult(
        weights=W,
        strategy_returns=sr,
        returns=R,
        turnover=turnover,
        growth=growth_factor(R),
    )


__all__ = ["PortfolioResult", "build_portfolio", "trailing_volatility", "compute_weights"]

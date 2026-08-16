from __future__ import annotations

import numpy as np

from ..core import BookWindow


def equal_weight_benchmark(window: BookWindow) -> float:
    """Equal-weight buy-and-hold growth factor over a window: buy every ticker
    equally at start, hold, no rebalancing. Average of per-ticker compounded factors."""
    per_ticker_factors = [
        float(np.prod(1.0 + window.ret[t])) for t in window.tickers
    ]
    return float(np.mean(per_ticker_factors))


def log_sharpe(fold_factors: np.ndarray) -> float:
    """score = mean(log(ff)) - 0.5*std(log(ff)) across folds."""
    ff = np.asarray(fold_factors, dtype=float)
    lf = np.log(np.maximum(ff, 1e-300))
    return float(np.mean(lf) - 0.5 * np.std(lf))

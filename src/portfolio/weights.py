from __future__ import annotations

import numpy as np

from ..config.settings import REBALANCE_EVERY, VOL_WINDOW


def trailing_volatility(ret: np.ndarray) -> np.ndarray:
    """Sample std of ret over the trailing VOL_WINDOW bars ending at t-1.

    The series is shifted one bar so that the value at index t excludes ret[t]:
    the inverse-vol weight applied to bar t's return never uses bar t's own
    return (no same-bar lookahead), matching the codebase's timing convention
    elsewhere (positions[t-1] decided from data <= t-1, applied to ret[t]).
    """
    import pandas as pd

    s = pd.Series(ret)
    return s.rolling(window=VOL_WINDOW, min_periods=2).std(ddof=1).shift(1).to_numpy()


def compute_weights(positions: np.ndarray, ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weight matrix W (n x T) per Step 4 rules.

    Returns (W, turnover): W[i, t] is ticker i's weight applied to bar t's return;
    turnover[t] = 0.5 * sum_i |W[i,t] - W[i,t-1]| (W[i,-1] = 0).
    """
    positions = np.asarray(positions, dtype=float)
    ret = np.asarray(ret, dtype=float)
    n, t_len = positions.shape
    vol = np.stack([trailing_volatility(ret[i]) for i in range(n)])

    W = np.zeros((n, t_len))
    turnover = np.zeros(t_len)
    prev_w = np.zeros(n)

    for t in range(t_len):
        active = positions[:, t - 1] == 1.0 if t > 0 else np.zeros(n, dtype=bool)
        is_rebalance = (t % REBALANCE_EVERY) == 0
        if is_rebalance:
            inv = np.zeros(n)
            valid = active & (vol[:, t] > 0) & np.isfinite(vol[:, t])
            inv[valid] = 1.0 / vol[valid, t]
            total = inv.sum()
            w = np.zeros(n)
            if total > 0:
                w[valid] = inv[valid] / total
        else:
            w = prev_w.copy()
            if t >= 2:
                turned_off = (positions[:, t - 2] == 1.0) & (positions[:, t - 1] == 0.0)
                w[turned_off] = 0.0
        W[:, t] = w
        turnover[t] = 0.5 * float(np.abs(w - prev_w).sum())
        prev_w = w
    return W, turnover

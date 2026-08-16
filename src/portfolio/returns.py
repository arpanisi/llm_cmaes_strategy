from __future__ import annotations

import numpy as np


def strategy_returns(positions: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """r_{i,t} = positions_{i,t-1} * ret_{i,t}; r_{i,0} = 0 (no prior position)."""
    positions = np.asarray(positions, dtype=float)
    ret = np.asarray(ret, dtype=float)
    n, t_len = ret.shape
    r = np.zeros((n, t_len))
    if t_len > 1:
        r[:, 1:] = positions[:, :-1] * ret[:, 1:]
    return r


def portfolio_returns(weights: np.ndarray, strategy_r: np.ndarray) -> np.ndarray:
    """R_t = sum_i W[i,t] * r_{i,t}; 0 when no ticker is active (all weights 0)."""
    weights = np.asarray(weights, dtype=float)
    strategy_r = np.asarray(strategy_r, dtype=float)
    R = np.sum(weights * strategy_r, axis=0)
    return R


def growth_factor(R: np.ndarray) -> float:
    """G = prod_t (1 + R_t)."""
    R = np.asarray(R, dtype=float)
    return float(np.prod(1.0 + R))

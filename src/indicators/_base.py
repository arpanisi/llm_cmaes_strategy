from __future__ import annotations

import numpy as np
import pandas as pd


def as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x, dtype=float))


def sma(x, w: int) -> np.ndarray:
    s = as_series(x)
    return s.rolling(window=w, min_periods=w).mean().to_numpy()


def rolling_max(x, w: int) -> np.ndarray:
    return as_series(x).rolling(window=w, min_periods=w).max().to_numpy()


def rolling_min(x, w: int) -> np.ndarray:
    return as_series(x).rolling(window=w, min_periods=w).min().to_numpy()


def rolling_std(x, w: int, ddof: int = 1) -> np.ndarray:
    return as_series(x).rolling(window=w, min_periods=w).std(ddof=ddof).to_numpy()


def rolling_mean(x, w: int) -> np.ndarray:
    return as_series(x).rolling(window=w, min_periods=w).mean().to_numpy()


def _ewm_seeded(x, w: int, alpha: float) -> np.ndarray:
    """Exponential smoother with alpha, seeded at index w-1 with SMA(x[0:w]).

    Formula matches the recursive definition exactly:
      e[t] = alpha*x[t] + (1-alpha)*e[t-1],  e[w-1] = SMA(x[0:w]).
    Implemented via a causal pandas ewm pass plus a closed-form re-seed
    correction, so values at index t depend only on x[0..t] (prefix-invariant,
    bit-identical on truncated input) and warmup is w-1 NaNs.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    out = np.full(n, np.nan)
    if n < w:
        return out
    f = pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    seed = np.mean(x[0:w])
    shift = seed - f[w - 1]
    decay = (1.0 - alpha) ** np.arange(n - (w - 1))
    out[w - 1:] = f[w - 1:] + decay * shift
    return out


def ema(x, w: int) -> np.ndarray:
    """Exponential moving average, alpha = 2/(w+1), seeded with SMA(x[0:w])."""
    w = int(w)
    if w <= 0:
        return np.full(len(x), np.nan)
    return _ewm_seeded(x, w, 2.0 / (w + 1))


def wilders(x, w: int) -> np.ndarray:
    """Wilder smoothing: EMA with alpha = 1/w, seeded with SMA(x[0:w])."""
    w = int(w)
    if w <= 0:
        return np.full(len(x), np.nan)
    return _ewm_seeded(x, w, 1.0 / w)


def nan_array_like(x) -> np.ndarray:
    return np.full(len(x), np.nan)

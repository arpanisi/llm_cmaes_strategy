from __future__ import annotations

import numpy as np

from ._base import ema, rolling_max, rolling_min, rolling_std, sma, wilders


def indicator_sma(x, w: int) -> np.ndarray:
    """Simple moving average over the trailing w bars. First w-1 values NaN."""
    return sma(x, int(w))


def indicator_ema(x, w: int) -> np.ndarray:
    """EMA(alpha=2/(w+1)), seeded with SMA of first w bars. First w-1 NaN."""
    return ema(x, int(w))


def indicator_macd(close, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray]:
    """(macd line, signal line). macd = EMA(fast)-EMA(slow); signal = EMA(macd, signal)."""
    fast = max(1, int(fast))
    slow = max(1, int(slow))
    signal = max(1, int(signal))
    macd_line = ema(close, fast) - ema(close, slow)
    sig = ema(np.nan_to_num(macd_line, nan=0.0), signal)
    sig = np.where(np.isnan(macd_line), np.nan, sig)
    return macd_line, sig


def indicator_donchian(high, low, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower) = (rolling max of high, rolling min of low) over trailing w bars."""
    w = int(w)
    return rolling_max(high, w), rolling_min(low, w)

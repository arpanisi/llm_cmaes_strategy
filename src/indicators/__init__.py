from __future__ import annotations

import numpy as np

from .flow import indicator_obv as OBV
from .momentum import (
    indicator_adx as ADX,
    indicator_rsi as RSI,
    indicator_stochastic as STOCHASTIC,
)
from .trend import (
    indicator_donchian as DONCHIAN,
    indicator_ema as EMA,
    indicator_macd as MACD,
    indicator_sma as SMA,
)
from .volatility import (
    indicator_atr as ATR,
    indicator_bollinger as BOLLINGER,
    indicator_realized_vol as REALIZED_VOL,
    indicator_zscore as ZSCORE,
)

__all__ = [
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "BOLLINGER",
    "ATR",
    "DONCHIAN",
    "ADX",
    "STOCHASTIC",
    "OBV",
    "REALIZED_VOL",
    "ZSCORE",
]

INDICATOR_DOCS = [
    ("SMA(x, w)", "simple moving average of x over trailing w bars"),
    ("EMA(x, w)", "exponential moving average of x, alpha=2/(w+1), seeded with SMA of first w bars"),
    ("RSI(close, w)", "Wilder's Relative Strength Index, scaled 0..100"),
    ("MACD(close, fast, slow, signal)", "returns (macd_line, signal_line); macd=EMA(fast)-EMA(slow), signal=EMA(macd, signal)"),
    ("BOLLINGER(close, w, k)", "returns (upper, lower) = SMA +/- k*rolling_std(ddof=1)"),
    ("ATR(high, low, close, w)", "Wilder's Average True Range"),
    ("DONCHIAN(high, low, w)", "returns (upper, lower) = rolling max of high, min of low over w bars"),
    ("ADX(high, low, close, w)", "Wilder's Average Directional Index"),
    ("STOCHASTIC(high, low, close, w)", "returns (%K, %D); %K=100*(close-rollmin)/(rollmax-rollmin), %D=3-bar SMA of %K"),
    ("OBV(close, volume)", "On-Balance Volume: cumulative volume signed by sign(close diff)"),
    ("REALIZED_VOL(close, w, bars_per_year)", "annualized realized vol: std(log rets, w)*sqrt(bars_per_year)"),
    ("ZSCORE(x, w)", "(x - rollmean)/rollstd; NaN where rolling std is 0"),
]


def round_int(x) -> int:
    """Round a continuous optimizer value to an integer for window sizing."""
    return max(1, int(round(float(x))))


def _demo_arrays(n: int = 120):
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    close[0] = 100.0
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.uniform(1e6, 5e6, n)
    return close, high, low, volume

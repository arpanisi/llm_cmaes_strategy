from __future__ import annotations

import numpy as np

from ._base import rolling_std, sma, wilders


def indicator_atr(high, low, close, w: int) -> np.ndarray:
    """Wilder's Average True Range. First w values NaN."""
    w = int(w)
    n = len(close)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    out = np.full(n, np.nan)
    if n < w + 1:
        return out
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    out[w:] = wilders(tr, w)[w:]
    return out


def indicator_realized_vol(close, w: int, bars_per_year: int) -> np.ndarray:
    """Annualized realized vol: std(log rets, trailing w) * sqrt(bars_per_year)."""
    w = int(w)
    n = len(close)
    c = np.asarray(close, dtype=float)
    out = np.full(n, np.nan)
    if n < w + 1:
        return out
    log_ret = np.diff(np.log(np.maximum(c, np.finfo(float).tiny)))
    if np.any(np.isinf(log_ret)) or not np.any(np.isfinite(log_ret)):
        return out
    sd = rolling_std(log_ret, w)
    vol = sd * np.sqrt(bars_per_year)
    out[w:] = vol[w - 1:]
    return out


def indicator_bollinger(close, w: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower) = SMA(close,w) +/- k * rolling_std(close,w) (ddof=1)."""
    w = int(w)
    mid = sma(close, w)
    sd = rolling_std(close, w)
    return mid + float(k) * sd, mid - float(k) * sd


def indicator_zscore(x, w: int) -> np.ndarray:
    """(x - rollmean)/rollstd; NaN when rolling std is 0."""
    w = int(w)
    from ._base import rolling_mean

    x = np.asarray(x, dtype=float)
    mean = rolling_mean(x, w)
    sd = rolling_std(x, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (x - mean) / np.where(sd == 0.0, np.nan, sd)
    return out

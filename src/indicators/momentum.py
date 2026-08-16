from __future__ import annotations

import numpy as np

from ._base import rolling_max, rolling_min, sma, wilders


def indicator_rsi(close, w: int) -> np.ndarray:
    """Wilder's RSI scaled to 0..100. First w values NaN."""
    w = int(w)
    n = len(close)
    x = np.asarray(close, dtype=float)
    out = np.full(n, np.nan)
    if n < w + 1:
        return out
    delta = np.diff(x)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = wilders(gain, w)
    avg_loss = wilders(loss, w)
    rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
    out[w:] = (100.0 - 100.0 / (1.0 + rs))[w - 1:]
    return out


def indicator_adx(high, low, close, w: int) -> np.ndarray:
    """Wilder's Average Directional Index. First ~2w-1 values NaN."""
    w = int(w)
    n = len(close)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    out = np.full(n, np.nan)
    if n < 2 * w:
        return out
    prev_h = np.roll(h, 1)
    prev_l = np.roll(l, 1)
    prev_c = np.roll(c, 1)
    prev_h[0] = h[0]
    prev_l[0] = l[0]
    prev_c[0] = c[0]

    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    up = h - prev_h
    dn = prev_l - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    atr = wilders(tr, w)
    plus_di = 100.0 * wilders(plus_dm, w) / np.where(atr == 0.0, np.nan, atr)
    minus_di = 100.0 * wilders(minus_dm, w) / np.where(atr == 0.0, np.nan, atr)
    denom = plus_di + minus_di
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where(denom == 0.0, np.nan, denom)
    adx = wilders(np.nan_to_num(dx, nan=0.0), w)
    first_valid = 2 * w - 1
    out[first_valid:] = adx[first_valid:]
    return out


def indicator_stochastic(high, low, close, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(%K, %D). %K = 100*(close-rollmin)/(rollmax-rollmin); %D = 3-bar SMA of %K."""
    w = int(w)
    hmax = rolling_max(high, w)
    lmin = rolling_min(low, w)
    denom = hmax - lmin
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_k = 100.0 * (np.asarray(close, dtype=float) - lmin) / np.where(denom == 0.0, np.nan, denom)
    pct_d = sma(np.nan_to_num(pct_k, nan=0.0), 3)
    pct_d = np.where(np.isnan(pct_k), np.nan, pct_d)
    return pct_k, pct_d

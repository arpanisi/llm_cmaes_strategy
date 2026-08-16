from __future__ import annotations

import numpy as np

from src import indicators as ind


def test_sma_known():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ind.SMA(x, 3)
    assert np.all(np.isnan(out[:2]))
    np.testing.assert_allclose(out[2:], [2.0, 3.0, 4.0])


def test_ema_warmup_and_prefix():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 120)) + 100
    w = 10
    out = ind.EMA(x, w)
    assert np.all(np.isnan(out[: w - 1]))
    assert not np.any(np.isnan(out[w - 1 :]))
    t = 60
    truncated = ind.EMA(x[: t + 1], w)
    assert out[t] == truncated[-1]


def test_rsi_bounds_and_warmup():
    rng = np.random.default_rng(2)
    close = np.cumsum(rng.normal(0, 1, 100)) + 50
    w = 14
    out = ind.RSI(close, w)
    assert np.all(np.isnan(out[:w]))
    valid = out[w:]
    assert np.all((valid >= 0) & (valid <= 100))
    assert np.all(np.isfinite(valid))


def test_all_indicators_deterministic():
    rng = np.random.default_rng(3)
    n = 150
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    high = close * 1.01
    low = close * 0.99
    volume = rng.uniform(1e5, 1e6, n)
    w = 20

    fns = [
        lambda: ind.SMA(close, w),
        lambda: ind.EMA(close, w),
        lambda: ind.RSI(close, w),
        lambda: ind.MACD(close, 12, 26, 9),
        lambda: ind.BOLLINGER(close, w, 2.0),
        lambda: ind.ATR(high, low, close, w),
        lambda: ind.DONCHIAN(high, low, w),
        lambda: ind.ADX(high, low, close, w),
        lambda: ind.STOCHASTIC(high, low, close, w),
        lambda: ind.OBV(close, volume),
        lambda: ind.REALIZED_VOL(close, w, 252),
        lambda: ind.ZSCORE(close, w),
    ]
    for fn in fns:
        a = fn()
        b = fn()
        if isinstance(a, tuple):
            for a_i, b_i in zip(a, b):
                np.testing.assert_array_equal(a_i, b_i)
        else:
            np.testing.assert_array_equal(a, b)


def test_indicators_prefix_invariant():
    """The property the lookahead check relies on: value at t depends only on data[:t+1]."""
    rng = np.random.default_rng(4)
    n = 150
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    high = close * 1.01
    low = close * 0.99
    volume = rng.uniform(1e5, 1e6, n)
    w = 20
    t = 80

    def check_tuple(fn_full, fn_trunc):
        full = fn_full()
        trunc = fn_trunc()
        for a, b in zip(full, trunc):
            assert a[t] == b[t], "prefix-invariance violated"

    check_tuple(lambda: ind.BOLLINGER(close, w, 2.0), lambda: ind.BOLLINGER(close[: t + 1], w, 2.0))
    check_tuple(lambda: ind.DONCHIAN(high, low, w), lambda: ind.DONCHIAN(high[: t + 1], low[: t + 1], w))
    check_tuple(lambda: ind.STOCHASTIC(high, low, close, w), lambda: ind.STOCHASTIC(high[: t + 1], low[: t + 1], close[: t + 1], w))

    assert ind.SMA(close, w)[t] == ind.SMA(close[: t + 1], w)[t]
    assert ind.EMA(close, w)[t] == ind.EMA(close[: t + 1], w)[t]
    assert ind.RSI(close, w)[t] == ind.RSI(close[: t + 1], w)[t]
    assert ind.ATR(high, low, close, w)[t] == ind.ATR(high[: t + 1], low[: t + 1], close[: t + 1], w)[t]
    assert ind.ADX(high, low, close, w)[t] == ind.ADX(high[: t + 1], low[: t + 1], close[: t + 1], w)[t]
    assert ind.REALIZED_VOL(close, w, 252)[t] == ind.REALIZED_VOL(close[: t + 1], w, 252)[t]
    assert ind.ZSCORE(close, w)[t] == ind.ZSCORE(close[: t + 1], w)[t]
    assert ind.MACD(close, 12, 26, 9)[0][t] == ind.MACD(close[: t + 1], 12, 26, 9)[0][t]


def test_realized_vol_zero_for_constant():
    close = np.full(100, 100.0)
    out = ind.REALIZED_VOL(close, 20, 252)
    assert np.all(np.isnan(out[:20]))
    assert np.all(out[20:] == 0.0)


def test_obv_known():
    close = np.array([10.0, 11.0, 10.5, 10.5])
    volume = np.array([100.0, 200.0, 300.0, 400.0])
    out = ind.OBV(close, volume)
    # +200, -300, 0
    np.testing.assert_allclose(out, [0.0, 200.0, -100.0, -100.0])

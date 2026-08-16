from __future__ import annotations

import numpy as np

from src.portfolio import build_portfolio, compute_weights


def test_single_ticker_always_on_known_answer():
    """One always-on ticker: weights 0 until the first rebalance bar (t=5), then 1."""
    n = 12
    ret = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01, 0.01, 0.005])
    positions = np.ones(n)

    W, turnover = compute_weights(positions.reshape(1, -1), ret.reshape(1, -1))
    assert np.all(W[0, :5] == 0.0)
    assert np.all(W[0, 5:] == 1.0)
    assert turnover[5] == 0.5
    assert np.all(turnover[:5] == 0.0)
    assert np.all(turnover[6:] == 0.0)

    pf = build_portfolio([positions], [ret])
    expected_R = np.zeros(n)
    expected_R[5:] = ret[5:]
    np.testing.assert_allclose(pf.returns, expected_R)
    np.testing.assert_allclose(pf.growth, np.prod(1 + ret[5:]))


def test_all_inactive_is_cash():
    n = 30
    ret = np.random.default_rng(0).normal(0, 0.01, (2, n))
    positions = np.zeros((2, n))
    W, turnover = compute_weights(positions, ret)
    assert np.all(W == 0.0)
    assert np.all(turnover == 0.0)
    pf = build_portfolio([positions[0], positions[1]], [ret[0], ret[1]])
    assert np.all(pf.returns == 0.0)
    assert pf.growth == 1.0


def test_inactive_ticker_contributes_zero():
    rng = np.random.default_rng(1)
    n = 60
    ret_a = rng.normal(0.001, 0.01, n)
    ret_b = rng.normal(-0.001, 0.02, n)
    pos_a = np.ones(n)
    pos_b = np.zeros(n)
    W, _ = compute_weights(np.stack([pos_a, pos_b]), np.stack([ret_a, ret_b]))
    assert np.all(W[1] == 0.0)
    # weights of active ticker A still normalize to 1 on rebalances
    assert np.allclose(W[0][W[0] != 0], 1.0)


def test_weights_inverse_vol():
    rng = np.random.default_rng(2)
    n = 60
    ret_high = rng.normal(0, 0.05, n)
    ret_low = rng.normal(0, 0.005, n)
    pos = np.ones((2, n))
    W, _ = compute_weights(pos, np.stack([ret_high, ret_low]))
    # low-vol ticker should get a much larger weight than the high-vol one at each rebalance
    rebalances = np.arange(0, n, 5)
    ratios = W[1, rebalances] / W[0, rebalances]
    assert np.all(ratios[~np.isnan(ratios)] > 2.0)


def test_inverse_vol_weight_excludes_current_bar_return():
    """Weight at rebalance bar t must use ret[<=t-1] only, never ret[t] (no lookahead)."""
    import pandas as pd

    n = 12
    ret_a = np.array([0.01, 0.011, 0.009, 0.01, 0.0105, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ret_b = np.array([0.02, 0.021, 0.019, 0.02, 0.021, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    positions = np.ones((2, n))
    W, _ = compute_weights(positions, np.stack([ret_a, ret_b]))

    # Expected weight at rebalance bar 5 = inverse-vol of ret[0..4] only (excludes ret[5]).
    va = float(pd.Series(ret_a[:5]).std(ddof=1))
    vb = float(pd.Series(ret_b[:5]).std(ddof=1))
    exp_a = (1.0 / va) / (1.0 / va + 1.0 / vb)
    assert np.isclose(W[0, 5], exp_a)

    # Changing the same bar's own return must not change that bar's weight.
    ret_a2 = ret_a.copy()
    ret_a2[5] = 0.80
    W2, _ = compute_weights(positions, np.stack([ret_a2, ret_b]))
    assert np.isclose(W2[0, 5], W[0, 5])
    assert np.isclose(W2[1, 5], W[1, 5])

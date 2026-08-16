from __future__ import annotations

import numpy as np

from src.config import RunConfig
from src.walkforward import (
    equal_weight_benchmark,
    evaluate_strategy,
    log_sharpe,
    make_folds,
    make_window,
    metrics,
)
from tests.synthetic import make_synthetic_aligned

TICKERS = ["AAA", "BBB"]


def test_make_folds_non_overlapping_contiguous():
    folds = make_folds(n=500, train=200, test=40, step=40)
    assert folds[0].test_start == 200
    assert folds[0].test_end == 240
    for a, b in zip(folds[:-1], folds[1:]):
        assert b.test_start == a.test_end
    assert folds[-1].test_end <= 500


def test_log_sharpe_known():
    factors = np.array([1.1, 0.9, 1.05])
    lf = np.log(factors)
    expected = float(np.mean(lf) - 0.5 * np.std(lf))
    assert log_sharpe(factors) == expected


def test_benchmark_known():
    aligned = make_synthetic_aligned(TICKERS, n=20, seed=1)
    win = make_window(aligned, TICKERS, slice(0, 20))
    per_ticker_factors = [
        float(np.prod(1.0 + win.ret[t])) for t in TICKERS
    ]
    expected = float(np.mean(per_ticker_factors))
    assert np.isclose(equal_weight_benchmark(win), expected)


def test_metrics_known():
    R = np.array([0.1, -0.05, 0.2])
    turnover = np.array([0.1, 0.2, 0.3])
    m = metrics(R, turnover, bars_per_year=252)
    assert np.isclose(m.total_return, np.prod(1 + R) - 1)
    eq = np.cumprod(1 + R)
    assert np.isclose(m.max_drawdown, np.min(eq / np.maximum.accumulate(eq) - 1))
    assert np.isclose(m.mean_turnover, 0.2)
    sd = np.std(R, ddof=1)
    assert np.isclose(m.annualized_sharpe, np.mean(R) / sd * np.sqrt(252))


def test_make_window_continuous_returns_across_fold_slice():
    aligned = make_synthetic_aligned(TICKERS, n=100, seed=42)
    win = make_window(aligned, TICKERS, slice(20, 50))
    for t in TICKERS:
        c_curr = aligned.iloc[20][f"{t}_close"]
        c_prev = aligned.iloc[19][f"{t}_close"]
        expected_ret_0 = float(c_curr / c_prev - 1.0)
        assert np.isclose(win.ret[t][0], expected_ret_0)
        assert win.ret[t][0] != 0.0


def test_test_fold_lookback_buffer_prevents_nan():
    from src.contract import validate_candidate
    from src.seeds import seed_sources
    from src.walkforward.evaluate import _eval_test_fold

    aligned = make_synthetic_aligned(TICKERS, n=300, seed=42)
    folds = make_folds(n=300, train=200, test=40, step=40)
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])

    # x_star with sma_period = 90.0, which exceeds test fold length of 40
    x_star = np.array([5.0, 90.0, 10.0, 0.05])

    fold = folds[0]
    test_win = make_window(aligned, TICKERS, fold.test_slice)

    # Without lookback buffer -> indicators all NaN, zero trades
    pf_unbuffered = _eval_test_fold(spec, test_win, x_star)
    assert pf_unbuffered.growth == 1.0
    assert float(np.mean(pf_unbuffered.turnover)) == 0.0

    # With lookback buffer -> indicators warmed up, real trades executed
    pf_buffered = _eval_test_fold(spec, test_win, x_star, aligned=aligned, fold=fold)
    assert pf_buffered.growth != 1.0
    assert float(np.mean(pf_buffered.turnover)) > 0.0


def test_to_dict_persists_full_fold_records():
    """Step 6 acceptance: per-fold breakdown (dates, x*, growth, turnover) is persisted."""
    from src.contract import validate_candidate
    from src.seeds import seed_sources

    aligned = make_synthetic_aligned(TICKERS, n=300, seed=11)
    cfg = RunConfig(book="equity", seed=1, train_bars=200, test_bars=40, step_bars=40,
                    cma_restarts=1, cma_evals=30, tickers_override=TICKERS)
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])
    res = evaluate_strategy(spec, aligned, cfg)
    d = res.to_dict()
    assert d["n_folds"] >= 1
    assert len(d["fold_records"]) == d["n_folds"]
    for rec in d["fold_records"]:
        for key in ("train_date_start", "train_date_end", "test_date_start", "test_date_end",
                    "x_star", "raw_factor", "benchmark_factor", "factor", "turnover"):
            assert key in rec
        assert rec["test_date_start"] is not None
        assert len(rec["x_star"]) >= 1
        assert isinstance(rec["raw_factor"], float)

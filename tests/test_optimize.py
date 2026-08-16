from __future__ import annotations

import numpy as np

from src.contract import validate_candidate
from src.optimize import calibration_check, optimize_strategy
from src.seeds import seed_sources
from src.walkforward import evaluate_strategy, make_folds, make_window
from src.config import RunConfig
from tests.synthetic import make_synthetic_aligned

TICKERS = ["AAA", "BBB", "CCC"]


def _cfg(book="equity", **kw) -> RunConfig:
    return RunConfig(book=book, seed=1, tickers_override=tuple(TICKERS), **kw)


def test_optimizer_in_bounds():
    aligned = make_synthetic_aligned(TICKERS, n=160, seed=3)
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])
    win = make_window(aligned, TICKERS, slice(0, 160))
    res = optimize_strategy(spec, win, seed=1, n_restarts=2, evals_per_restart=50)
    assert np.all(res.x >= spec.lower)
    assert np.all(res.x <= spec.upper)
    assert np.isfinite(res.value)


def test_calibration_check_runs():
    aligned = make_synthetic_aligned(TICKERS, n=160, seed=4)
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])
    win = make_window(aligned, TICKERS, slice(0, 160))
    out = calibration_check(spec, win, seeds=[11, 22, 33], n_restarts=2, evals_per_restart=50)
    assert len(out["values"]) == 3
    assert np.isfinite(out["spread"])


def test_evaluate_strategy_end_to_end_tiny():
    aligned = make_synthetic_aligned(TICKERS, n=300, seed=5)
    cfg = _cfg(train_bars=200, test_bars=40, step_bars=40, cma_restarts=1, cma_evals=30)
    spec = validate_candidate(seed_sources(252)["seed2_macro_trend"])
    folds = make_folds(len(aligned), cfg.train_bars, cfg.test_bars, cfg.step_bars)
    assert len(folds) >= 1
    res = evaluate_strategy(spec, aligned, cfg)
    assert len(res.folds) == len(folds)
    assert np.isfinite(res.score)
    assert res.last_fold_x is not None
    assert np.all(np.array(res.last_fold_x) >= spec.lower)
    assert np.all(np.array(res.last_fold_x) <= spec.upper)


def test_cmaes_restarts_use_independent_random_initializations(monkeypatch):
    captured_x0s = []
    import cma

    orig_init = cma.CMAEvolutionStrategy.__init__

    def mock_init(self, x0, sigma0, inopts=None):
        captured_x0s.append(np.array(x0, dtype=float))
        orig_init(self, x0, sigma0, inopts)

    monkeypatch.setattr(cma.CMAEvolutionStrategy, "__init__", mock_init)

    aligned = make_synthetic_aligned(TICKERS, n=160, seed=3)
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])
    win = make_window(aligned, TICKERS, slice(0, 160))

    seed = 42
    n_restarts = 3
    optimize_strategy(spec, win, seed=seed, n_restarts=n_restarts, evals_per_restart=20)

    assert len(captured_x0s) == n_restarts
    for r, x0 in enumerate(captured_x0s):
        expected_x0 = np.random.default_rng(seed + r).uniform(spec.lower, spec.upper)
        np.testing.assert_allclose(x0, expected_x0)
        assert np.all(x0 >= spec.lower)
        assert np.all(x0 <= spec.upper)

    assert not np.allclose(captured_x0s[0], captured_x0s[1])

from __future__ import annotations

import numpy as np

from src.config import RunConfig
from src.contract import run_simulate, validate_candidate
from src.live import replay_paper
from src.portfolio import compute_weights, strategy_returns
from src.seeds import seed_sources
from src.walkforward import evaluate_strategy, make_folds, make_window
from tests.synthetic import make_synthetic_aligned

TICKERS = ["AAA", "BBB", "CCC"]


def _cfg() -> RunConfig:
    return RunConfig(book="equity", train_bars=120, test_bars=30, step_bars=30,
                     cma_restarts=1, cma_evals=40, seed=1,
                     tickers_override=tuple(TICKERS))


def test_live_replay_matches_backtest():
    """Step 9 acceptance: replaying the frozen x* bar-by-bar over a window must
    reproduce the exact returns/turnover Step 6's backtest produces on that window."""
    cfg = _cfg()
    aligned = make_synthetic_aligned(TICKERS, n=200, seed=11)
    spec = validate_candidate(seed_sources(252)["seed2_macro_trend"])
    res = evaluate_strategy(spec, aligned, cfg)
    x_star = np.asarray(res.last_fold_x)

    folds = make_folds(len(aligned), cfg.train_bars, cfg.test_bars, cfg.step_bars)
    last = folds[-1]
    window_df = aligned.iloc[last.test_start:last.test_end]
    reports = replay_paper(cfg, spec, x_star, window_df, start_index=0)

    win = make_window(aligned, cfg.tickers, last.test_slice)
    pos = {
        t: run_simulate(
            spec.simulate, win.close[t], win.high[t], win.low[t],
            win.volume[t], win.macro, x_star,
        ).positions
        for t in cfg.tickers
    }
    pm = np.stack([pos[t] for t in cfg.tickers])
    rm = np.stack([win.ret[t] for t in cfg.tickers])
    W, turn = compute_weights(pm, rm)
    sr = strategy_returns(pm, rm)
    R = np.sum(W * sr, axis=0)

    rep_returns = np.array([r["return"] for r in reports])
    rep_turn = np.array([r["turnover"] for r in reports])
    assert len(rep_returns) == len(R)
    np.testing.assert_allclose(rep_returns, R)
    np.testing.assert_allclose(rep_turn, turn)
    assert reports[-1]["value"] > 0

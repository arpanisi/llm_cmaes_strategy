from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..config.settings import BOOKS
from ..contract import CrashRecord, StrategySpec, run_simulate
from ..core import BookWindow, StrategyCrash
from ..optimize import OptimizeResult, optimize_strategy
from ..portfolio import build_portfolio
from .folds import Fold, make_folds, make_window
from .metrics import PerformanceMetrics, metrics
from .scoring import equal_weight_benchmark, log_sharpe


def _fmt_date(v) -> str | None:
    if v is None:
        return None
    try:
        return v.strftime("%Y-%m-%d")
    except AttributeError:
        return str(v)


@dataclass
class FoldRecord:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_date_start: object
    train_date_end: object
    test_date_start: object
    test_date_end: object
    x_star: list[float]
    raw_factor: float
    benchmark_factor: float
    factor: float
    turnover: float

    def to_dict(self) -> dict:
        """Full per-fold breakdown for persistence (Step 6 acceptance)."""
        return {
            "fold": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_date_start": _fmt_date(self.train_date_start),
            "train_date_end": _fmt_date(self.train_date_end),
            "test_date_start": _fmt_date(self.test_date_start),
            "test_date_end": _fmt_date(self.test_date_end),
            "x_star": list(self.x_star),
            "raw_factor": self.raw_factor,
            "benchmark_factor": self.benchmark_factor,
            "factor": self.factor,
            "turnover": self.turnover,
        }


@dataclass
class CandidateResult:
    spec: StrategySpec
    folds: list[FoldRecord] = field(default_factory=list)
    score: float = float("nan")
    metrics: PerformanceMetrics | None = None
    factors: np.ndarray = None
    last_fold_x: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.spec.name,
            "score": self.score,
            "metrics": {
                "total_return": self.metrics.total_return if self.metrics else None,
                "annualized_sharpe": self.metrics.annualized_sharpe if self.metrics else None,
                "max_drawdown": self.metrics.max_drawdown if self.metrics else None,
                "mean_turnover": self.metrics.mean_turnover if self.metrics else None,
            },
            "fold_factors": [f.factor for f in self.folds],
            "fold_records": [f.to_dict() for f in self.folds],
            "last_fold_x": list(self.last_fold_x) if self.last_fold_x is not None else None,
            "n_folds": len(self.folds),
        }


def _eval_test_fold(
    spec: StrategySpec,
    test_win: BookWindow,
    x: np.ndarray,
    aligned: pd.DataFrame | None = None,
    fold: Fold | None = None,
    lookback: int = 200,
):
    if aligned is not None and fold is not None:
        buf_start = max(0, fold.test_start - lookback)
        buf_len = fold.test_start - buf_start
        buf_win = make_window(aligned, test_win.tickers, slice(buf_start, fold.test_end))
        eval_win = buf_win
    else:
        buf_len = 0
        eval_win = test_win

    positions = []
    for ticker in test_win.tickers:
        res = run_simulate(
            spec.simulate,
            eval_win.close[ticker],
            eval_win.high[ticker],
            eval_win.low[ticker],
            eval_win.volume[ticker],
            eval_win.macro,
            x,
        )
        if isinstance(res, CrashRecord):
            raise StrategyCrash(res)
        positions.append(res.positions[buf_len:])
    return build_portfolio(positions, [test_win.ret[t] for t in test_win.tickers])


def evaluate_strategy(
    spec: StrategySpec,
    aligned: pd.DataFrame,
    cfg: RunConfig,
    fold_list: list[Fold] | None = None,
) -> CandidateResult:
    tickers = cfg.tickers
    n = len(aligned)
    folds = fold_list if fold_list is not None else make_folds(n, cfg.train_bars, cfg.test_bars, cfg.step_bars)

    records = []
    raw_factors = []
    adj_factors = []
    test_R = []
    test_turnover = []
    last_x = None

    for fold in folds:
        train_win = make_window(aligned, tickers, fold.train_slice)
        test_win = make_window(aligned, tickers, fold.test_slice)
        opt: OptimizeResult = optimize_strategy(
            spec,
            train_win,
            seed=cfg.seed,
            n_restarts=cfg.cma_restarts,
            evals_per_restart=cfg.cma_evals,
        )
        x_star = opt.x
        last_x = x_star

        pf = _eval_test_fold(spec, test_win, x_star, aligned=aligned, fold=fold)
        raw_factor = pf.growth
        benchmark = equal_weight_benchmark(test_win) if cfg.book in BOOKS and cfg.book == "crypto" else 1.0
        adj_factor = raw_factor / benchmark if benchmark > 0 else raw_factor

        records.append(
            FoldRecord(
                index=fold.index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_date_start=aligned.index[fold.train_start],
                train_date_end=aligned.index[fold.train_end - 1],
                test_date_start=aligned.index[fold.test_start],
                test_date_end=aligned.index[fold.test_end - 1],
                x_star=[float(v) for v in x_star],
                raw_factor=raw_factor,
                benchmark_factor=benchmark,
                factor=adj_factor,
                turnover=float(np.mean(pf.turnover)),
            )
        )
        raw_factors.append(raw_factor)
        adj_factors.append(adj_factor)
        test_R.append(pf.returns)
        test_turnover.append(pf.turnover)

    if not records:
        return CandidateResult(spec=spec)

    factors = np.array(adj_factors)
    score = log_sharpe(factors)
    all_R = np.concatenate(test_R) if test_R else np.array([])
    all_T = np.concatenate(test_turnover) if test_turnover else np.array([])
    perf = metrics(all_R, all_T, cfg.bars_per_year)

    return CandidateResult(
        spec=spec,
        folds=records,
        score=score,
        metrics=perf,
        factors=factors,
        last_fold_x=last_x,
    )

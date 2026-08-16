from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RunConfig
from ..contract import StrategySpec
from ..portfolio import compute_weights, strategy_returns
from ..walkforward import make_window
from .feed import LiveSource, ReplaySource
from .paper import PaperTrader, simulate_positions


def _step(trader: PaperTrader, spec: StrategySpec, cfg: RunConfig, aligned: pd.DataFrame,
          k: int, x_star: np.ndarray) -> dict:
    window = make_window(aligned, cfg.tickers, slice(0, k + 1))
    pos = simulate_positions(spec, window, x_star)
    pos_mat = np.stack([pos[t] for t in cfg.tickers])
    ret_mat = np.stack([window.ret[t] for t in cfg.tickers])
    W, _ = compute_weights(pos_mat, ret_mat)
    sr = strategy_returns(pos_mat, ret_mat)
    R_k = float(np.sum(W[:, -1] * sr[:, -1]))
    trader.value = trader.value * (1.0 + R_k)
    prices = {t: float(window.close[t][-1]) for t in cfg.tickers}
    date = aligned.index[k]
    report = trader.rebalance(W[:, -1], prices, date)
    report["return"] = R_k
    return report


def replay_paper(cfg: RunConfig, spec: StrategySpec, x_star: np.ndarray,
                 aligned: pd.DataFrame, start_index: int,
                 initial_capital: float = 1_000_000.0) -> list[dict]:
    """Run the frozen strategy bar-by-bar over a historical window (Step 9 replay)."""
    trader = PaperTrader(cfg.tickers, initial_capital)
    reports = []
    source = ReplaySource(aligned, next_index=start_index)
    k = start_index
    while source.next() is not None:
        reports.append(_step(trader, spec, cfg, aligned, k, x_star))
        k += 1
    return reports


def live_run(cfg: RunConfig, spec: StrategySpec, x_star: np.ndarray,
             aligned: pd.DataFrame, initial_capital: float = 1_000_000.0) -> list[dict]:
    """Incremental live paper execution, appending each new bar before stepping."""
    from ..data.fred import fetch_macro_series

    trader = PaperTrader(cfg.tickers, initial_capital)
    source = LiveSource(cfg.book, cfg.tickers)
    macro_value = float(fetch_macro_series("2019-01-01", "2100-01-01").iloc[-1])
    reports = []
    k = len(aligned)
    while True:
        new_bar = source.next()
        if new_bar is None:
            break
        row = _bar_to_row(new_bar, cfg.tickers, macro_value)
        aligned = pd.concat([aligned, row])
        reports.append(_step(trader, spec, cfg, aligned, k, x_star))
        k += 1
    return reports


def _bar_to_row(new_bar: dict, tickers: list[str], macro_value: float) -> pd.DataFrame:
    idx = pd.to_datetime(new_bar[tickers[0]]["date"])
    data = {}
    for t in tickers:
        b = new_bar[t]
        data[f"{t}_close"] = b["close"]
        data[f"{t}_high"] = b["high"]
        data[f"{t}_low"] = b["low"]
        data[f"{t}_volume"] = b["volume"]
    data["macro"] = macro_value
    return pd.DataFrame([data], index=[idx])

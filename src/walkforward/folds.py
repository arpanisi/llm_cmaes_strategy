from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import BookWindow


@dataclass
class Fold:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)


def make_folds(n: int, train: int, test: int, step: int) -> list[Fold]:
    folds = []
    k = 0
    while True:
        train_start = k * step
        train_end = train_start + train
        test_start = train_end
        test_end = test_start + test
        if test_end > n:
            break
        folds.append(Fold(k, train_start, train_end, test_start, test_end))
        k += 1
    return folds


def make_window(aligned: pd.DataFrame, tickers: list[str], s: slice) -> BookWindow:
    rows = aligned.iloc[s]
    close = {}
    high = {}
    low = {}
    volume = {}
    ret = {}
    for t in tickers:
        c_full = aligned[f"{t}_close"].to_numpy(dtype=float)
        r_full = np.zeros_like(c_full)
        if len(c_full) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                r_full[1:] = c_full[1:] / c_full[:-1] - 1.0
        r_full = np.nan_to_num(r_full, nan=0.0, posinf=0.0, neginf=0.0)
        ret[t] = r_full[s]

        close[t] = rows[f"{t}_close"].to_numpy(dtype=float)
        high[t] = rows[f"{t}_high"].to_numpy(dtype=float)
        low[t] = rows[f"{t}_low"].to_numpy(dtype=float)
        volume[t] = rows[f"{t}_volume"].to_numpy(dtype=float)

    macro = rows["macro"].to_numpy(dtype=float)
    return BookWindow(close=close, high=high, low=low, volume=volume, macro=macro, ret=ret, tickers=tickers)


def make_full_window(aligned: pd.DataFrame, tickers: list[str]) -> BookWindow:
    return make_window(aligned, tickers, slice(0, len(aligned)))

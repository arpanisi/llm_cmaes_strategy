from __future__ import annotations

import pandas as pd

from .yahoo import load_ticker


def _book_calendar(book: str, tickers: list[str], start: str, end: str, force: bool = False) -> "pd.DatetimeIndex":
    dates = []
    for t in tickers:
        df = load_ticker(book, t, start, end, force=force)
        dates.append(df.index)
    union = pd.DatetimeIndex(sorted(set().union(*[set(d) for d in dates])))
    return union


def load_book_aligned(book: str, tickers: list[str], start: str, end: str,
                      macro: "pd.Series" | None = None, force: bool = False) -> "pd.DataFrame":
    """Return one DataFrame on the book's shared bar calendar.

    Columns: f"{ticker}_close|high|low|volume" for each ticker, plus "macro".
    Missing bars (rare gap) are forward-filled for price columns, 0 for volume.
    """
    calendar = _book_calendar(book, tickers, start, end, force=force)
    out = pd.DataFrame(index=calendar)
    for t in tickers:
        df = load_ticker(book, t, start, end, force=force)
        df = df.reindex(calendar)
        for col in ("close", "high", "low"):
            df[col] = df[col].ffill()
        df["volume"] = df["volume"].fillna(0.0)
        for col in ("close", "high", "low", "volume"):
            out[f"{t}_{col}"] = df[col].values
    if macro is not None:
        out["macro"] = macro.reindex(calendar).ffill().values
    else:
        out["macro"] = pd.NA
    return out


def ticker_arrays(aligned: "pd.DataFrame", tickers: list[str], ticker: str
                  ) -> dict[str, "np.ndarray"]:
    import numpy as np

    return {
        "close": np.asarray(aligned[f"{ticker}_close"].to_numpy(dtype=float)),
        "high": np.asarray(aligned[f"{ticker}_high"].to_numpy(dtype=float)),
        "low": np.asarray(aligned[f"{ticker}_low"].to_numpy(dtype=float)),
        "volume": np.asarray(aligned[f"{ticker}_volume"].to_numpy(dtype=float)),
        "macro": np.asarray(aligned["macro"].to_numpy(dtype=float)),
    }

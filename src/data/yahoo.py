from __future__ import annotations

import json

import pandas as pd

from ..config import paths
from ..config.settings import DATE_END, DATE_START


def fetch_ticker_history(ticker: str, start: str, end: str) -> "pd.DataFrame":
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker} in [{start}, {end})")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _cache_matches_span(path, meta) -> bool:
    if not path.exists() or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()) == {"start": DATE_START, "end": DATE_END}
    except (ValueError, OSError):
        return False


def load_ticker(book: str, ticker: str, start: str, end: str, force: bool = False) -> "pd.DataFrame":
    """Full-span (DATE_START..DATE_END) OHLCV, sliced to [start, end).

    The on-disk file always holds the full book span (verified by a sidecar
    metadata file), so any requested range is a pure slice and the data is
    downloaded only once. A file written by an older partial-range fetch or for a
    different span is detected and refetched.
    """
    from .io import read_csv, write_csv

    path = paths.ticker_ohlcv_file(book, ticker)
    meta = paths.ticker_ohlcv_meta_file(book, ticker)

    df = None
    if not force and _cache_matches_span(path, meta):
        df = read_csv(path)

    if df is None:
        df = fetch_ticker_history(ticker, DATE_START, DATE_END)
        write_csv(path, df)
        meta.write_text(json.dumps({"start": DATE_START, "end": DATE_END}))

    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return df.loc[(df.index >= lo) & (df.index < hi)]

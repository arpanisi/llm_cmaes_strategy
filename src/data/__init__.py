from __future__ import annotations

from ..config import RunConfig, paths
from .alignment import load_book_aligned, ticker_arrays
from .fred import MissingAPIKeyError, fetch_macro_series, load_macro
from .io import read_csv, write_csv
from .yahoo import fetch_ticker_history, load_ticker


def load_book_data(cfg: RunConfig, force: bool = False) -> "pd.DataFrame":
    """Step 1 end-to-end for one book: OHLCV per ticker + aligned macro."""
    import pandas as pd

    paths.ensure_dirs()
    macro = load_macro(cfg.date_start, cfg.date_end, force=force)
    return load_book_aligned(cfg.book, cfg.tickers, cfg.date_start, cfg.date_end, macro=macro, force=force)


__all__ = [
    "load_book_aligned",
    "ticker_arrays",
    "load_book_data",
    "read_csv",
    "write_csv",
    "MissingAPIKeyError",
    "fetch_macro_series",
    "load_macro",
    "fetch_ticker_history",
    "load_ticker",
]

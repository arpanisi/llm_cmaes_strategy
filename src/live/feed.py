from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ReplaySource:
    """Replays a historical window bar-by-bar (used for the Step 9 acceptance check)."""

    aligned: pd.DataFrame
    next_index: int = 0

    def next(self) -> pd.Series | None:
        if self.next_index >= len(self.aligned):
            return None
        row = self.aligned.iloc[self.next_index]
        self.next_index += 1
        return row


class LiveSource:
    """Fetches each ticker's latest bar from yfinance (same source as Step 1)."""

    def __init__(self, book: str, tickers: list[str]):
        self.book = book
        self.tickers = tickers
        self.last_dates: dict[str, str] = {}

    def _latest_bars(self) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        out = {}
        for t in self.tickers:
            df = yf.Ticker(t).history(period="5d", interval="1d", auto_adjust=True)
            if df.empty:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            out[t] = df
        return out

    def next(self) -> dict[str, dict[str, float]] | None:
        """Return the newest bar (ticker -> ohlcv) once ALL tickers report the same
        new date (the book's shared calendar); otherwise None."""
        bars = self._latest_bars()
        dates = {}
        for t, df in bars.items():
            if df.empty:
                return None
            dates[t] = str(df.index[-1].date())
        if len(set(dates.values())) != 1:
            return None
        date = dates[self.tickers[0]]
        if self.last_dates.get(self.tickers[0]) == date:
            return None
        new_bars: dict[str, dict[str, float]] = {}
        for t, df in bars.items():
            row = df.iloc[-1]
            new_bars[t] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "date": date,
            }
            self.last_dates[t] = date
        return new_bars if new_bars else None

from __future__ import annotations

from dataclasses import dataclass

from ..contract import CrashRecord


@dataclass
class BookWindow:
    """One window over the book's shared calendar: per-ticker arrays + macro + returns."""

    close: dict[str, "np.ndarray"]
    high: dict[str, "np.ndarray"]
    low: dict[str, "np.ndarray"]
    volume: dict[str, "np.ndarray"]
    macro: "np.ndarray"
    ret: dict[str, "np.ndarray"]
    tickers: list[str]

    def for_ticker(self, ticker: str) -> tuple["np.ndarray", ...]:
        return (
            self.close[ticker],
            self.high[ticker],
            self.low[ticker],
            self.volume[ticker],
        )


class StrategyCrash(RuntimeError):
    def __init__(self, record: CrashRecord):
        self.record = record
        super().__init__(record.formatted())


__all__ = ["BookWindow", "StrategyCrash"]

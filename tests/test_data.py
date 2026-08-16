from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import paths as config_paths
from src.config.settings import DATE_END, DATE_START
from src.data import fred, yahoo
from src.data.io import write_csv


def _full_span_df() -> pd.DataFrame:
    idx = pd.date_range(DATE_START, DATE_END, freq="D")
    n = len(idx)
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1e6},
        index=idx,
    )


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config_paths, "ticker_ohlcv_file", lambda book, t: tmp_path / f"{book}_{t}.csv")
    monkeypatch.setattr(config_paths, "ticker_ohlcv_meta_file", lambda book, t: tmp_path / f"{book}_{t}.meta.json")
    monkeypatch.setattr(config_paths, "macro_series_file", lambda: tmp_path / "macro.csv")
    monkeypatch.setattr(config_paths, "macro_series_meta_file", lambda: tmp_path / "macro.meta.json")
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    return tmp_path


def test_ticker_cache_slices_range_and_downloads_once(fake_cache, monkeypatch):
    calls = []

    def fake_fetch(ticker, start, end):
        calls.append((ticker, start, end))
        return _full_span_df()

    monkeypatch.setattr(yahoo, "fetch_ticker_history", fake_fetch)

    a = yahoo.load_ticker("equity", "AAA", "2020-06-01", "2021-06-01")
    b = yahoo.load_ticker("equity", "AAA", "2022-01-01", "2022-06-01")

    assert len(calls) == 1  # downloaded once, full span
    assert calls[0][1] == DATE_START and calls[0][2] == DATE_END
    # [start, end) slicing
    assert str(a.index[0])[:10] == "2020-06-01"
    assert str(a.index[-1])[:10] < "2021-06-01"
    assert len(a) == len(pd.date_range("2020-06-01", "2021-05-31", freq="D"))
    assert str(b.index[0])[:10] == "2022-01-01"


def test_ticker_cache_refetches_stale_partial(fake_cache, monkeypatch):
    calls = []

    def fake_fetch(ticker, start, end):
        calls.append(1)
        return _full_span_df()

    monkeypatch.setattr(yahoo, "fetch_ticker_history", fake_fetch)

    # legacy partial-range cache with no sidecar metadata
    partial = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                            "close": [1.0], "volume": [1e6]},
                           index=pd.to_datetime(["2019-06-01"]))
    write_csv(config_paths.ticker_ohlcv_file("equity", "AAA"), partial)

    out = yahoo.load_ticker("equity", "AAA", "2020-01-01", "2020-02-01")
    assert len(calls) == 1  # refetched because the cache lacks span metadata
    assert len(out) == len(pd.date_range("2020-01-01", "2020-01-31", freq="D"))


def test_macro_cache_slices_inclusive(fake_cache, monkeypatch):
    idx = pd.date_range(DATE_START, DATE_END, freq="D")
    series = pd.Series(np.linspace(0.1, 0.4, len(idx)), index=idx)

    def fake_fetch(start, end):
        return series

    monkeypatch.setattr(fred, "fetch_macro_series", fake_fetch)

    out = fred.load_macro("2020-06-01", "2020-06-30")
    assert str(out.index[0])[:10] == "2020-06-01"
    assert str(out.index[-1])[:10] == "2020-06-30"  # inclusive end for FRED
    # second call reuses the cache (no refetch happens; series object identity)
    out2 = fred.load_macro("2021-01-01", "2021-03-01")
    assert str(out2.index[0])[:10] == "2021-01-01"
    meta = json.loads(config_paths.macro_series_meta_file().read_text())
    assert meta == {"start": DATE_START, "end": DATE_END}

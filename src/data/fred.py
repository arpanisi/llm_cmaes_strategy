from __future__ import annotations

import json
import os

import pandas as pd

from ..config import paths
from ..config.settings import DATE_END, DATE_START, FRED_API_KEY_ENV, FRED_SERIES


class MissingAPIKeyError(RuntimeError):
    pass


def _require_api_key() -> str:
    key = os.environ.get(FRED_API_KEY_ENV, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"Environment variable {FRED_API_KEY_ENV} is unset or empty. "
            f"Set it to a FRED API key (free, one-time registration at "
            f"https://fred.stlouisfed.org/docs/api/api_key.html) before running. "
            f"The system will not silently fall back to a cached or stale macro series."
        )
    return key


def fetch_macro_series(start: str, end: str) -> "pd.Series":
    import requests

    key = _require_api_key()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": FRED_SERIES,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    rows = []
    for obs in resp.json()["observations"]:
        date = obs["date"]
        value = obs["value"]
        if value == ".":
            continue
        rows.append((pd.Timestamp(date), float(value)))
    series = pd.Series(dict(rows), name=FRED_SERIES)
    series.index.name = "date"
    series.sort_index(inplace=True)
    return series


def load_macro(start: str, end: str, force: bool = False) -> "pd.Series":
    from .io import read_csv, write_csv

    _require_api_key()  # fail fast at startup: key required even if a file exists
    path = paths.macro_series_file()
    meta = paths.macro_series_meta_file()

    df = None
    if not force and path.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == {"start": DATE_START, "end": DATE_END}:
                df = read_csv(path)
        except (ValueError, OSError):
            df = None

    if df is None:
        df = fetch_macro_series(DATE_START, DATE_END).to_frame()
        write_csv(path, df)
        meta.write_text(json.dumps({"start": DATE_START, "end": DATE_END}))

    s = df[df.columns[0]]
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    return s.loc[(s.index >= lo) & (s.index <= hi)]

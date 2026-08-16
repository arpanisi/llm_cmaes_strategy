from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_aligned(tickers: list[str], n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {}
    for i, t in enumerate(tickers):
        ret = rng.normal(0.0004, 0.01 + 0.005 * i, n)
        close = 100 * np.exp(np.cumsum(ret))
        spread = np.abs(rng.normal(0, 0.003, n))
        data[f"{t}_close"] = close
        data[f"{t}_high"] = close * (1 + spread)
        data[f"{t}_low"] = close * (1 - spread)
        data[f"{t}_volume"] = rng.uniform(1e5, 1e6, n)
    data["macro"] = 0.3 * np.sin(np.linspace(0, 8 * np.pi, n))
    return pd.DataFrame(data, index=idx)

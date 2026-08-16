from __future__ import annotations

from pathlib import Path


def read_csv(path: Path) -> "pd.DataFrame":
    import pandas as pd

    return pd.read_csv(path, index_col=0, parse_dates=True)


def write_csv(path: Path, df: "pd.DataFrame") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

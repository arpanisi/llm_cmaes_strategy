from __future__ import annotations

import numpy as np

from src.config import RunConfig
from src.loop import run_book
from src.seeds import seed_sources
from src.walkforward import evaluate_strategy
from tests.synthetic import make_synthetic_aligned

TICKERS = ["AAA", "BBB"]


def _cfg(book="equity", **kw) -> RunConfig:
    return RunConfig(book=book, seed=7, offline=True, rounds=2,
                     train_bars=150, test_bars=30, step_bars=30,
                     cma_restarts=1, cma_evals=20,
                     tickers_override=tuple(TICKERS), **kw)


def test_crypto_benchmark_applied():
    """Crypto fold factors are divided by the equal-weight B&H benchmark."""
    from src.contract import validate_candidate

    aligned = make_synthetic_aligned(TICKERS, n=300, seed=8)
    macro = aligned["macro"]
    spec = validate_candidate(seed_sources(365)["seed1_ema_sma_vol"])
    cfg = _cfg("crypto")
    res = evaluate_strategy(spec, aligned, cfg)
    # each record should carry a benchmark factor and adjusted factor
    for fold in res.folds:
        assert fold.benchmark_factor > 0
        assert np.isclose(fold.factor, fold.raw_factor / fold.benchmark_factor)


def test_run_book_offline_completes(tmp_path, monkeypatch):
    from src.loop import runner

    monkeypatch.setattr(runner.paths, "OUTPUTS_DIR", tmp_path)
    aligned = make_synthetic_aligned(TICKERS, n=260, seed=9)
    cfg = _cfg("equity")
    summary = run_book(cfg, aligned)
    assert summary["best"]["score"] is not None
    assert summary["model_used"] == "offline-mock"
    assert len(summary["rounds"]) == 2
    # every round either scored or logged a failure
    for r in summary["rounds"]:
        assert r["kept"] in (True, False) or r.get("error")
    assert summary["best"]["last_fold_x"] is not None


def test_extract_code_finds_fence_inside_prose():
    from src.loop.openrouter import extract_code

    prose = '''Here is my proposed strategy:

```python
def get_strategy():
    return {"name": "smoke", "variables": ["a"], "bounds": ([1], [2]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    return np.zeros(len(close)), 0
```

Let me know if you want changes.'''
    assert extract_code(prose).startswith("def get_strategy():")


def test_alignment_forward_fills_macro():
    """The macro reindex + ffill pattern used in alignment: a sparse weekday macro
    series must hold its last published value on the book's denser calendar."""
    import pandas as pd

    calendar = pd.date_range("2020-01-01", periods=10, freq="D")
    macro_dates = calendar[::2]
    macro = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=macro_dates)
    filled = macro.reindex(calendar).ffill()
    assert int(filled.isna().sum()) == 0
    assert filled.iloc[1] == 1.0
    assert filled.iloc[3] == 2.0

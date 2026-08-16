from __future__ import annotations

from ..config.settings import (
    BARS_PER_YEAR,
    CRYPTO_TICKERS,
    EQUITY_TICKERS,
    MAX_REFINEMENT_ATTEMPTS,
    RECENT_RESULTS_WINDOW,
)
from ..indicators import INDICATOR_DOCS

CONTRACT_TEXT = """\
## Strategy contract (must be followed exactly)
A strategy is a Python module exposing:

    def get_strategy():
        return {
            "name": "short_snake_case_name",
            "description": "one line",
            "variables": ["var1", "var2", ...],        # 4 to 15 tunable params
            "bounds": ([low1, low2, ...], [high1, high2, ...]),  # ALL lower first, then ALL upper
            "simulate": simulate,
        }

    def simulate(close, high, low, volume, macro, x):
        ...
        return positions, num_trades

Rules:
- `close`, `high`, `low`, `volume` are 1-D float arrays for one ticker over the evaluation
  window. `macro` is the aligned T10Y2Y (10Y-2Y Treasury spread) series over the same window.
  `x` is the trial parameter vector, same length as `variables`.
- `positions` is a float array of 0.0/1.0 (long-only; no shorts). `positions[t]` is the
  position held going INTO bar t+1; it may only depend on data up to and including bar t
  (STRICT no-lookahead: never index any array at t+1 or later while computing positions[t]).
- `num_trades` = integer count of position changes (reporting only).
- `bounds` is a 2-tuple `(lower, upper)` where each is a list aligned index-for-index with
  `variables`. Example: ema_period in [5,20], sma_period in [50,200] is
  `bounds = ([5, 50], [20, 200])`. NEVER use per-variable (low, high) pairs.
- Variables named with a `_period`, `_w`, or `_window` suffix are lookback lengths in bars.
  The optimizer treats them as continuous; you MUST convert with `round_int(x[i])` before
  using them as an array index/window size inside `simulate`.
- Use exactly the indicators listed below (imported for you, do not redefine them).
- Do NOT use `except Exception` / `except ZeroDivisionError` / bare `except`.
- Do NOT return a hardcoded all-zeros/all-ones positions array regardless of input.
- Do NOT use any data beyond bar t when computing positions[t] (this is machine-checked).
"""


def _book_brief(book: str) -> str:
    tickers = EQUITY_TICKERS if book == "equity" else CRYPTO_TICKERS
    calendar = "calendar days, 24/7 market" if book == "crypto" else "exchange trading days"
    if book == "equity":
        basis = ("absolute return basis: the passive alternative is cash, so raw compounded "
                 "return is the meaningful basis (no benchmark division).")
    else:
        basis = ("excess-return basis: each test window is scored as the ratio to an equal-weight "
                 "buy-and-hold benchmark over the same window, so the strategy must beat passive crypto drift.")
    return (
        f"Book: {book}. Tickers: {', '.join(tickers)}. Bars per year: {BARS_PER_YEAR[book]} "
        f"({calendar}). {basis}"
    )


def build_context(current_best, history: list[dict], book: str) -> dict:
    ind_list = "\n".join(f"  - {sig}: {desc}" for sig, desc in INDICATOR_DOCS)

    system = f"""\
You are a quantitative trading strategy researcher working inside an automated loop.
You propose COMPLETE Python strategy modules. Another system tunes their numeric
parameters with CMA-ES and walk-forward validation, so your job is only the structure:
which indicators, entry/exit logic, and which parameters to expose (4-15 of them).
Propose ONE complete replacement strategy per message. Output ONLY valid Python code
(the full module), no prose, no markdown fences.

Available indicators (already imported, do not redefine):
{ind_list}

{CONTRACT_TEXT}
"""

    recent = history[-RECENT_RESULTS_WINDOW:]
    recent_lines = []
    if recent:
        for h in recent:
            status = "KEPT" if h.get("kept") else "DISCARDED"
            recent_lines.append(
                f"- {h['name']}: score={h.get('score')}, status={status}, note={h.get('description', '')}"
            )
    else:
        recent_lines.append("- (no prior rounds yet)")

    best_code = current_best.spec.source if current_best is not None else "(none)"
    best_score = current_best.score if current_best is not None else "n/a"

    user = f"""\
{_book_brief(book)}

CURRENT BEST STRATEGY (full code):
```python
{best_code}
```
CURRENT BEST SCORE (log-Sharpe, higher is better): {best_score}

MOST RECENT PRIOR RESULTS (up to {RECENT_RESULTS_WINDOW}):
{chr(10).join(recent_lines)}

Propose ONE complete replacement strategy module that you believe will score higher.
Also return, in a comment at the very top of the module, a one-line description of what
you changed and why (format: `# change: <what and why>`).
"""
    return {"system": system, "user": user, "book": book}


def parse_description(source: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# change:"):
            return stripped[len("# change:"):].strip()
    return ""

# Runbook

Two independent strategy-mining loops (equity book, crypto book) over free daily data.
Each round proposes an LLM strategy, tunes its parameters with CMA-ES, and scores it
with walk-forward validation (log-Sharpe). Follows `../coding-plan.md`.

This is a CPU-only workload; no GPU needed.

## Setup Environment

From the project root:

```bash
cd autoresearch/llm_cmaes_strategy
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required Keys (in `.env`)

- `FRED_API_KEY` — REQUIRED. Free one-time registration at
  https://fred.stlouisfed.org/docs/api/api_key.html. The pipeline fails immediately
  at startup if this is unset; it never falls back to a stale/cached macro series.
  Add it to `.env` as `FRED_API_KEY=...`.
- `OPENROUTER_API_KEY` — required for LLM rounds (Step 7). Already present in `.env`.
  For cost-free pipeline testing, add `--offline` to any tier script to use mock
  candidates instead of the API.

`.env` is gitignored. Do not commit it.

## Verify Install

```bash
PYTHONPATH=. python -m pytest
PYTHONPATH=. python scripts/run_broken_check.py
```

## Data

Step 1 downloads and stores locally the full 2019-2024 span once per ticker
(yfinance) under `data/market/<book>/`, and the `T10Y2Y` economic series (FRED)
under `data/economic/`. Every requested range is sliced from that stored span, so
reruns never re-download. `data/` is gitignored (only `.gitkeep` placeholders are
committed) and acts as the data source on every machine, including VAS. Delete the
files under `data/market/` and `data/economic/` to force a fresh download.

## Outputs

Every run writes its results to the `outputs/` folder (gitignored):
tier summaries, per-book `_run_summary.json` + `_best_strategy.py`, and live reports.

## Tier 1 — Smoke Test

Both books separately, 2019 only, Step 8 seeds only (no LLM rounds), reduced folds
(train 200 / test 40). Confirms Steps 1-6 + 8 end to end. Already verified.

```bash
PYTHONPATH=. python scripts/run_tier1.py --book equity
PYTHONPATH=. python scripts/run_tier1.py --book crypto
```

Output: `outputs/tier1_equity.json`, `outputs/tier1_crypto.json`.
Expected time: minutes (CMA-ES: 24 restarts x 500 evals per fold).

## Tier 1b — Real LLM Round (Step 7 smoke)

Tier 1 footprint (2019, train 200 / test 40) plus **one real OpenRouter round** per
book. Confirms the Step 7 loop (propose -> validate -> walk-forward score, with up to
3 refinement attempts on failure) against the actual API before Tier 2/3 runs are
trusted. Refuses `--offline`; requires `OPENROUTER_API_KEY`. Verified.

```bash
PYTHONPATH=. python scripts/run_llm_smoke.py --book equity
PYTHONPATH=. python scripts/run_llm_smoke.py --book crypto
```

Output: `outputs/llm_smoke_equity.json`, `outputs/llm_smoke_crypto.json` (includes
`attempts`, `kept`, and any `error` per round). Expected time: ~2-3 min per book
(seed calibration + one model call). Override the model with `--model <openrouter-id>`.

## Tier 2 — Small Test

Each book separately, 2019-2021 (756 bars, 4 folds of 365/90/90), seeds + 5 LLM
rounds per book with refinement. Expected time per book: ~15-20 min
(measured ~36 s per fold-optimize; 7 strategies x 4 folds). Run offline first
to confirm the loop without OpenRouter cost:

```bash
PYTHONPATH=. python scripts/run_tier2.py --book equity --offline
PYTHONPATH=. python scripts/run_tier2.py --book crypto --offline
```

Then, to use the real LLM loop (OpenRouter, costs API calls):

```bash
PYTHONPATH=. python scripts/run_tier2.py --book equity
PYTHONPATH=. python scripts/run_tier2.py --book crypto
```

Optional flags: `--rounds N` to change the round count, `--model <id>` to override
the OpenRouter model. Output per book: `outputs/<book>_run_summary.json` and
`outputs/<book>_best_strategy.py`. A combined `outputs/tier2_summary.json` is
also written.

## Tier 3 — Full Run

Each book separately, full 2019-2024 range, full fold set, 50 LLM rounds per book.
Launch only after Tiers 1 and 2 pass. Expected time per book: ~3-4 h (measured
~2.4 min per round). The best-per-book strategies here are the deliverable and feed
the live path.

```bash
PYTHONPATH=. python scripts/run_tier3.py --book equity
PYTHONPATH=. python scripts/run_tier3.py --book crypto
```

Output per book: `outputs/<book>_run_summary.json`, `outputs/<book>_best_strategy.py`,
plus combined `outputs/tier3_summary.json`.

## Live Paper Execution (Step 9)

Loads the book's current-best strategy with its frozen `x*` (last walk-forward fold)
and runs it bar-by-bar. Replay mode validates against the historical window;
`--live` appends fresh yfinance bars.

```bash
# replay the last 90 bars (default):
PYTHONPATH=. python scripts/run_live.py --book equity
# incremental live (appends latest yfinance bars):
PYTHONPATH=. python scripts/run_live.py --book equity --live
```

Output: `outputs/<book>_live_report.json`.

## Common Flags

Every script accepts `--book {equity,crypto}` (required), `--seed N`, `--start/--end
YYYY-MM-DD`, `--force` (re-download), and `--offline`. Tier scripts also accept
`--rounds N` and `--model <openrouter-model-id>`.

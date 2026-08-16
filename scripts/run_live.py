"""Live paper-execution path (Step 9): load a book's current-best strategy (frozen x*
from its last walk-forward fold) and run it bar-by-bar over a replay window, or in
incremental live mode appending fresh yfinance bars."""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.config import DATE_END, DATE_START, paths
from src.contract import validate_candidate
from src.data import load_book_data
from src.live import live_run, replay_paper
from src.walkforward import make_folds, make_window, evaluate_strategy
from scripts.common import add_common_args, build_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 9 live paper execution")
    add_common_args(parser)
    parser.add_argument("--start-index", type=int, default=None,
                        help="replay start bar index (default: len - 90)")
    parser.add_argument("--live", action="store_true", help="incremental live mode (yfinance)")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    cfg = build_cfg(args, train=365, test=90, step=90, rounds=0,
                    default_start=DATE_START, default_end=DATE_END)

    best_json = paths.OUTPUTS_DIR / f"{cfg.book}_best_strategy.py"
    if not best_json.exists():
        raise SystemExit(f"No current-best strategy found for book '{cfg.book}'. "
                         f"Run a Tier 2/3 first: {best_json}")
    with open(best_json, encoding="utf-8") as fh:
        source = fh.read()
    spec = validate_candidate(source)

    summary_path = paths.OUTPUTS_DIR / f"{cfg.book}_run_summary.json"
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    last_fold_x = summary["best"]["last_fold_x"]
    if not last_fold_x:
        raise SystemExit(f"No frozen x* recorded for '{cfg.book}' best strategy.")
    x_star = np.asarray(last_fold_x, dtype=float)
    print(f"Frozen strategy '{spec.name}' with x* = {x_star.tolist()}")

    aligned = load_book_data(cfg, force=args.force)

    if args.live:
        reports = live_run(cfg, spec, x_star, aligned, initial_capital=args.capital)
        print(f"\nLive paper run: {len(reports)} new bars processed.")
    else:
        start = args.start_index if args.start_index is not None else max(0, len(aligned) - 90)
        reports = replay_paper(cfg, spec, x_star, aligned, start, initial_capital=args.capital)
        print(f"\nReplay over bars [{start}, {len(aligned)}): {len(reports)} bars.")

    if reports:
        print(f"{'date':<12}{'value':>14}{'turnover':>12}  weights")
        for r in reports[-10:]:
            w = {t: round(v, 3) for t, v in r["weights"].items() if v > 0.001}
            print(f"{str(r['date'])[:10]:<12}{r['value']:>14,.2f}{r['turnover']:>12.4f}  {w}")
        print(f"\nFinal paper value: {reports[-1]['value']:,.2f} "
              f"(start {args.capital:,.2f})")
        final = {
            "book": cfg.book,
            "strategy": spec.name,
            "x_star": x_star.tolist(),
            "final_value": reports[-1]["value"],
            "n_bars": len(reports),
        }
        with open(paths.OUTPUTS_DIR / f"{cfg.book}_live_report.json", "w", encoding="utf-8") as fh:
            json.dump(final, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

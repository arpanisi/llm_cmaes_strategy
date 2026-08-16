"""Tier 1 smoke test: both books, 2019 only, Step 8 seeds only (no LLM rounds),
reduced folds (train 200 / test 40). Confirms Steps 1-6 + 8 run end to end."""

from __future__ import annotations

import argparse
import json

from src.config import paths
from src.contract import validate_candidate
from src.data import load_book_data
from src.loop.runner import calibrate_seeds
from src.seeds import seed_sources
from src.walkforward import evaluate_strategy
from scripts.common import add_common_args, build_cfg

TIER1_TRAIN, TIER1_TEST, TIER1_STEP = 200, 40, 40


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 1 smoke test (seeds only, 2019)")
    add_common_args(parser)
    args = parser.parse_args()

    for book in (args.book,):
        cfg = build_cfg(args, train=TIER1_TRAIN, test=TIER1_TEST, step=TIER1_STEP,
                        rounds=0, default_start="2019-01-01", default_end="2019-12-31")
        aligned = load_book_data(cfg, force=args.force)
        print(f"\n=== Tier 1: {book} book, {len(aligned)} bars ({cfg.date_start}..{cfg.date_end}) ===")
        seed_results, best = calibrate_seeds(cfg, aligned)
        print(f"[{book}] seeds scored; best = {best.spec.name} with score {best.score:.4f}")

        paths.ensure_dirs()
        out = {
            "tier": 1,
            "book": book,
            "bars": len(aligned),
            "fold_sizes": {"train": TIER1_TRAIN, "test": TIER1_TEST, "step": TIER1_STEP},
            "seed_scores": {k: v.score for k, v in seed_results.items()},
            "best": {"name": best.spec.name, "score": best.score,
                     "metrics": {
                         "total_return": best.metrics.total_return,
                         "annualized_sharpe": best.metrics.annualized_sharpe,
                         "max_drawdown": best.metrics.max_drawdown,
                         "mean_turnover": best.metrics.mean_turnover}},
            "fold_factors": [f.factor for f in best.folds],
            "fold_records": [f.to_dict() for f in best.folds],
        }
        with open(paths.OUTPUTS_DIR / f"tier1_{book}.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"[{book}] wrote {paths.OUTPUTS_DIR / f'tier1_{book}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

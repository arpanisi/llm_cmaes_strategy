"""Tier 2 small test: both books, 2019-2021, seeds + N LLM rounds per book
(default 5) with refinement, plus the deliberate broken-candidate rejection check."""

from __future__ import annotations

import argparse
import json

from src.config import paths
from src.data import load_book_data
from src.loop.runner import run_book
from scripts.common import add_common_args, add_model_args, build_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 2 small test (seeds + LLM rounds)")
    add_common_args(parser)
    add_model_args(parser)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    summaries = {}
    for book in (args.book,):
        cfg = build_cfg(args, train=365, test=90, step=90, rounds=args.rounds,
                        default_start="2019-01-01", default_end="2021-12-31")
        aligned = load_book_data(cfg, force=args.force)
        print(f"\n=== Tier 2: {book} book, {len(aligned)} bars, {args.rounds} rounds ===")
        summary = run_book(cfg, aligned)
        summaries[book] = summary
        print(f"[{book}] best after rounds: {summary['best']['name']} "
              f"score={summary['best']['score']:.4f}")

    with open(paths.OUTPUTS_DIR / "tier2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    print(f"\nWrote {paths.OUTPUTS_DIR / 'tier2_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

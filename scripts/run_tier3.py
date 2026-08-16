"""Tier 3 full-scale run: both books, 2019-2024, full fold set, 50 LLM rounds per book.
This is the deliverable run: its best-per-book strategies load into the live path."""

from __future__ import annotations

import argparse
import json

from src.config import DATE_END, DATE_START, paths
from src.data import load_book_data
from src.loop.runner import run_book
from scripts.common import add_common_args, add_model_args, build_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 3 full run (2019-2024, 50 rounds/book)")
    add_common_args(parser)
    add_model_args(parser)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    summaries = {}
    for book in (args.book,):
        cfg = build_cfg(args, train=365, test=90, step=90, rounds=args.rounds,
                        default_start=DATE_START, default_end=DATE_END)
        aligned = load_book_data(cfg, force=args.force)
        print(f"\n=== Tier 3: {book} book, {len(aligned)} bars, {args.rounds} rounds ===")
        summary = run_book(cfg, aligned)
        summaries[book] = summary
        print(f"[{book}] best after rounds: {summary['best']['name']} "
              f"score={summary['best']['score']:.4f}")

    with open(paths.OUTPUTS_DIR / "tier3_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, default=str)
    print(f"\nWrote {paths.OUTPUTS_DIR / 'tier3_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

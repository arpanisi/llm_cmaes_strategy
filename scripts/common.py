from __future__ import annotations

import argparse

from src.config import RunConfig, load_dotenv_silently


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--book", choices=["equity", "crypto"], required=True)
    parser.add_argument("--start", default=None, help="data start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="data end date (YYYY-MM-DD)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="re-download cached data")
    parser.add_argument("--offline", action="store_true", help="use mock candidates instead of OpenRouter")


def build_cfg(args: argparse.Namespace, *, train: int, test: int, step: int,
              rounds: int, default_start: str, default_end: str) -> RunConfig:
    load_dotenv_silently()
    return RunConfig(
        book=args.book,
        date_start=args.start or default_start,
        date_end=args.end or default_end,
        train_bars=train,
        test_bars=test,
        step_bars=step,
        seed=args.seed,
        rounds=rounds,
        model=getattr(args, "model", None),
        offline=getattr(args, "offline", False),
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="OpenRouter model id (default: config/models.yaml)")

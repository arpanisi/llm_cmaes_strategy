"""Step 7 smoke check: one real OpenRouter round, Tier 1 fold sizes.

Confirms the LLM candidate loop works end to end (propose -> validate -> walk-forward
score, with 3-attempt refinement on failure) before Tier 2/3 runs are trusted.
Requires OPENROUTER_API_KEY in .env; fails fast if it is unset. Use the --model flag
to override the OpenRouter model (default from config/models.yaml)."""

from __future__ import annotations

import argparse
import json

from src.config import paths
from src.data import load_book_data
from src.loop.context import build_context
from src.loop.refine import run_round
from src.loop.runner import _make_proposer, calibrate_seeds
from src.walkforward import evaluate_strategy
from scripts.common import add_common_args, add_model_args, build_cfg

TIER1_TRAIN, TIER1_TEST, TIER1_STEP = 200, 40, 40


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 7 smoke: one real OpenRouter round per book (Tier 1 folds)"
    )
    add_common_args(parser)
    add_model_args(parser)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.offline:
        raise SystemExit("run_llm_smoke.py exercises the real OpenRouter path; drop --offline.")

    cfg = build_cfg(args, train=TIER1_TRAIN, test=TIER1_TEST, step=TIER1_STEP,
                    rounds=args.rounds, default_start="2019-01-01", default_end="2019-12-31")
    model = cfg.model or "openrouter-default"

    paths.ensure_dirs()
    aligned = load_book_data(cfg, force=args.force)
    print(f"=== LLM smoke: {cfg.book}, {len(aligned)} bars, {cfg.rounds} round(s), model={model} ===")

    seed_results, best = calibrate_seeds(cfg, aligned)
    print(f"[{cfg.book}] seeds calibrated; best = {best.spec.name} with score {best.score:.4f}")

    history: list[dict] = []
    round_records = []
    for r in range(1, cfg.rounds + 1):
        context = build_context(best, history, cfg.book)
        proposer = _make_proposer(context, r, cfg)
        outcome = run_round(proposer, lambda spec: evaluate_strategy(spec, aligned, cfg))

        record = {"round": r, "model": model, "attempts": outcome.attempts}
        if outcome.ok:
            res = outcome.result
            kept = res.score > best.score
            record.update({"name": res.spec.name, "score": res.score, "kept": kept})
            print(f"[{cfg.book}] round {r}: {res.spec.name} score={res.score:.4f} "
                  f"{'KEPT' if kept else 'DISCARDED'} (best={best.score:.4f})")
            if kept:
                best = res
            history.append({"name": res.spec.name, "score": res.score, "kept": kept,
                            "description": res.spec.description})
        else:
            record.update({"name": None, "score": None, "kept": False, "error": outcome.error})
            print(f"[{cfg.book}] round {r}: FAILED after {outcome.attempts} attempts: {outcome.error}")
        round_records.append(record)

    any_kept = any(r["kept"] for r in round_records)
    summary = {
        "step": "llm-smoke",
        "book": cfg.book,
        "model": model,
        "rounds": round_records,
        "best_after": {
            "name": best.spec.name,
            "score": best.score,
            "kept_from": "llm-round" if any_kept else "seed",
        },
    }
    out_path = paths.OUTPUTS_DIR / f"llm_smoke_{cfg.book}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[{cfg.book}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

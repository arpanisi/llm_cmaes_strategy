from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import RunConfig, paths
from ..contract import StrategySpec, validate_candidate
from ..seeds import seed_sources
from ..walkforward import CandidateResult, evaluate_strategy
from .context import build_context
from .mock import mock_propose
from .openrouter import extract_code, generate
from .refine import RoundOutcome, run_round


def _dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _make_proposer(context: dict, round_idx: int, cfg: RunConfig):
    if cfg.offline:
        return lambda last_error: mock_propose(cfg.bars_per_year, round_idx)
    model = cfg.model

    def proposer(last_error: str | None) -> str:
        user = context["user"]
        if last_error:
            user = (
                user
                + "\n\nYour previous attempt was REJECTED with this error:\n"
                + last_error
                + "\n\nPropose a corrected complete module."
            )
        text = generate(context["system"], user, model=model)
        return extract_code(text)

    return proposer


def calibrate_seeds(cfg: RunConfig, aligned: pd.DataFrame) -> tuple[dict, CandidateResult]:
    """Step 8 calibration: run both seeds through Step 6; current best = higher score."""
    results = {}
    for name, src in seed_sources(cfg.bars_per_year).items():
        spec = validate_candidate(src)
        res = evaluate_strategy(spec, aligned, cfg)
        results[name] = res
        print(f"[{cfg.book}] seed {name}: score={res.score:.4f} "
              f"(total_return={res.metrics.total_return:.4f} "
              f"sharpe={res.metrics.annualized_sharpe:.4f} "
              f"max_dd={res.metrics.max_drawdown:.4f} turnover={res.metrics.mean_turnover:.4f})")
    best = max(results.values(), key=lambda r: r.score)
    return results, best


def run_book(cfg: RunConfig, aligned: pd.DataFrame) -> dict:
    seed_results, best = calibrate_seeds(cfg, aligned)
    best_spec = best.spec

    history: list[dict] = []
    round_records = []
    model_used = cfg.model or ("offline-mock" if cfg.offline else "openrouter-default")

    for round_idx in range(1, cfg.rounds + 1):
        context = build_context(best, history, cfg.book)
        proposer = _make_proposer(context, round_idx, cfg)
        outcome: RoundOutcome = run_round(proposer, lambda spec: evaluate_strategy(spec, aligned, cfg))

        record = {
            "round": round_idx,
            "model": model_used,
            "attempts": outcome.attempts,
        }
        if outcome.ok:
            res: CandidateResult = outcome.result
            kept = res.score > best.score
            record.update(
                {
                    "name": res.spec.name,
                    "description": res.spec.description,
                    "score": res.score,
                    "kept": kept,
                    "metrics": {
                        "total_return": res.metrics.total_return if res.metrics else None,
                        "annualized_sharpe": res.metrics.annualized_sharpe if res.metrics else None,
                        "max_drawdown": res.metrics.max_drawdown if res.metrics else None,
                        "mean_turnover": res.metrics.mean_turnover if res.metrics else None,
                    },
                }
            )
            print(f"[{cfg.book}] round {round_idx}: {res.spec.name} score={res.score:.4f} "
                  f"{'KEPT' if kept else 'DISCARDED'} (best={best.score:.4f})")
            if kept:
                best = res
                best_spec = res.spec
            history.append(
                {
                    "name": res.spec.name,
                    "score": res.score,
                    "kept": kept,
                    "description": res.spec.description,
                }
            )
        else:
            record.update({"name": None, "score": None, "kept": False, "error": outcome.error})
            print(f"[{cfg.book}] round {round_idx}: FAILED after {outcome.attempts} attempts: {outcome.error}")
            history.append({"name": "(failed round)", "score": None, "kept": False, "error": outcome.error})
        round_records.append(record)

    summary = {
        "run_config": cfg.as_record(),
        "model_used": model_used,
        "seed_scores": {k: v.score for k, v in seed_results.items()},
        "best": {
            "name": best.spec.name,
            "score": best.score,
            "source": best.spec.source,
            "metrics": {
                "total_return": best.metrics.total_return if best.metrics else None,
                "annualized_sharpe": best.metrics.annualized_sharpe if best.metrics else None,
                "max_drawdown": best.metrics.max_drawdown if best.metrics else None,
                "mean_turnover": best.metrics.mean_turnover if best.metrics else None,
            },
            "last_fold_x": list(best.last_fold_x) if best.last_fold_x is not None else None,
            "fold_factors": [f.factor for f in best.folds],
            "fold_records": [f.to_dict() for f in best.folds],
        },
        "rounds": round_records,
    }
    _dump_json(summary, paths.OUTPUTS_DIR / f"{cfg.book}_run_summary.json")
    with open(paths.OUTPUTS_DIR / f"{cfg.book}_best_strategy.py", "w", encoding="utf-8") as fh:
        fh.write(best.spec.source)
    return summary

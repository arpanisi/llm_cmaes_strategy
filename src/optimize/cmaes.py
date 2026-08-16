from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

from ..config.settings import (
    CMA_EVALS_PER_RESTART,
    CMA_POPULATION,
    CMA_RESTARTS,
)
from ..contract import StrategySpec
from ..core import BookWindow
from .fitness import objective


@dataclass
class OptimizeResult:
    x: np.ndarray
    value: float
    restart_count: int


def optimize_strategy(
    spec: StrategySpec,
    window: BookWindow,
    seed: int,
    n_restarts: int = CMA_RESTARTS,
    evals_per_restart: int = CMA_EVALS_PER_RESTART,
    population: int = CMA_POPULATION,
) -> OptimizeResult:
    import cma

    lower = np.asarray(spec.lower, dtype=float)
    upper = np.asarray(spec.upper, dtype=float)
    sigma0 = max(0.15 * float(np.mean(upper - lower)), 1e-3)

    # cma always instantiates a CMADataLogger (final files like all_stoppings.json2
    # are written even with verb_log 0). Point it at a throwaway temp dir and delete
    # it afterwards so no optimizer output ever lands in the project tree.
    log_dir = tempfile.mkdtemp(prefix="cma_log_")
    try:
        best_x, best_f = None, np.inf
        for r in range(n_restarts):
            rng = np.random.default_rng(int(seed) + r)
            x0_r = rng.uniform(lower, upper)
            es = cma.CMAEvolutionStrategy(
                x0_r.tolist(),
                sigma0,
                {
                    "popsize": population,
                    "maxfevals": evals_per_restart,
                    "bounds": [lower.tolist(), upper.tolist()],
                    "seed": int(seed) + r,
                    "verbose": -9,
                    "verb_disp": 0,
                    "verb_log": 0,
                    "verb_filenameprefix": os.path.join(log_dir, "outcmaes"),
                },
            )
            es.optimize(lambda x: objective(x, spec, window))
            fx = float(es.result.fbest)
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(es.result.xbest, dtype=float)

        best_x = np.clip(best_x, lower, upper)
        return OptimizeResult(x=best_x, value=best_f, restart_count=n_restarts)
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def calibration_check(
    spec: StrategySpec,
    window: BookWindow,
    seeds: list[int],
    n_restarts: int,
    evals_per_restart: int,
) -> dict:
    """Design-time sanity check: spread across seeds should be small relative to
    differences between strategies/windows observed in the same pass."""
    results = []
    for s in seeds:
        res = optimize_strategy(
            spec, window, seed=s, n_restarts=n_restarts, evals_per_restart=evals_per_restart
        )
        results.append(res.value)
    return {
        "values": results,
        "spread": max(results) - min(results),
    }

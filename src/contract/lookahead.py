from __future__ import annotations

import numpy as np

from .spec import StrategyError, StrategySpec

SYNTHETIC_LEN = 300
N_SAMPLES = 10


def _synthetic_window(n: int = SYNTHETIC_LEN) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(12345)
    ret = rng.normal(0.0004, 0.012, n)
    ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.004, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    volume = rng.uniform(1e5, 1e6, n)
    macro = 0.5 * np.sin(np.linspace(0, 12 * np.pi, n)) + 0.3
    return {"close": close, "high": high, "low": low, "volume": volume, "macro": macro}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    eq = (a == b) | (np.isnan(a) & np.isnan(b))
    return bool(np.all(eq))


def check_lookahead(spec: StrategySpec) -> None:
    """Prefix-invariance check: positions[t_k] must be identical whether simulate is
    given the full window or the array truncated to [0 : t_k+1]."""
    win = _synthetic_window()
    close, high, low, volume, macro = (
        win["close"], win["high"], win["low"], win["volume"], win["macro"],
    )
    x = np.array([0.5 * (lo + hi) for lo, hi in zip(spec.lower, spec.upper)])

    warmup = max(30, spec.n_variables * 5)
    indices = np.linspace(warmup, len(close) - 1, N_SAMPLES).astype(int)
    indices = np.unique(indices)

    try:
        full_pos, _ = spec.simulate(close, high, low, volume, macro, x)
    except Exception as exc:
        raise StrategyError(
            f"lookahead check: simulate raised on the fixed synthetic window: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if np.asarray(full_pos).shape != close.shape:
        raise StrategyError(
            f"lookahead check: simulate returned positions of shape "
            f"{np.asarray(full_pos).shape}, expected {close.shape}."
        )

    for t in indices:
        truncated = {
            "close": close[: t + 1],
            "high": high[: t + 1],
            "low": low[: t + 1],
            "volume": volume[: t + 1],
            "macro": macro[: t + 1],
        }
        try:
            tr_pos, _ = spec.simulate(
                truncated["close"], truncated["high"], truncated["low"],
                truncated["volume"], truncated["macro"], x,
            )
        except Exception as exc:  # surface as a lookahead/robustness failure
            raise StrategyError(
                f"lookahead check: simulate raised on a truncated input at index {int(t)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        a = np.asarray(full_pos[t], dtype=float)
        b = np.asarray(tr_pos[-1], dtype=float)
        if not _same(np.array([a]), np.array([b])):
            raise StrategyError(
                f"lookahead check: positions[{int(t)}] differs between full and truncated "
                f"inputs ({a} vs {b}). positions[t] may only depend on data up to and "
                f"including bar t. Rewrite simulate to be strictly causal."
            )

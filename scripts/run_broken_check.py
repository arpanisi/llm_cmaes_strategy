"""Deliberately broken hand-written strategies — each must be rejected by Step 2
validation with a specific, actionable error. Fails loudly if any is accepted."""

from __future__ import annotations

import sys

from src.contract import StrategyError, validate_candidate

BROKEN_WRONG_SIGNATURE = '''
def get_strategy():
    return {"name": "bad", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro):
    return np.zeros(len(close)), 0
'''

BROKEN_LOOKAHEAD = '''
def get_strategy():
    return {"name": "peek", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    positions = np.zeros(len(close))
    for t in range(1, len(close) - 6):
        positions[t] = 1.0 if close[t + 5] > close[t] else 0.0
    return positions, int(np.sum(np.abs(np.diff(positions))))
'''

BROKEN_BARE_EXCEPT = '''
def get_strategy():
    return {"name": "swallow", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    try:
        positions = np.where(close > SMA(close, 10), 1.0, 0.0)
    except Exception:
        positions = np.zeros(len(close))
    return positions, 0
'''

BROKEN_CONSTANT_FALLBACK = '''
def get_strategy():
    return {"name": "flat", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    return np.ones(len(close)), 0
'''

BROKEN_BOUNDS_SHAPE = '''
def get_strategy():
    return {"name": "wrong_bounds", "variables": ["a", "b", "c", "d"],
            "bounds": ([5, 50], [20, 200]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    return np.zeros(len(close)), 0
'''


def main() -> int:
    cases = [
        ("wrong signature", BROKEN_WRONG_SIGNATURE, "signature"),
        ("non-causal lookahead", BROKEN_LOOKAHEAD, "lookahead"),
        ("bare except", BROKEN_BARE_EXCEPT, "exception"),
        ("constant fallback", BROKEN_CONSTANT_FALLBACK, "constant"),
        ("bounds shape", BROKEN_BOUNDS_SHAPE, "bounds"),
    ]
    failures = 0
    for name, source, expected_substr in cases:
        try:
            validate_candidate(source)
        except StrategyError as exc:
            ok = expected_substr.lower() in str(exc).lower()
            status = "PASS" if ok else f"FAIL (wrong error: {exc})"
            print(f"[{status}] {name}: rejected")
            if not ok:
                failures += 1
        except Exception as exc:
            print(f"[FAIL] {name}: rejected with unexpected {type(exc).__name__}: {exc}")
            failures += 1
        else:
            print(f"[FAIL] {name}: NOT rejected (accepted broken candidate!)")
            failures += 1
    if failures:
        print(f"\n{len(cases) - failures}/{len(cases)} checks passed.")
        return 1
    print(f"\nAll {len(cases)} broken-candidate checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

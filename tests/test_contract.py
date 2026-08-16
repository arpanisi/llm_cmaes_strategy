from __future__ import annotations

import numpy as np
import pytest

from src.contract import (
    CrashRecord,
    StrategyError,
    run_simulate,
    validate_candidate,
)
from src.seeds import seed_sources

NON_CAUSAL = '''
def get_strategy():
    return {"name": "peek", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    positions = np.zeros(len(close))
    for t in range(1, len(close) - 6):
        positions[t] = 1.0 if close[t + 5] > close[t] else 0.0
    return positions, int(np.sum(np.abs(np.diff(positions))))
'''

CAUSAL_STATE = '''
BARS_PER_YEAR = 252

def get_strategy():
    return {"name": "causal", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    w = round_int(x[0])
    rsi = RSI(close, w)
    positions = np.zeros(len(close))
    for t in range(1, len(close)):
        if rsi[t] > 55:
            positions[t] = 1.0
        elif rsi[t] < 45:
            positions[t] = 0.0
        else:
            positions[t] = positions[t - 1]
    return positions, int(np.sum(np.abs(np.diff(positions))))
'''


def test_seed1_validates():
    spec = validate_candidate(seed_sources(252)["seed1_ema_sma_vol"])
    assert len(spec.variables) == 4
    assert spec.lower == [5, 50, 10, 0.05]
    assert spec.upper == [20, 200, 60, 0.50]


def test_seed2_validates():
    spec = validate_candidate(seed_sources(365)["seed2_macro_trend"])
    assert len(spec.variables) == 5


def test_non_causal_rejected():
    with pytest.raises(StrategyError, match="lookahead"):
        validate_candidate(NON_CAUSAL)


def test_causal_strategy_passes():
    spec = validate_candidate(CAUSAL_STATE)
    close = np.linspace(100, 120, 200)
    high = close * 1.005
    low = close * 0.995
    volume = np.full(200, 1e5)
    macro = np.full(200, 0.5)
    x = np.array([0.5 * (a + b) for a, b in zip(spec.lower, spec.upper)])
    result = run_simulate(spec.simulate, close, high, low, volume, macro, x)
    assert isinstance(result, CrashRecord) is False
    assert result.positions.shape == (200,)
    assert set(np.unique(result.positions)) <= {0.0, 1.0}


def test_crash_captured_not_silent():
    source = CAUSAL_STATE.replace("RSI(close, w)", "RSI(close, -1)")
    with pytest.raises(StrategyError):
        validate_candidate(source)


def test_non_python_candidate_wrapped_as_strategy_error():
    """A response that is not valid Python (e.g. LLM prose around a fence) must
    surface as a StrategyError so the refinement loop can feed it back, not crash."""
    with pytest.raises(StrategyError, match="not valid Python"):
        validate_candidate('''Here is my idea:

* a volatility gate
* then entry rules

def get_strategy():
    return {
        "name": "oops",
        "variables": ["a", "b", "c", "d"],
        "bounds": ([1, 1, 1, 1], [5, 5, 5, 5]),
        "simulate": simulate,
    }

def simulate(close, high, low, volume, macro, x):
    unclosed = "
''')


def test_runtime_crash_capture():
    source = '''
def get_strategy():
    return {"name": "boom", "variables": ["w", "k", "j", "v"],
            "bounds": ([5, 1.0, 10, 0.05], [20, 3.0, 60, 0.5]), "simulate": simulate}

def simulate(close, high, low, volume, macro, x):
    x = 1.0 / 0.0
    return np.zeros(len(close)), 0
'''
    with pytest.raises(StrategyError, match="get_strategy|validation|ZeroDivisionError"):
        validate_candidate(source)


def test_shape_mismatch_captured_as_crash_record():
    def simulate_bad_shape(close, high, low, volume, macro, x):
        return np.zeros(len(close) - 5), 0

    close = np.linspace(100, 120, 20)
    high = close * 1.005
    low = close * 0.995
    volume = np.full(20, 1e5)
    macro = np.full(20, 0.5)
    x = np.array([10, 2.0, 30, 0.1])

    res = run_simulate(simulate_bad_shape, close, high, low, volume, macro, x)
    assert isinstance(res, CrashRecord)
    assert res.exception_type == "ValueError"
    assert "simulate returned positions of shape" in res.message

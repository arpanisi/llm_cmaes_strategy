from __future__ import annotations

from ..seeds import seed_sources

MOCK_A = """\
# change: RSI momentum with a macro gate instead of the seed's SMA crossover.
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "rsi_macro_momentum",
        "description": "long when RSI is strong and the T10Y2Y spread is above threshold",
        "variables": ["rsi_w", "rsi_buy", "rsi_sell", "macro_threshold"],
        "bounds": ([5, 50, 40, -2.0], [20, 70, 60, 2.0]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    rsi_w = round_int(x[0])
    rsi_buy = float(x[1])
    rsi_sell = float(x[2])
    macro_threshold = float(x[3])
    rsi = RSI(close, rsi_w)
    macro_ok = macro > macro_threshold
    positions = np.zeros(len(close))
    for t in range(1, len(close)):
        if rsi[t] > rsi_buy and macro_ok[t]:
            positions[t] = 1.0
        elif rsi[t] < rsi_sell:
            positions[t] = 0.0
        else:
            positions[t] = positions[t - 1]
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
"""

MOCK_B = """\
# change: Donchian channel breakout instead of moving-average crossover.
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "donchian_breakout",
        "description": "long above Donchian upper with vol floor, exit below Donchian lower",
        "variables": ["entry_w", "exit_w", "vol_w", "vol_threshold"],
        "bounds": ([10, 5, 10, 0.05], [100, 40, 60, 0.50]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    entry_w = round_int(x[0])
    exit_w = round_int(x[1])
    vol_w = round_int(x[2])
    vol_threshold = float(x[3])
    up, _ = DONCHIAN(high, low, entry_w)
    _, exit_dn = DONCHIAN(high, low, exit_w)
    vol = REALIZED_VOL(close, vol_w, BARS_PER_YEAR)
    positions = np.zeros(len(close))
    for t in range(1, len(close)):
        if close[t] > up[t] and vol[t] > vol_threshold:
            positions[t] = 1.0
        elif close[t] < exit_dn[t]:
            positions[t] = 0.0
        else:
            positions[t] = positions[t - 1]
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
"""

MOCK_C = """\
# change: Bollinger mean reversion, buy under the lower band.
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "bollinger_reversion",
        "description": "buy below lower Bollinger band, exit at the middle band",
        "variables": ["bb_w", "bb_k", "vol_w", "vol_threshold"],
        "bounds": ([10, 1.0, 10, 0.05], [60, 3.0, 60, 0.50]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    bb_w = round_int(x[0])
    bb_k = float(x[1])
    vol_w = round_int(x[2])
    vol_threshold = float(x[3])
    _, dn = BOLLINGER(close, bb_w, bb_k)
    mid = SMA(close, bb_w)
    vol = REALIZED_VOL(close, vol_w, BARS_PER_YEAR)
    positions = np.zeros(len(close))
    for t in range(1, len(close)):
        if close[t] < dn[t] and vol[t] > vol_threshold:
            positions[t] = 1.0
        elif close[t] > mid[t]:
            positions[t] = 0.0
        else:
            positions[t] = positions[t - 1]
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
"""

MOCK_D = """\
# change: stochastic oversold recovery confirmed by an EMA trend filter.
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "stoch_ema_trend",
        "description": "long when %K is low, price is above its EMA, and macro is supportive",
        "variables": ["stoch_w", "k_low", "ema_w", "macro_threshold"],
        "bounds": ([5, 10, 5, -2.0], [20, 40, 50, 2.0]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    stoch_w = round_int(x[0])
    k_low = float(x[1])
    ema_w = round_int(x[2])
    macro_threshold = float(x[3])
    pct_k, _ = STOCHASTIC(high, low, close, stoch_w)
    ema = EMA(close, ema_w)
    positions = np.zeros(len(close))
    for t in range(1, len(close)):
        if pct_k[t] < k_low and close[t] > ema[t] and macro[t] > macro_threshold:
            positions[t] = 1.0
        elif pct_k[t] > 80.0:
            positions[t] = 0.0
        else:
            positions[t] = positions[t - 1]
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
"""


def mock_sources(bars_per_year: int) -> list[str]:
    bpy = int(bars_per_year)
    sources = [MOCK_A, MOCK_B, MOCK_C, MOCK_D]
    seeds = seed_sources(bpy)
    return [s.format(bars_per_year=bpy) for s in sources] + list(seeds.values())


def mock_propose(bars_per_year: int, round_idx: int) -> str:
    pool = mock_sources(bars_per_year)
    return pool[round_idx % len(pool)]

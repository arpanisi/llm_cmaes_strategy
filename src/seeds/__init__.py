from __future__ import annotations

TEMPLATE_SEED1 = '''\
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "ema_sma_vol_filter",
        "description": "EMA/SMA crossover, long only when trailing realized volatility is above a floor",
        "variables": ["ema_period", "sma_period", "vol_period", "vol_threshold"],
        "bounds": ([5, 50, 10, 0.05], [20, 200, 60, 0.50]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    ema_period = round_int(x[0])
    sma_period = round_int(x[1])
    vol_period = round_int(x[2])
    vol_threshold = float(x[3])
    ema = EMA(close, ema_period)
    sma = SMA(close, sma_period)
    vol = REALIZED_VOL(close, vol_period, BARS_PER_YEAR)
    signal = (ema > sma) & (vol > vol_threshold)
    positions = np.where(signal, 1.0, 0.0)
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
'''

TEMPLATE_SEED2 = '''\
BARS_PER_YEAR = {bars_per_year}

def get_strategy():
    return {{
        "name": "macro_trend_filter",
        "description": "EMA/SMA crossover with realized-vol floor, additionally requiring the T10Y2Y macro series above a threshold",
        "variables": ["ema_period", "sma_period", "vol_period", "vol_threshold", "macro_threshold"],
        "bounds": ([5, 50, 10, 0.05, -2.0], [20, 200, 60, 0.50, 2.0]),
        "simulate": simulate,
    }}

def simulate(close, high, low, volume, macro, x):
    ema_period = round_int(x[0])
    sma_period = round_int(x[1])
    vol_period = round_int(x[2])
    vol_threshold = float(x[3])
    macro_threshold = float(x[4])
    ema = EMA(close, ema_period)
    sma = SMA(close, sma_period)
    vol = REALIZED_VOL(close, vol_period, BARS_PER_YEAR)
    macro_ok = macro > macro_threshold
    signal = (ema > sma) & (vol > vol_threshold) & macro_ok
    positions = np.where(signal, 1.0, 0.0)
    num_trades = int(np.sum(np.abs(np.diff(positions))))
    return positions, num_trades
'''


def seed_sources(bars_per_year: int) -> dict[str, str]:
    return {
        "seed1_ema_sma_vol": TEMPLATE_SEED1.format(bars_per_year=int(bars_per_year)),
        "seed2_macro_trend": TEMPLATE_SEED2.format(bars_per_year=int(bars_per_year)),
    }

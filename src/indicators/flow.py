from __future__ import annotations

import numpy as np


def indicator_obv(close, volume) -> np.ndarray:
    """On-Balance Volume: cumulative volume signed by sign(close_t - close_{t-1})."""
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    out = np.full(len(c), 0.0)
    if len(c) > 1:
        signs = np.sign(c[1:] - c[:-1])
        out[1:] = np.cumsum(signs * v[1:])
    return out

"""Fair value gap — the three-candle imbalance, traded on its own.

The claim: an imbalance is unfinished business, and price returns to fill it. This
strategy takes the weaker version of that claim — enter in the direction of a
newly formed gap and hold — because it is the version that can be measured without
also having to model where the retracement ends.

It exists so the ICT concept can be scored against the baselines rather than
argued about. See tjr.py for the same primitive used as one leg of a larger setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .primitives import atr as _atr


def fvg(df: pd.DataFrame, min_atr: float = 0.25, atr_len: int = 14,
        hold: int = 10) -> pd.Series:
    """Fair value gap strategy — same detection logic as the Pine indicator.

    Enters in the direction of a newly formed gap, holds for `hold` bars.
    This exists so you can measure the ICT-style concept against the baselines
    rather than argue about it.

    The gap test is the vectorized twin of primitives.find_fvg: bull requires
    low[t] > high[t-2] with the space between them at least `min_atr` x ATR. It is
    written as a mask rather than a per-bar call because nothing here is
    path-dependent, so there is no state machine to step through and the whole
    series can be computed at once.
    """
    atr = pd.Series(_atr(df, atr_len), index=df.index)

    bull = (df["low"] > df["high"].shift(2)) & ((df["low"] - df["high"].shift(2)) >= atr * min_atr)
    bear = (df["high"] < df["low"].shift(2)) & ((df["low"].shift(2) - df["high"]) >= atr * min_atr)

    raw = pd.Series(0.0, index=df.index)
    raw[bull] = 1.0
    raw[bear] = -1.0
    # hold the signal for `hold` bars, most recent wins
    return raw.replace(0.0, np.nan).ffill(limit=hold).fillna(0.0)

"""Benchmarks — neither family. These are what both families have to beat.

buy_hold is the opportunity cost: if a strategy cannot beat simply owning the
thing, its rules are subtracting value, however good the Sharpe looks in isolation.

random_entry is the null hypothesis with the trade frequency held constant. It is
the more uncomfortable of the two, because a strategy that trades often will
usually beat buy_hold on some metric by accident, and only the random benchmark
asks whether the *rules* did anything.

Neither takes a view. Neither is trying to. They are the calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def random_entry(df: pd.DataFrame, trade_rate: float = 0.02, seed: int = 0,
                 hold: int = 10) -> pd.Series:
    """Coin-flip benchmark with a comparable trade frequency.

    If your strategy cannot beat a distribution of these, it has no edge —
    it just has exposure.

    It fails validate.causality_check, and that is a false positive rather than a
    leak. `side` is drawn after `flips`, so the number of values pulled from the
    stream before it depends on len(df): recomputing on truncated history shifts
    the RNG position and produces different coin flips for the same bars. No future
    bar is read — the signal at bar t is drawn without looking at any price at all.

    Left alone deliberately. Reseeding per bar to make it prefix-stable would
    change every flip, and random_benchmark draws from this function to score every
    other strategy's random gate, so it would silently move the gauntlet's verdicts.
    Read a LOOKAHEAD verdict on `--strategy random_entry` as this, not as a bug.
    """
    rng = np.random.default_rng(seed)
    flips = rng.random(len(df)) < trade_rate
    side = rng.choice([-1.0, 1.0], len(df))
    raw = pd.Series(np.where(flips, side, np.nan), index=df.index)
    return raw.ffill(limit=hold).fillna(0.0)

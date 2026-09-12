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


def equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """Own the whole universe, equally. The panel answer to buy_hold.

    A cross-sectional strategy that cannot beat this has not earned its ranking;
    it has earned the universe's return with extra steps and extra costs.
    """
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def random_panel(prices: pd.DataFrame, cut: float = 0.2, hold: int = 21,
                 seed: int = 0) -> pd.DataFrame:
    """Coin flips with the same shape as a real cross-sectional book.

    Picks the same number of longs and shorts, at the same rebalance frequency,
    as the strategy it stands in for — the names are the only thing chosen at
    random. That is the point: holding turnover and gross exposure fixed isolates
    whether the *ranking* did anything, rather than flattering a benchmark for
    simply trading less.
    """
    rng = np.random.default_rng(seed)
    cols = list(prices.columns)
    k = max(1, int(len(cols) * cut))
    w = pd.DataFrame(0.0, index=prices.index, columns=cols)
    for i in range(0, len(prices), hold):
        picks = rng.choice(len(cols), size=min(2 * k, len(cols)), replace=False)
        block = np.zeros(len(cols))
        block[picks[:k]] = 0.5 / k
        block[picks[k:2 * k]] = -0.5 / k
        w.iloc[i:i + hold] = block
    return w


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

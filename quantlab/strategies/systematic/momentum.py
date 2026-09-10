"""Trend and momentum — strategies that harvest return autocorrelation.

The shared claim: a return over a trailing window carries information about the
next return, on average, across the whole sample. That is a prediction, and a
falsifiable one. What separates it from the predictive family is what the claim
rests on — a statistical property that has persisted across decades, assets and
research groups, rather than an individual chart formation implying intent.

None of these look at a setup. They look at one number computed the same way on
every bar, and they are wrong on most of them; the claim is only ever about the
average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def tsmom(df: pd.DataFrame, lookback: int = 252, long_only: bool = False) -> pd.Series:
    """Time-series momentum. Long if trailing return > 0, else short (or flat).

    This is the Moskowitz/Ooi/Pedersen rule — the most-documented effect in the
    literature and the sanity baseline every other strategy should be measured against.
    """
    past = df["close"].pct_change(lookback)
    sig = np.sign(past)
    if long_only:
        sig = sig.clip(lower=0)
    return sig


def ma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200, long_only: bool = False) -> pd.Series:
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    sig = np.sign(f - s)
    if long_only:
        sig = sig.clip(lower=0)
    return sig


def donchian(df: pd.DataFrame, entry: int = 55, exit_n: int = 20) -> pd.Series:
    """Classic turtle-style breakout. Long on N-bar high, out on M-bar low."""
    hi = df["high"].rolling(entry).max().shift(1)
    lo = df["low"].rolling(entry).min().shift(1)
    xhi = df["high"].rolling(exit_n).max().shift(1)
    xlo = df["low"].rolling(exit_n).min().shift(1)

    pos = np.zeros(len(df))
    state = 0.0
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    hi_v, lo_v, xhi_v, xlo_v = hi.values, lo.values, xhi.values, xlo.values
    for i in range(len(df)):
        if np.isnan(hi_v[i]):
            pos[i] = 0.0
            continue
        if state <= 0 and h[i] > hi_v[i]:
            state = 1.0
        elif state >= 0 and l[i] < lo_v[i]:
            state = -1.0
        elif state > 0 and l[i] < xlo_v[i]:
            state = 0.0
        elif state < 0 and h[i] > xhi_v[i]:
            state = 0.0
        pos[i] = state
    return pd.Series(pos, index=df.index)


def trend_filter(df: pd.DataFrame, ma_len: int = 50, confirm: int = 1,
                 long_only: bool = True) -> pd.Series:
    """Hold while price is above its own moving average. Flat otherwise.

    Written for a venue constraint rather than for a backtest. Alpaca does not
    permit shorting crypto, so a long/short signal there is fiction — the broker
    clamps the shorts to flat and you end up trading something you never tested.
    This is long_only by default for that reason.

    `confirm` requires N consecutive closes on the right side of the average before
    flipping. It exists because the binding constraint on crypto is cost, not
    signal: at 25bps a side, whipsaw around the average is what kills a filter, and
    turnover is the thing worth spending a parameter on. It is not a return knob.

    Against buy-and-hold, this can only add value by avoiding drawdowns — it will
    never out-earn a held position in a bull market, because it is the same position
    minus some of the time. Judge it on Sharpe, not CAGR.
    """
    ma = df["close"].rolling(ma_len).mean()
    above = df["close"] > ma
    if confirm > 1:
        # rolling(...).sum() over a boolean: only flip once N in a row agree
        above = above.rolling(confirm).sum() >= confirm
    sig = above.astype(float)
    if not long_only:
        sig = sig * 2.0 - 1.0
    return sig

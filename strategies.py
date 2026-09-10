"""Strategies.

A strategy is a function: (df, **params) -> pd.Series of DESIRED position,
aligned to df.index, where the value at bar t is the position you want to hold
*going into* bar t+1.

The engine shifts signals forward by one bar before applying returns, so a
strategy may freely use df.loc[:t] without introducing lookahead. Never use
.shift(-n) or any future-referencing operation in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    name: str
    fn: Callable[..., pd.Series]
    params: Dict[str, Any] = field(default_factory=dict)
    # grid used by sensitivity + walk-forward optimization
    grid: Dict[str, list] = field(default_factory=dict)

    def signal(self, df: pd.DataFrame, **overrides) -> pd.Series:
        p = {**self.params, **overrides}
        sig = self.fn(df, **p)
        return sig.reindex(df.index).fillna(0.0).clip(-1, 1)


# ─────────────────────────── signal functions ───────────────────────────

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


def rsi_meanrev(df: pd.DataFrame, length: int = 2, lower: float = 10, upper: float = 90) -> pd.Series:
    """Short-term mean reversion. Included because it backtests beautifully and
    usually dies on costs — a useful demonstration."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    pos = np.where(rsi < lower, 1.0, np.where(rsi > upper, -1.0, np.nan))
    return pd.Series(pos, index=df.index).ffill().fillna(0.0)


def fvg(df: pd.DataFrame, min_atr: float = 0.25, atr_len: int = 14,
        hold: int = 10) -> pd.Series:
    """Fair value gap strategy — same detection logic as the Pine indicator.

    Enters in the direction of a newly formed gap, holds for `hold` bars.
    This exists so you can measure the ICT-style concept against the baselines
    rather than argue about it.
    """
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_len, adjust=False).mean()

    bull = (df["low"] > df["high"].shift(2)) & ((df["low"] - df["high"].shift(2)) >= atr * min_atr)
    bear = (df["high"] < df["low"].shift(2)) & ((df["low"].shift(2) - df["high"]) >= atr * min_atr)

    raw = pd.Series(0.0, index=df.index)
    raw[bull] = 1.0
    raw[bear] = -1.0
    # hold the signal for `hold` bars, most recent wins
    return raw.replace(0.0, np.nan).ffill(limit=hold).fillna(0.0)


def random_entry(df: pd.DataFrame, trade_rate: float = 0.02, seed: int = 0,
                 hold: int = 10) -> pd.Series:
    """Coin-flip benchmark with a comparable trade frequency.

    If your strategy cannot beat a distribution of these, it has no edge —
    it just has exposure.
    """
    rng = np.random.default_rng(seed)
    flips = rng.random(len(df)) < trade_rate
    side = rng.choice([-1.0, 1.0], len(df))
    raw = pd.Series(np.where(flips, side, np.nan), index=df.index)
    return raw.ffill(limit=hold).fillna(0.0)


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


# ─────────────────────────── registry ───────────────────────────

REGISTRY: Dict[str, Strategy] = {
    "tsmom": Strategy("tsmom", tsmom, {"lookback": 252, "long_only": False},
                      {"lookback": [63, 126, 189, 252, 378, 504]}),
    "ma_cross": Strategy("ma_cross", ma_cross, {"fast": 50, "slow": 200},
                         {"fast": [10, 20, 50, 100], "slow": [100, 150, 200, 300]}),
    "donchian": Strategy("donchian", donchian, {"entry": 55, "exit_n": 20},
                         {"entry": [20, 40, 55, 80, 120], "exit_n": [10, 20, 40]}),
    "rsi_meanrev": Strategy("rsi_meanrev", rsi_meanrev, {"length": 2, "lower": 10, "upper": 90},
                            {"length": [2, 3, 5, 10], "lower": [5, 10, 20, 30]}),
    "fvg": Strategy("fvg", fvg, {"min_atr": 0.25, "atr_len": 14, "hold": 10},
                    {"min_atr": [0.0, 0.25, 0.5, 1.0], "hold": [3, 5, 10, 20, 40]}),
    "buy_hold": Strategy("buy_hold", buy_hold, {}, {}),
}

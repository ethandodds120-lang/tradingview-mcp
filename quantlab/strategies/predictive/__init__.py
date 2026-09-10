"""Predictive strategies — a pattern implies what price does next.

The claim these make is about a *specific setup*: a liquidity sweep implies a
reversal, an imbalance implies a retracement. Underneath it is a story about
intent — someone's stops were taken, someone has an unfilled order to defend —
and the strategy is that story made mechanical enough to test.

This is discretionary logic turned into rules. That is what makes it worth
measuring: the discretionary version cannot be falsified, and this version can.
Both families predict. The difference is that a predictive strategy's claim rests
on a formation being read correctly, so it stands or falls on whether that
formation carries information at all — which is exactly what the gauntlet is for.

Everything here is built from primitives.py. Adding the next model should be
composition over that vocabulary, not another copy of the pivot loop.
"""

from __future__ import annotations

import pandas as pd

from . import primitives, tjr
from .fvg import fvg
from .primitives import (
    Bars,
    Gap,
    Swing,
    Sweep,
    atr,
    confirmed_swings,
    detect_bos,
    detect_sweep,
    find_fvg,
    in_killzone,
    is_displacement,
    structure_level,
)


def tjr_signal(df: pd.DataFrame, **params) -> pd.Series:
    """Sweep -> BOS -> FVG, with stops and targets. See tjr.py.

    Path-dependent, so the actual work happens in its own module; this is only the
    adapter that hands the engine a position series.
    """
    return tjr.signal(df, **params)


__all__ = [
    "primitives", "tjr", "tjr_signal", "fvg",
    "Bars", "Gap", "Swing", "Sweep", "atr", "confirmed_swings", "detect_bos",
    "detect_sweep", "find_fvg", "in_killzone", "is_displacement", "structure_level",
]

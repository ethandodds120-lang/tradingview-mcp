"""Strategies, split into two families that make different kinds of claim.

A strategy is a function: (df, **params) -> pd.Series of DESIRED position,
aligned to df.index, where the value at bar t is the position you want to hold
*going into* bar t+1.

The engine shifts signals forward by one bar before applying returns, so a
strategy may freely use df.loc[:t] without introducing lookahead. Never use
.shift(-n) or any future-referencing operation in here.

The two families
----------------
PREDICTIVE  a chart pattern or market structure tells you what price does next.
            A sweep implies a reversal; an imbalance implies a retracement. The
            claim is about a specific setup, and underneath it is a story about
            intent. Discretionary logic made mechanical.

SYSTEMATIC  no view on any individual setup. Harvests a statistical property
            measured across the whole dataset — return autocorrelation,
            volatility clustering, drift. No narrative about why price should
            move; the claim is only ever about an average.

BENCHMARK   neither. What both families have to beat.

Both families predict, and both can be wrong. The distinction is what the claim
rests on: a pattern implying intent, versus a property that has persisted. That
difference shows up in the evidence tag, and it shows up in turnover — predictive
strategies trade far more often, so they need a much larger gross edge to survive
the same cost model. `run.py --head-to-head` puts those two columns side by side.

The metadata is not decoration. `evidence` is the honest state of what is known
about the effect, and `folklore` is not an insult — it means untested here, which
is the entire reason this harness exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import pandas as pd

from .. import engine
from . import benchmarks, predictive, systematic
from .benchmarks import buy_hold, equal_weight, random_entry, random_panel
from .predictive import fvg
from .predictive import tjr_signal as tjr
from .systematic import (donchian, ma_cross, rsi_meanrev, trend_filter,
                         tsmom, xs_momentum, xs_reversal)

#: The families a strategy may belong to. 'benchmark' is not a family in the
#: sense the other two are — it is the control group.
FAMILIES: tuple[str, ...] = ("predictive", "systematic", "benchmark")

#: What is actually known about the effect a strategy claims to harvest.
#:   published  replicated in peer-reviewed or equivalently scrutinised work
#:   folklore   widely traded, widely believed, no independent support found
#:   untested   makes no claim, or the claim has never been examined anywhere
EVIDENCE: tuple[str, ...] = ("published", "folklore", "untested")


@dataclass
class Strategy:
    name: str
    fn: Callable[..., pd.Series]
    params: Dict[str, Any] = field(default_factory=dict)
    # grid used by sensitivity + walk-forward optimization
    grid: Dict[str, list] = field(default_factory=dict)
    # ── metadata ──
    family: str = ""            # 'predictive' | 'systematic' | 'benchmark'
    thesis: str = ""            # one sentence: what has to be true for this to work
    evidence: str = ""          # 'published' | 'folklore' | 'untested'
    source: str = ""            # citation or origin
    # 'single' takes an OHLCV frame and returns a position Series.
    # 'panel' takes a dates x tickers price frame and returns a weight DataFrame.
    kind: str = "single"

    def __post_init__(self):
        # Defaults exist only so the dataclass is constructible; a strategy that
        # reaches the registry without declaring what kind of claim it makes is a
        # mistake, and a silent one if it is not caught here.
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.name}: family must be one of {FAMILIES}, got {self.family!r}")
        if self.evidence not in EVIDENCE:
            raise ValueError(
                f"{self.name}: evidence must be one of {EVIDENCE}, got {self.evidence!r}")
        if self.kind not in ("single", "panel"):
            raise ValueError(
                f"{self.name}: kind must be 'single' or 'panel', got {self.kind!r}")

    @property
    def is_panel(self) -> bool:
        return self.kind == "panel"

    def signal(self, df: pd.DataFrame, **overrides):
        """A position Series, or a weight DataFrame for a panel strategy.

        The same three corrections apply either way — reindex to the data, fill
        gaps flat, and clip to a unit position per name — because a DataFrame
        takes them elementwise.
        """
        p = {**self.params, **overrides}
        sig = self.fn(df, **p)
        return sig.reindex(df.index).fillna(0.0).clip(-1, 1)

    def backtest(self, df: pd.DataFrame, signal, costs, **bt_kwargs):
        """Route to the right engine. This is the single dispatch point.

        Everything in validate.py goes through here rather than calling
        engine.run directly, so a validator never has to know whether it is
        looking at one instrument or twenty.
        """
        if self.is_panel:
            return engine.run_panel(df, signal, costs, **bt_kwargs)
        return engine.run(df, signal, costs, **bt_kwargs)


# ─────────────────────────── registry ───────────────────────────
# Ordered by family so that --compare reads top to bottom as predictive,
# systematic, benchmark.

REGISTRY: Dict[str, Strategy] = {
    # ── predictive: a pattern implies what happens next ──
    "tjr": Strategy(
        "tjr", tjr,
        {"swing_left": 3, "swing_right": 3, "bos_window": 20,
         "fvg_window": 12, "min_fvg_atr": 0.10, "stop_buffer_atr": 0.25,
         "rr": 2.0, "max_hold": 60},
        {"rr": [1.0, 1.5, 2.0, 3.0],
         "bos_window": [10, 20, 40],
         "min_fvg_atr": [0.0, 0.10, 0.25]},
        family="predictive",
        thesis="Stops resting under a swing low are deliberately taken, and the "
               "reversal that follows leaves an imbalance price returns to fill.",
        evidence="folklore",
        source="TJR / ICT trading community. No peer-reviewed support found for "
               "any component of the sequence.",
    ),
    "fvg": Strategy(
        "fvg", fvg, {"min_atr": 0.25, "atr_len": 14, "hold": 10},
        {"min_atr": [0.0, 0.25, 0.5, 1.0], "hold": [3, 5, 10, 20, 40]},
        family="predictive",
        thesis="A three-candle imbalance is unfinished business, so price "
               "continues in the direction that created it.",
        evidence="folklore",
        source="ICT trading community. No peer-reviewed support.",
    ),

    # ── systematic: a statistical property persists ──
    "tsmom": Strategy(
        "tsmom", tsmom, {"lookback": 252, "long_only": False},
        {"lookback": [63, 126, 189, 252, 378, 504]},
        family="systematic",
        thesis="Returns are positively autocorrelated at horizons of months, so "
               "the sign of the trailing return predicts the sign of the next one.",
        evidence="published",
        source="Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum'; "
               "Hurst, Ooi & Pedersen (2017), 'A Century of Evidence on "
               "Trend-Following Investing'.",
    ),
    "ma_cross": Strategy(
        "ma_cross", ma_cross, {"fast": 50, "slow": 200},
        {"fast": [10, 20, 50, 100], "slow": [100, 150, 200, 300]},
        family="systematic",
        thesis="The same trend autocorrelation as tsmom, read off the crossing of "
               "two moving averages instead of a single lookback return.",
        evidence="published",
        source="Trend-following literature; a smoothed restatement of the tsmom "
               "effect rather than an independent one.",
    ),
    "donchian": Strategy(
        "donchian", donchian, {"entry": 55, "exit_n": 20},
        {"entry": [20, 40, 55, 80, 120], "exit_n": [10, 20, 40]},
        family="systematic",
        thesis="A breach of an N-bar extreme marks a trend that continues long "
               "enough to pay for the false breakouts.",
        evidence="published",
        source="Trend-following literature; the original Turtle rules "
               "(Dennis & Eckhardt, 1983).",
    ),
    "trend_filter": Strategy(
        "trend_filter", trend_filter,
        {"ma_len": 50, "confirm": 1, "long_only": True},
        {"ma_len": [20, 30, 50, 80, 100, 150],
         "confirm": [1, 2, 3]},
        family="systematic",
        thesis="Holding only while price is above its own moving average removes "
               "enough of the left tail to beat the asset on risk-adjusted terms — "
               "a drawdown overlay, not a source of return.",
        evidence="published",
        source="Trend-following literature; same mechanism as ma_cross, long-only "
               "because Alpaca does not permit shorting crypto.",
    ),
    "rsi_meanrev": Strategy(
        "rsi_meanrev", rsi_meanrev, {"length": 2, "lower": 10, "upper": 90},
        {"length": [2, 3, 5, 10], "lower": [5, 10, 20, 30]},
        family="systematic",
        thesis="Short-horizon returns are negatively autocorrelated, so an "
               "unusually oversold reading is partially given back.",
        evidence="folklore",
        source="Retail trading literature. Short-horizon reversal does appear in "
               "the academic record, but the 2-period RSI thresholds traded here "
               "are not from it, and the effect rarely clears costs.",
    ),

    # ── systematic, cross-sectional: ranked across names rather than over time ──
    "xs_momentum": Strategy(
        "xs_momentum", xs_momentum, {"lookback": 126, "cut": 0.2, "hold": 21},
        {"lookback": [63, 126, 252], "cut": [0.1, 0.2], "hold": [5, 21]},
        family="systematic", kind="panel",
        thesis="Relative performance persists: the names that led the universe "
               "over the past six months keep leading it over the next month.",
        evidence="published",
        source="Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013), "
               "'Value and Momentum Everywhere'. Crash risk documented in "
               "Daniel & Moskowitz (2016), 'Momentum Crashes'.",
    ),
    "xs_reversal": Strategy(
        "xs_reversal", xs_reversal, {"lookback": 5, "cut": 0.2, "hold": 5},
        {"lookback": [3, 5, 10], "cut": [0.1, 0.2]},
        family="systematic", kind="panel",
        thesis="Over a week, the names that fell most bounce back relative to "
               "the names that rose most.",
        evidence="folklore",
        source="Short-horizon reversal is in the record, but Heston, Korajczyk & "
               "Sadka attribute most of it to sub-hour liquidity imbalance and "
               "bid-ask bounce — costs a retail taker pays rather than earns.",
    ),

    # ── benchmark: what both families have to beat ──
    "buy_hold": Strategy(
        "buy_hold", buy_hold, {}, {},
        family="benchmark",
        thesis="The asset goes up over the sample. No rules, no turnover, no "
               "cost drag.",
        evidence="published",
        source="Equity risk premium; Dimson, Marsh & Staunton (2002).",
    ),
    "equal_weight": Strategy(
        "equal_weight", equal_weight, {}, {},
        family="benchmark", kind="panel",
        thesis="Own the universe. No ranking, no view, minimal turnover — what a "
               "cross-sectional strategy has to beat to have earned its ranking.",
        evidence="published",
        source="The panel analogue of buy_hold.",
    ),
    "random_panel": Strategy(
        "random_panel", random_panel, {"cut": 0.2, "hold": 21, "seed": 0}, {},
        family="benchmark", kind="panel",
        thesis="Nothing has to be true. Same leg count and same rebalance "
               "frequency as a real cross-sectional book, with the names drawn "
               "at random — so the comparison isolates the ranking.",
        evidence="untested",
        source="Not a claim about markets. A control.",
    ),
    "random_entry": Strategy(
        "random_entry", random_entry, {"trade_rate": 0.02, "seed": 0, "hold": 10}, {},
        family="benchmark",
        thesis="Nothing has to be true. This is the null hypothesis with trade "
               "frequency held constant — if a strategy cannot beat it, the rules "
               "contributed nothing and only the exposure did.",
        evidence="untested",
        source="Not a claim about markets. A control. Note it trips the causality "
               "gate for RNG-stream reasons, not lookahead — see benchmarks.py.",
    ),
}


# ─────────────────────────── family helpers ───────────────────────────

def by_family(family: str = "all", kind: str | None = None) -> Dict[str, Strategy]:
    """The registry filtered to one family, and optionally to one kind.

    `kind` matters because the two are not interchangeable: handing a panel
    strategy a single close column, or a single-asset strategy a frame of twenty
    tickers, does not fail cleanly. Callers that have data in hand should say
    which shape it is.
    """
    sel = REGISTRY if family == "all" else None
    if sel is None:
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; have {FAMILIES + ('all',)}")
        sel = {k: s for k, s in REGISTRY.items() if s.family == family}
    if kind is not None:
        sel = {k: s for k, s in sel.items() if s.kind == kind}
    return dict(sel)


def family_of(name: str) -> str:
    """Which family a strategy belongs to."""
    return REGISTRY[name].family


def names_by_family(family: str = "all") -> list[str]:
    """Strategy names in registry order, filtered to one family."""
    return list(by_family(family))


def grouped(family: str = "all", kind: str | None = None) -> Dict[str, list[str]]:
    """{family: [names]} in FAMILIES order, skipping families with no members."""
    sel = by_family(family, kind)
    out: Dict[str, list[str]] = {}
    for fam in FAMILIES:
        members = [n for n, s in sel.items() if s.family == fam]
        if members:
            out[fam] = members
    return out


__all__ = [
    "Strategy", "REGISTRY", "FAMILIES", "EVIDENCE",
    "by_family", "family_of", "names_by_family", "grouped",
    "predictive", "systematic", "benchmarks",
    "tjr", "fvg", "tsmom", "ma_cross", "donchian", "trend_filter",
    "rsi_meanrev", "buy_hold", "random_entry",
    "xs_momentum", "xs_reversal", "equal_weight", "random_panel",
]

"""Shared price-action building blocks for the predictive family.

Every predictive strategy here is assembled from these. They exist so that adding
the next pattern model — order blocks, SMT divergence, turtle soup — is
composition rather than another copy of the same swing-detection loop, and so that
the lookahead guard in `confirmed_swings` is written once, in one place, where it
can be audited.

All primitives take `bars`: either a DataFrame with open/high/low/close columns,
or a `Bars` view built from one. `Bars` holds plain numpy arrays plus a
precomputed ATR. The per-bar primitives are called inside tight loops, and
indexing a DataFrame there costs more than the rest of the model put together, so
build the view once and reuse it:

    bars = Bars.from_frame(df, atr_len=14)

Passing a DataFrame directly still works; it just rebuilds the view on each call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ────────────────────────────── the bar view ──────────────────────────────

@dataclass
class Bars:
    """Column-major view of an OHLC frame, with ATR precomputed."""
    index: pd.Index
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.close)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, atr_len: int = 14) -> "Bars":
        return cls(
            index=df.index,
            open=df["open"].values.astype(float),
            high=df["high"].values.astype(float),
            low=df["low"].values.astype(float),
            close=df["close"].values.astype(float),
            atr=atr(df, atr_len),
        )


def _as_bars(bars, atr_len: int = 14) -> Bars:
    return bars if isinstance(bars, Bars) else Bars.from_frame(bars, atr_len)


def atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Wilder-style ATR as a numpy array, back-filled over the warmup.

    Carried over verbatim from tjr.py so that the gap-size floor keeps measuring
    exactly what it measured before this package existed.
    """
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(1, length), adjust=False).mean().bfill().values


# ────────────────────────────── record types ──────────────────────────────

@dataclass(frozen=True)
class Swing:
    """A confirmed pivot.

    `index` is the bar the extreme printed on. `confirmed_at` is the bar it became
    knowable on. They are not the same bar, and conflating them is the bug this
    whole module is organised around — see `confirmed_swings`.
    """
    index: int
    price: float
    confirmed_at: int


@dataclass(frozen=True)
class Sweep:
    """A pierced level. `reclaimed` says whether price closed back inside it.

    The two are reported separately on purpose. A pierced level has had its resting
    liquidity taken whether or not price recovered, so the caller must retire it
    from the live list either way; only a reclaimed sweep starts a setup. Folding
    them into one boolean loses that distinction and leaves swept levels in the
    book to be swept a second time.
    """
    level: Swing
    reclaimed: bool


@dataclass(frozen=True)
class Gap:
    """A three-candle imbalance. `index` is the bar that completed it."""
    index: int
    low: float
    high: float


# ────────────────────────────── primitives ──────────────────────────────

def confirmed_swings(bars, left: int = 3, right: int = 3) -> tuple[list[Swing], list[Swing]]:
    """Pivot highs and lows, each tagged with the bar it became knowable on.

    THIS LOOKAHEAD GUARD IS THE MOST IMPORTANT LINE IN THE CODEBASE. A pivot at
    index p is not knowable at bar p. It needs `right` bars to its right before
    anyone can say it was a pivot at all, so it is only admissible from bar
    p + right onward — which is what `confirmed_at` records. Every caller must gate
    on `confirmed_at`, never on `index`.

    Detecting pivots with a centred window instead — rolling(..., center=True),
    scipy.signal.argrelextrema, or any max()/idxmax() over a window straddling p —
    marks the pivot at the bar it printed rather than the bar it was confirmed.
    Every sweep downstream then gets identified using bars that had not printed
    yet, and the backtest reports an edge that could never have been traded.
    Skipping this is the single most common way an ICT-style backtest produces fake
    results.

    Carried over exactly from the loop that used to be inlined in tjr.py, including
    the asymmetric comparisons: `>=` against the left window and `>` against the
    right, so that a flat double top confirms on its later bar rather than its
    earlier one.
    """
    b = _as_bars(bars)
    left, right = max(1, int(left)), max(1, int(right))
    h, l, n = b.high, b.low, b.n

    highs: list[Swing] = []
    lows: list[Swing] = []
    # p is the pivot bar; i is the bar its right-hand window closes on. The upper
    # bound stops at n - right because a pivot whose confirmation bar has not
    # printed yet is not a pivot yet.
    for p in range(left, n - right):
        i = p + right
        if h[p] >= h[p - left:p].max() and h[p] > h[p + 1:i + 1].max():
            highs.append(Swing(p, float(h[p]), i))
        if l[p] <= l[p - left:p].min() and l[p] < l[p + 1:i + 1].min():
            lows.append(Swing(p, float(l[p]), i))
    return highs, lows


def detect_sweep(bars, i: int, levels: list[Swing], direction: int = 1) -> Sweep | None:
    """Did bar i take the liquidity resting at the most recent level?

    `levels` is the live list of confirmed swings, oldest first — only the newest
    one has liquidity worth taking, because anything older sits behind it. Returns
    None if nothing was pierced, otherwise a Sweep saying whether price also closed
    back inside. direction 1 sweeps lows (a long setup), -1 sweeps highs.
    """
    b = _as_bars(bars)
    if not levels:
        return None
    lvl = levels[-1]
    if direction > 0:
        if b.low[i] < lvl.price:
            return Sweep(lvl, bool(b.close[i] > lvl.price))
    else:
        if b.high[i] > lvl.price:
            return Sweep(lvl, bool(b.close[i] < lvl.price))
    return None


def structure_level(swings: list[Swing], after_idx: int, direction: int) -> float | None:
    """The level whose break would confirm a change of structure.

    The most recent opposing pivot that formed *after* the sweep. Picked once, when
    the setup opens, and then frozen: re-deriving it on every bar would let a pivot
    that confirms during the BOS window move the goalposts mid-setup, which is a
    different model from the one this package has been testing.

    `direction` is accepted for symmetry with the rest of the vocabulary and to
    keep call sites self-documenting; the caller passes the opposing swing list, so
    the selection rule itself is the same for longs and shorts.
    """
    return next((s.price for s in reversed(swings) if s.index > after_idx), None)


def detect_bos(bars, i: int, level: float, direction: int) -> bool:
    """Break of structure: bar i *closes* through the frozen level.

    Takes the level rather than the swing list precisely because it must be frozen
    at sweep time — see `structure_level`. A close, not a wick: an intrabar poke
    through a high that closes back below it is the sweep pattern, not the break.
    """
    b = _as_bars(bars)
    return bool(b.close[i] > level) if direction > 0 else bool(b.close[i] < level)


def find_fvg(bars, i: int, direction: int, min_atr: float = 0.0) -> Gap | None:
    """The three-candle imbalance completed at bar i, if it clears the size floor.

    Bullish: bar i's low prints above bar i-2's high, leaving untraded space
    between them. Bearish is the mirror. `min_atr` measures the gap against ATR at
    bar i, so the floor scales with volatility instead of being a fixed number of
    points.
    """
    b = _as_bars(bars)
    if i < 2:
        return None
    floor = min_atr * b.atr[i]
    if direction > 0 and b.low[i] > b.high[i - 2] and (b.low[i] - b.high[i - 2]) >= floor:
        return Gap(i, float(b.high[i - 2]), float(b.low[i]))
    if direction < 0 and b.high[i] < b.low[i - 2] and (b.low[i - 2] - b.high[i]) >= floor:
        return Gap(i, float(b.high[i]), float(b.low[i - 2]))
    return None


def is_displacement(bars, i: int, body_frac: float = 0.5) -> bool:
    """Impulse test: is bar i's body at least `body_frac` of its whole range?

    NOT USED BY ANY STRATEGY TODAY. It is here because order-block and turtle-soup
    models need it and it belongs with the rest of the vocabulary. tjr's break of
    structure is a close-through test with no impulse filter, and bolting this onto
    it would change tjr's results — which this refactor is not permitted to do.
    Wire it in deliberately, as a new strategy, and let the gauntlet judge it.
    """
    b = _as_bars(bars)
    rng = b.high[i] - b.low[i]
    if rng <= 0:
        return False
    return bool(abs(b.close[i] - b.open[i]) / rng >= body_frac)


#: Session windows in UTC hours, [start, end). These are the ICT convention
#: (London 02:00-05:00 ET, New York 07:00-10:00 ET) expressed against a fixed
#: UTC-5 offset. They do NOT track daylight saving, so for roughly half the year
#: they sit an hour off the New York clock. Anything that needs the boundary to be
#: exact should localise the index itself rather than rely on these.
KILLZONES: dict[str, tuple[int, int]] = {
    "london": (7, 10),
    "newyork": (12, 15),
}


def in_killzone(ts, zones: tuple[str, ...] = ("london", "newyork")) -> bool:
    """Is this timestamp inside one of the named session windows?

    NOT USED BY ANY STRATEGY TODAY, for the same reason as `is_displacement`: tjr
    takes setups at any hour, and adding a session filter would change its results.
    It is also meaningless on daily bars — check your timeframe before reaching for
    it. Provided as vocabulary for the next model, not as a default.
    """
    hour = pd.Timestamp(ts).hour
    return any(KILLZONES[z][0] <= hour < KILLZONES[z][1] for z in zones if z in KILLZONES)

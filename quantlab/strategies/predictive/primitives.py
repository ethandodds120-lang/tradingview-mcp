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

import math
from dataclasses import dataclass, field
from typing import NamedTuple

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


# ────────────────────────────── the wider vocabulary (DESIGN-tjr-human.md §2) ──────────────────────────────
#
# Everything from here down was added for the wider TJR spec: resampled swings,
# volume-at-price, order blocks and breakers. None of it is wired into a
# registered strategy. tjr_intraday.build_context does the 5-minute case of
# `resample_context` with its own code; it could delegate here later, but that
# is a separate change with a before/after diff of its own.

_ET = "America/New_York"
_SESSION_MINUTES = 23 * 60      # CME index futures: 18:00 -> 17:00 ET the next day


@dataclass
class Context:
    """Resampled bars, and how each input bar maps to them.

    Mirrors tjr_intraday.Context field for field so a strategy can read either.
    `bars` is the resampled view; `idx` says which resampled bar each input bar
    sits in; `done` marks the input bar that completes its bin; `last` is the
    newest COMPLETE bin an input bar may read (-1 before the first); `first` is,
    per bin, the first input bar inside it. `frame` keeps the resampled OHLCV
    for anything Bars does not carry (volume). Nothing should read a bin
    through `idx` alone — a bin that is still printing is not a bar yet.
    """
    bars: Bars
    idx: np.ndarray
    done: np.ndarray
    last: np.ndarray
    first: np.ndarray
    frame: pd.DataFrame | None = None
    pair_high: np.ndarray | None = None
    pair_low: np.ndarray | None = None


def _hhmm(text: str) -> int:
    h, m = str(text).split(":")
    return int(h) * 60 + int(m)


def resample_context(df: pd.DataFrame, minutes: int, anchor_et: str = "18:00",
                     session_minutes: int = _SESSION_MINUTES, bar_minutes: int | None = None,
                     atr_len: int = 14) -> Context:
    """`minutes`-wide bins on the New York clock, anchored at the CME open.

    The index is tz-naive UTC, as everywhere in the codebase; it is converted to
    ET so the bins follow the wall clock through DST (a 4-hour bin opens at
    18:00 ET in both January and July). Bin k of a session covers minutes
    [k*minutes, (k+1)*minutes) after the anchor, and the last bin of a session is
    clipped at the session close, so on a 23-hour session the 14:00 4-hour bin
    runs 14:00-16:55 and completes at the 16:55 bar.

    A bin is complete at the input bar whose stamp is the LAST slot the bin can
    hold — by minute arithmetic, never by looking at the row after it. That is
    what keeps a truncation in the middle of a bin causal: the partial bin is
    never readable as the newest bin, and every completed bar is the same
    whether or not the frame continues. A bin whose last slot never prints (a
    session that closes early, such as Good Friday's 09:10 close) is still a
    row of `frame` and of `bars`: it never completes at its own time, so `last`
    skips it, but once the next bin completes it is readable like any other bar
    and `confirmed_swings` may pivot on it. That is the bar a chart shows, and
    it is causal: at any input bar the readable bins are the same whether the
    frame is cut there or runs on. (This differs from tjr_intraday's 1-minute
    rule, which skips a 5-minute bin missing its :04 minute -- a gap inside a
    session, not the end of one.) Empty bins do not exist, so the resampled
    series has no phantom bars for `confirmed_swings` to count.

    Bins are labelled by their nominal open time, in the same tz-naive UTC
    convention as the input. `bar_minutes` is the input bar size, inferred from
    the smallest gap in the index when not given.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
        raise ValueError("resample_context: needs a non-empty DatetimeIndex")
    minutes = int(minutes)
    if bar_minutes is None:
        gaps = np.diff(df.index.to_numpy().astype("datetime64[m]").astype(np.int64))
        bar_minutes = int(gaps[gaps > 0].min()) if (gaps > 0).any() else minutes
    et = (df.index.tz_localize("UTC") if df.index.tz is None else df.index).tz_convert(_ET)
    wall = (et.hour * 60 + et.minute).to_numpy()
    anchor = _hhmm(anchor_et)
    since = (wall - anchor) % (24 * 60)          # minutes into the session
    # The session is named for the calendar date it ends on: bars at or after
    # the anchor belong to the next date. DST is already in the wall clock.
    midnight = et.tz_localize(None).normalize().as_unit("ns").to_numpy()
    day = midnight + np.where(wall >= anchor, 1, 0).astype("timedelta64[D]")
    k = since // minutes
    last_slot = np.minimum((k + 1) * minutes, session_minutes) - bar_minutes
    done = since == last_slot

    # Bins numbered in time order: (day, k) is monotone along the index.
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays([day, k]), sort=True)
    idx = codes.astype(np.int64)
    first = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    # The nominal open is the first bar's stamp minus its offset into the bin.
    # Within a session there is no DST step, so this minute arithmetic holds in
    # UTC exactly as it does on the wall clock.
    offset = (since[first] - k[first] * minutes).astype("timedelta64[m]")
    label = pd.DatetimeIndex(df.index.to_numpy()[first] - offset, name=df.index.name)

    cols = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        cols["volume"] = "sum"
    has_pair = {"pair_high", "pair_low"} <= set(df.columns)
    if has_pair:
        cols.update(pair_high="max", pair_low="min")
    ctx = df[list(cols)].groupby(idx).agg(cols)
    ctx.index = label
    # idx never decreases, so the running maximum of the completed bins is the
    # newest one each input bar may read.
    last = np.maximum.accumulate(np.where(done, idx, -1))
    return Context(
        bars=Bars.from_frame(ctx, atr_len), idx=idx, done=done, last=last, first=first,
        frame=ctx,
        pair_high=ctx["pair_high"].to_numpy(dtype=float) if has_pair else None,
        pair_low=ctx["pair_low"].to_numpy(dtype=float) if has_pair else None,
    )


# ────────────────────────────── volume at price ──────────────────────────────

class Profile(NamedTuple):
    """Volume at price. `edges` has one more entry than `volume`; bin b covers
    [edges[b], edges[b+1]) and its price is the midpoint."""
    edges: np.ndarray
    volume: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def width(self) -> float:
        return float(self.edges[1] - self.edges[0])


def volume_profile(high, low, volume, tick: float = 0.25, bin_ticks: int = 4,
                   edges: np.ndarray | None = None) -> Profile:
    """Volume-at-price over a set of bars, each bar's volume spread uniformly
    across the bins its [low, high] covers.

    Bin width is `tick * bin_ticks`, and the grid is anchored at multiples of
    that width so profiles of different bar sets line up. Pass `edges` to build
    on a shared grid — a composite is then the sum of its sessions' `volume`
    arrays. A bar with no range puts all its volume in the bin holding its
    price; a bar wholly outside a given grid contributes nothing.

    Which bars go in is the caller's business, and so is the lookahead: a
    profile of bars 18:00 (D-2) .. 16:55 (D-1) read on day D is causal because
    every bar in it printed before D's first bar.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    volume = np.asarray(volume, dtype=float)
    w = float(tick) * int(bin_ticks)
    if edges is None:
        if len(high) == 0:
            return Profile(np.array([0.0, w]), np.zeros(1))
        lo = np.floor(np.nanmin(low) / w + 1e-9) * w
        hi = np.floor(np.nanmax(high) / w + 1e-9) * w + w
        edges = lo + w * np.arange(int(round((hi - lo) / w)) + 1)
    edges = np.asarray(edges, dtype=float)
    nb = len(edges) - 1
    lo, w = float(edges[0]), float(edges[1] - edges[0])
    vols = np.zeros(nb)
    for h, l, v in zip(high, low, volume):
        if not (v > 0) or np.isnan(h) or np.isnan(l):
            continue
        b0 = int(np.floor((l - lo) / w + 1e-9))
        b1 = int(np.floor((h - lo) / w + 1e-9))
        if h <= l or b0 == b1:
            if 0 <= b0 < nb:
                vols[b0] += v
            continue
        b0, b1 = max(b0, 0), min(b1, nb - 1)
        if b0 > b1:
            continue
        overlap = np.minimum(h, edges[b0 + 1:b1 + 2]) - np.maximum(l, edges[b0:b1 + 1])
        vols[b0:b1 + 1] += v * np.clip(overlap, 0.0, None) / (h - l)
    return Profile(edges, vols)


def vwap(high, low, close, volume) -> float:
    """Volume-weighted typical price, (h+l+c)/3. The tie-breaker `poc` uses."""
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3.0
    v = np.asarray(volume, float)
    return float((tp * v).sum() / v.sum()) if v.sum() > 0 else math.nan


def poc(profile: Profile, near: float | None = None) -> float:
    """Point of control: the price of the bin with the most volume.

    Ties go to the bin nearest `near` (the session's VWAP, in the §2 use), or
    to the lowest-priced of them when nothing is given to break the tie with.
    """
    vols = np.asarray(profile.volume, dtype=float)
    top = np.flatnonzero(vols == vols.max())
    centers = profile.centers[top]
    if near is not None and len(top) > 1 and not math.isnan(near):
        return float(centers[int(np.argmin(np.abs(centers - near)))])
    return float(centers[0])


def hvn(profile: Profile, smooth: int = 5, frac: float = 0.5, max_n: int = 6,
        near: float | None = None) -> list[float]:
    """High-volume nodes: local maxima of the smoothed profile that hold at
    least `frac` of its peak.

    The smoothing is a centred `smooth`-bin moving average on the PRICE axis of
    a profile built from past bars. It is not a rolling(center=True) on time:
    every bin already holds only bars that have printed, so a bin's neighbours
    above and below it are just as past as it is, and looking at both is not
    lookahead. At the ends of the grid the window is shorter, and the mean is
    over the bins it actually covers, so an edge bin is not penalised for
    having no neighbour.

    A local maximum is `>=` its lower neighbour and `>` its upper one, the same
    asymmetry `confirmed_swings` uses, so a flat plateau yields one node, at
    its top. Returns at most `max_n` prices, nearest to `near` first when it is
    given (the funnel names them HVN_1.. by that order), else largest volume
    first.
    """
    vols = np.asarray(profile.volume, dtype=float)
    nb = len(vols)
    if nb == 0 or not vols.max() > 0:
        return []
    smooth = max(1, int(smooth))
    kernel = np.ones(smooth)
    s = np.convolve(vols, kernel, mode="same") / np.convolve(np.ones(nb), kernel, mode="same")
    below = np.r_[-np.inf, s[:-1]]
    above = np.r_[s[1:], -np.inf]
    peaks = np.flatnonzero((s >= below) & (s > above) & (s >= frac * s.max()))
    centers = profile.centers[peaks]
    if near is not None and not math.isnan(near):
        order = np.argsort(np.abs(centers - near), kind="stable")
    else:
        order = np.argsort(-s[peaks], kind="stable")
    return [float(p) for p in centers[order][:max(0, int(max_n))]]


# ────────────────────────────── blocks ──────────────────────────────

@dataclass(frozen=True)
class Block:
    """A candle used as a zone. `index` is the bar it became usable on — the
    candle itself for an order block, the bar that violated it for a breaker —
    and `candle` is the bar whose [low, high] the zone is."""
    index: int
    low: float
    high: float
    candle: int


def _opposing(b: Bars, k: int, side: int) -> bool:
    """Is bar k a candle against `side`: bearish for a long, bullish for a short?"""
    return bool(b.close[k] < b.open[k]) if side > 0 else bool(b.close[k] > b.open[k])


def order_block(bars, sweep_idx: int, side: int, start: int = 0) -> Block | None:
    """The last opposing candle at or before the sweep bar, as a zone.

    Long: the last bearish candle (close < open) with index <= sweep_idx, zone
    [low, high]. Shorts mirror on bullish candles. `start` bounds the search
    below (a day start, say); the default searches back to the first bar.
    """
    b = _as_bars(bars)
    for k in range(min(sweep_idx, b.n - 1), max(start, 0) - 1, -1):
        if _opposing(b, k, side):
            return Block(k, float(b.low[k]), float(b.high[k]), k)
    return None


def breaker_block(bars, day_start: int, sweep_idx: int, i: int, side: int,
                  swings: list[Swing]) -> Block | None:
    """A breaker: the last bullish candle before a swing high of the day, which
    a bar since the sweep has closed above — checked through bar i.

    Long side: `swings` are swing HIGHS. For each with day_start <= index <
    sweep_idx that is admissible at bar i (confirmed_at <= i — the guard is
    re-applied here even when the caller passes an admitted list), the
    candidate candle is the last close > open bar strictly before the swing
    bar and on or after day_start. It is a breaker once some bar j in
    [sweep_idx, i] closes above its high; the zone is then [low, high] and its
    `index` is that j. Where several qualify, the most recently violated wins
    (ties: the one built on the later swing). Shorts mirror on swing lows,
    bearish candles, and a close below the low.
    """
    b = _as_bars(bars)
    best, best_swing = None, -1
    for s in swings:
        if s.confirmed_at > i or not (day_start <= s.index < sweep_idx):
            continue
        # The candle is on the side of the move, i.e. opposing the *short* side
        # for a long breaker: the last bullish bar before the high.
        candle = next((k for k in range(s.index - 1, day_start - 1, -1) if _opposing(b, k, -side)), None)
        if candle is None:
            continue
        lo, hi = float(b.low[candle]), float(b.high[candle])
        j = next((j for j in range(sweep_idx, i + 1)
                  if ((b.close[j] > hi) if side > 0 else (b.close[j] < lo))), None)
        if j is None:
            continue
        if best is None or (j, s.index) > (best.index, best_swing):
            best, best_swing = Block(j, lo, hi, candle), s.index
    return best

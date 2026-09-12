"""TJR intraday: his session model on 5-minute ES / NQ, made mechanical.

The daily-bar `tjr` has no clock. This one is all clock: six liquidity levels
that are fixed before the New York open, a sweep of one of them in the first
twenty minutes, a break of structure and an entry in the next twenty, flat by
the close. Every choice he leaves to the trader is written down in
DESIGN-tjr-intraday.md and made here with no discretion left. Same surface as
tjr, plus a per-day log:

    simulate(df, **params) -> SimResult      trades + position series
    signal(df, **params)   -> pd.Series      adapter for the engine
    day_log(df, **params)  -> pd.DataFrame   one row per trading day: how far it got

The model, in words (every time below is ET, bar stamps are bar opens)
----------------------------------------------------------------------
LEVELS  Asia high/low (bars 18:00-02:55), London high/low (03:00-08:25), the
        previous session's high/low (its 18:00-16:55). All six are final before
        the 09:30 bar prints.
SWEEP   a bar stamped 09:30-09:45 wicks through one of the three lows and closes
        back above it: long bias, and the day's direction is fixed. Several lows
        in one wick: the lowest. Shorts mirror on the highs. A wick through a low
        AND a high is ambiguous and skipped.
SMT     (variant) on that bar the other index must NOT have swept its matching
        level. Divergence says manipulation; agreement says breakdown.
BOS     no later than 10:05, a close above the last confirmed swing high that
        existed at the sweep bar.
ZONE    a bullish FVG on the displacement leg, or an earlier bearish FVG of the
        day that a bar since the sweep closed above (inverted). Its top must sit
        at or below the midpoint of [sweep low, BOS level]: discount only.
FILL    a bar stamped 09:50-10:05, after both the BOS bar and the zone's bar,
        trades into the zone. Stop under the sweep wick, target at the nearest of
        the three highs above entry. Nothing above: no trade.
EXIT    stop, target, or the 15:55 close. One setup per day, first one wins.

Built from primitives
---------------------
confirmed_swings, detect_bos and find_fvg are the shared vocabulary; the state
machine below only decides which one is consulted when. The sweep test is not
primitives.detect_sweep: that one retires the newest swing from a live list,
while here the levels are six session numbers that never move during the day,
so `sweep_of_levels` lives in this module.

Lookahead
---------
Same guard as tjr: a pivot is admitted only once bar i has reached its
`confirmed_at` tag. The levels come from one groupby over the whole frame, which
is safe for the reason written at `session_levels`: every window that feeds a
day's levels has closed before that day's sweep window opens, so no bar that
reads a level can see a bar that moved it. The position series keeps tjr's
eod rule for the same reason tjr does — see the comment at the exit block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .primitives import Bars, atr, confirmed_swings, detect_bos, find_fvg  # noqa: F401  (atr re-exported for tests)
from .tjr import _TRADE_COLS as _TJR_COLS
from .tjr import SimResult, format_trade_stats, trade_stats  # noqa: F401  (reused, not copied)

ET = ZoneInfo("America/New_York")

#: The clock, in minutes after midnight ET. Half-open windows, [start, end).
SESSION_OPEN = 18 * 60                        # 18:00 (D-1) opens trading day D
ASIA_END = 3 * 60                             # Asia bars: 18:00 .. 02:55
LONDON_END = 8 * 60 + 30                      # London bars: 03:00 .. 08:25
SWEEP_WINDOW = (9 * 60 + 30, 9 * 60 + 50)     # 09:30 09:35 09:40 09:45
ENTRY_WINDOW = (9 * 60 + 50, 10 * 60 + 10)    # 09:50 09:55 10:00 10:05

LOWS = ("ASIA_L", "LON_L", "PDL")
HIGHS = ("ASIA_H", "LON_H", "PDH")

_TRADE_COLS = _TJR_COLS + ["sweep_level", "target_level", "zone_kind", "smt", "session_day"]
_DAY_COLS = ["session_day", "side", "sweep_time", "sweep_level", "bos_time",
             "zone_kind", "fill_time", "outcome"]

#: The defaults of `simulate`, in one place so day_log and the registry agree with it.
DEFAULTS = {"swing_left": 2, "swing_right": 2, "atr_len": 14, "stop_buffer_atr": 0.25,
            "min_fvg_atr": 0.0, "allow_ifvg": True, "smt": False, "flat_at": "15:55"}


# ────────────────────────────── the clock ──────────────────────────────

@dataclass
class Clock:
    """Per-bar ET view of a tz-naive UTC index. All arrays are one per bar."""
    minutes: np.ndarray      # hour * 60 + minute, ET
    day: np.ndarray          # datetime64[ns] midnight: the CME day the bar belongs to
    in_asia: np.ndarray
    in_london: np.ndarray
    in_sweep: np.ndarray
    in_entry: np.ndarray
    is_flat: np.ndarray      # stamped at or after flat_at, still inside its own day


def _hhmm(text: str) -> int:
    h, m = str(text).split(":")
    return int(h) * 60 + int(m)


def session_clock(index: pd.Index, flat_at: str = "15:55") -> Clock | None:
    """The ET clock for every bar, or None when the frame has no 09:30 bar.

    None is the "nothing to do" signal: daily bars, an RTH-only equity file, the
    synthetic walk. The strategy goes all-flat on it instead of raising, so
    `--compare --synthetic` keeps working with this strategy registered.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    et = (index.tz_localize("UTC") if index.tz is None else index).tz_convert(ET)
    minutes = (et.hour * 60 + et.minute).to_numpy()
    if not (minutes == SWEEP_WINDOW[0]).any():
        return None
    # The CME day is named for the date it ends on, so bars stamped 18:00 or
    # later belong to the next calendar date. Wall-clock ET: DST already applied.
    midnight = et.tz_localize(None).normalize().as_unit("ns").to_numpy()
    day = midnight + np.where(minutes >= SESSION_OPEN, 1, 0).astype("timedelta64[D]")
    flat = _hhmm(flat_at)
    return Clock(
        minutes=minutes,
        day=day,
        in_asia=(minutes >= SESSION_OPEN) | (minutes < ASIA_END),
        in_london=(minutes >= ASIA_END) & (minutes < LONDON_END),
        in_sweep=(minutes >= SWEEP_WINDOW[0]) & (minutes < SWEEP_WINDOW[1]),
        in_entry=(minutes >= ENTRY_WINDOW[0]) & (minutes < ENTRY_WINDOW[1]),
        is_flat=(minutes >= flat) & (minutes < SESSION_OPEN),
    )


# ────────────────────────────── the levels ──────────────────────────────

def session_levels(clock: Clock, high: np.ndarray, low: np.ndarray) -> dict[str, np.ndarray]:
    """The six levels, per bar, read off the bar's own trading day. NaN = no level.

    One groupby over the whole frame. That is causal, not lookahead, because of
    when each window closes relative to the only bars that read it — bars stamped
    09:30 (D) or later:
      ASIA_*  bars 18:00 (D-1) .. 02:55 (D)     closed six and a half hours earlier
      LON_*   bars 03:00 .. 08:25 (D)           closed an hour earlier
      PD*     every bar of the previous session  closed at 17:00 the day before
    Truncate the frame at any bar of D from 09:30 on and every window feeding D's
    levels is already complete, so the numbers it reads do not change. Truncate
    earlier and D has no sweep-window bar and never reads them. That is exactly
    what validate.causality_check probes.
    """
    f = pd.DataFrame({"day": clock.day, "high": high, "low": low})
    lv = pd.DataFrame(index=pd.Index(np.unique(clock.day), name="day"))
    asia = f[clock.in_asia].groupby("day")
    lon = f[clock.in_london].groupby("day")
    lv["ASIA_H"] = asia["high"].max()
    lv["ASIA_L"] = asia["low"].min()
    lv["LON_H"] = lon["high"].max()
    lv["LON_L"] = lon["low"].min()
    # The previous trading day in the frame, not the previous calendar date:
    # Monday's PD candle is Friday's. The first day in the frame has none.
    sess = f.groupby("day")
    lv["PDH"] = sess["high"].max().shift(1)
    lv["PDL"] = sess["low"].min().shift(1)
    per_bar = lv.reindex(clock.day)
    return {name: per_bar[name].to_numpy(dtype=float) for name in lv.columns}


# ────────────────────────────── the sweep ──────────────────────────────

def sweep_of_levels(high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int,
                    lows: dict[str, float], highs: dict[str, float]) -> tuple[int, str, float] | None:
    """Did bar i take one of the resting pools? -> (side, level name, level) or None.

    Wick through the level and close back inside. When one wick took several lows
    the lowest is the sweep — the most liquidity taken — and closing back above it
    means closing above the rest too. A wick through a low AND a high is ambiguous
    and skipped whichever side it closed on. NaN levels do not exist.
    """
    below = {k: v for k, v in lows.items() if not math.isnan(v) and low[i] < v}
    above = {k: v for k, v in highs.items() if not math.isnan(v) and high[i] > v}
    if below and above:
        return None
    if below:
        name = min(below, key=below.get)
        if close[i] > below[name]:
            return 1, name, below[name]
    elif above:
        name = max(above, key=above.get)
        if close[i] < above[name]:
            return -1, name, above[name]
    return None


# ────────────────────────────── the state machine ──────────────────────────────

@dataclass(frozen=True)
class Zone:
    """An entry zone. `index` is the bar it became usable on: the bar that
    completed an FVG, or the bar that inverted a bearish one."""
    low: float
    high: float
    index: int
    kind: str        # 'fvg' | 'ifvg'
    formed: int      # the bar the gap itself completed on


def _discount(z, side: int, eq: float) -> bool:
    """His rule, not a filter: longs enter at or below equilibrium, shorts above."""
    return z.high <= eq if side > 0 else z.low >= eq


def _inverted(gap, side: int, close: float) -> bool:
    """A bearish gap inverts when a bar closes above it; a bullish one, below."""
    return close > gap.high if side > 0 else close < gap.low


def _freshest(zones: list[Zone]) -> Zone | None:
    # Most recently usable wins. Same bar: the FVG that completed there over the
    # gap that inverted there; among inversions, the more recently formed gap.
    return max(zones, key=lambda z: (z.index, z.kind == "fvg", z.formed)) if zones else None


def _run(df: pd.DataFrame, swing_left: int, swing_right: int, atr_len: int,
         stop_buffer_atr: float, min_fvg_atr: float, allow_ifvg: bool, smt: bool,
         flat_at: str) -> tuple[list, np.ndarray, list]:
    """Bar by bar. Returns (trades, position array, per-day log rows)."""
    swing_right = max(1, int(swing_right))
    swing_left = max(1, int(swing_left))
    n = len(df)
    pos = np.zeros(n)
    trades: list = []
    days: list = []

    clock = session_clock(df.index, flat_at)
    if clock is None:
        return trades, pos, days
    if smt and not {"pair_high", "pair_low"} <= set(df.columns):
        raise ValueError("tjr_intraday: smt=True needs pair_high/pair_low columns; "
                         "load the frame with data.load_futures_pair (run.py --pair-csv)")

    bars = Bars.from_frame(df, atr_len)
    o, h, l, c, atr_ = bars.open, bars.high, bars.low, bars.close, bars.atr
    idx = bars.index
    minutes, day = clock.minutes, clock.day
    levels = session_levels(clock, h, l)
    if smt:
        pair_h = df["pair_high"].to_numpy(dtype=float)
        pair_l = df["pair_low"].to_numpy(dtype=float)
        pair_levels = session_levels(clock, pair_h, pair_l)

    # Every pivot in the frame, tagged with the bar it became knowable on. The
    # cursors below admit one only once bar i has reached that tag.
    all_highs, all_lows = confirmed_swings(bars, swing_left, swing_right)
    hi_cur = lo_cur = 0
    swing_highs: list = []
    swing_lows: list = []

    setup = None        # {side, sweep_idx, extreme, level, bos_level, stage, bos_idx, eq, zone, pending}
    trade = None        # {side, entry_idx, entry, stop, target, risk, names..., day}
    cur_day = None
    day_start = 0
    day_done = False    # the day's one setup has been used, one way or another
    row = None          # the log row for the day in progress

    def close_trade(i, price, reason):
        nonlocal trade
        trades.append({
            "side": trade["side"],
            "entry_time": idx[trade["entry_idx"]],
            "exit_time": idx[i],
            "entry": trade["entry"],
            "exit": price,
            "stop": trade["stop"],
            "target": trade["target"],
            "r": trade["side"] * (price - trade["entry"]) / trade["risk"],
            "bars": i - trade["entry_idx"],
            "reason": reason,
            "sweep_level": trade["sweep_level"],
            "target_level": trade["target_level"],
            "zone_kind": trade["zone_kind"],
            "smt": bool(smt),
            "session_day": pd.Timestamp(trade["day"]),
        })
        trade = None

    for i in range(n):
        # -- 0. a new trading day: forget the old setup, keep an open trade --------
        if day[i] != cur_day:
            if row is not None:
                days.append(row)
            cur_day, day_start, setup, day_done, row = day[i], i, None, False, None

        # -- 1. admit the pivots whose right-hand window just closed ---------------
        while hi_cur < len(all_highs) and all_highs[hi_cur].confirmed_at <= i:
            swing_highs.append(all_highs[hi_cur])
            hi_cur += 1
        while lo_cur < len(all_lows) and all_lows[lo_cur].confirmed_at <= i:
            swing_lows.append(all_lows[lo_cur])
            lo_cur += 1

        # -- 2. manage the open position -------------------------------------------
        if trade is not None:
            side = trade["side"]
            price, reason = None, None
            if side > 0:
                if l[i] <= trade["stop"]:
                    price, reason = trade["stop"], "stop"
                elif h[i] >= trade["target"]:
                    price, reason = trade["target"], "target"
            else:
                if h[i] >= trade["stop"]:
                    price, reason = trade["stop"], "stop"
                elif l[i] <= trade["target"]:
                    price, reason = trade["target"], "target"
            # Flat at the 15:55 close. If the day had no such bar (an early close),
            # the first bar of the next session is the first chance to be flat.
            if price is None and (clock.is_flat[i] or day[i] != trade["day"]):
                price, reason = c[i], "flat"
            if price is None and i == n - 1:
                price, reason = c[i], "eod"

            if price is not None:
                close_trade(i, price, reason)
                # On the last bar the position value is never used (the engine
                # lags by one and there is no next bar), so hold it rather than
                # flatten it. That keeps the series identical to the one a run on
                # a longer history would produce, which is what makes
                # validate.causality_check able to compare every overlapping bar.
                if reason == "eod":
                    pos[i] = side
            else:
                pos[i] = side
            continue        # no new setup work on a bar we were already in

        # -- 3. no setup yet: is this a sweep-window bar that took a pool? ----------
        if setup is None:
            if day_done or not clock.in_sweep[i]:
                continue
            if row is None:
                row = {"session_day": pd.Timestamp(day[i]), "side": 0, "sweep_time": pd.NaT,
                       "sweep_level": None, "bos_time": pd.NaT, "zone_kind": None,
                       "fill_time": pd.NaT, "outcome": "no_sweep"}
            lows = {k: levels[k][i] for k in LOWS}
            highs = {k: levels[k][i] for k in HIGHS}
            swept = sweep_of_levels(h, l, c, i, lows, highs)
            if swept is None:
                continue
            side, name, level = swept
            # The first qualifying bar fixes the day's direction, whatever comes of it.
            day_done = True
            row.update(side=side, sweep_time=idx[i], sweep_level=name)

            if smt:
                # The pair must have held its matching level. A NaN pair level cannot
                # be held, and it cannot happen anyway: the frames are inner-joined,
                # so the pair has bars wherever the traded index has a level.
                held = (pair_l[i] >= pair_levels[name][i]) if side > 0 else (pair_h[i] <= pair_levels[name][i])
                if not held:
                    row["outcome"] = "smt"
                    continue
            # The most recent opposing swing that existed at this bar. Frozen now:
            # a pivot confirming later must not move the goalposts.
            opposing = swing_highs if side > 0 else swing_lows
            if not opposing:
                row["outcome"] = "no_swing"
                continue
            extreme = l[i] if side > 0 else h[i]
            setup = {"side": side, "sweep_idx": i, "extreme": extreme, "level": name,
                     "bos_level": opposing[-1].price, "stage": "bos", "bos_idx": None,
                     "eq": None, "zone": None, "pending": []}
            row["outcome"] = "no_bos"
            continue        # the break has to come on a later bar

        # -- 4. a setup is live: it dies at 10:10 whatever stage it is in -----------
        side = setup["side"]
        if minutes[i] >= ENTRY_WINDOW[1]:
            setup = None
            continue

        if setup["stage"] == "bos":
            if not detect_bos(bars, i, setup["bos_level"], side):
                continue
            setup["stage"], setup["bos_idx"] = "entry", i
            row["bos_time"], row["outcome"] = idx[i], "no_zone"
            # Dealing range [sweep extreme, BOS level]; entries only on the cheap half.
            eq = 0.5 * (setup["extreme"] + setup["bos_level"])
            setup["eq"] = eq
            zones: list[Zone] = []
            # The displacement leg may already have left a gap behind it. Freshest
            # first, and the first one on the right side of EQ is the one.
            for k in range(i, setup["sweep_idx"] + 1, -1):
                z = find_fvg(bars, k, side, min_fvg_atr)
                if z and _discount(z, side, eq):
                    zones.append(Zone(z.low, z.high, k, "fvg", k))
                    break
            if allow_ifvg:
                # Opposing gaps from anywhere in the day up to here. One that a bar
                # since the sweep has closed through is inverted from that bar on;
                # the rest stay pending and may invert during the entry window.
                for k in range(day_start, i + 1):
                    g = find_fvg(bars, k, -side, min_fvg_atr)
                    if g is None:
                        continue
                    j = next((j for j in range(max(setup["sweep_idx"], k + 1), i + 1)
                              if _inverted(g, side, c[j])), None)
                    if j is None:
                        setup["pending"].append(g)
                    elif _discount(g, side, eq):
                        zones.append(Zone(g.low, g.high, j, "ifvg", k))
            setup["zone"] = _freshest(zones)
            continue        # no fill on the BOS bar itself: order inside it is unknown

        # -- 5. entry stage: fresh zones first, then the fill --------------------------
        # A BOS on the 09:35 or 09:40 bar lands the setup here while the sweep
        # window is still running. Those bars can still invert a pending gap —
        # the contract lets an inversion happen on any bar after the sweep — but
        # they cannot fill (entries are 09:50–10:10 only) and cannot supply a
        # fresh FVG (the contract admits gaps from the displacement leg or from
        # inside the entry window, nothing in between). Killing the setup here
        # was the bug review caught: it threw away every early-BOS day, which on
        # the seven-month NQ frame was most of the days that would have traded.
        eq = setup["eq"]
        fresh: list[Zone] = []
        if clock.in_entry[i]:
            z = find_fvg(bars, i, side, min_fvg_atr)
            if z and _discount(z, side, eq):
                fresh.append(Zone(z.low, z.high, i, "fvg", i))
        if allow_ifvg and setup["pending"]:
            still = []
            for g in setup["pending"]:
                if not _inverted(g, side, c[i]):
                    still.append(g)
                elif _discount(g, side, eq):
                    fresh.append(Zone(g.low, g.high, i, "ifvg", g.index))
            setup["pending"] = still
        if fresh:
            setup["zone"] = _freshest(fresh)         # freshest zone wins
        zone = setup["zone"]
        if zone is None:
            continue
        row["zone_kind"], row["outcome"] = zone.kind, "no_fill"
        if not clock.in_entry[i]:
            continue        # a zone may exist before 09:50; the fill may not
        if i <= max(setup["bos_idx"], zone.index):
            continue

        entry = None
        if side > 0 and l[i] <= zone.high:
            entry = min(zone.high, o[i])
        elif side < 0 and h[i] >= zone.low:
            entry = max(zone.low, o[i])
        if entry is None:
            continue

        # Touched. Whatever happens next, the day's setup is spent.
        live, setup = setup, None
        row["fill_time"] = idx[i]
        # Stop under the sweep wick, not under the lowest low since: the wick is
        # the level his model says was defended.
        stop = live["extreme"] - side * stop_buffer_atr * atr_[i]
        risk = side * (entry - stop)
        if risk <= 0:
            row["outcome"] = "no_risk"
            continue
        # His target is the next pool, not an R multiple: the nearest of the three
        # opposing levels strictly beyond the entry. Nothing beyond, nothing to aim at.
        pools = {k: levels[k][i] for k in (HIGHS if side > 0 else LOWS)}
        beyond = {k: v for k, v in pools.items() if not math.isnan(v) and side * (v - entry) > 0}
        if not beyond:
            row["outcome"] = "no_target"
            continue
        tname = min(beyond, key=lambda k: side * (beyond[k] - entry))
        trade = {"side": side, "entry_idx": i, "entry": entry, "stop": stop,
                 "target": beyond[tname], "risk": risk, "day": day[i],
                 "sweep_level": live["level"], "target_level": tname,
                 "zone_kind": zone.kind}
        row["outcome"] = "traded"
        pos[i] = side
        # the same bar can take out the stop - assume it did
        if (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop):
            pos[i] = 0.0
            close_trade(i, stop, "stop")

    if row is not None:
        days.append(row)
    return trades, pos, days


# ────────────────────────────── public surface ──────────────────────────────

def simulate(df: pd.DataFrame,
             swing_left: int = 2,
             swing_right: int = 2,
             atr_len: int = 14,
             stop_buffer_atr: float = 0.25,
             min_fvg_atr: float = 0.0,
             allow_ifvg: bool = True,
             smt: bool = False,
             flat_at: str = "15:55") -> SimResult:
    """Run the state machine. One setup and one position per trading day."""
    params = {"swing_left": swing_left, "swing_right": swing_right, "atr_len": atr_len,
              "stop_buffer_atr": stop_buffer_atr, "min_fvg_atr": min_fvg_atr,
              "allow_ifvg": allow_ifvg, "smt": smt, "flat_at": flat_at}
    trades, pos, _ = _run(df, **params)
    return SimResult(
        trades=pd.DataFrame(trades, columns=_TRADE_COLS),
        position=pd.Series(pos, index=df.index),
        params=params,
    )


def signal(df: pd.DataFrame, **params) -> pd.Series:
    """Adapter so the engine can treat this like any other strategy."""
    return simulate(df, **params).position


def day_log(df: pd.DataFrame, **params) -> pd.DataFrame:
    """One row per trading day that had a sweep-window bar, and how far it got.

    `outcome` is the stage the day died at: no_sweep, smt, no_swing, no_bos,
    no_zone, no_fill, no_risk, no_target, or traded. This is the setups-vs-fills
    view a trade list cannot give, and the one to read before the trade stats.
    """
    _, _, days = _run(df, **{**DEFAULTS, **params})
    return pd.DataFrame(days, columns=_DAY_COLS)

"""tjr_human signal detector: closed 1-minute bars in, events out (DESIGN-tjr-human.md section 9.2).

PURE and INCREMENTAL. No I/O, no clock of its own, no broker, no network. A
`Detector` is fed closed 1-minute bars of ONE instrument — a whole frame, or a
minute at a time, or anything between — and returns events as plain dicts.
There is one code path: `feed` walks the rows in time order and `_step(i)`
may read rows `<= i` only (every array view is cut at the row being stepped),
so one call or forty thousand give the same events, and truncating the frame
at any minute reproduces every earlier event (section 9.9). The tests check
both on the TradingView files anyway.

What is read where
------------------
Structure is read on COMPLETED bars only, by clock arithmetic:

* 5-minute context bins are built here, minute by minute, exactly as
  `tjr_intraday.build_context` builds them (UTC floor to 5 minutes, labelled by
  open time, complete at the minute stamped :x4, a bin whose :x4 minute never
  prints never completes but stays a bar). `_last` is the newest completed bin.
  Swings on the context are `primitives.confirmed_swings(left=2, right=2)` run
  on the five-bin window that closes on each newly completed bin — the
  primitive, not a copy of its loop — and are therefore admitted exactly at
  `confirmed_at`.
* 1-hour and 4-hour bins are `primitives.resample_context` on the 1-minute
  frame as of the 09:29 minute, once a day; swings admitted by
  `confirmed_at <= ctx.last`.
* ATR is `primitives.atr` (ATR 14) of the context bins `<=` the newest
  completed one. Anything decided DURING minute i (the dead-setup test, the
  stops of a fill at minute i's open) reads the ATR known BEFORE minute i:
  the newest bin completed by minute i-1.

Definitions are `tjr_wide_funnel.py --mode final` (sections 2.1, 2.3, 2.5)
where they apply on the context, and section 9.2 on the 1-minute bars. The
funnel script lives at the repo root and is not importable from a package, so
its small helpers (class names, the per-class combination rule, the composite,
the target merge) are restated here and `tests/test_tjr_human.py` asserts they
agree with the script's.

Events (every one a JSON-safe dict; the header is the same on all of them)
--------------------------------------------------------------------------
    levels         09:29 close: the day's levels, names, classes; composite depth; warm
    no_sweep       09:50: the sweep window closed without a sweep
    sweep          the minute completing the 5-minute sweep bar
    confirmation   the confirming 1-minute close: types, dealing range, EQ, ATR
    signal         confirmation AND a zone known: everything section 3.2 draws
    zone           after the signal: the zone that will be in play changed
    touch          a minute touched the zone in play (eq: having opened on the far side)
    entry_trigger  a minute closed out of the zone after a touch; entry = next minute's open
    fill           the entry minute: entry = its open, exact stops, both target pairs
    invalidated    dead setup (wick stop traded before the entry), or no risk at the open
    expired        10:10 (or the session ended) with a live setup and no fill

Header: kind, instrument, day (session date), setup_id (instrument-day),
bar_time (UTC stamp of the 1-minute bar that produced the event), et (HH:MM of
that bar), knowable_at (bar_time + 60 s: the bar's close; for `fill` the bar's
own stamp — a fill is known at the open).

Warm history
------------
The levels need the 20 completed sessions of the composite and the 4-hour
swing history over them (section 2.4). `levels["warm"]` says whether day D had
them; `check_warm(frame)` / `Detector.warm_status()` are what the runner calls
before it starts. A detector that is not warm still emits — its HVNs come from
a shallower composite and its H4 swings from a shorter history — so tests can
run on the 30-session files; the runner is the one that refuses.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..strategies.predictive import primitives as P
from ..strategies.predictive import tjr_intraday as M
from ..strategies.predictive.primitives import Bars, confirmed_swings, detect_bos, find_fvg

# ────────────────────────────── constants ──────────────────────────────

#: Section 9.10 item 7. "nearest" is section 2.5 as written; "untaken" restricts
#: the same candidates to levels price has not traded at since the 09:30 open.
#: BOTH pairs are computed and emitted on every signal and fill; this constant
#: (or Detector(target_rule=...)) only chooses which pair the trade exits at.
TARGET_RULE = "nearest"
TARGET_RULES = ("nearest", "untaken")

TICK = 0.25
COMPOSITE_SESSIONS = 20
OTE = 0.79
EQ_EXCURSION_ATR = 0.5
WICK_STOP_ATR = float(M.DEFAULTS["stop_buffer_atr"])          # 0.25, round 1's buffer
SWING_LEFT = SWING_RIGHT = 2
ATR_LEN = 14
CONTEXT_MINUTES = M.CONTEXT_MINUTES
RESAMPLES = (("H1", 60), ("H4", 240))
CONF_TYPES = ("bos", "ifvg", "ote")
ZONE_TYPES = ("fvg", "eq", "ob", "breaker")
ZONE_RANK = {"ob": 0, "eq": 1, "breaker": 2, "fvg": 3}
LEVEL_ORDER = ("ASIA_H", "ASIA_L", "LON_H", "LON_L", "PDH", "PDL",
               "H1_SH", "H1_SL", "H4_SH", "H4_SL", "POC_PREV")
DIRECTIONAL_LOWS = ("ASIA_L", "LON_L", "PDL", "H1_SL", "H4_SL")
DIRECTIONAL_HIGHS = ("ASIA_H", "LON_H", "PDH", "H1_SH", "H4_SH")
CLASSES = ("SESSION", "H1", "H4")
SESSION_TYPES = ("ASIA", "LON", "PD")
STOP_MODES = M.STOP_MODES                                       # wick, atr1.0, atr1.5, atr2.0, session

FREEZE_MINUTE = M.SWEEP_WINDOW[0] - 1                           # 09:29: the minute that closes the 09:25 bar
SWEEP_START, SWEEP_END = M.SWEEP_WINDOW                         # 09:30 .. 09:49 stamps
ENTRY_START, ENTRY_END = M.ENTRY_WINDOW                         # 09:50 .. 10:09 stamps
TRIGGER_FIRST, TRIGGER_LAST = ENTRY_START - 1, ENTRY_END - 2    # closing-out minutes 09:49 .. 10:08

WARM_SESSIONS = COMPOSITE_SESSIONS
WARM_H4_BINS = 100                # 20 sessions of 4-hour bins at the early-close floor of five a session
THIN_SESSION_BARS = 1000          # reported by check_warm, never enforced

_MIN = np.timedelta64(1, "m")


# ────────────────────────────── small pure helpers ──────────────────────────────

def level_type(name: str) -> str:
    """ASIA_L -> ASIA, PDH -> PD, H1_SH -> H1, POC_PREV -> POC, HVN_3 -> HVN."""
    return "PD" if name in ("PDH", "PDL") else name.split("_")[0]


def level_class(name: str) -> str:
    """Section 2.5 item 2: ASIA_*, LON_*, PDH, PDL -> SESSION; H1_* -> H1; H4_* -> H4;
    POC_PREV -> POC and HVN_* -> HVN (targets only, never swept)."""
    t = level_type(name)
    return "SESSION" if t in SESSION_TYPES else t


def level_sort_key(name: str) -> tuple[int, int]:
    if name in LEVEL_ORDER:
        return LEVEL_ORDER.index(name), 0
    return len(LEVEL_ORDER), int(name.split("_")[1])


def hhmm(minutes: int) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def et_clock(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(ET minute of day, CME session day as datetime64[ns]) per bar of a tz-naive
    UTC index. The same arithmetic as `tjr_intraday.session_clock`, without its
    "no 09:30 bar -> None" rule, which a one-row chunk would always trip."""
    et = (index.tz_localize("UTC") if index.tz is None else index).tz_convert(M.ET)
    minutes = (et.hour * 60 + et.minute).to_numpy().astype(np.int64)
    midnight = et.tz_localize(None).normalize().as_unit("ns").to_numpy()
    day = midnight + np.where(minutes >= M.SESSION_OPEN, 1, 0).astype("timedelta64[D]")
    return minutes, day.astype("datetime64[ns]")


def _clock(minutes: np.ndarray, day: np.ndarray) -> M.Clock:
    flat = 15 * 60 + 55
    return M.Clock(
        minutes=minutes, day=day,
        in_asia=(minutes >= M.SESSION_OPEN) | (minutes < M.ASIA_END),
        in_london=(minutes >= M.ASIA_END) & (minutes < M.LONDON_END),
        in_sweep=(minutes >= SWEEP_START) & (minutes < SWEEP_END),
        in_entry=(minutes >= ENTRY_START) & (minutes < ENTRY_END),
        is_flat=(minutes >= flat) & (minutes < M.SESSION_OPEN),
    )


def _clean(x):
    """JSON-safe: NaN -> None, numpy scalars -> Python ones, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else float(x)
    return x


def _mk_bars(o, h, l, c) -> Bars:
    """A `Bars` view over arrays. The ATR slot is zeros: every primitive called
    through this view is called with its ATR floor at 0, and the ATR the model
    uses is computed separately by `primitives.atr` on completed context bins."""
    return Bars(index=None, open=o, high=h, low=l, close=c, atr=np.zeros(len(c)))


def class_sweep(high, low, close, ci: int, lows: dict, highs: dict) -> dict:
    """Round 1's sweep test (`tjr_intraday.sweep_of_levels`) per level class on
    context bar `ci` — section 2.3 item 3 with the three classes of section 2.5.

    A class whose wick took one of its lows AND one of its highs abstains. Among
    the classes that return a sweep: one side -> the bar sweeps on that side and
    the setup's level is the deepest breached level of that side across those
    classes; two sides -> ambiguous, skipped. `verdict` is one of
    "sweep" | "ambiguous" | "abstain_only" | "none".
    """
    below = {k: p for k, p in lows.items() if low[ci] < p}
    above = {k: p for k, p in highs.items() if high[ci] > p}
    returns: dict[str, tuple] = {}
    abstained: list[str] = []
    state: dict[str, str] = {}
    for cls in CLASSES:
        lc = {k: p for k, p in lows.items() if level_class(k) == cls}
        hc = {k: p for k, p in highs.items() if level_class(k) == cls}
        r = M.sweep_of_levels(high, low, close, ci, lc, hc)
        if r is None:
            took_low, took_high = any(k in below for k in lc), any(k in above for k in hc)
            if took_low and took_high:
                abstained.append(cls)
            state[cls] = ("abstained" if took_low and took_high else
                          "breached_no_close_back" if took_low or took_high else
                          "untouched" if lc or hc else "absent")
            continue
        returns[cls] = r
        state[cls] = "returned"
    out = {"verdict": "none", "side": 0, "name": None, "price": None, "below": below, "above": above,
           "returned": {cls: r[1] for cls, r in returns.items()}, "abstained": abstained, "state": state}
    sides = {r[0] for r in returns.values()}
    if len(sides) == 2:
        out["verdict"] = "ambiguous"
        return out
    if not returns:
        if abstained:
            out["verdict"] = "abstain_only"
        return out
    side = sides.pop()
    taken = below if side > 0 else above
    cand = {k: p for k, p in taken.items() if level_class(k) in returns}
    name = min(cand, key=cand.get) if side > 0 else max(cand, key=cand.get)
    out.update(verdict="sweep", side=int(side), name=name, price=float(cand[name]))
    return out


def target_pairs(levels: dict, side: int, entry: float, seen_extreme: float | None) -> dict:
    """Both target pairs of section 9.10 item 7 for an entry (or reference price).

    Candidates (section 2.5 item 3): the opposing-NAMED directional levels plus
    POC_PREV and HVN_*, strictly beyond `entry`; levels at one price are one
    target carrying every name; T1 the nearest, T2 the next distinct price. A
    same-side-named level is never a candidate.

    "nearest" takes them as they come. "untaken" keeps only candidates strictly
    beyond `seen_extreme` — the session's highest high (long) / lowest low
    (short) over the completed minutes from the 09:30 open up to the event — so
    a level price has already traded at is not a target. None -> nothing taken.
    """
    named = DIRECTIONAL_HIGHS if side > 0 else DIRECTIONAL_LOWS
    pool = {k: float(p) for k, p in levels.items()
            if (k in named or level_type(k) in ("POC", "HVN")) and side * (p - entry) > 0}

    def pair(cands: dict) -> dict:
        by_price: dict[float, list[str]] = {}
        for k, p in cands.items():
            by_price.setdefault(round(p, 6), []).append(k)
        prices = sorted(by_price, key=lambda p: side * (p - entry))
        tg = [{"names": "+".join(sorted(by_price[p], key=level_sort_key)), "price": float(p)} for p in prices[:2]]
        return {"t1": tg[0] if tg else None, "t2": tg[1] if len(tg) > 1 else None}

    if seen_extreme is None or (isinstance(seen_extreme, float) and math.isnan(seen_extreme)):
        untaken = pool
    else:
        untaken = {k: p for k, p in pool.items() if side * (p - seen_extreme) > 0}
    return {"nearest": pair(pool), "untaken": pair(untaken)}


def candidate_stops(side: int, entry: float, extreme: float, session_extreme: float, atr_value: float) -> dict:
    """The five candidate stops of section 3.3, each by `tjr_intraday.stop_price`."""
    return {mode: float(M.stop_price(mode, side, entry, extreme, session_extreme, atr_value, WICK_STOP_ATR))
            for mode in STOP_MODES}


class Zone:
    """A continuation zone. `formed` is the freshness stamp and `usable` the
    minute the detector could first know it on, both 1-minute row numbers
    (context stamps are converted to the last minute of their bin). It is in
    play on minutes strictly after `usable` (section 9.10 item 6)."""
    __slots__ = ("kind", "low", "high", "formed", "usable")

    def __init__(self, kind: str, low: float, high: float, formed: int, usable: int):
        self.kind, self.low, self.high = kind, float(low), float(high)
        self.formed, self.usable = int(formed), int(usable)


def freshest(zones: list) -> Zone | None:
    return max(zones, key=lambda z: (z.formed, ZONE_RANK[z.kind])) if zones else None


# ────────────────────────────── growable arrays ──────────────────────────────

class _Buf:
    __slots__ = ("a", "n")

    def __init__(self, dtype, cap: int = 8192):
        self.a = np.empty(cap, dtype=dtype)
        self.n = 0

    def _room(self, k: int) -> None:
        if self.n + k > len(self.a):
            grown = np.empty(max(2 * len(self.a), self.n + k), dtype=self.a.dtype)
            grown[:self.n] = self.a[:self.n]
            self.a = grown

    def extend(self, vals) -> None:
        k = len(vals)
        self._room(k)
        self.a[self.n:self.n + k] = vals
        self.n += k

    def append(self, v) -> None:
        self._room(1)
        self.a[self.n] = v
        self.n += 1


class _Day:
    """Everything the detector knows about the session in progress."""

    def __init__(self, day, start: int, start_ci: int, ordinal: int):
        self.day = day
        self.day_str = str(np.datetime_as_string(day, unit="D"))
        self.start, self.start_ci, self.ordinal = start, start_ci, ordinal
        self.frozen = False
        self.done = False
        self.stage = "pre"           # pre -> sweep -> confirm -> entry -> pending -> (done)
        self.i_open = None           # first minute stamped >= 09:30
        self.levels: dict[str, float] = {}
        self.lows: dict[str, float] = {}
        self.highs: dict[str, float] = {}
        self.ambiguous: list[str] = []
        self.abstain_only: list[str] = []
        # the setup
        self.side = 0
        self.sweep_i = None
        self.sweep_ci = None
        self.sweep_first = None
        self.extreme = math.nan
        self.extreme_1m = None
        self.level = None
        self.pre_swing = None
        self.ote_level = None
        self.fresh_gaps: list = []
        self.conf_any: dict[str, str] = {}
        self.conf_i = None
        self.conf_types: list[str] = []
        self.range = (math.nan, math.nan)
        self.eq = math.nan
        self.eq_atr = math.nan
        self.eq_thr = math.nan
        self.eq_pending = False
        self.zones: list[Zone] = []
        self.seen_breakers: set = set()
        self.signalled = False
        self.announced: Zone | None = None
        self.inplay: Zone | None = None
        self.touched = False
        self.touch_i = None
        self.spent_patterns = 0
        self.eq_refused: list[str] = []
        self.trigger_i = None
        self.trigger_zone: Zone | None = None


# ────────────────────────────── the detector ──────────────────────────────

class Detector:
    """One instrument. `feed(frame)` -> the new events; `events` holds all of them.

    `frame` is closed 1-minute bars: a tz-naive UTC DatetimeIndex (as
    `quantlab.data.load_csv` gives), columns open/high/low/close and optionally
    volume. Rows at or before the newest row already fed are ignored (counted in
    `dropped`), so a runner may feed overlapping reads of its store.
    """

    def __init__(self, instrument: str, target_rule: str | None = None, bin_ticks: int = 4,
                 hvn_frac: float = 0.5, hvn_max: int = 6):
        rule = TARGET_RULE if target_rule is None else target_rule
        if rule not in TARGET_RULES:
            raise ValueError(f"tjr_human.detector: target_rule must be one of {TARGET_RULES}, got {rule!r}")
        self.instrument = str(instrument)
        self.target_rule = rule
        self.bin_ticks, self.hvn_frac, self.hvn_max = int(bin_ticks), float(hvn_frac), int(hvn_max)
        self.events: list[dict] = []
        self.dropped = 0
        # 1-minute rows
        self._t = _Buf("datetime64[ns]")
        self._o, self._h, self._l, self._c, self._v = (_Buf(float) for _ in range(5))
        self._m = _Buf(np.int64)
        self._day = _Buf("datetime64[ns]")
        self._n = 0                                   # rows VISIBLE to _step: every view is cut here
        # 5-minute context bins
        self._bt = _Buf("datetime64[ns]")
        self._bo, self._bh, self._bl, self._bc = (_Buf(float) for _ in range(4))
        self._bfirst = _Buf(np.int64)                 # first 1-minute row of the bin
        self._bend = _Buf(np.int64)                   # last 1-minute row of the bin so far
        self._last = -1                               # newest bin completed by the clock
        self._last_prev = -1                          # the same, as of the previous minute
        self._done_now = False                        # did the minute being stepped complete its bin?
        self._sw_hi: list = []                        # admitted context swings (confirmed_at <= _last)
        self._sw_lo: list = []
        self._atr_cache: dict[int, float] = {}
        # sessions
        self._sessions: list[list] = []               # [day, start row, end row or None]
        self._profiles: dict[int, tuple] = {}
        self._ds: _Day | None = None
        self._out: list[dict] = []

    # ---- feeding -----------------------------------------------------------------

    def feed(self, frame: pd.DataFrame) -> list[dict]:
        if frame is None or len(frame) == 0:
            return []
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("tjr_human.detector: feed needs a DatetimeIndex (tz-naive UTC)")
        idx = frame.index
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        t = idx.to_numpy().astype("datetime64[ns]")
        if len(t) > 1 and not (np.diff(t.astype(np.int64)) > 0).all():
            raise ValueError("tjr_human.detector: bars must be strictly increasing in time")
        keep = np.ones(len(t), dtype=bool)
        if self._n:
            keep = t > self._t.a[self._n - 1]
            self.dropped += int((~keep).sum())
            if not keep.any():
                return []
        cols = {k: frame[k].to_numpy(dtype=float)[keep] for k in ("open", "high", "low", "close")}
        if any(np.isnan(a).any() for a in cols.values()):
            raise ValueError("tjr_human.detector: NaN in open/high/low/close")
        vol = (frame["volume"].to_numpy(dtype=float)[keep] if "volume" in frame.columns
               else np.zeros(int(keep.sum())))
        t = t[keep]
        minutes, day = et_clock(pd.DatetimeIndex(t))
        self._t.extend(t)
        self._o.extend(cols["open"]); self._h.extend(cols["high"])
        self._l.extend(cols["low"]); self._c.extend(cols["close"])
        self._v.extend(np.nan_to_num(vol, nan=0.0))
        self._m.extend(minutes)
        self._day.extend(day)
        self._out = []
        for i in range(self._n, self._n + len(t)):
            snap = self._snapshot(i)
            try:
                self._n = i + 1                       # row i becomes visible; nothing after it is
                self._advance_context(i)
                self._step(i)
            except BaseException as exc:
                # EACH ROW IS ATOMIC. Row i did not happen: the state is what it was before it, row i
                # and the rest of the chunk leave the buffers (they were never visible), and the rows
                # before it stay processed — their events are in `events` and ride on the exception
                # (`events_before`), since this call cannot return them. Feeding the same rows again
                # is the retry.
                self._restore(snap)
                out, self._out = self._out, []
                self.events.extend(out)
                try:
                    exc.events_before = out
                except Exception:
                    pass
                raise
        out, self._out = self._out, []
        self.events.extend(out)
        return out

    _DAY_CONTAINERS = (list, dict, set)

    def _snapshot(self, i: int) -> tuple:
        """What one row can change, cheaply: counters, the forming context bin, the lengths of the
        append-only lists, and the day's state one level deep (zones and gaps are never mutated in
        place). The ATR and profile caches are pure functions of completed bins / sessions, so an
        entry a failed row left behind is still right."""
        nb = self._bt.n
        last_bin = (float(self._bh.a[nb - 1]), float(self._bl.a[nb - 1]), float(self._bc.a[nb - 1]),
                    int(self._bend.a[nb - 1])) if nb else None
        ds = self._ds
        day = None if ds is None else {k: (v.copy() if isinstance(v, self._DAY_CONTAINERS) else v)
                                       for k, v in ds.__dict__.items()}
        return (i, nb, last_bin, self._last, self._last_prev, self._done_now, len(self._sw_hi), len(self._sw_lo),
                len(self._sessions), self._sessions[-1][2] if self._sessions else None, ds, day, len(self._out))

    def _restore(self, snap: tuple) -> None:
        i, nb, last_bin, last, last_prev, done_now, n_hi, n_lo, n_sess, sess_end, ds, day, n_out = snap
        for buf in (self._t, self._o, self._h, self._l, self._c, self._v, self._m, self._day):
            buf.n = i
        self._n = i
        for buf in (self._bt, self._bo, self._bh, self._bl, self._bc, self._bfirst, self._bend):
            buf.n = nb
        if last_bin is not None:
            self._bh.a[nb - 1], self._bl.a[nb - 1], self._bc.a[nb - 1], self._bend.a[nb - 1] = last_bin
        self._last, self._last_prev, self._done_now = last, last_prev, done_now
        del self._sw_hi[n_hi:], self._sw_lo[n_lo:], self._sessions[n_sess:], self._out[n_out:]
        if self._sessions:
            self._sessions[-1][2] = sess_end
        self._ds = ds
        if ds is not None:
            ds.__dict__.clear()
            ds.__dict__.update(day)

    # ---- views, all cut at the visible row count ------------------------------------

    def _bars1(self) -> Bars:
        n = self._n
        return _mk_bars(self._o.a[:n], self._h.a[:n], self._l.a[:n], self._c.a[:n])

    def _ctx_bars(self, upto: int) -> Bars:
        """Context bins 0..upto inclusive; callers pass a COMPLETED bin."""
        k = upto + 1
        return _mk_bars(self._bo.a[:k], self._bh.a[:k], self._bl.a[:k], self._bc.a[:k])

    def _frame(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"open": self._o.a[:n], "high": self._h.a[:n], "low": self._l.a[:n],
                             "close": self._c.a[:n], "volume": self._v.a[:n]},
                            index=pd.DatetimeIndex(self._t.a[:n], name="date"))

    def _atr_at(self, k: int) -> float:
        """ATR(14) of the context at completed bin k: `primitives.atr` over bins <= k."""
        if k < 0:
            return math.nan
        got = self._atr_cache.get(k)
        if got is None:
            f = pd.DataFrame({"high": self._bh.a[:k + 1], "low": self._bl.a[:k + 1], "close": self._bc.a[:k + 1]})
            got = float(P.atr(f, ATR_LEN)[-1])
            if len(self._atr_cache) > 64:
                self._atr_cache.clear()
            self._atr_cache[k] = got
        return got

    # ---- the 5-minute context, one minute at a time ---------------------------------------

    def _advance_context(self, i: int) -> None:
        t = self._t.a[i]
        tm = int(t.astype("datetime64[m]").astype(np.int64))
        label = np.datetime64(tm - tm % CONTEXT_MINUTES, "m").astype("datetime64[ns]")
        nb = self._bt.n
        if nb == 0 or self._bt.a[nb - 1] != label:
            self._bt.append(label)
            self._bo.append(self._o.a[i]); self._bh.append(self._h.a[i])
            self._bl.append(self._l.a[i]); self._bc.append(self._c.a[i])
            self._bfirst.append(i); self._bend.append(i)
        else:
            b = nb - 1
            if self._h.a[i] > self._bh.a[b]:
                self._bh.a[b] = self._h.a[i]
            if self._l.a[i] < self._bl.a[b]:
                self._bl.a[b] = self._l.a[i]
            self._bc.a[b] = self._c.a[i]
            self._bend.a[b] = i
        self._last_prev = self._last
        self._done_now = tm % CONTEXT_MINUTES == CONTEXT_MINUTES - 1
        if self._done_now:
            new_last = self._bt.n - 1
            for k in range(self._last + 1, new_last + 1):
                self._admit_swings(k)
            self._last = new_last

    def _admit_swings(self, k: int) -> None:
        """The pivots whose right-hand window closes on bin k: confirmed_swings on
        the window [k-4, k] finds exactly the pivot at k-2, confirmed at k."""
        w = SWING_LEFT + SWING_RIGHT
        if k < w:
            return
        a = k - w
        highs, lows = confirmed_swings(_mk_bars(self._bo.a[a:k + 1], self._bh.a[a:k + 1],
                                                self._bl.a[a:k + 1], self._bc.a[a:k + 1]),
                                       SWING_LEFT, SWING_RIGHT)
        for s in highs:
            self._sw_hi.append(P.Swing(a + s.index, s.price, a + s.confirmed_at))
        for s in lows:
            self._sw_lo.append(P.Swing(a + s.index, s.price, a + s.confirmed_at))

    # ---- events ----------------------------------------------------------------------------

    def _iso(self, t) -> str:
        return str(np.datetime_as_string(t, unit="s"))

    def _time(self, i: int) -> str:
        return self._iso(self._t.a[i])

    def _emit(self, kind: str, i: int, at_open: bool = False, **body) -> dict:
        ds = self._ds
        t = self._t.a[i]
        ev = {"kind": kind, "instrument": self.instrument, "day": ds.day_str,
              "setup_id": f"{self.instrument}-{ds.day_str}",
              "bar_time": self._iso(t), "et": hhmm(self._m.a[i]),
              "knowable_at": self._iso(t if at_open else t + _MIN)}
        ev.update(_clean(body))
        self._out.append(ev)
        return ev

    def _zone_dict(self, z: Zone | None) -> dict | None:
        if z is None:
            return None
        return {"kind": z.kind, "low": z.low, "high": z.high,
                "formed_time": self._time(z.formed), "usable_time": self._time(z.usable)}

    def _setup_body(self) -> dict:
        ds = self._ds
        return {"side": ds.side, "direction": "long" if ds.side > 0 else "short",
                "level": {"name": ds.level, "price": ds.levels[ds.level], "class": level_class(ds.level)},
                "extreme": ds.extreme}

    # ---- one minute ---------------------------------------------------------------------------

    def _step(self, i: int) -> None:
        m = int(self._m.a[i])
        day = self._day.a[i]
        if self._ds is None or day != self._ds.day:
            self._roll_day(i, day)
        ds = self._ds
        if ds.done:
            return
        daypart = m < M.SESSION_OPEN                 # evening minutes belong to the next session's start
        if not ds.frozen:
            if not (daypart and m >= FREEZE_MINUTE):
                return
            self._freeze(i, m)
        if ds.i_open is None and m >= SWEEP_START:
            ds.i_open = i
        if ds.stage == "sweep":
            self._sweep_stage(i, m)
        elif ds.stage == "pending":
            self._fill_stage(i, m)
        elif ds.stage in ("confirm", "entry"):
            self._setup_stage(i, m)

    def _roll_day(self, i: int, day) -> None:
        ds = self._ds
        if ds is not None and not ds.done and ds.frozen:
            # the session ended under a live stage: say so on the first bar that shows it
            if ds.stage == "sweep":
                self._emit("no_sweep", i, ambiguous_bars=list(ds.ambiguous), abstain_only_bars=list(ds.abstain_only),
                           reason="session_ended")
            else:
                self._expire(i, "session_ended")
        if self._sessions:
            self._sessions[-1][2] = i
        self._sessions.append([day, i, None])
        self._ds = _Day(day, i, self._bt.n - 1, len(self._sessions) - 1)

    # ---- levels, frozen at the 09:25 close ---------------------------------------------------------

    def _profile(self, s: int) -> tuple:
        got = self._profiles.get(s)
        if got is None:
            _, a, e = self._sessions[s]
            h, l, c, v = self._h.a[a:e], self._l.a[a:e], self._c.a[a:e], self._v.a[a:e]
            prof = P.volume_profile(h, l, v, TICK, self.bin_ticks)
            w = TICK * self.bin_ticks
            got = (prof, P.vwap(h, l, c, v), int(round(prof.edges[0] / w)), float(v.sum()))
            self._profiles[s] = got
        return got

    def _composite(self, s: int):
        lo_s = max(0, s - COMPOSITE_SESSIONS)
        parts = [self._profile(k) for k in range(lo_s, s)]
        if not parts:
            return None, 0
        w = TICK * self.bin_ticks
        kmin = min(k for _, _, k, _ in parts)
        kmax = max(k + len(p.volume) for p, _, k, _ in parts)
        vols = np.zeros(kmax - kmin)
        for p, _, k, _ in parts:
            vols[k - kmin:k - kmin + len(p.volume)] += p.volume
        return P.Profile(w * np.arange(kmin, kmax + 1), vols), len(parts)

    def _freeze(self, i: int, m: int) -> None:
        ds = self._ds
        n = i + 1
        h, l, c = self._h.a[:n], self._l.a[:n], self._c.a[:n]
        lv: dict[str, float] = {}
        six = M.session_levels(_clock(self._m.a[:n], self._day.a[:n]), h, l)
        for name in M.HIGHS + M.LOWS:
            val = float(six[name][i])
            if not math.isnan(val):
                lv[name] = val
        # The last minute of D stamped <= 09:29: row i itself when it IS the 09:29
        # minute, else the row before it (a gap swallowed 09:29), if that is in D.
        pre = i if m == FREEZE_MINUTE else (i - 1 if i - 1 >= ds.start else None)
        close_pre = float(c[pre]) if pre is not None else math.nan
        depth, n_hvn, h4_bins = 0, 0, 0
        if pre is not None:
            frame = self._frame(pre + 1)
            for name, mins in RESAMPLES:
                ctx = P.resample_context(frame, mins, bar_minutes=1, atr_len=ATR_LEN)
                newest = int(ctx.last[-1])
                if name == "H4":
                    h4_bins = newest + 1
                highs, lows = confirmed_swings(ctx.bars, SWING_LEFT, SWING_RIGHT)
                above = [s.price for s in highs if s.confirmed_at <= newest and s.price > close_pre]
                below = [s.price for s in lows if s.confirmed_at <= newest and s.price < close_pre]
                if above:
                    lv[f"{name}_SH"] = float(min(above))
                if below:
                    lv[f"{name}_SL"] = float(max(below))
            s = ds.ordinal
            if s >= 1:
                prof, vw, _, vsum = self._profile(s - 1)
                if vsum > 0:                          # a session with no volume has no point of control
                    lv["POC_PREV"] = float(P.poc(prof, near=vw))
            comp, depth = self._composite(s)
            if comp is not None:
                # primitives.hvn smooths over 5 bins and indexes past a profile narrower than that (a
                # 20-session composite under five points wide: toy prices only, never NQ or ES) — no nodes then
                nodes = (P.hvn(comp, smooth=5, frac=self.hvn_frac, max_n=self.hvn_max, near=close_pre)
                         if len(comp.volume) >= 5 else [])
                for k, price in enumerate(nodes, 1):
                    lv[f"HVN_{k}"] = float(price)
                n_hvn = len(nodes)
        ds.levels = lv
        ds.lows = {k: p for k, p in lv.items() if k in DIRECTIONAL_LOWS}
        ds.highs = {k: p for k, p in lv.items() if k in DIRECTIONAL_HIGHS}
        ds.frozen, ds.stage = True, "sweep"
        rows = []
        for name in sorted(lv, key=level_sort_key):
            cls = level_class(name)
            if name in DIRECTIONAL_LOWS:
                rests = "low"
            elif name in DIRECTIONAL_HIGHS:
                rests = "high"
            else:
                rests = "low" if lv[name] <= close_pre else "high"
            rows.append({"name": name, "price": lv[name], "class": cls, "rests": rests,
                         "swept": cls in CLASSES})
        self._emit("levels", i, levels=rows, close_pre=None if math.isnan(close_pre) else close_pre,
                   composite_depth=int(depth), n_hvn=int(n_hvn), h4_bins=int(h4_bins),
                   warm=bool(depth >= WARM_SESSIONS and h4_bins >= WARM_H4_BINS))

    # ---- the sweep: the 5-minute context bar, read at the minute that completes it ---------------------

    def _sweep_stage(self, i: int, m: int) -> None:
        ds = self._ds
        if m >= SWEEP_END:
            ds.done = True
            self._emit("no_sweep", i, ambiguous_bars=list(ds.ambiguous), abstain_only_bars=list(ds.abstain_only),
                       reason="window_closed")
            return
        if m < SWEEP_START or not self._done_now:
            return
        ci = self._last                                # the bin this minute completed
        k = ci + 1
        bh, bl, bc = self._bh.a[:k], self._bl.a[:k], self._bc.a[:k]
        res = class_sweep(bh, bl, bc, ci, ds.lows, ds.highs)
        label = hhmm(m - (CONTEXT_MINUTES - 1))
        if res["verdict"] == "ambiguous":
            ds.ambiguous.append(label)
        elif res["verdict"] == "abstain_only":
            ds.abstain_only.append(label)
        if res["verdict"] != "sweep":
            return
        side = res["side"]
        first = int(self._bfirst.a[ci])
        lo1, hi1 = self._l.a[first:i + 1], self._h.a[first:i + 1]
        ds.side, ds.sweep_i, ds.sweep_ci, ds.sweep_first = side, i, ci, first
        ds.level = res["name"]
        if side > 0:
            ds.extreme, ds.extreme_1m = float(bl[ci]), first + int(np.argmin(lo1))
        else:
            ds.extreme, ds.extreme_1m = float(bh[ci]), first + int(np.argmax(hi1))
        # the most recent opposing context swing existing at the sweep bar; frozen
        opp = self._sw_hi if side > 0 else self._sw_lo
        ds.pre_swing = float(opp[-1].price) if opp else None
        ds.ote_level = (ds.extreme + OTE * (ds.pre_swing - ds.extreme)) if ds.pre_swing is not None else None
        # opposing 1-minute gaps of day D up to here that no close has been through: the fresh ones
        b1 = self._bars1()
        c1 = b1.close
        for kk in range(ds.start, i + 1):
            g = find_fvg(b1, kk, -side)
            if g is None:
                continue
            after = c1[g.index + 1:i + 1]
            if len(after) and M._inverted(g, side, float(after.max() if side > 0 else after.min())):
                continue                               # stale: a close was through it by the sweep bar
            ds.fresh_gaps.append(g)
        below, above = res["below"], res["above"]
        ds.stage = "confirm"
        self._emit("sweep", i, **self._setup_body(),
                   sweep_bar=label, sweep_bar_time=self._iso(self._bt.a[ci]),
                   breached=(sorted(below, key=below.get) if side > 0 else sorted(above, key=above.get, reverse=True)),
                   breached_opposite=(sorted(above, key=above.get, reverse=True) if side > 0
                                      else sorted(below, key=below.get)),
                   classes_returned=res["returned"], classes_abstained=res["abstained"], class_state=res["state"],
                   pre_swing=ds.pre_swing, ote_level=ds.ote_level,
                   ambiguous_bars=list(ds.ambiguous), abstain_only_bars=list(ds.abstain_only))

    # ---- after the sweep: dead setup, confirmation, zones, touch, trigger ---------------------------------

    def _expire(self, i: int, why: str | None = None) -> None:
        ds = self._ds
        if ds.stage == "confirm":
            reason = "no_confirmation"
        elif ds.stage == "pending":
            reason = "no_entry_bar"
        elif not ds.zones:
            reason = "no_zone"
        elif ds.touch_i is None and ds.spent_patterns == 0:
            reason = "no_touch"
        else:
            reason = "no_close_out"
        ds.done = True
        self._emit("expired", i, **self._setup_body(), reason=reason, because=why or "window_closed",
                   stage=ds.stage, conf_types=list(ds.conf_types), conf_types_seen=dict(ds.conf_any),
                   zone=self._zone_dict(ds.announced), eq_refused=list(ds.eq_refused),
                   patterns_before_window=int(ds.spent_patterns))

    def _conf_types(self, i: int, b1: Bars) -> list[str]:
        ds = self._ds
        side, close = ds.side, float(b1.close[i])
        g = find_fvg(b1, i, -side)
        if g is not None:
            ds.fresh_gaps.append(g)                    # completed after the sweep bar: fresh by definition
        fired = []
        if ds.pre_swing is not None and detect_bos(b1, i, ds.pre_swing, side):
            fired.append("bos")
        if any(M._inverted(g, side, close) for g in ds.fresh_gaps):
            fired.append("ifvg")
        if ds.ote_level is not None and (close > ds.ote_level if side > 0 else close < ds.ote_level):
            fired.append("ote")
        for t in fired:
            ds.conf_any.setdefault(t, hhmm(self._m.a[i]))
        return fired

    def _add_zone(self, kind: str, lo: float, hi: float, formed: int, usable: int) -> None:
        ds = self._ds
        z = Zone(kind, lo, hi, formed, usable)
        if M._discount(z, ds.side, ds.eq):
            ds.zones.append(z)

    def _breaker_at(self, i: int) -> None:
        """A breaker on the context, judged through the newest COMPLETED bin at minute i."""
        ds = self._ds
        ci = self._last
        if ci < ds.sweep_ci:
            return
        swings = [s for s in (self._sw_hi if ds.side > 0 else self._sw_lo) if s.index >= ds.start_ci]
        blk = P.breaker_block(self._ctx_bars(ci), ds.start_ci, ds.sweep_ci, ci, ds.side, swings)
        if blk is not None and (blk.index, blk.candle) not in ds.seen_breakers:
            ds.seen_breakers.add((blk.index, blk.candle))
            self._add_zone("breaker", blk.low, blk.high, int(self._bend.a[blk.index]), i)

    def _setup_stage(self, i: int, m: int) -> None:
        ds = self._ds
        if m >= ENTRY_END:
            self._expire(i)
            return
        side = ds.side
        b1 = self._bars1()
        o, h, l, c = float(b1.open[i]), float(b1.high[i]), float(b1.low[i]), float(b1.close[i])

        # -- dead setup: price trades at or beyond the wick stop before the entry. The
        #    stop uses the ATR known before this minute began.
        atr_prev = self._atr_at(self._last_prev)
        wick = float(M.stop_price("wick", side, math.nan, ds.extreme, math.nan, atr_prev, WICK_STOP_ATR))
        if (l <= wick) if side > 0 else (h >= wick):
            ds.done = True
            self._emit("invalidated", i, **self._setup_body(), reason="wick_stop_traded", stage=ds.stage,
                       wick_stop=wick, atr=atr_prev, traded=l if side > 0 else h,
                       conf_types=list(ds.conf_types), zone=self._zone_dict(ds.announced))
            return

        fired = self._conf_types(i, b1)
        if ds.stage == "confirm":
            if not fired:
                return
            self._confirm(i, fired, b1)
        else:
            # -- the zone in play on this minute: the freshest known BEFORE it --------------
            z = freshest([z for z in ds.zones if z.usable < i])
            if z is not ds.inplay:
                ds.inplay, ds.touched, ds.touch_i = z, False, None
            if z is not None:
                touch = (l <= z.high) if side > 0 else (h >= z.low)
                if touch and z.kind == "eq" and not ((o > ds.eq) if side > 0 else (o < ds.eq)):
                    touch = False                     # opened at or through the midpoint: not a retrace
                    if not ds.touched:                # a refusal is one only while a touch is still wanted
                        ds.eq_refused.append(hhmm(m))
                if touch and not ds.touched:
                    ds.touched, ds.touch_i = True, i
                    self._emit("touch", i, side=side, zone=self._zone_dict(z), open=o, low=l, high=h)
                closed_out = (c > z.high) if side > 0 else (c < z.low)
                if ds.touched and closed_out:
                    if TRIGGER_FIRST <= m <= TRIGGER_LAST:
                        ds.stage, ds.trigger_i, ds.trigger_zone = "pending", i, z
                        atr_now = self._atr_at(self._last)
                        self._emit("entry_trigger", i, **self._setup_body(), zone=self._zone_dict(z), close=c,
                                   touch_time=self._time(ds.touch_i),
                                   entry_at=self._iso(self._t.a[i] + _MIN), atr=atr_now,
                                   wick_stop=float(M.stop_price("wick", side, math.nan, ds.extreme, math.nan,
                                                                atr_now, WICK_STOP_ATR)))
                        return
                    # closed out too early for an entry stamped 09:50 or later: the pattern is spent
                    ds.touched, ds.touch_i = False, None
                    ds.spent_patterns += 1
            # -- what this minute itself completed: in play from the next minute -----------
            if ENTRY_START <= m < ENTRY_END:
                g = find_fvg(b1, i, side)
                if g is not None:
                    self._add_zone("fvg", g.low, g.high, i, i)
            if self._done_now:
                self._breaker_at(i)
            if ds.eq_pending and ((h >= ds.eq_thr) if side > 0 else (l <= ds.eq_thr)):
                ds.eq_pending = False
                self._add_zone("eq", ds.eq, ds.eq, ds.conf_i, i)

        known = freshest(ds.zones)                     # every zone here has usable <= i
        if known is None:
            return
        if not ds.signalled:
            ds.signalled, ds.announced = True, known
            self._emit("signal", i, **self._signal_body(i, known))
        elif known is not ds.announced:
            ds.announced = known
            self._emit("zone", i, **self._signal_body(i, known))

    def _confirm(self, i: int, fired: list[str], b1: Bars) -> None:
        ds = self._ds
        side = ds.side
        ds.stage, ds.conf_i, ds.conf_types = "entry", i, list(fired)
        span_h, span_l = b1.high[ds.sweep_first:i + 1], b1.low[ds.sweep_first:i + 1]
        ds.range = (ds.extreme, float(span_h.max())) if side > 0 else (float(span_l.min()), ds.extreme)
        ds.eq = 0.5 * (ds.range[0] + ds.range[1])
        ds.eq_atr = self._atr_at(self._last)
        ds.eq_thr = ds.eq + side * EQ_EXCURSION_ATR * ds.eq_atr
        for k in range(ds.extreme_1m + 2, i + 1):                       # 1-minute gaps on the displacement leg
            g = find_fvg(b1, k, side)
            if g is not None:
                self._add_zone("fvg", g.low, g.high, k, i)
        ob = P.order_block(self._ctx_bars(ds.sweep_ci), ds.sweep_ci, side)
        if ob is not None:
            self._add_zone("ob", ob.low, ob.high, int(self._bend.a[ob.candle]), i)
        far = ds.range[1] >= ds.eq_thr if side > 0 else ds.range[0] <= ds.eq_thr
        if far:
            self._add_zone("eq", ds.eq, ds.eq, i, i)
        else:
            ds.eq_pending = True                                        # NaN ATR lands here and stays
        self._breaker_at(i)
        self._emit("confirmation", i, **self._setup_body(), types=list(fired), close=float(b1.close[i]),
                   range=[ds.range[0], ds.range[1]], eq=ds.eq, atr=ds.eq_atr,
                   eq_usable=bool(far), zones=[self._zone_dict(z) for z in ds.zones])

    def _seen_extreme(self, upto: int) -> float | None:
        """Highest high (long) / lowest low (short) over completed minutes i_open .. upto-1."""
        ds = self._ds
        if ds.i_open is None or upto <= ds.i_open:
            return None
        return float(self._h.a[ds.i_open:upto].max() if ds.side > 0 else self._l.a[ds.i_open:upto].min())

    def _signal_body(self, i: int, z: Zone) -> dict:
        ds = self._ds
        side = ds.side
        ref = z.high if side > 0 else z.low            # indicative entry: the zone's near edge
        atr_now = self._atr_at(self._last)
        sess = float(self._l.a[ds.start:i + 1].min() if side > 0 else self._h.a[ds.start:i + 1].max())
        tg = target_pairs(ds.levels, side, ref, self._seen_extreme(i + 1))
        chosen = tg[self.target_rule]
        body = self._setup_body()
        body.update(conf_types=list(ds.conf_types), conf_time=self._time(ds.conf_i),
                    sweep_time=self._time(ds.sweep_i), sweep_bar_time=self._iso(self._bt.a[ds.sweep_ci]),
                    zone=self._zone_dict(z), zones_known=[self._zone_dict(k) for k in ds.zones],
                    range=[ds.range[0], ds.range[1]], eq=ds.eq, atr=atr_now, ref_price=float(ref),
                    stops_indicative=candidate_stops(side, float(ref), ds.extreme, sess, atr_now),
                    targets=tg, target_rule=self.target_rule, t1=chosen["t1"], t2=chosen["t2"],
                    levels=dict(sorted(ds.levels.items(), key=lambda kv: level_sort_key(kv[0]))))
        return body

    # ---- the fill: the open of the minute after the closing-out minute -----------------------------------

    def _fill_stage(self, i: int, m: int) -> None:
        ds = self._ds
        if not (ENTRY_START <= m < ENTRY_END):
            self._expire(i, "entry_bar_outside_window")
            return
        # known at the open: the bins completed by the minute before, the rows before this one
        kind, body = self._fill_parts(float(self._o.a[i]), i, self._atr_at(self._last_prev))
        ds.done = True
        if kind == "fill":
            body["entry_time"] = self._time(i)
        self._emit(kind, i, at_open=True, **body)

    def _fill_parts(self, entry: float, upto: int, atr_f: float) -> tuple[str, dict]:
        """The fill (or the no-risk invalidation) for an entry at `entry`, the open of
        the minute that follows rows [0, upto). Reads nothing at or after row `upto`."""
        ds = self._ds
        side, z = ds.side, ds.trigger_zone
        sess = float(self._l.a[ds.start:upto].min() if side > 0 else self._h.a[ds.start:upto].max())
        stops = candidate_stops(side, entry, ds.extreme, sess, atr_f)
        if not side * (entry - stops["wick"]) > 0:
            body = self._setup_body()
            body.update(reason="no_risk", stage="pending", wick_stop=stops["wick"], atr=atr_f, traded=entry,
                        conf_types=list(ds.conf_types), zone=self._zone_dict(z))
            return "invalidated", body
        seen = self._seen_extreme(upto)
        tg = target_pairs(ds.levels, side, entry, seen)
        chosen = tg[self.target_rule]
        near1 = tg["nearest"]["t1"]
        known = [k for k in ds.zones if k.usable < upto]
        body = self._setup_body()
        body.update(entry=entry, conf_types=list(ds.conf_types),
                    conf_types_seen=dict(ds.conf_any), conf_time=self._time(ds.conf_i),
                    sweep_time=self._time(ds.sweep_i), touch_time=self._time(ds.touch_i),
                    trigger_time=self._time(ds.trigger_i), zone=self._zone_dict(z),
                    cooccur=sorted({k.kind for k in known if k.kind != z.kind}, key=ZONE_TYPES.index),
                    eq_refused=list(ds.eq_refused), range=[ds.range[0], ds.range[1]], eq=ds.eq,
                    atr=atr_f, session_extreme=sess, stops=stops,
                    risk={k: side * (entry - p) for k, p in stops.items()},
                    targets=tg, target_rule=self.target_rule, t1=chosen["t1"], t2=chosen["t2"],
                    seen_extreme=seen,
                    nearest_t1_already_traded=(None if near1 is None or seen is None
                                               else bool(side * (seen - near1["price"]) >= 0)),
                    levels=dict(sorted(ds.levels.items(), key=lambda kv: level_sort_key(kv[0]))))
        return "fill", body

    def preview_fill(self, entry_open: float) -> dict | None:
        """Between an `entry_trigger` and the bar that carries the fill: the event the
        entry minute WILL produce if it opens at `entry_open` — the same stops,
        targets and fields, flagged `provisional`. A feed of closed bars delivers
        the entry minute a minute after the entry happened; the runner observes
        the open itself and asks here, so the human's window opens on time. Pure:
        changes nothing, emits nothing. None when no entry is pending. The real
        `fill` follows when the closed bar is fed and is the one that is journaled;
        they differ only if the feed's open differs or the next bar is not the
        next minute."""
        ds = self._ds
        if ds is None or ds.done or ds.stage != "pending":
            return None
        kind, body = self._fill_parts(float(entry_open), self._n, self._atr_at(self._last))
        t = self._t.a[ds.trigger_i] + _MIN
        ev = {"kind": kind, "instrument": self.instrument, "day": ds.day_str,
              "setup_id": f"{self.instrument}-{ds.day_str}", "bar_time": self._iso(t),
              "et": hhmm(int(self._m.a[ds.trigger_i]) + 1), "knowable_at": self._iso(t)}
        ev.update(_clean(body))
        if kind == "fill":
            ev["entry_time"] = ev["bar_time"]
        ev["provisional"] = True
        return ev

    # ---- what the runner asks -------------------------------------------------------------------------------

    def warm_status(self) -> dict:
        """Is the history fed so far enough for the NEXT levels to be the contract's?"""
        if self._n == 0:
            return _warm_dict(0, 0, None, None, [])
        return check_warm(self._frame(self._n))

    def snapshot(self) -> dict:
        """Where the detector is. Machinery facts only."""
        ds = self._ds
        return {"instrument": self.instrument, "target_rule": self.target_rule, "bars": int(self._n),
                "last_bar": self._time(self._n - 1) if self._n else None,
                "sessions": len(self._sessions), "day": ds.day_str if ds else None,
                "stage": ("done" if ds.done else ds.stage) if ds else None,
                "events": len(self.events), "dropped": int(self.dropped)}


# ────────────────────────────── module surface ──────────────────────────────

def detect(frame: pd.DataFrame, instrument: str, **kwargs) -> list[dict]:
    """Every event of a 1-minute frame, in one call."""
    d = Detector(instrument, **kwargs)
    d.feed(frame)
    return d.events


def _warm_dict(sessions: int, h4_bins: int, first, last, thin: list) -> dict:
    ok = sessions >= WARM_SESSIONS and h4_bins >= WARM_H4_BINS
    why = []
    if sessions < WARM_SESSIONS:
        why.append(f"{sessions} completed sessions, {WARM_SESSIONS} needed for the composite")
    if h4_bins < WARM_H4_BINS:
        why.append(f"{h4_bins} completed 4-hour bins, {WARM_H4_BINS} needed for the H4 swings")
    return {"ok": bool(ok), "sessions": int(sessions), "sessions_required": WARM_SESSIONS,
            "h4_bins": int(h4_bins), "h4_bins_required": WARM_H4_BINS,
            "first_bar": first, "last_bar": last, "thin_sessions": thin,
            "reason": "; ".join(why) if why else "warm"}


def check_warm(frame: pd.DataFrame, for_day=None) -> dict:
    """Does a 1-minute store cover what the levels of `for_day` need?

    Needs: WARM_SESSIONS (20) completed sessions before `for_day` for the HVN
    composite (the previous one also gives POC_PREV, PDH and PDL), and the
    4-hour swing history over them — at least WARM_H4_BINS (100) completed
    4-hour bins. `for_day` is a session date; by default it is the session of
    the last bar in the frame, which is then treated as the one in progress.
    Sessions under THIN_SESSION_BARS bars are listed, not refused: an early
    close is thin and legitimate, a PC that was off is thin and is not, and
    only the person reading can tell which.
    """
    if frame is None or len(frame) == 0 or not isinstance(frame.index, pd.DatetimeIndex):
        return _warm_dict(0, 0, None, None, [])
    _, day = et_clock(frame.index)
    target = day[-1] if for_day is None else np.datetime64(pd.Timestamp(for_day).normalize(), "ns")
    before = day < target
    days, counts = np.unique(day[before], return_counts=True)
    thin = [str(np.datetime_as_string(d, unit="D")) for d, k in zip(days, counts) if k < THIN_SESSION_BARS]
    h4 = 0
    if before.any():
        ctx = P.resample_context(frame.loc[before, ["open", "high", "low", "close"]], 240, bar_minutes=1)
        h4 = int(ctx.last[-1]) + 1
    return _warm_dict(len(days), h4, str(frame.index[0]), str(frame.index[-1]), thin)

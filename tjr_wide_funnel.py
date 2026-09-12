#!/usr/bin/env python3
"""Part 2 of DESIGN-tjr-human.md: the wider TJR spec, run as a funnel.

Round 1 read TJR narrowly: six session levels, one confirmation (BOS), two
zones (FVG, inverted FVG). His own description is wider, and section 2 of the
human contract writes the wide version down: the six levels plus 1-hour and
4-hour swings, the previous session's POC and the composite profile's
high-volume nodes; three confirmations (BOS, inverted FVG, OTE); four zones
(FVG, equilibrium, order block, breaker). This runs that spec over the same
149 sessions and counts: sessions -> sweeps -> confirmations -> zones in
discount/premium -> fills, broken out by level, confirmation and zone type,
next to round 1's numbers.

FUNNEL ONLY. No stop, no target, no exit, no Sharpe, no gate, no verdict,
nothing registered, no trial added to the count. The question is whether the
narrow reading starved the test or the wide one produces more of the same.

Composition, not a copy: the clock, the six levels and the sweep test are
tjr_intraday's (session_clock, session_levels, sweep_of_levels); the swings,
gaps and BOS are the round-1 primitives; the resample, the volume profile, the
POC, the HVNs, the order block and the breaker are the section-2 additions to
primitives.py. The per-day walk below only decides which one is consulted when.

    python tjr_wide_funnel.py --mode literal > results/tjr_intraday/part2_wide_funnel.txt
    python tjr_wide_funnel.py                 # decided mode (section 2.3), tees to part2_decided_funnel.txt
    python tjr_wide_funnel.py --csv data/NQ_5min_tv.csv --hvn-frac 0.6

Causality
---------
Every level of day D is frozen at the 09:25 bar's close from bars <= that bar:
the six levels by session_levels' own argument, the 1h/4h swings by
resample_context's completion rule plus confirmed_at, POC_PREV from the previous
session's bars, the HVNs from the twenty completed sessions before D. Every
confirmation and zone at bar i reads bars <= i. The proof section at the end
truncates each frame at six mid-session bars and shows that the day-level rows
for every earlier day, and the cut day's row up to the cut, do not move.
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab import data
from quantlab.strategies.predictive import primitives as P
from quantlab.strategies.predictive import tjr_intraday as M
from quantlab.strategies.predictive.primitives import Bars, confirmed_swings, detect_bos, find_fvg

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

DEFAULT_CSVS = ("data/NQ_5min_tv.csv", "data/ES_5min_tv.csv")
OUT_DIR = os.path.join("results", "tjr_intraday")
TICK = 0.25                      # ES and NQ
COMPOSITE_SESSIONS = 20          # the fixed composite behind the HVNs
OTE = 0.79                       # the retracement the ote confirmation must clear
SWING_LEFT, SWING_RIGHT = 2, 2
RESAMPLES = OrderedDict(H1=60, H4=240)
CONF_TYPES = ("bos", "ifvg", "ote")
ZONE_TYPES = ("fvg", "eq", "ob", "breaker")
#: Same-bar tie order for "freshest zone": the gap that formed there over the
#: violation that happened there over the level over the candle from before.
ZONE_RANK = {"ob": 0, "eq": 1, "breaker": 2, "fvg": 3}
#: Round 1's funnel (DESIGN-tjr-intraday.md section 9): sweeps, BOS, zone, fills.
ROUND1 = {"NQ": (93, 50, 34, 7), "ES": (97, 46, 24, 1)}
STAGES = ("sweeps", "confirmations", "zones", "fills")
LEVEL_ORDER = ("ASIA_H", "ASIA_L", "LON_H", "LON_L", "PDH", "PDL",
               "H1_SH", "H1_SL", "H4_SH", "H4_SL", "POC_PREV")
DIRECTIONAL_LOWS = ("ASIA_L", "LON_L", "PDL", "H1_SL", "H4_SL")
DIRECTIONAL_HIGHS = ("ASIA_H", "LON_H", "PDH", "H1_SH", "H4_SH")
#: The five level classes of DESIGN-tjr-human.md section 2.3 (decided mode):
#: the sweep test runs per class; POC_PREV and HVN_* are targets only.
CLASSES = ("ASIA", "LON", "PD", "H1", "H4")
MODES = ("decided", "literal")
EQ_EXCURSION_ATR = 0.5           # section 2.3 item 2: the retrace the eq touch must follow


def level_type(name: str) -> str:
    """ASIA_L -> ASIA, PDH -> PD, H1_SH -> H1, POC_PREV -> POC, HVN_3 -> HVN."""
    return "PD" if name in ("PDH", "PDL") else name.split("_")[0]


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass(frozen=True)
class Zone:
    """A continuation zone. `formed` is the contract's freshness stamp (the gap's
    completion bar, the breaker's violation bar, the order block's candle, the
    confirmation bar for eq); `usable` is the bar the funnel could first know it
    on, which is the later of `formed`, the confirmation bar and — for a breaker —
    the bar its swing confirmed on. A fill must be strictly after `usable`."""
    kind: str
    low: float
    high: float
    formed: int
    usable: int


def freshest(zones: list[Zone]) -> Zone | None:
    return max(zones, key=lambda z: (z.formed, ZONE_RANK[z.kind])) if zones else None


# ────────────────────────────── the funnel ──────────────────────────────

class WideFunnel:
    """One instrument, one frame: the wide spec walked day by day.

    Frame-level work happens once in __init__ (clock, six levels, the two
    resamples and their swings, the 5-minute swings, one volume profile per
    session); `day()` then reads only what is admissible at each bar.
    """

    def __init__(self, df: pd.DataFrame, bin_ticks: int = 4, hvn_frac: float = 0.5,
                 hvn_max: int = 6, level_types: tuple[str, ...] | None = None,
                 ifvg_fresh: bool = False, mode: str = "literal"):
        """`level_types` restricts the sweep targets by type (None = all seven);
        ("ASIA", "LON", "PD") is round 1's set and must reproduce its sweeps.
        `ifvg_fresh` is the sensitivity reading of the ifvg confirmation: only a
        gap no bar closed through between its completion and the sweep bar
        counts (the contract's literal reading, the default, counts any gap of
        the day).

        `mode` is "literal" (section 2.1 as built: the committed printout) or
        "decided" (section 2.3): ifvg fresh-only, the eq zone usable only after
        a 0.5-ATR excursion beyond the midpoint, the sweep judged per level
        class on the directional set with POC/HVN as targets only, and T1/T2
        logged per fill. `level_types` and `ifvg_fresh` are literal-mode knobs;
        decided mode fixes both."""
        if mode not in MODES:
            raise ValueError(f"WideFunnel: mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.decided = mode == "decided"
        self.df = df
        self.bin_ticks, self.hvn_frac, self.hvn_max = int(bin_ticks), float(hvn_frac), int(hvn_max)
        self.level_types = tuple(level_types) if level_types is not None else None
        self.ifvg_fresh = True if self.decided else bool(ifvg_fresh)
        self.rows: list[dict] = []
        self.clock = M.session_clock(df.index)
        if self.clock is None:
            return                      # daily bars, no 09:30: nothing to do, as round 1
        self.bars = Bars.from_frame(df, 14)
        b = self.bars
        self.o, self.h, self.l, self.c = b.open, b.high, b.low, b.close
        self.v = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(len(df))
        self.levels6 = M.session_levels(self.clock, self.h, self.l)

        # 1h / 4h bins on the NY clock anchored 18:00 ET; swings on them, admitted
        # per input bar by confirmed_at <= ctx.last[i] (the newest COMPLETED bin).
        self.ctx = {name: P.resample_context(df, mins, atr_len=14) for name, mins in RESAMPLES.items()}
        self.rswings = {name: confirmed_swings(ctx.bars, SWING_LEFT, SWING_RIGHT)
                        for name, ctx in self.ctx.items()}
        # 5-minute structure: the same primitive round 1 used, admitted by confirmed_at <= i.
        self.highs5, self.lows5 = confirmed_swings(b, SWING_LEFT, SWING_RIGHT)
        self.highs5_at = [s.confirmed_at for s in self.highs5]
        self.lows5_at = [s.confirmed_at for s in self.lows5]

        # Sessions in frame order. clock.day never decreases along the index.
        day = self.clock.day
        starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
        ends = np.r_[starts[1:], len(df)]
        self.sessions = [(pd.Timestamp(day[a]), int(a), int(e)) for a, e in zip(starts, ends)]
        # One profile per session on its own $-grid (edges anchored at multiples
        # of the bin width, so sessions line up bin for bin by integer key).
        self.w = TICK * self.bin_ticks
        self.profiles: list[tuple[P.Profile, float, int]] = []      # (profile, vwap, first bin key)
        for _, a, e in self.sessions:
            prof = P.volume_profile(self.h[a:e], self.l[a:e], self.v[a:e], TICK, self.bin_ticks)
            vw = P.vwap(self.h[a:e], self.l[a:e], self.c[a:e], self.v[a:e])
            self.profiles.append((prof, vw, int(round(prof.edges[0] / self.w))))

        for s, (_, a, e) in enumerate(self.sessions):
            row = self.day(s, a, e)
            if row is not None:
                self.rows.append(row)

    # ---- levels, all from bars <= the 09:25 bar ------------------------------

    def resampled_levels(self, name: str, i: int, close: float) -> dict[str, float]:
        """H1_SH/H1_SL (or H4): nearest admissible swing above/below `close` at bar i."""
        ctx = self.ctx[name]
        newest = ctx.last[i]
        highs, lows = self.rswings[name]
        out = {}
        above = [s.price for s in highs if s.confirmed_at <= newest and s.price > close]
        below = [s.price for s in lows if s.confirmed_at <= newest and s.price < close]
        if above:
            out[f"{name}_SH"] = min(above)
        if below:
            out[f"{name}_SL"] = max(below)
        return out

    def composite(self, s: int) -> tuple[P.Profile | None, int]:
        """The fixed composite before session s: the last COMPOSITE_SESSIONS
        completed sessions, summed bin for bin. Returns (profile, depth)."""
        lo_s = max(0, s - COMPOSITE_SESSIONS)
        parts = self.profiles[lo_s:s]
        if not parts:
            return None, 0
        kmin = min(k for _, _, k in parts)
        kmax = max(k + len(p.volume) for p, _, k in parts)
        vols = np.zeros(kmax - kmin)
        for p, _, k in parts:                       # chronological, so the sums are reproducible
            vols[k - kmin:k - kmin + len(p.volume)] += p.volume
        edges = self.w * np.arange(kmin, kmax + 1)
        return P.Profile(edges, vols), len(parts)

    # ---- the sweep per level class (decided mode, section 2.3 item 3) ----------

    def class_sweep(self, row: dict, i: int, lows: dict, highs: dict,
                    below: dict, above: dict) -> tuple[int, str, float] | None:
        """Round 1's test run per class on the directional set at bar i.

        A class whose wick took one of its lows AND one of its highs abstains
        (sweep_of_levels returns None for it; told apart from "no breach" by
        the class's own below/above). Among the classes that return a sweep:
        all one side -> that side, and the setup's level is the deepest
        breached level of that side across those classes; two sides -> the bar
        is ambiguous (class disagreement) and skipped. No class returning is
        not a sweep; if some class abstained it is counted as abstention-only,
        separately from ambiguity. `row` receives the bar's verdict.
        """
        h, l, c = self.h, self.l, self.c
        returns: dict[str, tuple[int, str, float]] = {}
        abstained: list[str] = []
        for cls in CLASSES:
            lc = {k: p for k, p in lows.items() if level_type(k) == cls}
            hc = {k: p for k, p in highs.items() if level_type(k) == cls}
            r = M.sweep_of_levels(h, l, c, i, lc, hc)
            if r is None:
                if any(k in below for k in lc) and any(k in above for k in hc):
                    abstained.append(cls)
                continue
            returns[cls] = r
        sides = {r[0] for r in returns.values()}
        if len(sides) == 2:
            row["ambiguous"].append(i)
            row["amb_detail"][i] = ";".join(f"{cls}:{'L' if r[0] > 0 else 'S'}" for cls, r in returns.items())
            if abstained:
                row["amb_detail"][i] += ";" + ";".join(f"{cls}:abstain" for cls in abstained)
            return None
        if not returns:
            if abstained:
                row["abstain_only"].append(i)
            return None
        side = sides.pop()
        taken = below if side > 0 else above
        cand = {k: p for k, p in taken.items() if level_type(k) in returns}
        name = min(cand, key=cand.get) if side > 0 else max(cand, key=cand.get)
        row["cls_ret"] = {cls: r[1] for cls, r in returns.items()}
        row["cls_abs"] = list(abstained)
        return side, name, cand[name]

    # ---- one day ----------------------------------------------------------------

    def day(self, s: int, a: int, e: int) -> dict | None:
        clock, o, h, l, c, bars = self.clock, self.o, self.h, self.l, self.c, self.bars
        minutes = clock.minutes
        sw = np.flatnonzero(clock.in_sweep[a:e])
        if len(sw) == 0:
            return None                              # no sweep-window bar: not a session for the funnel
        i_open = a + int(sw[0])
        i_pre = i_open - 1 if i_open > a else None   # the 09:25 bar (the last bar before the window)
        row = {"day": self.sessions[s][0], "day_start": a, "i_open": i_open, "i_pre": i_pre,
               "has_0930": bool(minutes[i_open] == M.SWEEP_WINDOW[0]),
               "close_pre": float(c[i_pre]) if i_pre is not None else math.nan,
               "levels": OrderedDict(), "n_hvn": 0, "depth": 0,
               "sweep_idx": None, "side": 0, "breached": [], "level": None, "extreme": math.nan,
               "pre_swing": None, "ambiguous": [],
               "conf_idx": None, "conf_types": [], "conf_any": {}, "ifvg_fresh_any": {},
               "eq": math.nan, "range": (math.nan, math.nan),
               "zones": [], "fill_idx": None, "entry": math.nan, "fill_zone": None,
               # decided mode only (section 2.3); untouched in literal mode
               "abstain_only": [], "amb_detail": {}, "cls_ret": {}, "cls_abs": [],
               "breached_opp": [], "eq_atr": math.nan, "eq_usable": None,
               "eq_open_far": None, "cooccur": [], "targets": {}}

        # -- levels, frozen at the 09:25 close ---------------------------------------
        lv = row["levels"]
        for name in M.HIGHS + M.LOWS:
            val = self.levels6[name][i_open]
            if not math.isnan(val):
                lv[name] = float(val)
        if i_pre is not None:
            close_pre = float(c[i_pre])
            for name in RESAMPLES:
                lv.update(self.resampled_levels(name, i_pre, close_pre))
            if s >= 1:
                prof, vw, _ = self.profiles[s - 1]
                lv["POC_PREV"] = P.poc(prof, near=vw)
            comp, depth = self.composite(s)
            row["depth"] = depth
            if comp is not None:
                nodes = P.hvn(comp, smooth=5, frac=self.hvn_frac, max_n=self.hvn_max, near=close_pre)
                for k, price in enumerate(nodes, 1):
                    lv[f"HVN_{k}"] = price
                row["n_hvn"] = len(nodes)
        if self.level_types is not None:
            for name in [n for n in lv if level_type(n) not in self.level_types]:
                del lv[name]
        # Which side each level rests on. Directional levels by name; POC and HVNs
        # by where they sit against the 09:25 close (at the close -> below).
        # Decided mode (section 2.3 item 3): the sweep set is the directional
        # levels only; POC_PREV and HVN_* stay in `lv` as targets.
        if self.decided:
            lows = {n: p for n, p in lv.items() if n in DIRECTIONAL_LOWS}
            highs = {n: p for n, p in lv.items() if n in DIRECTIONAL_HIGHS}
        else:
            lows = {n: p for n, p in lv.items()
                    if n in DIRECTIONAL_LOWS or (level_type(n) in ("POC", "HVN") and p <= row["close_pre"])}
            highs = {n: p for n, p in lv.items()
                     if n in DIRECTIONAL_HIGHS or (level_type(n) in ("POC", "HVN") and p > row["close_pre"])}

        # -- sweep: round 1's test against the union of levels (literal) or per class (decided) --
        sweep_idx = None
        for i in range(i_open, e):
            if minutes[i] >= M.SWEEP_WINDOW[1]:
                break
            if not clock.in_sweep[i]:
                continue
            below = {k: p for k, p in lows.items() if l[i] < p}
            above = {k: p for k, p in highs.items() if h[i] > p}
            if self.decided:
                swept = self.class_sweep(row, i, lows, highs, below, above)
            else:
                swept = M.sweep_of_levels(h, l, c, i, lows, highs)
                if swept is None and below and above:
                    row["ambiguous"].append(i)
            if swept is None:
                continue
            side, name, _ = swept
            sweep_idx = i
            row.update(sweep_idx=i, side=side, level=name,
                       extreme=float(l[i]) if side > 0 else float(h[i]))
            # every level the wick took, deepest first; the setup's is the deepest
            # (decided: the deepest among the classes that returned the side; a
            # deeper level in an abstaining class is in `breached` but not the setup's)
            row["breached"] = (sorted(below, key=below.get) if side > 0
                               else sorted(above, key=above.get, reverse=True))
            # decided: the opposite-side levels the same wick took (abstaining
            # classes, or a class whose high was run without a close back inside)
            row["breached_opp"] = (sorted(above, key=above.get, reverse=True) if side > 0
                                   else sorted(below, key=below.get))
            break
        if sweep_idx is None:
            return row
        side, extreme = row["side"], row["extreme"]

        # The most recent opposing 5-minute swing existing at the sweep bar. Frozen.
        opp, opp_at = (self.highs5, self.highs5_at) if side > 0 else (self.lows5, self.lows5_at)
        n_adm = bisect.bisect_right(opp_at, sweep_idx)
        pre = opp[n_adm - 1] if n_adm else None
        row["pre_swing"] = pre.price if pre is not None else None
        ote_level = extreme + OTE * (pre.price - extreme) if pre is not None else None

        # -- confirmations: first bar after the sweep, <= 10:05, on which any fires -----
        # Opposing gaps completed on day D at or before bar i, for the inversion
        # test. A gap is STALE when a bar between its completion and the sweep bar
        # (inclusive) already closed through it: the literal definition still
        # counts it; the `ifvg_fresh` reading does not. Both are recorded.
        opp_gaps = [g for g in (find_fvg(bars, k, -side) for k in range(a, sweep_idx + 1)) if g]
        stale = {g.index: any(M._inverted(g, side, c[j]) for j in range(g.index + 1, sweep_idx + 1))
                 for g in opp_gaps}
        conf_idx = None
        for i in range(sweep_idx + 1, e):
            if minutes[i] >= M.ENTRY_WINDOW[1]:
                break
            g = find_fvg(bars, i, -side)
            if g:
                opp_gaps.append(g)
                stale[g.index] = False
            fired = []
            if pre is not None and detect_bos(bars, i, pre.price, side):
                fired.append("bos")
            inverted = [g for g in opp_gaps if M._inverted(g, side, c[i])]
            fresh = [g for g in inverted if not stale[g.index]]
            if fresh if self.ifvg_fresh else inverted:
                fired.append("ifvg")
            if fresh:
                row["ifvg_fresh_any"].setdefault(i, True)
            if ote_level is not None and (c[i] > ote_level if side > 0 else c[i] < ote_level):
                fired.append("ote")
            for t in fired:
                row["conf_any"].setdefault(t, i)
            if fired and conf_idx is None:
                conf_idx = i
                row.update(conf_idx=i, conf_types=fired)
        if conf_idx is None:
            return row

        # -- dealing range at confirmation, zones, fill ---------------------------------
        if side > 0:
            rng = (extreme, float(h[sweep_idx:conf_idx + 1].max()))
        else:
            rng = (float(l[sweep_idx:conf_idx + 1].min()), extreme)
        eq = 0.5 * (rng[0] + rng[1])
        row.update(eq=eq, range=rng)
        zones: list[Zone] = row["zones"]
        seen = set()

        def add(kind, lo, hi, formed, usable):
            z = Zone(kind, float(lo), float(hi), int(formed), int(usable))
            if M._discount(z, side, eq):
                zones.append(z)

        def breaker_at(i):
            blk = P.breaker_block(bars, a, sweep_idx, i, side, self.highs5 if side > 0 else self.lows5)
            if blk is not None and (blk.index, blk.candle) not in seen:
                seen.add((blk.index, blk.candle))
                add("breaker", blk.low, blk.high, blk.index, i)

        for k in range(sweep_idx + 2, conf_idx + 1):            # gaps on the displacement leg
            z = find_fvg(bars, k, side)
            if z:
                add("fvg", z.low, z.high, k, conf_idx)
        ob = P.order_block(bars, sweep_idx, side)
        if ob is not None:
            add("ob", ob.low, ob.high, ob.candle, conf_idx)
        # eq. Literal: usable at the confirmation bar. Decided (section 2.3 item
        # 2): usable only from the first completed bar j >= sweep bar on which
        # price was >= 0.5 x ATR beyond the midpoint on the far side, ATR(14) of
        # this frame read at the confirmation bar. The range extreme through the
        # confirmation bar covers j <= conf_idx (then j = conf_idx); otherwise
        # the zone is not in the list until the bar that extends that far, and
        # its `formed` stamp stays conf_idx. A fill needs i > usable.
        eq_pending = False
        if self.decided:
            atr_c = float(bars.atr[conf_idx])
            thr = eq + side * EQ_EXCURSION_ATR * atr_c
            row["eq_atr"] = atr_c
            far = (lambda j: h[j] >= thr) if side > 0 else (lambda j: l[j] <= thr)   # noqa: E731
            if (rng[1] >= thr) if side > 0 else (rng[0] <= thr):
                add("eq", eq, eq, conf_idx, conf_idx)
                row["eq_usable"] = conf_idx
            else:
                eq_pending = True
        else:
            add("eq", eq, eq, conf_idx, conf_idx)
        breaker_at(conf_idx)

        for i in range(conf_idx + 1, e):
            if minutes[i] >= M.ENTRY_WINDOW[1]:
                break
            if clock.in_entry[i]:                               # a fresh gap inside the entry window
                z = find_fvg(bars, i, side)
                if z:
                    add("fvg", z.low, z.high, i, i)
            breaker_at(i)                                        # a violation on any bar after the sweep
            if eq_pending and far(i):                            # the excursion bar: eq usable from here
                add("eq", eq, eq, conf_idx, i)
                row["eq_usable"] = i
                eq_pending = False
            if row["fill_idx"] is not None or not clock.in_entry[i]:
                continue
            zone = freshest(zones)
            if zone is None or i <= max(conf_idx, zone.usable):
                continue
            entry = None
            if side > 0 and l[i] <= zone.high:
                entry = min(zone.high, o[i])
            elif side < 0 and h[i] >= zone.low:
                entry = max(zone.low, o[i])
            if entry is not None:
                row.update(fill_idx=i, entry=float(entry), fill_zone=zone)
                if self.decided:
                    self.fill_log(row, i, zone, zones, lv)
        return row

    def fill_log(self, row: dict, i: int, zone: Zone, zones: list[Zone], lv: dict) -> None:
        """Decided-mode diagnostics at the fill bar i: for an eq fill, the other
        zone types in discount / premium at that bar and whether the bar opened
        on the far side of EQ; for every fill, T1 / T2 (section 2.3 item 3 and
        the task's item 2d): the nearest level beyond the entry among the
        directional set, among POC_PREV + HVN_*, and among all levels (T1), and
        the one beyond T1 (T2). Logged only; nothing here changes a fill."""
        side, entry, eq = row["side"], row["entry"], row["eq"]
        if zone.kind == "eq":
            row["cooccur"] = sorted({z.kind for z in zones if z.kind != "eq"}, key=ZONE_TYPES.index)
            row["eq_open_far"] = bool(self.o[i] > eq) if side > 0 else bool(self.o[i] < eq)
        beyond = {k: p for k, p in lv.items() if side * (p - entry) > 0}
        ranked = sorted(beyond, key=lambda k: side * (beyond[k] - entry))
        dirl = [k for k in ranked if level_type(k) in CLASSES]
        vol = [k for k in ranked if level_type(k) in ("POC", "HVN")]
        row["targets"] = {
            "t1_dir": (dirl[0], beyond[dirl[0]]) if dirl else None,
            "t1_vol": (vol[0], beyond[vol[0]]) if vol else None,
            "t1": (ranked[0], beyond[ranked[0]]) if ranked else None,
            "t2": (ranked[1], beyond[ranked[1]]) if len(ranked) > 1 else None,
        }


# ────────────────────────────── rows: as-of, formatting ──────────────────────────────

def as_of(row: dict, cut: int) -> dict:
    """The full-run row with everything at bar >= cut forgotten. What a frame
    truncated at `cut` must reproduce, if the funnel is causal."""
    r = dict(row)
    r["ambiguous"] = [i for i in row["ambiguous"] if i < cut]
    r["abstain_only"] = [i for i in row["abstain_only"] if i < cut]
    r["amb_detail"] = {i: d for i, d in row["amb_detail"].items() if i < cut}
    no_fill = dict(fill_idx=None, entry=math.nan, fill_zone=None, eq_open_far=None, cooccur=[], targets={})
    no_conf = dict(conf_idx=None, conf_types=[], eq=math.nan, range=(math.nan, math.nan), zones=[],
                   eq_atr=math.nan, eq_usable=None, **no_fill)
    if row["sweep_idx"] is None or row["sweep_idx"] >= cut:
        r.update(sweep_idx=None, side=0, breached=[], breached_opp=[], level=None, extreme=math.nan,
                 pre_swing=None, conf_any={}, ifvg_fresh_any={}, cls_ret={}, cls_abs=[], **no_conf)
        return r
    r["conf_any"] = {t: j for t, j in row["conf_any"].items() if j < cut}
    r["ifvg_fresh_any"] = {j: v for j, v in row["ifvg_fresh_any"].items() if j < cut}
    if row["conf_idx"] is None or row["conf_idx"] >= cut:
        r.update(**no_conf)
        return r
    r["zones"] = [z for z in row["zones"] if z.usable < cut]
    if row["eq_usable"] is not None and row["eq_usable"] >= cut:
        r["eq_usable"] = None
    if row["fill_idx"] is not None and row["fill_idx"] >= cut:
        r.update(**no_fill)
    return r


def zone_used(row: dict) -> Zone | None:
    return row["fill_zone"] if row["fill_idx"] is not None else freshest(row["zones"])


def outcome(row: dict) -> str:
    if row["fill_idx"] is not None:
        return "filled"
    if row["zones"]:
        return "no_fill"
    if row["conf_idx"] is not None:
        return "no_zone"
    if row["sweep_idx"] is not None:
        return "no_confirmation"
    return "no_sweep"


def zone_any(row: dict) -> dict[str, int]:
    """Zone type -> the first bar it qualified on (in discount / premium)."""
    out: dict[str, int] = {}
    for z in sorted(row["zones"], key=lambda z: z.usable):
        out.setdefault(z.kind, z.usable)
    return out


def format_row(row: dict, minutes: np.ndarray, decided: bool = False) -> OrderedDict:
    """The day-level CSV row. Times are ET bar stamps. `decided` appends the
    section-2.3 columns after the literal ones (the literal row is unchanged)."""
    t = lambda i: hhmm(int(minutes[i])) if i is not None else ""     # noqa: E731
    f = lambda x: "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"  # noqa: E731
    zu = zone_used(row)
    out = literal_row(row, t, f, zu)
    if not decided:
        return out
    tg = row["targets"]
    lvl = lambda kv: f"{kv[0]}={kv[1]:.2f}" if kv else ""              # noqa: E731
    out.update([
        ("ambiguous_detail", ";".join(f"{t(i)}[{row['amb_detail'].get(i, '')}]" for i in row["ambiguous"])),
        ("abstain_only_bars", ";".join(t(i) for i in row["abstain_only"])),
        ("classes_returned", ";".join(f"{cls}:{name}" for cls, name in row["cls_ret"].items())),
        ("classes_abstained", ";".join(row["cls_abs"])),
        ("wick_also_took", ";".join(row["breached_opp"])),
        ("eq_atr", f(row["eq_atr"])),
        ("eq_usable_time", t(row["eq_usable"])),
        ("eq_wait_bars", "" if row["eq_usable"] is None else row["eq_usable"] - row["conf_idx"]),
        ("eq_fill_open_far_side", "" if row["eq_open_far"] is None else int(row["eq_open_far"])),
        ("cooccur_zones", "+".join(row["cooccur"])),
        ("t1_dir", lvl(tg.get("t1_dir"))),
        ("t1_vol", lvl(tg.get("t1_vol"))),
        ("t1", lvl(tg.get("t1"))),
        ("t2", lvl(tg.get("t2"))),
    ])
    return out


def literal_row(row: dict, t, f, zu) -> OrderedDict:
    return OrderedDict([
        ("session_day", row["day"].strftime("%Y-%m-%d")),
        ("has_0930", int(row["has_0930"])),
        ("close_0925", f(row["close_pre"])),
        ("composite_depth", row["depth"]),
        ("n_hvn", row["n_hvn"]),
        ("levels_present", ";".join(f"{k}={v:.2f}" for k, v in row["levels"].items())),
        ("ambiguous_bars", ";".join(t(i) for i in row["ambiguous"])),
        ("side", row["side"]),
        ("sweep_time", t(row["sweep_idx"])),
        ("swept_levels", ";".join(row["breached"])),
        ("swept_level", row["level"] or ""),
        ("swept_type", level_type(row["level"]) if row["level"] else ""),
        ("sweep_extreme", f(row["extreme"])),
        ("pre_sweep_swing", f(row["pre_swing"])),
        ("conf_time", t(row["conf_idx"])),
        ("conf_types", "+".join(row["conf_types"])),
        ("conf_any", ";".join(f"{k}@{t(j)}" for k, j in sorted(row["conf_any"].items(), key=lambda kv: (kv[1], CONF_TYPES.index(kv[0]))))),
        ("ifvg_fresh_on_conf_bar", "" if row["conf_idx"] is None or "ifvg" not in row["conf_types"]
         else int(row["conf_idx"] in row["ifvg_fresh_any"])),
        ("range_lo", f(row["range"][0])),
        ("range_hi", f(row["range"][1])),
        ("eq", f(row["eq"])),
        ("zones_any", ";".join(f"{k}@{t(j)}" for k, j in sorted(zone_any(row).items(), key=lambda kv: (kv[1], ZONE_TYPES.index(kv[0]))))),
        ("zone_used", zu.kind if zu else ""),
        ("zone_low", f(zu.low) if zu else ""),
        ("zone_high", f(zu.high) if zu else ""),
        ("zone_time", t(zu.formed) if zu else ""),
        ("fill_time", t(row["fill_idx"])),
        ("entry", f(row["entry"])),
        ("outcome", outcome(row)),
    ])


# ────────────────────────────── reporting ──────────────────────────────

def counter_line(cnt: Counter, keys=None) -> str:
    keys = list(keys) if keys is not None else sorted(cnt, key=lambda k: (-cnt[k], str(k)))
    return "  ".join(f"{k}:{cnt.get(k, 0)}" for k in keys) if keys else "(none)"


def funnel_counts(fn: WideFunnel) -> dict:
    rows = fn.rows
    swept = [r for r in rows if r["sweep_idx"] is not None]
    conf = [r for r in swept if r["conf_idx"] is not None]
    zoned = [r for r in conf if r["zones"]]
    filled = [r for r in zoned if r["fill_idx"] is not None]
    return {"sessions": len(rows), "sweeps": len(swept), "confirmations": len(conf),
            "zones": len(zoned), "fills": len(filled)}


def round1_crosscheck(fn6: WideFunnel, r1log: pd.DataFrame) -> tuple[int, bool]:
    """This code with round 1's six levels against tjr_intraday.day_log: same
    sweep days, and on each the same bar, side and level."""
    r1 = r1log.set_index("session_day")
    mine = {r["day"]: r for r in fn6.rows if r["sweep_idx"] is not None}
    theirs = set(r1.index[r1["side"] != 0])
    same = set(mine) == theirs and all(
        int(r1.loc[d, "side"]) == r["side"] and r1.loc[d, "sweep_level"] == r["level"]
        and pd.Timestamp(r1.loc[d, "sweep_time"]) == pd.Timestamp(fn6.df.index[r["sweep_idx"]])
        for d, r in mine.items())
    return len(mine), bool(same)


def report(label: str, path: str, fn: WideFunnel, fn6: WideFunnel, r1log: pd.DataFrame,
           fn_fresh: WideFunnel) -> dict:
    rows = fn.rows
    df = fn.df
    print(f"=== {label}  {path}  {len(df)} bars  {df.index[0]} .. {df.index[-1]} UTC ===")
    print()
    print("1. levels at 09:25")
    n = len(rows)
    print(f"   sessions with a sweep-window bar: {n}   (with a bar stamped exactly 09:30: {sum(r['has_0930'] for r in rows)})")
    present = Counter()
    for r in rows:
        for k in r["levels"]:
            present[k] += 1
        if r["n_hvn"]:
            present["HVN(any)"] += 1
    names = [k for k in LEVEL_ORDER] + ["HVN(any)"]
    print("   present: " + counter_line(present, names))
    print("   HVN count per day: " + counter_line(Counter(r["n_hvn"] for r in rows), range(0, fn.hvn_max + 1)))
    depth = Counter(r["depth"] for r in rows)
    print("   composite depth per day (sessions in the profile): "
          + counter_line(depth, sorted(depth, reverse=True)))
    print(f"   POC_PREV absent on: {[r['day'].strftime('%Y-%m-%d') for r in rows if 'POC_PREV' not in r['levels']] or 'no day'}"
          f"   (first session in the file has no previous session)")

    swept = [r for r in rows if r["sweep_idx"] is not None]
    conf = [r for r in swept if r["conf_idx"] is not None]
    zoned = [r for r in conf if r["zones"]]
    zoned_x = [r for r in conf if any(z.kind != "eq" for z in r["zones"])]
    filled = [r for r in zoned if r["fill_idx"] is not None]
    counts = {"sessions": n, "sweeps": len(swept), "confirmations": len(conf),
              "zones": len(zoned), "fills": len(filled)}
    print()
    print("2. funnel")
    print(f"   sessions {n} -> sweeps {len(swept)} -> confirmations {len(conf)} -> "
          f"zones in discount/premium {len(zoned)} -> fills {len(filled)}")
    print(f"   zones excluding the eq touch (which is always in discount by construction): {len(zoned_x)}")
    print(f"   sweep-window bars skipped as ambiguous (a low and a high in one wick): "
          f"{sum(len(r['ambiguous']) for r in rows)} bars on {sum(1 for r in rows if r['ambiguous'])} days")
    print(f"   sweep days with no admissible opposing 5-minute swing (bos/ote cannot fire): "
          f"{sum(1 for r in swept if r['pre_swing'] is None)}")
    degenerate = [r for r in swept if r["pre_swing"] is not None
                  and r["side"] * (r["pre_swing"] - r["extreme"]) <= 0]
    print(f"   sweep days whose pre-sweep swing sits on the wrong side of the sweep extreme (degenerate leg): "
          f"{len(degenerate)}")
    print(f"   sweeps by side: long {sum(1 for r in swept if r['side'] > 0)}  short {sum(1 for r in swept if r['side'] < 0)}")
    n6, same = round1_crosscheck(fn6, r1log)
    print(f"   cross-check, this code with round 1's six levels only: sweeps {n6}; sweep day, bar, side and level "
          f"identical to tjr_intraday.day_log: {'yes' if same else 'NO'}")
    six = {r["day"]: r for r in fn6.rows}
    lost = [r for r in rows if r["sweep_idx"] is None and six[r["day"]]["sweep_idx"] is not None]
    gained = [r for r in rows if r["sweep_idx"] is not None and six[r["day"]]["sweep_idx"] is None]
    both = [r for r in swept if six[r["day"]]["sweep_idx"] is not None]
    print(f"   six-level sweep days lost under the wide set: {len(lost)} (of which by the ambiguity rule: "
          f"{sum(1 for r in lost if r['ambiguous'])}); days swept only under the wide set: {len(gained)}; "
          f"swept under both: {len(both)} (same side {sum(1 for r in both if r['side'] == six[r['day']]['side'])}, "
          f"same bar {sum(1 for r in both if r['sweep_idx'] == six[r['day']]['sweep_idx'])})")
    ifvg_conf = [r for r in conf if "ifvg" in r["conf_types"]]
    fresh = [r for r in ifvg_conf if r["conf_idx"] in r["ifvg_fresh_any"]]
    print(f"   ifvg on the confirmation bar: {len(ifvg_conf)}; resting on a FRESH inversion (no bar closed through "
          f"the gap between its completion and the sweep bar): {len(fresh)}; on already-inverted gaps only: "
          f"{len(ifvg_conf) - len(fresh)}")
    print(f"   sweep bar: " + counter_line(Counter(hhmm(int(fn.clock.minutes[r['sweep_idx']])) for r in swept),
                                             sorted({hhmm(int(fn.clock.minutes[r['sweep_idx']])) for r in swept})))
    print(f"   confirmation bar: " + counter_line(Counter(hhmm(int(fn.clock.minutes[r['conf_idx']])) for r in conf),
                                                    sorted({hhmm(int(fn.clock.minutes[r['conf_idx']])) for r in conf})))

    print()
    print("3. breakdowns")
    every = Counter(level_type(k) for r in swept for k in r["breached"])
    every_name = Counter(k for r in swept for k in r["breached"])
    deepest = Counter(level_type(r["level"]) for r in swept)
    deepest_name = Counter(r["level"] for r in swept)
    types = ("ASIA", "LON", "PD", "H1", "H4", "POC", "HVN")
    print("   sweeps by level type, every breached level counted:  " + counter_line(every, types)
          + f"   (levels per sweep bar: {sum(every.values()) / max(1, len(swept)):.2f})")
    print("   sweeps by level type, deepest level (the setup's):   " + counter_line(deepest, types))
    print("     by name, every breached:  " + counter_line(every_name))
    print("     by name, deepest:         " + counter_line(deepest_name))
    only_wide = [r for r in swept if not any(level_type(k) in ("ASIA", "LON", "PD") for k in r["breached"])]
    print(f"   sweep days where NO round-1 level was breached (the wide levels alone made the sweep): {len(only_wide)}")
    first = Counter(t for r in conf for t in r["conf_types"])
    combo = Counter("+".join(r["conf_types"]) for r in conf)
    anyby = Counter(t for r in swept for t in r["conf_any"])
    print("   confirmations by type, on the confirmation bar:  " + counter_line(first, CONF_TYPES)
          + "   combos: " + counter_line(combo))
    print("   confirmations by type, any bar by 10:05:        " + counter_line(anyby, CONF_TYPES))
    print("   confirmations, first-to-fire ONLY that type:    "
          + counter_line(Counter(r["conf_types"][0] for r in conf if len(r["conf_types"]) == 1), CONF_TYPES))
    zany = Counter(k for r in conf for k in zone_any(r))
    zused = Counter(zone_used(r).kind for r in zoned)
    print("   zones by type, any qualifying (in discount/premium): " + counter_line(zany, ZONE_TYPES))
    print("   zones by type, the one used (freshest at the end or at the fill): " + counter_line(zused, ZONE_TYPES))
    print("   fills by (level type, confirmation types, zone used):")
    fills = Counter((level_type(r["level"]), "+".join(r["conf_types"]), r["fill_zone"].kind) for r in filled)
    if fills:
        for (lt, ct, zk), cnt in sorted(fills.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"     {lt:<5} {ct:<13} {zk:<8} {cnt}")
    else:
        print("     (none)")
    print("   fills, per day: " + ("; ".join(
        f"{r['day'].strftime('%Y-%m-%d')} {'L' if r['side'] > 0 else 'S'} {r['level']} "
        f"{'+'.join(r['conf_types'])}@{hhmm(int(fn.clock.minutes[r['conf_idx']]))} "
        f"{r['fill_zone'].kind}@{hhmm(int(fn.clock.minutes[r['fill_idx']]))} {r['entry']:.2f}"
        for r in filled) or "(none)"))

    print()
    print("3b. sensitivity: ifvg counted only on a fresh inversion (everything else as above)")
    fc = funnel_counts(fn_fresh)
    fconf = [r for r in fn_fresh.rows if r["conf_idx"] is not None]
    ffill = [r for r in fconf if r["fill_idx"] is not None]
    print(f"   sessions {fc['sessions']} -> sweeps {fc['sweeps']} -> confirmations {fc['confirmations']} -> "
          f"zones in discount/premium {fc['zones']} -> fills {fc['fills']}")
    print("   confirmations by type, on the confirmation bar: "
          + counter_line(Counter(t for r in fconf for t in r["conf_types"]), CONF_TYPES)
          + "   combos: " + counter_line(Counter("+".join(r["conf_types"]) for r in fconf)))
    print("   confirmations by type, any bar by 10:05:       "
          + counter_line(Counter(t for r in fn_fresh.rows if r["sweep_idx"] is not None for t in r["conf_any"]), CONF_TYPES))
    print("   zones used: " + counter_line(Counter(zone_used(r).kind for r in fconf if r["zones"]), ZONE_TYPES)
          + "   fills by zone used: " + counter_line(Counter(r["fill_zone"].kind for r in ffill), ZONE_TYPES)
          + "   fills by level type: " + counter_line(Counter(level_type(r["level"]) for r in ffill), types))
    counts["fresh"] = fc

    print()
    print("4. side-by-side with round 1 (DESIGN-tjr-intraday.md section 9, base variant)")
    r1 = ROUND1.get(label)
    if r1 is None:
        print(f"   no round-1 row for {label!r}")
    else:
        wide = tuple(counts[s] for s in STAGES)
        print(f"   {'stage':<14} {'round 1':>8} {'wide':>6} {'delta':>7} {'x':>6}")
        ratios = {}
        for stage, a, w in zip(STAGES, r1, wide):
            ratio = w / a if a else math.inf
            ratios[stage] = ratio
            print(f"   {stage:<14} {a:>8} {w:>6} {w - a:>+7} {ratio:>6.2f}")
        # the stage that widened most: the largest multiple of its round-1 count
        top = max(ratios, key=lambda s: ratios[s])
        print(f"   widened most: {top} ({ratios[top]:.2f}x round 1)")
        conv = [(f"{s}->{t}", r1[i + 1] / r1[i] if r1[i] else math.nan, wide[i + 1] / wide[i] if wide[i] else math.nan)
                for i, (s, t) in enumerate(zip(STAGES, STAGES[1:]))]
        print("   stage-to-stage conversion, round 1 vs wide: "
              + "  ".join(f"{k} {a:.2f}/{b:.2f}" for k, a, b in conv))
        fw = tuple(fc[s] for s in STAGES)
        print("   with fresh-ifvg only:  " + "  ".join(f"{s} {w} ({w / a:.2f}x)" if a else f"{s} {w}"
                                                        for s, a, w in zip(STAGES, r1, fw)))
    print()
    return counts


def _levels_block(fn: WideFunnel) -> None:
    """Section 1 of the report: the levels at 09:25. Shared verbatim by both modes."""
    rows = fn.rows
    n = len(rows)
    print("1. levels at 09:25")
    print(f"   sessions with a sweep-window bar: {n}   (with a bar stamped exactly 09:30: {sum(r['has_0930'] for r in rows)})")
    present = Counter()
    for r in rows:
        for k in r["levels"]:
            present[k] += 1
        if r["n_hvn"]:
            present["HVN(any)"] += 1
    names = [k for k in LEVEL_ORDER] + ["HVN(any)"]
    print("   present: " + counter_line(present, names))
    print("   HVN count per day: " + counter_line(Counter(r["n_hvn"] for r in rows), range(0, fn.hvn_max + 1)))
    depth = Counter(r["depth"] for r in rows)
    print("   composite depth per day (sessions in the profile): "
          + counter_line(depth, sorted(depth, reverse=True)))
    print(f"   POC_PREV absent on: {[r['day'].strftime('%Y-%m-%d') for r in rows if 'POC_PREV' not in r['levels']] or 'no day'}"
          f"   (first session in the file has no previous session)")


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100.0 * a / b:.0f}%)" if b else f"{a}/{b}"


def _quantiles(xs: list[float]) -> str:
    if not xs:
        return "(none)"
    q = np.quantile(np.asarray(xs, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return f"min {q[0]:.2f}  q1 {q[1]:.2f}  median {q[2]:.2f}  q3 {q[3]:.2f}  max {q[4]:.2f}  (n={len(xs)})"


def report_decided(label: str, path: str, fn: WideFunnel, fn6: WideFunnel, r1log: pd.DataFrame,
                   fn_lit: WideFunnel, fn_fresh: WideFunnel) -> dict:
    """The decided-mode printout: the literal report's structure, plus the eq,
    ambiguity and target statistics of section 2.3, and a four-column
    side-by-side (round 1 / literal wide / literal fresh-ifvg / decided).
    `fn_lit` and `fn_fresh` are literal-mode runs of this same code on the same
    frame (they reproduce the committed printout); `fn6` is the literal run on
    round 1's six levels, which reproduces round 1's sweeps day for day."""
    rows, df, minutes = fn.rows, fn.df, fn.clock.minutes
    tm = lambda i: hhmm(int(minutes[i]))                                       # noqa: E731
    print(f"=== {label}  {path}  {len(df)} bars  {df.index[0]} .. {df.index[-1]} UTC ===")
    print()
    _levels_block(fn)
    n = len(rows)

    swept = [r for r in rows if r["sweep_idx"] is not None]
    conf = [r for r in swept if r["conf_idx"] is not None]
    zoned = [r for r in conf if r["zones"]]
    zoned_x = [r for r in conf if any(z.kind != "eq" for z in r["zones"])]
    filled = [r for r in zoned if r["fill_idx"] is not None]
    counts = {"sessions": n, "sweeps": len(swept), "confirmations": len(conf),
              "zones": len(zoned), "fills": len(filled)}
    print()
    print("2. funnel (section 2.3: fresh ifvg, eq as a retrace, per-class sweep on the directional set)")
    print(f"   sessions {n} -> sweeps {len(swept)} -> confirmations {len(conf)} -> "
          f"zones in discount/premium {len(zoned)} -> fills {len(filled)}")
    print(f"   zones excluding eq: {len(zoned_x)}   (eq is no longer in discount by construction: it needs the excursion first)")
    print(f"   sweep-window bars skipped as ambiguous (two classes returned opposite sides): "
          f"{sum(len(r['ambiguous']) for r in rows)} bars on {sum(1 for r in rows if r['ambiguous'])} days")
    print(f"   sweep-window bars on which a class abstained and no class returned (not a sweep, not ambiguous): "
          f"{sum(len(r['abstain_only']) for r in rows)} bars on {sum(1 for r in rows if r['abstain_only'])} days")
    print(f"   sweep days with no admissible opposing 5-minute swing (bos/ote cannot fire): "
          f"{sum(1 for r in swept if r['pre_swing'] is None)}")
    degenerate = [r for r in swept if r["pre_swing"] is not None
                  and r["side"] * (r["pre_swing"] - r["extreme"]) <= 0]
    print(f"   sweep days whose pre-sweep swing sits on the wrong side of the sweep extreme (degenerate leg): "
          f"{len(degenerate)}")
    print(f"   sweeps by side: long {sum(1 for r in swept if r['side'] > 0)}  short {sum(1 for r in swept if r['side'] < 0)}")
    n6, same = round1_crosscheck(fn6, r1log)
    print(f"   cross-check, literal code with round 1's six levels only: sweeps {n6}; sweep day, bar, side and level "
          f"identical to tjr_intraday.day_log: {'yes' if same else 'NO'}")
    six = {r["day"]: r for r in fn6.rows}
    lost = [r for r in rows if r["sweep_idx"] is None and six[r["day"]]["sweep_idx"] is not None]
    lost_dis = [r for r in lost if r["ambiguous"]]
    lost_abs = [r for r in lost if not r["ambiguous"] and r["abstain_only"]]
    gained = [r for r in rows if r["sweep_idx"] is not None and six[r["day"]]["sweep_idx"] is None]
    both = [r for r in swept if six[r["day"]]["sweep_idx"] is not None]
    print(f"   round-1 six-level sweep days lost under the decided rule: {len(lost)} (by class disagreement: {len(lost_dis)}; "
          f"abstention-only: {len(lost_abs)}; other: {len(lost) - len(lost_dis) - len(lost_abs)}); "
          f"days swept only under the decided rule: {len(gained)}; swept under both: {len(both)} "
          f"(same side {sum(1 for r in both if r['side'] == six[r['day']]['side'])}, "
          f"same bar {sum(1 for r in both if r['sweep_idx'] == six[r['day']]['sweep_idx'])}, "
          f"same level {sum(1 for r in both if r['level'] == six[r['day']]['level'])})")
    lit = {r["day"]: r for r in fn_lit.rows}
    lost_l = [r for r in rows if r["sweep_idx"] is None and lit[r["day"]]["sweep_idx"] is not None]
    gained_l = [r for r in rows if r["sweep_idx"] is not None and lit[r["day"]]["sweep_idx"] is None]
    both_l = [r for r in swept if lit[r["day"]]["sweep_idx"] is not None]
    print(f"   literal-wide sweep days lost under the decided rule: {len(lost_l)}; gained: {len(gained_l)}; both: {len(both_l)} "
          f"(same side {sum(1 for r in both_l if r['side'] == lit[r['day']]['side'])}, "
          f"same bar {sum(1 for r in both_l if r['sweep_idx'] == lit[r['day']]['sweep_idx'])}, "
          f"same level {sum(1 for r in both_l if r['level'] == lit[r['day']]['level'])})")
    ifvg_conf = [r for r in conf if "ifvg" in r["conf_types"]]
    print(f"   ifvg on the confirmation bar: {len(ifvg_conf)} (fresh inversions by definition; "
          f"literal wide counted {sum(1 for r in fn_lit.rows if r['conf_idx'] is not None and 'ifvg' in r['conf_types'])})")
    print(f"   sweep bar: " + counter_line(Counter(tm(r['sweep_idx']) for r in swept),
                                             sorted({tm(r['sweep_idx']) for r in swept})))
    print(f"   confirmation bar: " + counter_line(Counter(tm(r['conf_idx']) for r in conf),
                                                    sorted({tm(r['conf_idx']) for r in conf})))

    print()
    print("3. breakdowns")
    every = Counter(level_type(k) for r in swept for k in r["breached"])
    every_name = Counter(k for r in swept for k in r["breached"])
    deepest = Counter(level_type(r["level"]) for r in swept)
    deepest_name = Counter(r["level"] for r in swept)
    print("   sweeps by level class, every breached level of the sweep side counted:  " + counter_line(every, CLASSES)
          + f"   (levels per sweep bar: {sum(every.values()) / max(1, len(swept)):.2f})")
    print("   sweeps by level class, the setup's level (deepest among the returning classes): " + counter_line(deepest, CLASSES))
    print("     by name, every breached:  " + counter_line(every_name))
    print("     by name, the setup's:     " + counter_line(deepest_name))
    opp = Counter(k for r in swept for k in r["breached_opp"])
    print(f"   opposite-side levels the sweep wick also took (abstaining classes, or a high run without a close back): "
          f"{sum(opp.values())} on {sum(1 for r in swept if r['breached_opp'])} sweep bars   " + counter_line(opp))
    deeper = [r for r in swept if r["breached"] and r["breached"][0] != r["level"]]
    print(f"   sweep bars where a deeper level of the sweep side sat in an abstaining class (recorded, not the setup's): {len(deeper)}")
    only_wide = [r for r in swept if not any(level_type(k) in ("ASIA", "LON", "PD") for k in r["breached"])]
    print(f"   sweep days where NO round-1 level was breached (H1/H4 alone made the sweep): {len(only_wide)}")
    first = Counter(t for r in conf for t in r["conf_types"])
    combo = Counter("+".join(r["conf_types"]) for r in conf)
    anyby = Counter(t for r in swept for t in r["conf_any"])
    print("   confirmations by type, on the confirmation bar:  " + counter_line(first, CONF_TYPES)
          + "   combos: " + counter_line(combo))
    print("   confirmations by type, any bar by 10:05:        " + counter_line(anyby, CONF_TYPES))
    print("   confirmations, first-to-fire ONLY that type:    "
          + counter_line(Counter(r["conf_types"][0] for r in conf if len(r["conf_types"]) == 1), CONF_TYPES))
    zany = Counter(k for r in conf for k in zone_any(r))
    zused = Counter(zone_used(r).kind for r in zoned)
    print("   zones by type, any qualifying (in discount/premium, eq once usable): " + counter_line(zany, ZONE_TYPES))
    print("   zones by type, the one used (freshest at the end or at the fill): " + counter_line(zused, ZONE_TYPES))
    print("   confirmed days with no zone at all (no leg gap, no ob/breaker in discount, no eq excursion): "
          f"{len(conf) - len(zoned)}")
    print("   fills by (level class, confirmation types, zone used):")
    fills = Counter((level_type(r["level"]), "+".join(r["conf_types"]), r["fill_zone"].kind) for r in filled)
    if fills:
        for (lt, ct, zk), cnt in sorted(fills.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"     {lt:<5} {ct:<13} {zk:<8} {cnt}")
    else:
        print("     (none)")
    print("   fills, per day: " + ("; ".join(
        f"{r['day'].strftime('%Y-%m-%d')} {'L' if r['side'] > 0 else 'S'} {r['level']} "
        f"{'+'.join(r['conf_types'])}@{tm(r['conf_idx'])} "
        f"{r['fill_zone'].kind}@{tm(r['fill_idx'])} {r['entry']:.2f}"
        + (f" T1={r['targets']['t1'][0]}" if r["targets"].get("t1") else " T1=none")
        for r in filled) or "(none)"))

    # ---- 3b. eq --------------------------------------------------------------
    print()
    print("3b. eq zone (section 2.3 item 2): usable only from the first completed bar j, sweep bar <= j, on which "
          f"price was >= {EQ_EXCURSION_ATR} x ATR(14, this frame, at the confirmation bar) beyond the midpoint")
    eq_days = [r for r in conf if r["eq_usable"] is not None]
    at_conf = [r for r in eq_days if r["eq_usable"] == r["conf_idx"]]
    later = [r for r in eq_days if r["eq_usable"] > r["conf_idx"]]
    never = [r for r in conf if r["eq_usable"] is None]
    print(f"   confirmed days {len(conf)}: eq usable on {len(eq_days)} (at the confirmation bar {len(at_conf)}, "
          f"on a later bar {len(later)}); never by 10:05: {len(never)}")
    print("   wait, bars after confirmation (later days): " + counter_line(Counter(r["eq_usable"] - r["conf_idx"] for r in later))
          + "   excursion bar: " + counter_line(Counter(tm(r["eq_usable"]) for r in later), sorted({tm(r["eq_usable"]) for r in later})))
    widths = [(r["range"][1] - r["range"][0]) / r["eq_atr"] for r in conf if r["eq_atr"] > 0]
    print("   dealing range width at confirmation, in ATR: " + _quantiles(widths)
          + f"   >= 1.0 ATR (met at confirmation): {sum(1 for w in widths if w >= 2 * EQ_EXCURSION_ATR)}")
    eq_fills = [r for r in filled if r["fill_zone"].kind == "eq"]
    ef_conf = [r for r in eq_fills if r["eq_usable"] == r["conf_idx"]]
    ef_later = [r for r in eq_fills if r["eq_usable"] > r["conf_idx"]]
    print(f"   eq fills: {len(eq_fills)} of {len(filled)} fills; excursion met at confirmation on {len(ef_conf)}, "
          f"on a later bar on {len(ef_later)}"
          + (" (" + ", ".join(f"{r['day'].strftime('%m-%d')} j={tm(r['eq_usable'])} fill={tm(r['fill_idx'])}" for r in ef_later) + ")"
             if ef_later else ""))
    far = [r for r in eq_fills if r["eq_open_far"]]
    print(f"   eq fills whose touching bar OPENED on the far side of EQ (a retrace through the bar): {len(far)}; "
          f"opened at or across EQ already (touched at the open): {len(eq_fills) - len(far)}")
    print("   co-occurrence at eq fills, other zone types in discount/premium at the fill bar: "
          + counter_line(Counter("+".join(r["cooccur"]) or "none" for r in eq_fills))
          + "   by type: " + counter_line(Counter(k for r in eq_fills for k in r["cooccur"]), ("fvg", "ob", "breaker")))
    eq_only = [r for r in zoned if all(z.kind == "eq" for z in r["zones"])]
    print(f"   zone-stage days whose only zone is eq: {len(eq_only)}; fills on those days: "
          f"{sum(1 for r in eq_only if r['fill_idx'] is not None)}")
    counts["eq"] = {"confirmed": len(conf), "eq_usable": len(eq_days), "at_confirmation": len(at_conf),
                    "later": len(later), "never": len(never),
                    "wait_bars": dict(sorted(Counter(r["eq_usable"] - r["conf_idx"] for r in later).items())),
                    "range_width_atr_median": float(np.median(widths)) if widths else None,
                    "eq_fills": len(eq_fills), "eq_fills_excursion_at_confirmation": len(ef_conf),
                    "eq_fills_excursion_later": len(ef_later),
                    "eq_fills_open_far_side": len(far), "eq_fills_open_already_crossed": len(eq_fills) - len(far),
                    "cooccurrence": dict(Counter("+".join(r["cooccur"]) or "none" for r in eq_fills)),
                    "zoned_days_eq_only": len(eq_only)}

    # ---- 3c. ambiguity ---------------------------------------------------------
    print()
    print("3c. ambiguity (section 2.3 item 3): the sweep judged per class on ASIA / LON / PD / H1 / H4; POC and HVN never swept")
    amb_bars = [(r, i) for r in rows for i in r["ambiguous"]]
    abs_bars = [(r, i) for r in rows for i in r["abstain_only"]]
    print(f"   class-disagreement bars: {len(amb_bars)} on {sum(1 for r in rows if r['ambiguous'])} days; "
          f"abstention-only bars: {len(abs_bars)} on {sum(1 for r in rows if r['abstain_only'])} days; "
          f"literal wide skipped {sum(len(r['ambiguous']) for r in fn_lit.rows)} bars on {sum(1 for r in fn_lit.rows if r['ambiguous'])} days")
    print("   disagreement patterns (class:side; abstaining classes noted): "
          + counter_line(Counter(r["amb_detail"][i] for r, i in amb_bars)))
    print(f"   days whose FIRST sweep-window verdict was ambiguous but a later bar swept: "
          f"{sum(1 for r in swept if r['ambiguous'] and min(r['ambiguous']) < r['sweep_idx'])}; "
          f"sweep days with an abstention-only bar before the sweep bar: "
          f"{sum(1 for r in swept if r['abstain_only'] and min(r['abstain_only']) < r['sweep_idx'])}")
    print("   classes returning on the sweep bar, count per sweep: "
          + counter_line(Counter(len(r["cls_ret"]) for r in swept), sorted({len(r["cls_ret"]) for r in swept}))
          + "   by class: " + counter_line(Counter(cls for r in swept for cls in r["cls_ret"]), CLASSES))
    print("   classes abstaining on the sweep bar: "
          + counter_line(Counter(cls for r in swept for cls in r["cls_abs"]), CLASSES)
          + f"   sweep bars with any abstention: {sum(1 for r in swept if r['cls_abs'])}")
    print(f"   round-1 six-level sweep days lost: {len(lost)} of {n6} (by class disagreement {len(lost_dis)}, "
          f"abstention-only {len(lost_abs)}, other {len(lost) - len(lost_dis) - len(lost_abs)}); "
          f"literal wide lost {sum(1 for r in fn_lit.rows if r['sweep_idx'] is None and six[r['day']]['sweep_idx'] is not None)}")
    lost_days = ", ".join(f"{r['day'].strftime('%m-%d')}[{';'.join(r['amb_detail'][i] for i in r['ambiguous'])}]" for r in lost_dis)
    print(f"   lost by class disagreement, per day: {lost_days or '(none)'}")
    counts["ambiguity"] = {"disagreement_bars": len(amb_bars), "disagreement_days": sum(1 for r in rows if r["ambiguous"]),
                           "abstention_only_bars": len(abs_bars), "abstention_only_days": sum(1 for r in rows if r["abstain_only"]),
                           "round1_sweep_days": n6, "round1_days_lost": len(lost),
                           "round1_days_lost_by_class_disagreement": len(lost_dis),
                           "round1_days_lost_by_abstention_only": len(lost_abs),
                           "round1_days_lost_other": len(lost) - len(lost_dis) - len(lost_abs),
                           "days_gained_vs_round1": len(gained), "swept_under_both": len(both),
                           "literal_wide_days_lost": len(lost_l), "literal_wide_days_gained": len(gained_l),
                           "disagreement_patterns": dict(Counter(r["amb_detail"][i] for r, i in amb_bars)),
                           "sweep_bars_with_abstention": sum(1 for r in swept if r["cls_abs"]),
                           "deeper_level_in_abstaining_class": len(deeper)}

    # ---- 3d. targets -----------------------------------------------------------
    print()
    print("3d. targets, logged only: the nearest level beyond the entry (long: above; short: below), T2 the one beyond T1")
    t1 = [r["targets"]["t1"] for r in filled]
    t2 = [r["targets"]["t2"] for r in filled]
    t1d = [r["targets"]["t1_dir"] for r in filled]
    t1v = [r["targets"]["t1_vol"] for r in filled]
    types7 = CLASSES + ("POC", "HVN")
    print("   T1 (all levels) by type over fills:  " + counter_line(Counter(level_type(t[0]) for t in t1 if t), types7)
          + f"   no T1: {sum(1 for t in t1 if t is None)}")
    print("   T1 by name:                          " + counter_line(Counter(t[0] for t in t1 if t)))
    print("   T2 (all levels) by type over fills:  " + counter_line(Counter(level_type(t[0]) for t in t2 if t), types7)
          + f"   no T2: {sum(1 for t in t2 if t is None)}")
    print("   nearest directional level by type:   " + counter_line(Counter(level_type(t[0]) for t in t1d if t), CLASSES)
          + f"   none: {sum(1 for t in t1d if t is None)}")
    print("   nearest POC/HVN level by type:       " + counter_line(Counter(level_type(t[0]) for t in t1v if t), ("POC", "HVN"))
          + f"   none: {sum(1 for t in t1v if t is None)}")
    dist = lambda r, t: abs(t[1] - r["entry"]) / r["eq_atr"]                    # noqa: E731
    print("   T1 distance from entry, in ATR: " + _quantiles([dist(r, r["targets"]["t1"]) for r in filled if r["targets"].get("t1")]))
    print("   T2 distance from entry, in ATR: " + _quantiles([dist(r, r["targets"]["t2"]) for r in filled if r["targets"].get("t2")]))
    print("   nearest directional level, in ATR: " + _quantiles([dist(r, r["targets"]["t1_dir"]) for r in filled if r["targets"].get("t1_dir")]))
    same_name = [r for r in filled if r["targets"].get("t1")
                 and ((r["side"] > 0 and r["targets"]["t1"][0] in DIRECTIONAL_LOWS)
                      or (r["side"] < 0 and r["targets"]["t1"][0] in DIRECTIONAL_HIGHS))]
    print(f"   fills whose T1 is a same-side-named directional level (a *_L above a long's entry, or mirror): {len(same_name)}"
          + (" (" + ", ".join(f"{r['day'].strftime('%m-%d')} {r['targets']['t1'][0]}" for r in same_name) + ")" if same_name else ""))
    t1_swept = [r for r in filled if r["targets"].get("t1") and r["targets"]["t1"][0] == r["level"]]
    print(f"   fills whose T1 is the swept level itself (the entry sits beyond the level the sweep took): {len(t1_swept)}; "
          f"T1 is any level the sweep wick breached: "
          f"{sum(1 for r in filled if r['targets'].get('t1') and r['targets']['t1'][0] in r['breached'])}")
    counts["targets"] = {"fills": len(filled),
                         "t1_type": dict(Counter(level_type(t[0]) for t in t1 if t)),
                         "t1_name": dict(Counter(t[0] for t in t1 if t)),
                         "t2_type": dict(Counter(level_type(t[0]) for t in t2 if t)),
                         "t1_dir_type": dict(Counter(level_type(t[0]) for t in t1d if t)),
                         "t1_vol_type": dict(Counter(level_type(t[0]) for t in t1v if t)),
                         "no_t1": sum(1 for t in t1 if t is None), "no_t2": sum(1 for t in t2 if t is None),
                         "t1_dist_atr_median": float(np.median([dist(r, r["targets"]["t1"]) for r in filled if r["targets"].get("t1")])) if t1 and any(t1) else None,
                         "t1_same_side_named": len(same_name), "t1_is_swept_level": len(t1_swept)}

    # ---- 4. side-by-side -------------------------------------------------------
    print()
    print("4. side-by-side: round 1 (DESIGN-tjr-intraday.md section 9, base) / literal wide / literal fresh-ifvg / decided")
    r1 = ROUND1.get(label)
    lc, fc = funnel_counts(fn_lit), funnel_counts(fn_fresh)
    counts["literal"], counts["fresh"] = lc, fc
    if r1 is None:
        print(f"   no round-1 row for {label!r}")
    else:
        dec = tuple(counts[s] for s in STAGES)
        print(f"   {'stage':<14} {'round 1':>8} {'literal':>8} {'fresh':>8} {'decided':>8} {'dec/r1':>8} {'dec/lit':>8} {'dec/fresh':>10}")
        for stage, a, w in zip(STAGES, r1, dec):
            lw, fw = lc[stage], fc[stage]
            fmt = lambda x, y: f"{x / y:.2f}" if y else "inf"                    # noqa: E731
            print(f"   {stage:<14} {a:>8} {lw:>8} {fw:>8} {w:>8} {fmt(w, a):>8} {fmt(w, lw):>8} {fmt(w, fw):>10}")
        conv = lambda seq: "  ".join(f"{s}->{t} {(seq[i + 1] / seq[i] if seq[i] else math.nan):.2f}"   # noqa: E731
                                     for i, (s, t) in enumerate(zip(STAGES, STAGES[1:])))
        print("   stage-to-stage conversion, round 1:       " + conv(r1))
        print("   stage-to-stage conversion, literal wide:  " + conv(tuple(lc[s] for s in STAGES)))
        print("   stage-to-stage conversion, literal fresh: " + conv(tuple(fc[s] for s in STAGES)))
        print("   stage-to-stage conversion, decided:       " + conv(dec))
    print()
    return counts


def print_definitions_as_built() -> None:
    """Every place section 2.3 is silent, and the reading this build took. The
    orchestrator writes section 2.4; this block is the source for it."""
    print("definitions as built (decided mode) -- where section 2.3 is silent, the choice made here")
    for line in (
        "ifvg: `ifvg_fresh` is the definition; a gap is stale if any bar from its completion+1 through the sweep bar",
        "      (inclusive) closed through it; the inverting close is on a bar strictly after the sweep bar. No 3b section.",
        "eq/ATR: ATR(14) is primitives.Bars.from_frame(df, 14).atr (Wilder ewm on the 5-minute frame, causal, back-filled",
        "      over the warmup), read at the confirmation bar once and frozen; the threshold EQ +/- 0.5 x ATR never moves.",
        "eq/excursion: tested on completed bars' high (long) / low (short), >= (inclusive). Through the confirmation bar",
        "      the range extreme is the test (j = conf_idx); after it, every bar conf_idx+1 .. the last bar stamped < 10:10",
        "      is tested, whether or not it is an entry-window bar. The zone is not in the zone list before the excursion",
        "      bar, so another zone type (fvg/ob/breaker) may fill first on those bars; from the excursion bar on it is",
        "      in the list with formed = conf_idx, usable = j, and the same rank rule as before (formed, then ob<eq<breaker<fvg).",
        "eq/fill: the touching bar must be strictly after usable (i > j), as for every zone. `open on the far side` is",
        "      open > EQ (long) / open < EQ (short); `already crossed` is open <= EQ (long), including open == EQ.",
        "eq/co-occurrence: the other zone kinds in the zone list at the fill bar (all in discount/premium by the add rule),",
        "      including one that only became usable on the fill bar itself.",
        "sweep/classes: the five classes are by level_type; a class with no level present on the day returns None and is",
        "      neither returning nor abstaining. Abstention is `a low AND a high of the class breached` regardless of close.",
        "      A class whose level was breached but not closed back inside returns None and is not an abstention.",
        "sweep/level: the setup's level is the deepest breached level of the sweep side among the RETURNING classes only;",
        "      a deeper level of the same side inside an abstaining class is recorded in `breached` (deepest first) but is",
        "      not the setup's level. `breached` = every directional level of the sweep side the wick took, in any class;",
        "      `wick_also_took` = every directional level of the other side the wick took. POC/HVN are never in either.",
        "sweep/first bar: the first sweep-window bar (09:30..09:45) with a one-sided verdict fixes the day; disagreement and",
        "      abstention-only bars are skipped and the next bar may sweep. `lost by class disagreement` = a round-1 sweep",
        "      day with no decided sweep and at least one disagreement bar; `abstention-only` = none, but an abstention bar.",
        "targets: `opposing` is by price, as the task states (a long's T1 is the nearest level strictly above the entry,",
        "      any name, POC/HVN included); the count of T1s that are a same-side-named level is reported. Ties in distance",
        "      keep the level order ASIA_H, ASIA_L, LON_H, LON_L, PDH, PDL, H1_*, H4_*, POC_PREV, HVN_1..6 (stable sort).",
        "      Distances are in the confirmation-bar ATR. Targets are logged; no exit, no outcome, no R.",
        "side-by-side: the literal and literal-fresh columns are literal-mode runs of this script in the same process,",
        "      which reproduce results/tjr_intraday/part2_wide_funnel.txt byte for byte (--mode literal).",
        "proof: decided mode adds a seventh cut at the excursion bar of a day whose eq zone became usable after the",
        "      confirmation bar (an eq-filled such day when one exists), so the zone and any fill on it must vanish.",
    ):
        print("   " + line)
    print()


def write_days_csv(fn: WideFunnel, path: str) -> None:
    frame = pd.DataFrame([format_row(r, fn.clock.minutes, fn.decided) for r in fn.rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


# ────────────────────────────── the causality proof ──────────────────────────────

def pick_cuts(fn: WideFunnel) -> list[tuple[int, str]]:
    """Six cut points: two inside the sweep window, two inside the entry
    window, two mid-afternoon, on days chosen so each cut forgets something
    the full run knew (a sweep, a confirmation, a fill)."""
    rows, minutes, day = fn.rows, fn.clock.minutes, fn.clock.day

    def bar_at(r, hm):
        a, e = r["day_start"], next(e for d, a2, e in fn.sessions if a2 == r["day_start"])
        hits = [i for i in range(a, e) if minutes[i] == hm]
        return hits[0] if hits else None

    cuts, used = [], set()
    want = 7 if fn.decided else 6

    def take(cands, hm, what, need=1):
        got = 0
        for r in cands:
            if r["day"] in used or got >= need:
                continue
            i = bar_at(r, hm)
            if i is None:
                continue
            used.add(r["day"])
            cuts.append((i, f"{r['day'].strftime('%Y-%m-%d')} cut before the {hhmm(hm)} bar ({what})"))
            got += 1

    if fn.decided:
        # Section 2.3 item 2: a day whose eq zone became usable AFTER the
        # confirmation bar, cut at the excursion bar itself, so the truncated
        # frame ends before it and the zone (and any fill on it) must vanish.
        # An eq-filled such day first, when one exists.
        eq_later = [r for r in rows if r["eq_usable"] is not None and r["eq_usable"] > r["conf_idx"]]
        eq_later.sort(key=lambda r: (not (r["fill_idx"] is not None and r["fill_zone"].kind == "eq"), r["day"]))
        for r in eq_later[:1]:
            take([r], int(minutes[r["eq_usable"]]),
                 "before the eq excursion bar: the eq zone became usable after the confirmation bar and must vanish")
        if not eq_later:
            print("   no day on this frame has an eq zone that became usable after the confirmation bar "
                  "(section 3b: later 0), so no excursion-bar cut exists; the seventh cut is a top-up")

    late_sweep = [r for r in rows if r["sweep_idx"] is not None and minutes[r["sweep_idx"]] >= 9 * 60 + 40]
    early_sweep_conf = [r for r in rows if r["conf_idx"] is not None and minutes[r["sweep_idx"]] == 9 * 60 + 30
                        and minutes[r["conf_idx"]] >= 9 * 60 + 45]
    filled = [r for r in rows if r["fill_idx"] is not None]
    confirmed = [r for r in rows if r["conf_idx"] is not None and r["fill_idx"] is None]
    unswept = [r for r in rows if r["sweep_idx"] is None]
    take(late_sweep, 9 * 60 + 40, "inside the sweep window: the sweep is not yet in the frame")
    take(early_sweep_conf, 9 * 60 + 45, "inside the sweep window: sweep in, confirmation not yet")
    if filled:
        take([filled[len(filled) // 2]], int(minutes[filled[len(filled) // 2]["fill_idx"]]),
             "inside the entry window: cut at the fill bar, the fill must vanish")
    take(confirmed, 9 * 60 + 55, "inside the entry window: confirmation in, zones may still be forming")
    take(filled[::-1], 13 * 60, "mid-afternoon on a filled day")
    take(unswept, 13 * 60, "mid-afternoon on a day with no sweep")
    # top up with generic cuts if a category had no candidate
    for r in rows[::7]:
        if len(cuts) >= want:
            break
        take([r], 10 * 60, "inside the entry window (top-up)")
    return cuts[:want]


def prove_causality(label: str, fn: WideFunnel) -> bool:
    print(f"5. causality proof: {label}")
    full = [format_row(r, fn.clock.minutes, fn.decided) for r in fn.rows]
    all_ok = True
    cuts = pick_cuts(fn)
    n_cuts = len(cuts)
    for cut, what in cuts:
        part = WideFunnel(fn.df.iloc[:cut], fn.bin_ticks, fn.hvn_frac, fn.hvn_max, mode=fn.mode)
        prows = [format_row(r, part.clock.minutes, fn.decided) for r in part.rows]
        cut_day = fn.clock.day[cut]
        k = next(j for j, r in enumerate(fn.rows) if r["day"] == pd.Timestamp(cut_day))
        earlier_ok = prows[:k] == full[:k] and len(prows) == k + 1
        expect = format_row(as_of(fn.rows[k], cut), fn.clock.minutes, fn.decided)
        got = prows[k] if len(prows) > k else None
        cut_ok = got == expect
        all_ok &= earlier_ok and cut_ok
        diff = "" if cut_ok else "  DIFF: " + "; ".join(f"{c}: {got.get(c) if got else None!r} != {expect[c]!r}"
                                                         for c in expect if got is None or got.get(c) != expect[c])
        print(f"   cut at bar {cut} ({fn.df.index[cut]} UTC): {what}")
        print(f"     earlier days: {k} rows identical: {'yes' if earlier_ok else 'NO'}   "
              f"cut day as-of the cut identical: {'yes' if cut_ok else 'NO'}{diff}")
        print(f"     cut day full-run outcome {full[k]['outcome']!r} -> truncated {got['outcome'] if got else None!r}"
              f"  (sweep {full[k]['sweep_time'] or '-'} conf {full[k]['conf_time'] or '-'} fill {full[k]['fill_time'] or '-'})")
    words = {6: 'six', 7: 'seven'}.get(n_cuts, str(n_cuts))
    print(f"   result: {f'all {words} cuts reproduce the full run' if all_ok else 'MISMATCH'}")
    print()
    return all_ok


# ────────────────────────────── main ──────────────────────────────

class _Tee:
    """stdout and a file at once, for the decided-mode printout."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", nargs="+", default=list(DEFAULT_CSVS))
    ap.add_argument("--mode", choices=MODES, default="decided",
                    help="decided: DESIGN-tjr-human.md section 2.3 (the default); literal: section 2.1 as built, "
                         "which reproduces results/tjr_intraday/part2_wide_funnel.txt byte for byte")
    ap.add_argument("--poc-bin-ticks", type=int, default=4)
    ap.add_argument("--hvn-frac", type=float, default=0.5)
    ap.add_argument("--hvn-max", type=int, default=6)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-proof", action="store_true", help="skip the truncation proof")
    args = ap.parse_args(argv)
    if args.mode == "literal":
        return run_literal(args)
    # Decided mode: the printout goes to stdout AND to <out-dir>/part2_decided_funnel.txt.
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "part2_decided_funnel.txt")
    real = sys.stdout
    with open(report_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(real, fh)
        try:
            rc = run_decided(args)
        finally:
            sys.stdout = real
    print(f"printout written: {report_path}")
    return rc


def run_literal(args) -> int:
    """Section 2.1 as built. Byte for byte the committed printout."""
    print("TJR wide funnel (DESIGN-tjr-human.md section 2). Funnel only: no stop, target, exit, Sharpe, gate or verdict.")
    print(f"params: poc_bin_ticks={args.poc_bin_ticks} (bin ${TICK * args.poc_bin_ticks:.2f})  hvn_frac={args.hvn_frac}  "
          f"hvn_max={args.hvn_max}  composite={COMPOSITE_SESSIONS} sessions  swings left/right={SWING_LEFT}/{SWING_RIGHT}  ote={OTE}")
    print()
    summary = {}
    proofs = {}
    funnels = []
    for path in args.csv:
        label = os.path.basename(path).split("_")[0].upper()
        df = data.load_csv(path)
        fn = WideFunnel(df, args.poc_bin_ticks, args.hvn_frac, args.hvn_max)
        if fn.clock is None:
            print(f"=== {label}  {path}: no 09:30 bar in the frame, nothing to do ===")
            continue
        fn6 = WideFunnel(df, args.poc_bin_ticks, args.hvn_frac, args.hvn_max, level_types=("ASIA", "LON", "PD"))
        fn_fresh = WideFunnel(df, args.poc_bin_ticks, args.hvn_frac, args.hvn_max, ifvg_fresh=True)
        summary[label] = report(label, path, fn, fn6, M.day_log(df), fn_fresh)
        out = os.path.join(args.out_dir, f"part2_funnel_days_{label}.csv")
        write_days_csv(fn, out)
        print(f"   day-level rows written: {out} ({len(fn.rows)} rows)")
        print()
        funnels.append((label, fn))
    if not args.no_proof:
        for label, fn in funnels:
            proofs[label] = prove_causality(label, fn)

    print("6. synthetic daily data (quantlab.data.synthetic): the no-09:30 path")
    syn = WideFunnel(data.synthetic(), args.poc_bin_ticks, args.hvn_frac, args.hvn_max)
    print(f"   clock is None: {syn.clock is None}; rows: {len(syn.rows)} -> all-flat, no exception")
    print()
    print("summary")
    for label, c in summary.items():
        print(f"   {label}: sessions {c['sessions']} -> sweeps {c['sweeps']} -> confirmations {c['confirmations']} "
              f"-> zones {c['zones']} -> fills {c['fills']}"
              + (f"   causality: {'ok' if proofs[label] else 'MISMATCH'}" if label in proofs else ""))
        f = c["fresh"]
        print(f"   {label} (fresh-ifvg sensitivity): sessions {f['sessions']} -> sweeps {f['sweeps']} -> "
              f"confirmations {f['confirmations']} -> zones {f['zones']} -> fills {f['fills']}")
    return 0 if all(proofs.values()) else 1


def run_decided(args) -> int:
    """Section 2.3: the one re-run of the wide funnel under the three decisions."""
    print("TJR wide funnel, DECIDED (DESIGN-tjr-human.md section 2.3: fresh ifvg; eq only as a retrace; "
          "ambiguity within a level class, POC/HVN targets only). Funnel only: no stop, target, exit, Sharpe, gate or verdict.")
    print(f"params: poc_bin_ticks={args.poc_bin_ticks} (bin ${TICK * args.poc_bin_ticks:.2f})  hvn_frac={args.hvn_frac}  "
          f"hvn_max={args.hvn_max}  composite={COMPOSITE_SESSIONS} sessions  swings left/right={SWING_LEFT}/{SWING_RIGHT}  "
          f"ote={OTE}  eq excursion={EQ_EXCURSION_ATR} x ATR(14)")
    print()
    summary = {}
    proofs = {}
    funnels = []
    for path in args.csv:
        label = os.path.basename(path).split("_")[0].upper()
        df = data.load_csv(path)
        fn = WideFunnel(df, args.poc_bin_ticks, args.hvn_frac, args.hvn_max, mode="decided")
        if fn.clock is None:
            print(f"=== {label}  {path}: no 09:30 bar in the frame, nothing to do ===")
            continue
        kw = dict(bin_ticks=args.poc_bin_ticks, hvn_frac=args.hvn_frac, hvn_max=args.hvn_max, mode="literal")
        fn6 = WideFunnel(df, level_types=("ASIA", "LON", "PD"), **kw)
        fn_lit = WideFunnel(df, **kw)
        fn_fresh = WideFunnel(df, ifvg_fresh=True, **kw)
        summary[label] = report_decided(label, path, fn, fn6, M.day_log(df), fn_lit, fn_fresh)
        out = os.path.join(args.out_dir, f"part2_decided_days_{label}.csv")
        write_days_csv(fn, out)
        print(f"   day-level rows written: {out} ({len(fn.rows)} rows)")
        print()
        funnels.append((label, fn))
    if not args.no_proof:
        for label, fn in funnels:
            proofs[label] = prove_causality(label, fn)

    print("6. synthetic daily data (quantlab.data.synthetic): the no-09:30 path")
    syn = WideFunnel(data.synthetic(), args.poc_bin_ticks, args.hvn_frac, args.hvn_max, mode="decided")
    print(f"   clock is None: {syn.clock is None}; rows: {len(syn.rows)} -> all-flat, no exception")
    print()
    print_definitions_as_built()
    print("summary")
    for label, c in summary.items():
        r1 = ROUND1.get(label, (None,) * 4)
        r1c = dict(sessions=c["sessions"], sweeps=r1[0], confirmations=r1[1], zones=r1[2], fills=r1[3])
        for name, cc in (("round 1", r1c), ("literal wide", c["literal"]), ("literal fresh-ifvg", c["fresh"]), ("decided", c)):
            print(f"   {label} ({name}): sessions {cc['sessions']} -> sweeps {cc['sweeps']} -> confirmations {cc['confirmations']} "
                  f"-> zones {cc['zones']} -> fills {cc['fills']}"
                  + (f"   causality: {'ok' if proofs[label] else 'MISMATCH'}" if name == "decided" and label in proofs else ""))
        e, a = c["eq"], c["ambiguity"]
        print(f"   {label} eq: usable on {e['eq_usable']} of {e['confirmed']} confirmed days (at confirmation {e['at_confirmation']}, "
              f"later {e['later']}, never {e['never']}); eq fills {e['eq_fills']} (opened far side {e['eq_fills_open_far_side']})")
        print(f"   {label} ambiguity: disagreement bars {a['disagreement_bars']} on {a['disagreement_days']} days; "
              f"abstention-only bars {a['abstention_only_bars']} on {a['abstention_only_days']} days; "
              f"round-1 sweep days lost {a['round1_days_lost']} (class disagreement {a['round1_days_lost_by_class_disagreement']})")
    return 0 if all(proofs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

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

Three modes, one code path; the two older ones reproduce their committed
printouts and day CSVs byte for byte (only the lines echoing --out-dir move):

    python tjr_wide_funnel.py                  # --mode final, the default (section 2.5): tees to
                                               #   <out-dir>/part2_final_funnel.txt, writes part2_final_days_{NQ,ES}.csv
    python tjr_wide_funnel.py --mode decided   # section 2.3 / 2.4: tees to <out-dir>/part2_decided_funnel.txt,
                                               #   writes part2_decided_days_{NQ,ES}.csv
    python tjr_wide_funnel.py --mode literal > results/tjr_intraday/part2_wide_funnel.txt
                                               # section 2.1 as built: prints to stdout only (no tee),
                                               #   writes part2_funnel_days_{NQ,ES}.csv
    python tjr_wide_funnel.py --mode decided --out-dir /tmp/check   # re-run an old mode without touching results/
    python tjr_wide_funnel.py --csv data/NQ_5min_tv.csv --hvn-frac 0.6 --no-proof

Causality
---------
Every level of day D is frozen at the 09:25 bar's close from bars <= that bar:
the six levels by session_levels' own argument, the 1h/4h swings by
resample_context's completion rule plus confirmed_at, POC_PREV from the previous
session's bars, the HVNs from the twenty completed sessions before D. Every
confirmation and zone at bar i reads bars <= i; the final mode's open-side
test reads the open of the completed bar it is judging, the zone it fills on
bar i is chosen from the zones known before bar i (the two older modes keep
round 1's convention, under which a zone completed by bar i withholds a fill
on bar i: intrabar lookahead that truncation cannot detect), and its targets and
wick stop are arithmetic on the day's frozen levels, the entry, the sweep
extreme and the confirmation bar's ATR. The proof section at the end truncates
each frame at mid-session bars (six in literal mode, seven in decided, up to
ten in final: the extra ones sit at the eq excursion bar, at a bar the
open-side test refused, at the later bar that then filled, and at the bar
after a fill bar that itself completed a fresher zone) and shows that
the day-level rows for every earlier day, and the cut day's row up to the cut,
do not move.
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
#: The three level classes of section 2.5 item 2 (final mode): round 1's six
#: session levels are ONE class again, the 1-hour and 4-hour swings two more.
CLASSES_FINAL = ("SESSION", "H1", "H4")
SESSION_TYPES = ("ASIA", "LON", "PD")
MODES = ("final", "decided", "literal")
EQ_EXCURSION_ATR = 0.5           # section 2.3 item 2: the retrace the eq touch must follow
WICK_STOP_ATR = 0.25             # round 1's wick stop buffer (tjr_intraday DEFAULTS["stop_buffer_atr"]); geometry only
#: Section 2.5 item 3: what a long may target (shorts: TARGET_LOWS), plus POC_PREV and HVN_* on either side.
TARGET_HIGHS = DIRECTIONAL_HIGHS
TARGET_LOWS = DIRECTIONAL_LOWS


def level_type(name: str) -> str:
    """ASIA_L -> ASIA, PDH -> PD, H1_SH -> H1, POC_PREV -> POC, HVN_3 -> HVN."""
    return "PD" if name in ("PDH", "PDL") else name.split("_")[0]


def level_class_final(name: str) -> str:
    """Section 2.5 item 2: ASIA_*, LON_*, PDH, PDL -> SESSION; H1_* -> H1; H4_* -> H4."""
    t = level_type(name)
    return "SESSION" if t in SESSION_TYPES else t


def level_sort_key(name: str) -> tuple[int, int]:
    """LEVEL_ORDER, then HVN_1..6: the order names are joined in when two levels share a price."""
    if name in LEVEL_ORDER:
        return LEVEL_ORDER.index(name), 0
    return len(LEVEL_ORDER), int(name.split("_")[1])


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
                 ifvg_fresh: bool = False, mode: str = "literal", eq_open_side: bool | None = None,
                 zone_same_bar: bool | None = None):
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
        decided mode fixes both.

        "final" (section 2.5) is decided mode with three changes and nothing
        else: a touch of the eq zone counts only on a bar that OPENS on the far
        side of the midpoint; the level classes are SESSION / H1 / H4 (round 1's
        six levels one class again); and the logged targets are the nearest
        opposing-NAMED level or POC/HVN beyond the entry, with the wick-stop
        geometry next to them. `self.decided` is true in both, so every
        section-2.3 rule the corrections did not touch runs the same code."""
        if mode not in MODES:
            raise ValueError(f"WideFunnel: mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.final = mode == "final"
        self.decided = mode in ("decided", "final")
        self.classes = CLASSES_FINAL if self.final else CLASSES
        self.class_of = level_class_final if self.final else level_type
        # Attribution knob for the final printout only: final mode with the
        # open-side test switched off isolates what the class change did alone.
        # None -> the mode's own rule (on in final, off elsewhere).
        self.eq_open_side = self.final if eq_open_side is None else bool(eq_open_side) and self.final
        # Which zone is in play on bar i. Round 1's convention (literal, decided;
        # tjr_intraday "fresh zones first, then the fill"): a zone bar i itself
        # completes is already the freshest on bar i, and since a fill needs
        # i > usable, bar i then fills nothing. That uses bar i's CLOSE to
        # withhold a fill that a resting order would have taken INSIDE bar i.
        # Final mode reads the zone in play from the zones known before bar i
        # (usable < i); a zone completed on bar i counts from bar i + 1.
        # True = round 1's convention; it is a diagnostic knob in final mode
        # only and cannot be switched off in the two older modes.
        self.zone_same_bar = (not self.final) if zone_same_bar is None else (bool(zone_same_bar) or not self.final)
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
        the class's own below/above). The classes are `self.classes`: the five
        of section 2.3 in decided mode, SESSION / H1 / H4 in final mode (section
        2.5 item 2), where SESSION holding all six session levels makes a wick
        through any session low and any session high round 1's ambiguity again.
        Among the classes that return a sweep:
        all one side -> that side, and the setup's level is the deepest
        breached level of that side across those classes; two sides -> the bar
        is ambiguous (class disagreement) and skipped. No class returning is
        not a sweep; if some class abstained it is counted as abstention-only,
        separately from ambiguity. `row` receives the bar's verdict.
        """
        h, l, c = self.h, self.l, self.c
        returns: dict[str, tuple[int, str, float]] = {}
        abstained: list[str] = []
        state: dict[str, str] = {}
        for cls in self.classes:
            lc = {k: p for k, p in lows.items() if self.class_of(k) == cls}
            hc = {k: p for k, p in highs.items() if self.class_of(k) == cls}
            r = M.sweep_of_levels(h, l, c, i, lc, hc)
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
        cand = {k: p for k, p in taken.items() if self.class_of(k) in returns}
        name = min(cand, key=cand.get) if side > 0 else max(cand, key=cand.get)
        row["cls_ret"] = {cls: r[1] for cls, r in returns.items()}
        row["cls_abs"] = list(abstained)
        row["cls_state"] = state
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
               "breached_opp": [], "eq_atr": math.nan, "eq_usable": None, "fill_bar_new_zone": None,
               "eq_open_far": None, "cooccur": [], "targets": {},
               # final mode only (section 2.5); untouched in the other two
               "cls_state": {}, "eq_refused": [], "geom": {}}

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
            if self.zone_same_bar:
                zone = freshest(zones)
                if zone is None or i <= max(conf_idx, zone.usable):
                    continue
            else:
                # Final mode: only zones known before bar i can be in play on it
                # (i > conf_idx always holds here, and usable >= conf_idx).
                zone = freshest([z for z in zones if z.usable < i])
                if zone is None:
                    continue
            # Final mode (section 2.5 item 1): while eq is the freshest zone it is
            # the zone in play, and it fills only on a bar that OPENS on the far
            # side of the midpoint (long: open > EQ; short: open < EQ; strict). A
            # bar that opens at or through EQ touches it by construction (its low
            # is <= its open), is refused, and does not fall through to an older
            # zone. o[i] is the open of the completed bar being read.
            if self.eq_open_side and zone.kind == "eq" and not (o[i] > eq if side > 0 else o[i] < eq):
                row["eq_refused"].append(i)
                continue
            entry = None
            if side > 0 and l[i] <= zone.high:
                entry = min(zone.high, o[i])
            elif side < 0 and h[i] >= zone.low:
                entry = max(zone.low, o[i])
            if entry is not None:
                row.update(fill_idx=i, entry=float(entry), fill_zone=zone)
                fz = freshest(zones)                             # did bar i itself complete a fresher zone?
                row["fill_bar_new_zone"] = fz.kind if fz is not zone and fz.usable >= i else None
                if self.decided:
                    self.fill_log(row, i, zone, zones, lv)
        return row

    def fill_log(self, row: dict, i: int, zone: Zone, zones: list[Zone], lv: dict) -> None:
        """Decided-mode diagnostics at the fill bar i: for an eq fill, the other
        zone types in discount / premium at that bar and whether the bar opened
        on the far side of EQ; for every fill, T1 / T2 (section 2.3 item 3 and
        the task's item 2d): the nearest level beyond the entry among the
        directional set, among POC_PREV + HVN_*, and among all levels (T1), and
        the one beyond T1 (T2). Logged only; nothing here changes a fill.

        Final mode adds section 2.5 item 3 next to those (the by-price reading
        above is kept under its own keys so the two can be compared): T1 = the
        nearest price strictly beyond the entry among the opposing-NAMED
        directional levels plus POC_PREV and HVN_*, T2 = the next distinct
        price; levels at one price are one target carrying every name. And the
        entry geometry against round 1's wick stop, with the confirmation
        bar's ATR. No bar after the fill bar is read; no outcome exists."""
        side, entry, eq = row["side"], row["entry"], row["eq"]
        if zone.kind == "eq":
            known = zones if self.zone_same_bar else [z for z in zones if z.usable < i]
            row["cooccur"] = sorted({z.kind for z in known if z.kind != "eq"}, key=ZONE_TYPES.index)
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
        if not self.final:
            return
        named = TARGET_HIGHS if side > 0 else TARGET_LOWS
        pool = {k: p for k, p in beyond.items() if k in named or level_type(k) in ("POC", "HVN")}
        by_price: dict[float, list[str]] = {}
        for k, p in pool.items():
            by_price.setdefault(round(p, 6), []).append(k)
        prices = sorted(by_price, key=lambda p: side * (p - entry))
        tgt = [("+".join(sorted(by_price[p], key=level_sort_key)), p) for p in prices[:2]]
        row["targets"]["f_t1"] = tgt[0] if tgt else None
        row["targets"]["f_t2"] = tgt[1] if len(tgt) > 1 else None
        atr_c = row["eq_atr"]
        stop = row["extreme"] - side * WICK_STOP_ATR * atr_c
        risk = side * (entry - stop)
        atr_f = float(self.bars.atr[i])              # round 1's own reading: the ATR at the fill bar
        # Had the day already traded through T1 on a bar before the fill bar? (09:30 .. i-1; geometry, not outcome)
        span = slice(row["i_open"], i)
        seen = (float(self.h[span].max()) if side > 0 else float(self.l[span].min())) if i > row["i_open"] else None
        # ... and through the wick stop, on a bar after the sweep bar and before the fill bar? The funnel has no
        # invalidation rule, so such a setup still fills; round 1's engine would book entry-beyond-stop as `no_risk`.
        pre = slice(row["sweep_idx"] + 1, i)
        stop_hit = bool(i > row["sweep_idx"] + 1 and
                        (self.l[pre].min() <= stop if side > 0 else self.h[pre].max() >= stop))
        row["geom"] = {"stop": stop, "risk": risk, "atr": atr_c, "stop_traded": stop_hit,
                       "t1_traded": (None if not tgt or seen is None else bool(side * (seen - tgt[0][1]) >= 0)),
                       "risk_fillbar_atr": side * (entry - (row["extreme"] - side * WICK_STOP_ATR * atr_f)),
                       "rr1": (side * (tgt[0][1] - entry) / risk) if tgt and risk > 0 else None,
                       "rr2": (side * (tgt[1][1] - entry) / risk) if len(tgt) > 1 and risk > 0 else None}


# ────────────────────────────── rows: as-of, formatting ──────────────────────────────

def as_of(row: dict, cut: int) -> dict:
    """The full-run row with everything at bar >= cut forgotten. What a frame
    truncated at `cut` must reproduce, if the funnel is causal."""
    r = dict(row)
    r["ambiguous"] = [i for i in row["ambiguous"] if i < cut]
    r["abstain_only"] = [i for i in row["abstain_only"] if i < cut]
    r["amb_detail"] = {i: d for i, d in row["amb_detail"].items() if i < cut}
    no_fill = dict(fill_idx=None, entry=math.nan, fill_zone=None, eq_open_far=None, cooccur=[], targets={}, geom={},
                   fill_bar_new_zone=None)
    no_conf = dict(conf_idx=None, conf_types=[], eq=math.nan, range=(math.nan, math.nan), zones=[],
                   eq_atr=math.nan, eq_usable=None, eq_refused=[], **no_fill)
    if row["sweep_idx"] is None or row["sweep_idx"] >= cut:
        r.update(sweep_idx=None, side=0, breached=[], breached_opp=[], level=None, extreme=math.nan,
                 pre_swing=None, conf_any={}, ifvg_fresh_any={}, cls_ret={}, cls_abs=[], cls_state={}, **no_conf)
        return r
    r["conf_any"] = {t: j for t, j in row["conf_any"].items() if j < cut}
    r["ifvg_fresh_any"] = {j: v for j, v in row["ifvg_fresh_any"].items() if j < cut}
    if row["conf_idx"] is None or row["conf_idx"] >= cut:
        r.update(**no_conf)
        return r
    r["zones"] = [z for z in row["zones"] if z.usable < cut]
    if row["eq_usable"] is not None and row["eq_usable"] >= cut:
        r["eq_usable"] = None
    r["eq_refused"] = [i for i in row["eq_refused"] if i < cut]
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


def after_refusal(row: dict) -> str:
    """What a day did after the open-side test refused a bar (final mode)."""
    if not row["eq_refused"]:
        return ""
    if row["fill_idx"] is None:
        return "no_fill"
    return "filled_later_on_eq" if row["fill_zone"].kind == "eq" else f"filled_on_other_zone:{row['fill_zone'].kind}"


def format_row(row: dict, minutes: np.ndarray, decided: bool = False, final: bool = False) -> OrderedDict:
    """The day-level CSV row. Times are ET bar stamps. `decided` appends the
    section-2.3 columns after the literal ones (the literal row is unchanged);
    `final` appends the section-2.5 columns after those (the decided columns
    keep their section-2.3 meaning: t1 / t2 there are still by price, any name)."""
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
    if not final:
        return out
    g = row["geom"]
    f1, f2 = tg.get("f_t1"), tg.get("f_t2")
    in_atr = lambda kv: f(abs(kv[1] - row["entry"]) / g["atr"]) if kv and g.get("atr", 0) > 0 else ""   # noqa: E731
    out.update([
        ("class_states", ";".join(f"{cls}:{st}" for cls, st in row["cls_state"].items())),
        ("session_class", row["cls_state"].get("SESSION", "")),
        ("eq_refused_bars", ";".join(t(i) for i in row["eq_refused"])),
        ("eq_refused_n", len(row["eq_refused"]) if row["conf_idx"] is not None else ""),
        ("eq_after_refusal", after_refusal(row)),
        ("fill_bar_new_zone", row.get("fill_bar_new_zone") or ""),
        ("final_t1", f1[0] if f1 else ""),
        ("final_t1_price", f(f1[1]) if f1 else ""),
        ("final_t1_atr", in_atr(f1)),
        ("final_t1_already_traded", "" if g.get("t1_traded") is None else int(g["t1_traded"])),
        ("final_t2", f2[0] if f2 else ""),
        ("final_t2_price", f(f2[1]) if f2 else ""),
        ("final_t2_atr", in_atr(f2)),
        ("wick_stop", f(g.get("stop"))),
        ("wick_stop_traded_before_fill", "" if not g else int(g["stop_traded"])),
        ("risk_pts", f(g.get("risk"))),
        ("risk_atr", f(g["risk"] / g["atr"]) if g and g["atr"] > 0 else ""),
        ("final_t1_rr", f(g.get("rr1"))),
        ("final_t2_rr", f(g.get("rr2"))),
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


def target_type(names: str) -> str:
    """A final-mode target's bucket: its level type, or `tie` when two names share the price."""
    return "tie" if "+" in names else level_type(names)


def report_final(label: str, path: str, fn: WideFunnel, fn6: WideFunnel, r1log: pd.DataFrame,
                 fn_lit: WideFunnel, fn_fresh: WideFunnel, fn_dec: WideFunnel, fn_cls: WideFunnel,
                 fn_cls_r1: WideFunnel, fn_r1z: WideFunnel) -> dict:
    """The final-mode printout (section 2.5): the decided report's structure with
    the open-side test, the three classes and the opposing-named targets, and a
    five-column side-by-side. `fn_dec` is the decided-mode run on the same
    frame (it reproduces part2_decided_funnel.txt); `fn_cls` is final mode with
    the open-side test switched off, used ONLY to check the refusals against
    the fills they replace. `fn_cls_r1` (open-side test off) and `fn_r1z`
    (open-side test on) are final mode under round 1's same-bar zone
    convention (zone_same_bar=True), used ONLY to attribute the change in
    fills between the class correction, the eq correction and the
    zone-in-play rule. None of the three is a mode or a spec."""
    rows, df, minutes = fn.rows, fn.df, fn.clock.minutes
    tm = lambda i: hhmm(int(minutes[i]))                                       # noqa: E731
    dstr = lambda r: r["day"].strftime("%m-%d")                                # noqa: E731
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
    print("2. funnel (section 2.5: fresh ifvg; eq as a retrace AND only on a bar that opens on the far side of EQ; "
          "per-class sweep with SESSION / H1 / H4)")
    print(f"   sessions {n} -> sweeps {len(swept)} -> confirmations {len(conf)} -> "
          f"zones in discount/premium {len(zoned)} -> fills {len(filled)}")
    print(f"   zones excluding eq: {len(zoned_x)}")
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
    print(f"   round-1 six-level sweep days lost under the final rule: {len(lost)} (by class disagreement: {len(lost_dis)}; "
          f"abstention-only: {len(lost_abs)}; other: {len(lost) - len(lost_dis) - len(lost_abs)}); "
          f"days swept only under the final rule: {len(gained)}; swept under both: {len(both)} "
          f"(same side {sum(1 for r in both if r['side'] == six[r['day']]['side'])}, "
          f"same bar {sum(1 for r in both if r['sweep_idx'] == six[r['day']]['sweep_idx'])}, "
          f"same level {sum(1 for r in both if r['level'] == six[r['day']]['level'])})")
    dec = {r["day"]: r for r in fn_dec.rows}
    lost_d = [r for r in rows if r["sweep_idx"] is None and dec[r["day"]]["sweep_idx"] is not None]
    gained_d = [r for r in rows if r["sweep_idx"] is not None and dec[r["day"]]["sweep_idx"] is None]
    both_d = [r for r in swept if dec[r["day"]]["sweep_idx"] is not None]
    print(f"   decided (five-class) sweep days lost under the final rule: {len(lost_d)}; gained: {len(gained_d)}; both: {len(both_d)} "
          f"(same side {sum(1 for r in both_d if r['side'] == dec[r['day']]['side'])}, "
          f"same bar {sum(1 for r in both_d if r['sweep_idx'] == dec[r['day']]['sweep_idx'])}, "
          f"same level {sum(1 for r in both_d if r['level'] == dec[r['day']]['level'])})")
    ifvg_conf = [r for r in conf if "ifvg" in r["conf_types"]]
    print(f"   ifvg on the confirmation bar: {len(ifvg_conf)} (fresh inversions by definition)")
    print(f"   sweep bar: " + counter_line(Counter(tm(r['sweep_idx']) for r in swept),
                                             sorted({tm(r['sweep_idx']) for r in swept})))
    print(f"   confirmation bar: " + counter_line(Counter(tm(r['conf_idx']) for r in conf),
                                                    sorted({tm(r['conf_idx']) for r in conf})))

    print()
    print("3. breakdowns")
    every = Counter(level_class_final(k) for r in swept for k in r["breached"])
    every_t = Counter(level_type(k) for r in swept for k in r["breached"])
    every_name = Counter(k for r in swept for k in r["breached"])
    deepest = Counter(level_class_final(r["level"]) for r in swept)
    deepest_t = Counter(level_type(r["level"]) for r in swept)
    deepest_name = Counter(r["level"] for r in swept)
    print("   sweeps by level class, every breached level of the sweep side counted:  " + counter_line(every, CLASSES_FINAL)
          + f"   (levels per sweep bar: {sum(every.values()) / max(1, len(swept)):.2f})   by type: " + counter_line(every_t, CLASSES))
    print("   sweeps by level class, the setup's level (deepest among the returning classes): " + counter_line(deepest, CLASSES_FINAL)
          + "   by type: " + counter_line(deepest_t, CLASSES))
    print("     by name, every breached:  " + counter_line(every_name))
    print("     by name, the setup's:     " + counter_line(deepest_name))
    opp = Counter(k for r in swept for k in r["breached_opp"])
    print(f"   opposite-side levels the sweep wick also took (abstaining classes, or a high run without a close back): "
          f"{sum(opp.values())} on {sum(1 for r in swept if r['breached_opp'])} sweep bars   " + counter_line(opp))
    deeper = [r for r in swept if r["breached"] and r["breached"][0] != r["level"]]
    print(f"   sweep bars where a deeper level of the sweep side sat in an abstaining class (recorded, not the setup's): {len(deeper)}")
    only_wide = [r for r in swept if not any(level_type(k) in SESSION_TYPES for k in r["breached"])]
    print(f"   sweep days where NO session level of the sweep side was breached (H1/H4 alone on that side): {len(only_wide)}")
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
    print("   fills by zone used: " + counter_line(Counter(r["fill_zone"].kind for r in filled), ZONE_TYPES)
          + "   by side: " + counter_line(Counter("long" if r["side"] > 0 else "short" for r in filled), ("long", "short"))
          + "   by fill bar: " + counter_line(Counter(tm(r["fill_idx"]) for r in filled), sorted({tm(r["fill_idx"]) for r in filled})))
    print("   fills by (level class, level type, confirmation types, zone used):")
    fills = Counter((level_class_final(r["level"]), level_type(r["level"]), "+".join(r["conf_types"]), r["fill_zone"].kind)
                    for r in filled)
    if fills:
        for (lc_, lt, ct, zk), cnt in sorted(fills.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"     {lc_:<8} {lt:<5} {ct:<13} {zk:<8} {cnt}")
    else:
        print("     (none)")
    print("   fills, per day: " + ("; ".join(
        f"{r['day'].strftime('%Y-%m-%d')} {'L' if r['side'] > 0 else 'S'} {r['level']} "
        f"{'+'.join(r['conf_types'])}@{tm(r['conf_idx'])} "
        f"{r['fill_zone'].kind}@{tm(r['fill_idx'])} {r['entry']:.2f}"
        + (f" T1={r['targets']['f_t1'][0]}" if r["targets"].get("f_t1") else " T1=none")
        for r in filled) or "(none)"))

    # ---- 3b. eq --------------------------------------------------------------
    print()
    print("3b. eq zone (section 2.3 item 2 + section 2.5 item 1): usable after the "
          f"{EQ_EXCURSION_ATR} x ATR excursion, and a touch counts only on a bar that OPENS on the far side of EQ")
    eq_days = [r for r in conf if r["eq_usable"] is not None]
    at_conf = [r for r in eq_days if r["eq_usable"] == r["conf_idx"]]
    later = [r for r in eq_days if r["eq_usable"] > r["conf_idx"]]
    never = [r for r in conf if r["eq_usable"] is None]
    print(f"   excursion (unchanged): confirmed days {len(conf)}: eq usable on {len(eq_days)} (at the confirmation bar {len(at_conf)}, "
          f"on a later bar {len(later)}); never by 10:05: {len(never)}")
    widths = [(r["range"][1] - r["range"][0]) / r["eq_atr"] for r in conf if r["eq_atr"] > 0]
    print("   dealing range width at confirmation, in ATR: " + _quantiles(widths))
    refused = [r for r in conf if r["eq_refused"]]
    n_ref_bars = sum(len(r["eq_refused"]) for r in refused)
    after = Counter(after_refusal(r).split(":")[0] for r in refused)
    other_kinds = Counter(r["fill_zone"].kind for r in refused if r["fill_idx"] is not None and r["fill_zone"].kind != "eq")
    print(f"   open-side test: bars refused (eligible entry-window bar, eq the freshest zone, open at or through EQ): "
          f"{n_ref_bars} bars on {len(refused)} setups;  refused bars per setup: "
          + counter_line(Counter(len(r["eq_refused"]) for r in refused), sorted({len(r["eq_refused"]) for r in refused})))
    print("   refused bar stamps: " + counter_line(Counter(tm(i) for r in refused for i in r["eq_refused"]),
                                                   sorted({tm(i) for r in refused for i in r["eq_refused"]}))
          + f"   refused bars that opened exactly AT EQ: {sum(1 for r in refused for i in r['eq_refused'] if fn.o[i] == r['eq'])}")
    print(f"   what the refused setups then did: filled later on eq {after.get('filled_later_on_eq', 0)}; "
          f"filled on another zone that became freshest {after.get('filled_on_other_zone', 0)}"
          + (f" ({counter_line(other_kinds)})" if other_kinds else "")
          + f"; no fill {after.get('no_fill', 0)}")
    for key, title in (("filled_later_on_eq", "filled later on eq"), ("filled_on_other_zone", "filled on another zone"),
                       ("no_fill", "no fill")):
        days = [r for r in refused if after_refusal(r).split(":")[0] == key]
        print(f"     {title}: " + (", ".join(
            f"{dstr(r)} {'L' if r['side'] > 0 else 'S'} refused {'/'.join(tm(i) for i in r['eq_refused'])}"
            + (f" -> {r['fill_zone'].kind}@{tm(r['fill_idx'])}" if r["fill_idx"] is not None else "")
            for r in days) or "(none)"))
    # The classes-only run (open-side test off) fills exactly where the first refusal happens.
    cls = {r["day"]: r for r in fn_cls.rows}
    cls_fills = [r for r in fn_cls.rows if r["fill_idx"] is not None]
    cls_eq = [r for r in cls_fills if r["fill_zone"].kind == "eq"]
    cls_eq_near = [r for r in cls_eq if not r["eq_open_far"]]
    inv = ({r["day"] for r in cls_eq_near} == {r["day"] for r in refused}
           and all(cls[r["day"]]["fill_idx"] == r["eq_refused"][0] for r in refused))
    print(f"   check: without the open-side test (same three classes, same zone-in-play rule) there are {len(cls_fills)} fills, {len(cls_eq)} on eq, "
          f"{len(cls_eq_near)} of them on a bar that opened at or through EQ; those are exactly the {len(refused)} refused setups, "
          f"each first refused on the bar it would have filled: {'yes' if inv else 'NO'}")
    eq_fills = [r for r in filled if r["fill_zone"].kind == "eq"]
    far = [r for r in eq_fills if r["eq_open_far"]]
    at_eq = [r for r in eq_fills if r["entry"] == r["eq"]]
    print(f"   eq fills: {len(eq_fills)} of {len(filled)} fills; touching bar opened on the far side: {len(far)} (all, by the rule); "
          f"entry exactly at EQ: {len(at_eq)}; fills on other zones: "
          + counter_line(Counter(r["fill_zone"].kind for r in filled if r["fill_zone"].kind != "eq"), ("fvg", "ob", "breaker")))
    print("   co-occurrence at eq fills, other zone types in discount/premium at the fill bar: "
          + counter_line(Counter("+".join(r["cooccur"]) or "none" for r in eq_fills))
          + "   by type: " + counter_line(Counter(k for r in eq_fills for k in r["cooccur"]), ("fvg", "ob", "breaker")))
    eq_only = [r for r in zoned if all(z.kind == "eq" for z in r["zones"])]
    print(f"   zone-stage days whose only zone is eq: {len(eq_only)}; fills on those days: "
          f"{sum(1 for r in eq_only if r['fill_idx'] is not None)}")
    counts["eq"] = {"confirmed": len(conf), "eq_usable": len(eq_days), "at_confirmation": len(at_conf),
                    "later": len(later), "never": len(never),
                    "refused_bars": n_ref_bars, "refused_setups": len(refused),
                    "refused_then_filled_on_eq": after.get("filled_later_on_eq", 0),
                    "refused_then_filled_on_other_zone": after.get("filled_on_other_zone", 0),
                    "refused_then_other_zone_kinds": dict(other_kinds),
                    "refused_then_no_fill": after.get("no_fill", 0),
                    "fills": len(filled), "eq_fills": len(eq_fills),
                    "fills_by_zone": dict(Counter(r["fill_zone"].kind for r in filled)),
                    "fills_without_open_side_test": len(cls_fills), "eq_fills_without_open_side_test": len(cls_eq),
                    "refusal_invariant_ok": bool(inv),
                    "cooccurrence": dict(Counter("+".join(r["cooccur"]) or "none" for r in eq_fills)),
                    "zoned_days_eq_only": len(eq_only)}

    # ---- 3c. classes -----------------------------------------------------------
    print()
    print("3c. level classes (section 2.5 item 2): SESSION = ASIA_H/L + LON_H/L + PDH/PDL as ONE class; H1; H4. POC and HVN never swept")
    amb_bars = [(r, i) for r in rows for i in r["ambiguous"]]
    abs_bars = [(r, i) for r in rows for i in r["abstain_only"]]
    print(f"   class-disagreement bars: {len(amb_bars)} on {sum(1 for r in rows if r['ambiguous'])} days; "
          f"abstention-only bars: {len(abs_bars)} on {sum(1 for r in rows if r['abstain_only'])} days; "
          f"decided (five classes) had {sum(len(r['ambiguous']) for r in fn_dec.rows)} and {sum(len(r['abstain_only']) for r in fn_dec.rows)} bars")
    print("   disagreement patterns (class:side; abstaining classes noted): "
          + counter_line(Counter(r["amb_detail"][i] for r, i in amb_bars)))
    print(f"   days whose FIRST sweep-window verdict was ambiguous but a later bar swept: "
          f"{sum(1 for r in swept if r['ambiguous'] and min(r['ambiguous']) < r['sweep_idx'])}; "
          f"sweep days with an abstention-only bar before the sweep bar: "
          f"{sum(1 for r in swept if r['abstain_only'] and min(r['abstain_only']) < r['sweep_idx'])}")
    print("   classes returning on the sweep bar, count per sweep: "
          + counter_line(Counter(len(r["cls_ret"]) for r in swept), sorted({len(r["cls_ret"]) for r in swept}))
          + "   by class: " + counter_line(Counter(c_ for r in swept for c_ in r["cls_ret"]), CLASSES_FINAL))
    print("   classes abstaining on the sweep bar: "
          + counter_line(Counter(c_ for r in swept for c_ in r["cls_abs"]), CLASSES_FINAL)
          + f"   sweep bars with any abstention: {sum(1 for r in swept if r['cls_abs'])}")
    states = ("returned", "abstained", "breached_no_close_back", "untouched", "absent")
    sess = Counter(r["cls_state"].get("SESSION", "absent") for r in swept)
    print("   SESSION on the sweep bar: " + counter_line(sess, states))
    print("     returned = round 1's own test swept this bar; abstained = a session low AND a session high in the wick (round 1's")
    print("     ambiguity: round 1 SKIPS the bar); breached_no_close_back / untouched = round 1 does NOT sweep the bar")
    swing_made = [r for r in swept if r["cls_state"].get("SESSION") != "returned"]
    sm_abst = [r for r in swing_made if r["cls_state"].get("SESSION") == "abstained"]
    sm_none = [r for r in swing_made if r["cls_state"].get("SESSION") != "abstained"]
    r1day = lambda r: six[r["day"]]["sweep_idx"]                                 # noqa: E731
    def split(rs):                                                               # noqa: E306
        no_r1 = [r for r in rs if r1day(r) is None]
        r1_later = [r for r in rs if r1day(r) is not None and r1day(r) > r["sweep_idx"]]
        r1_other = [r for r in rs if r1day(r) is not None and r1day(r) <= r["sweep_idx"]]
        return no_r1, r1_later, r1_other
    print(f"   final sweeps made by a swing class alone (SESSION did not return on the sweep bar): {len(swing_made)} of {len(swept)}")
    for title, rs in (("SESSION abstained, a swing class returned (round 1 would have SKIPPED the bar)", sm_abst),
                      ("SESSION returned nothing, a swing class returned (round 1 would NOT have swept the bar)", sm_none)):
        no_r1, r1_later, r1_other = split(rs)
        print(f"     {title}: {len(rs)}  -> round 1 had no sweep that day: {len(no_r1)}; round 1 swept a LATER bar that day: "
              f"{len(r1_later)} (same side {sum(1 for r in r1_later if six[r['day']]['side'] == r['side'])}); "
              f"round 1 swept an EARLIER bar that final skipped as a class disagreement: {len(r1_other)} "
              f"(of which that bar is a logged disagreement bar: {sum(1 for r in r1_other if r1day(r) in r['ambiguous'])})"
              f"   returning class: " + counter_line(Counter("+".join(r["cls_ret"]) for r in rs)))
    sess_ret = [r for r in swept if r["cls_state"].get("SESSION") == "returned"]
    sr_same = [r for r in sess_ret if r1day(r) == r["sweep_idx"]]
    sr_level = [r for r in sr_same if six[r["day"]]["level"] == r["level"]]
    print(f"   final sweeps on which SESSION returned: {len(sess_ret)}; round 1 swept the same bar: {len(sr_same)} "
          f"(same side {sum(1 for r in sr_same if six[r['day']]['side'] == r['side'])}, same setup level {len(sr_level)}; "
          f"a deeper H1/H4 level of the same side is the setup's on {len(sr_same) - len(sr_level)}); "
          f"round 1 swept a different bar: {len(sess_ret) - len(sr_same)}")
    print(f"   days only H1/H4 made (no round-1 sweep that day at all): {len(gained)}"
          + (" (" + ", ".join(f"{dstr(r)} {'L' if r['side'] > 0 else 'S'} {r['level']}[SESSION {r['cls_state'].get('SESSION')}]"
                              for r in gained) + ")" if gained else ""))
    print(f"   round-1 six-level sweep days lost: {len(lost)} of {n6} (by class disagreement {len(lost_dis)}, "
          f"abstention-only {len(lost_abs)}, other {len(lost) - len(lost_dis) - len(lost_abs)}); "
          f"decided lost {sum(1 for r in fn_dec.rows if r['sweep_idx'] is None and six[r['day']]['sweep_idx'] is not None)}")
    lost_days = ", ".join(f"{dstr(r)}[{';'.join(r['amb_detail'][i] for i in r['ambiguous'])}]" for r in lost_dis)
    print(f"   lost by class disagreement, per day: {lost_days or '(none)'}")
    gone = ", ".join(f"{dstr(r)}" for r in lost_d)
    print(f"   decided sweep days that are not final sweep days (the five-class split made them; SESSION as one class abstains): "
          f"{len(lost_d)}" + (f" ({gone})" if gone else ""))
    counts["classes"] = {"sweeps": len(swept), "disagreement_bars": len(amb_bars),
                         "disagreement_days": sum(1 for r in rows if r["ambiguous"]),
                         "abstention_only_bars": len(abs_bars), "abstention_only_days": sum(1 for r in rows if r["abstain_only"]),
                         "disagreement_patterns": dict(Counter(r["amb_detail"][i] for r, i in amb_bars)),
                         "round1_sweep_days": n6, "round1_days_lost": len(lost),
                         "round1_days_lost_by_class_disagreement": len(lost_dis),
                         "round1_days_lost_by_abstention_only": len(lost_abs),
                         "days_gained_vs_round1_only_H1_H4": len(gained), "swept_under_both": len(both),
                         "both_same_side": sum(1 for r in both if r["side"] == six[r["day"]]["side"]),
                         "both_same_bar": sum(1 for r in both if r["sweep_idx"] == six[r["day"]]["sweep_idx"]),
                         "both_same_level": sum(1 for r in both if r["level"] == six[r["day"]]["level"]),
                         "session_state_on_sweep_bar": dict(sess),
                         "swing_class_alone_sweeps": len(swing_made),
                         "session_abstained_swing_returned": len(sm_abst),
                         "session_abstained_swing_returned_no_round1_sweep_that_day": len(split(sm_abst)[0]),
                         "session_returned_nothing_swing_returned": len(sm_none),
                         "session_returned_nothing_swing_returned_no_round1_sweep_that_day": len(split(sm_none)[0]),
                         "decided_sweep_days_lost": len(lost_d), "decided_sweep_days_gained": len(gained_d),
                         "sweep_bars_with_abstention": sum(1 for r in swept if r["cls_abs"]),
                         "abstaining_by_class": dict(Counter(c_ for r in swept for c_ in r["cls_abs"])),
                         "deeper_level_in_abstaining_class": len(deeper)}

    # ---- 3d. targets -----------------------------------------------------------
    print()
    print("3d. targets, logged only (section 2.5 item 3): nearest opposing-NAMED level (long: ASIA_H LON_H PDH H1_SH H4_SH; "
          "short: the lows) or POC_PREV / HVN_*, strictly beyond the entry; T2 = the next distinct price")
    t1 = [r["targets"]["f_t1"] for r in filled]
    t2 = [r["targets"]["f_t2"] for r in filled]
    buckets = CLASSES + ("POC", "HVN", "tie")
    ties1 = Counter(t[0] for t in t1 if t and "+" in t[0])
    ties2 = Counter(t[0] for t in t2 if t and "+" in t[0])
    print("   T1 by type over fills:  " + counter_line(Counter(target_type(t[0]) for t in t1 if t), buckets)
          + f"   no T1: {sum(1 for t in t1 if t is None)}" + (f"   ties: {counter_line(ties1)}" if ties1 else ""))
    print("   T1 by name:             " + counter_line(Counter(t[0] for t in t1 if t)))
    print("   T2 by type over fills:  " + counter_line(Counter(target_type(t[0]) for t in t2 if t), buckets)
          + f"   no T2: {sum(1 for t in t2 if t is None)}" + (f"   ties: {counter_line(ties2)}" if ties2 else ""))
    print("   T2 by name:             " + counter_line(Counter(t[0] for t in t2 if t)))
    no_t1 = [r for r in filled if r["targets"]["f_t1"] is None]
    if no_t1:
        print("   fills with no T1: " + ", ".join(f"{dstr(r)} {'L' if r['side'] > 0 else 'S'} {r['entry']:.2f}" for r in no_t1))
    same_side = [r for r in filled for t in (r["targets"]["f_t1"], r["targets"]["f_t2"]) if t
                 for k in t[0].split("+") if k in (DIRECTIONAL_LOWS if r["side"] > 0 else DIRECTIONAL_HIGHS)]
    swept_t = [r for r in filled for t in (r["targets"]["f_t1"], r["targets"]["f_t2"]) if t and r["level"] in t[0].split("+")]
    print(f"   check: targets carrying a same-side-named level: {len(same_side)}; targets that are the swept level: {len(swept_t)} "
          f"(both 0 by construction)")
    old_same = [r for r in filled if r["targets"].get("t1")
                and r["targets"]["t1"][0] in (DIRECTIONAL_LOWS if r["side"] > 0 else DIRECTIONAL_HIGHS)]
    print(f"   for comparison, section 2.3's by-price T1 on these same fills would be a same-side-named level on {len(old_same)}")
    dist = lambda r, t: abs(t[1] - r["entry"]) / r["geom"]["atr"]              # noqa: E731
    d1 = [dist(r, r["targets"]["f_t1"]) for r in filled if r["targets"]["f_t1"]]
    d2 = [dist(r, r["targets"]["f_t2"]) for r in filled if r["targets"]["f_t2"]]
    print("   T1 distance from entry, in confirmation-bar ATR: " + _quantiles(d1))
    print("   T2 distance from entry, in confirmation-bar ATR: " + _quantiles(d2))
    traded = [r for r in filled if r["geom"].get("t1_traded")]
    print(f"   fills whose T1 price the day had already traded through before the fill bar (09:30 .. the bar before the fill): "
          f"{_pct(len(traded), len(d1))}   by T1 type: " + counter_line(Counter(target_type(r["targets"]["f_t1"][0]) for r in traded), buckets))
    print(f"   entry geometry, no outcome: wick stop = sweep extreme -/+ {WICK_STOP_ATR} x ATR(14) at the confirmation bar")
    ok_risk = [r for r in filled if r["geom"]["risk"] > 0]
    print(f"   fills with entry-to-stop distance <= 0 (no R defined): {len(filled) - len(ok_risk)}")
    stop_traded = [r for r in filled if r["geom"]["stop_traded"]]
    print(f"   fills on a setup whose wick stop price had ALREADY traded on a bar after the sweep bar and before the fill bar "
          f"(the funnel has no invalidation rule): {len(stop_traded)}"
          + (" (" + ", ".join(f"{dstr(r)} {r['fill_zone'].kind}" for r in stop_traded) + ")" if stop_traded else ""))
    print("   entry to wick stop, in ATR:    " + _quantiles([r["geom"]["risk"] / r["geom"]["atr"] for r in ok_risk]))
    print("   entry to wick stop, in points: " + _quantiles([r["geom"]["risk"] for r in ok_risk]))
    print("   (round 1 itself read the ATR at the fill bar, not the confirmation bar; on these fills that stop distance, in points: "
          + _quantiles([r["geom"]["risk_fillbar_atr"] for r in ok_risk]) + ")")
    rr1 = [r["geom"]["rr1"] for r in ok_risk if r["geom"]["rr1"] is not None]
    rr2 = [r["geom"]["rr2"] for r in ok_risk if r["geom"]["rr2"] is not None]
    print("   T1 reward-to-risk against the wick stop: " + _quantiles(rr1))
    print("   T2 reward-to-risk against the wick stop: " + _quantiles(rr2))
    close1 = [r for r in ok_risk if r["geom"]["rr1"] is not None and r["geom"]["rr1"] < 1.0]
    close2 = [r for r in ok_risk if r["geom"]["rr2"] is not None and r["geom"]["rr2"] < 1.0]
    print(f"   fills with T1 closer than 1.0 R: {_pct(len(close1), len(rr1))}; closer than 0.5 R: "
          f"{sum(1 for x in rr1 if x < 0.5)}; T2 closer than 1.0 R: {_pct(len(close2), len(rr2))}")
    print("   T1 closer than 1.0 R, by T1 type: " + counter_line(Counter(target_type(r["targets"]["f_t1"][0]) for r in close1), buckets))
    qd = lambda xs: [float(x) for x in np.quantile(np.asarray(xs, dtype=float), [0, .25, .5, .75, 1])] if xs else None  # noqa: E731
    counts["targets"] = {"fills": len(filled),
                         "t1_type": dict(Counter(target_type(t[0]) for t in t1 if t)),
                         "t1_name": dict(Counter(t[0] for t in t1 if t)),
                         "t2_type": dict(Counter(target_type(t[0]) for t in t2 if t)),
                         "t2_name": dict(Counter(t[0] for t in t2 if t)),
                         "t1_ties": dict(ties1), "t2_ties": dict(ties2),
                         "no_t1": sum(1 for t in t1 if t is None), "no_t2": sum(1 for t in t2 if t is None),
                         "t1_dist_atr_q": qd(d1), "t2_dist_atr_q": qd(d2),
                         "stop_dist_atr_q": qd([r["geom"]["risk"] / r["geom"]["atr"] for r in ok_risk]),
                         "t1_rr_q": qd(rr1), "t2_rr_q": qd(rr2),
                         "t1_closer_than_1R": len(close1), "t1_rr_n": len(rr1),
                         "t2_closer_than_1R": len(close2), "t2_rr_n": len(rr2),
                         "t1_already_traded_before_fill": len(traded),
                         "entry_beyond_stop": len(filled) - len(ok_risk),
                         "stop_traded_before_fill": len(stop_traded),
                         "same_side_named_targets": len(same_side), "swept_level_targets": len(swept_t)}

    # ---- 3e. final against decided, fill for fill --------------------------------
    print()
    print("3e. final against decided, fill for fill (no outcome: which days fill, and where)")
    dfill = {r["day"]: r for r in fn_dec.rows if r["fill_idx"] is not None}
    ffill = {r["day"]: r for r in filled}
    cfill = {r["day"]: r for r in fn_cls_r1.rows if r["fill_idx"] is not None}
    zfill = {r["day"]: r for r in fn_r1z.rows if r["fill_idx"] is not None}
    gone_f = [d for d in dfill if d not in ffill]
    new_f = [d for d in ffill if d not in dfill]
    same_f = [d for d in ffill if d in dfill and ffill[d]["fill_idx"] == dfill[d]["fill_idx"]
              and ffill[d]["entry"] == dfill[d]["entry"] and ffill[d]["side"] == dfill[d]["side"]]
    print(f"   decided fills {len(dfill)} -> with the three classes only {len(cfill)} -> with the open-side test too {len(zfill)} "
          f"(both still under round 1's same-bar zone convention, as decided is) -> with the zone in play read from "
          f"before the bar (final, section 3f) {len(ffill)}")
    print(f"   accounting: {len(dfill)} decided - {sum(1 for d in dfill if d not in cfill)} fill days the class change removed "
          f"+ {sum(1 for d in cfill if d not in dfill)} it added = {len(cfill)}; - {sum(1 for d in cfill if d not in zfill)} "
          f"the open-side test left unfilled + {sum(1 for d in zfill if d not in cfill)} it added = {len(zfill)}; "
          f"- {sum(1 for d in zfill if d not in ffill)} the zone-in-play rule removed "
          f"+ {sum(1 for d in ffill if d not in zfill)} it added = {len(ffill)}")
    print(f"   decided fill days that do not fill in final: {len(gone_f)} "
          f"(of which the class change alone already removed {sum(1 for d in gone_f if d not in cfill)}, "
          f"the open-side test removed {sum(1 for d in gone_f if d in cfill and d not in zfill)}, "
          f"the zone-in-play rule removed {sum(1 for d in gone_f if d in zfill)}); "
          f"final fill days that did not fill in decided: {len(new_f)}; "
          f"fill on both: {len(ffill) - len(new_f)} (same side, bar and entry {len(same_f)})")
    print("     gone: " + (", ".join(pd.Timestamp(d).strftime("%m-%d") for d in gone_f) or "(none)"))
    print("     new:  " + (", ".join(pd.Timestamp(d).strftime("%m-%d") for d in new_f) or "(none)"))
    counts["vs_decided"] = {"decided_fills": len(dfill), "classes_only_fills": len(cfill),
                            "classes_and_open_side_same_bar_convention_fills": len(zfill), "final_fills": len(ffill),
                            "removed_by_classes": sum(1 for d in dfill if d not in cfill),
                            "added_by_classes": sum(1 for d in cfill if d not in dfill),
                            "unfilled_by_open_side": sum(1 for d in cfill if d not in zfill),
                            "removed_by_zone_in_play": sum(1 for d in zfill if d not in ffill),
                            "added_by_zone_in_play": sum(1 for d in ffill if d not in zfill),
                            "decided_fill_days_gone": len(gone_f), "new_fill_days": len(new_f),
                            "same_side_bar_entry": len(same_f)}

    # ---- 3f. the zone in play on a bar ---------------------------------------------
    print()
    print("3f. zone in play on bar i: the freshest zone KNOWN BEFORE bar i (usable < i). A gap or breaker that bar i itself")
    print("    completes is known only at bar i's close; an order resting on the zone in play has traded inside bar i by then.")
    print("    Round 1's convention (literal, decided, tjr_intraday: `fresh zones first, then the fill`) makes the new zone the")
    print("    freshest ON bar i and so withholds any fill on bar i. No truncation cut can see the difference: both read bars <= i.")
    zrow = {r["day"]: r for r in fn_r1z.rows}
    sig = lambda r: (r["fill_idx"], None if r["fill_idx"] is None else r["fill_zone"].kind,                 # noqa: E731
                     None if r["fill_idx"] is None else r["entry"], tuple(r["eq_refused"]))
    fill_sig = lambda r: sig(r)[:3]                                                                          # noqa: E731
    moved = [r for r in conf if fill_sig(r) != fill_sig(zrow[r["day"]])]
    ref_only = [r for r in conf if fill_sig(r) == fill_sig(zrow[r["day"]]) and sig(r) != sig(zrow[r["day"]])]
    upstream = sum(1 for r in rows if (r["sweep_idx"], r["side"], r["conf_idx"]) !=
                   (zrow[r["day"]]["sweep_idx"], zrow[r["day"]]["side"], zrow[r["day"]]["conf_idx"]))
    same_bar_fills = [r for r in filled if r.get("fill_bar_new_zone")]
    z_filled = [r for r in fn_r1z.rows if r["fill_idx"] is not None]
    z_ref = [r for r in fn_r1z.rows if r["eq_refused"]]
    desc = lambda r: ("no fill" if r["fill_idx"] is None                                                     # noqa: E731
                      else f"{r['fill_zone'].kind}@{tm(r['fill_idx'])} {r['entry']:.2f}")
    print(f"   sweeps and confirmations are untouched by the rule (days differing upstream of the zone: {upstream})")
    print(f"   fills under round 1's same-bar convention: {len(z_filled)}; under the rule (final): {len(filled)}; "
          f"confirmed days whose fill differs (filled or not, bar, zone or entry): {len(moved)} of {len(conf)}")
    for r in moved:
        print(f"     {dstr(r)} {'L' if r['side'] > 0 else 'S'} EQ {r['eq']:.2f}: same-bar convention {desc(zrow[r['day']])} "
              f"-> final {desc(r)}" + (f"  (the fill bar itself completed a {r['fill_bar_new_zone']})" if r.get("fill_bar_new_zone") else ""))
    print(f"   final fills on a bar that itself completed a fresher zone: {len(same_bar_fills)}"
          + (" (" + ", ".join(f"{dstr(r)} {r['fill_zone'].kind}@{tm(r['fill_idx'])} new {r['fill_bar_new_zone']}" for r in same_bar_fills) + ")"
             if same_bar_fills else ""))
    print(f"   open-side refusals under the same-bar convention: {sum(len(r['eq_refused']) for r in z_ref)} bars on {len(z_ref)} setups; "
          f"under the rule: {n_ref_bars} bars on {len(refused)} setups. The extra ones are bars that completed a fresher zone: "
          f"under the old convention such a bar was not read for a fill at all, under the rule eq was still in play on it and the "
          f"bar opened at or through EQ. Days with the same fill but a different refusal list: {len(ref_only)}"
          + (" (" + ", ".join(dstr(r) for r in ref_only) + ")" if ref_only else ""))
    counts["zone_in_play"] = {"fills_same_bar_convention": len(z_filled), "fills_final": len(filled),
                              "days_fill_differs": [dstr(r) for r in moved],
                              "final_fills_on_bar_completing_fresher_zone": len(same_bar_fills),
                              "refused_bars_same_bar_convention": sum(len(r["eq_refused"]) for r in z_ref),
                              "refused_setups_same_bar_convention": len(z_ref),
                              "days_refusals_differ_only": [dstr(r) for r in ref_only],
                              "days_differing_upstream": upstream}

    # ---- 4. side-by-side -------------------------------------------------------
    print()
    print("4. side-by-side: round 1 (DESIGN-tjr-intraday.md section 9, base) / literal wide / literal fresh-ifvg / decided / final")
    r1 = ROUND1.get(label)
    lc, fc, dc, cc = funnel_counts(fn_lit), funnel_counts(fn_fresh), funnel_counts(fn_dec), funnel_counts(fn_cls_r1)
    zc, c2 = funnel_counts(fn_r1z), funnel_counts(fn_cls)
    counts["literal"], counts["fresh"], counts["decided"], counts["classes_only"] = lc, fc, dc, cc
    counts["classes_open_side_same_bar"], counts["classes_only_zone_rule"] = zc, c2
    if r1 is None:
        print(f"   no round-1 row for {label!r}")
    else:
        fin = tuple(counts[s] for s in STAGES)
        print(f"   {'stage':<14} {'round 1':>8} {'literal':>8} {'fresh':>8} {'decided':>8} {'final':>8} {'fin/r1':>8} {'fin/dec':>8}")
        fmt = lambda x, y: f"{x / y:.2f}" if y else "inf"                        # noqa: E731
        print(f"   {'sessions':<14} {n:>8} {lc['sessions']:>8} {fc['sessions']:>8} {dc['sessions']:>8} {n:>8} {'':>8} {'':>8}")
        for stage, a, w in zip(STAGES, r1, fin):
            print(f"   {stage:<14} {a:>8} {lc[stage]:>8} {fc[stage]:>8} {dc[stage]:>8} {w:>8} {fmt(w, a):>8} {fmt(w, dc[stage]):>8}")
        conv = lambda seq: "  ".join(f"{s}->{t} {(seq[i + 1] / seq[i] if seq[i] else math.nan):.2f}"   # noqa: E731
                                     for i, (s, t) in enumerate(zip(STAGES, STAGES[1:])))
        print("   stage-to-stage conversion, round 1:       " + conv(r1))
        print("   stage-to-stage conversion, literal wide:  " + conv(tuple(lc[s] for s in STAGES)))
        print("   stage-to-stage conversion, literal fresh: " + conv(tuple(fc[s] for s in STAGES)))
        print("   stage-to-stage conversion, decided:       " + conv(tuple(dc[s] for s in STAGES)))
        print("   stage-to-stage conversion, final:         " + conv(fin))
        print("   attribution, same-bar zone convention, open-side test off (= the class correction alone): "
              + "  ".join(f"{s} {cc[s]}" for s in STAGES))
        print("   attribution, same-bar zone convention, open-side test on (= final but for the zone-in-play rule): "
              + "  ".join(f"{s} {zc[s]}" for s in STAGES))
        print("   attribution, zone-in-play rule, open-side test off (the run the 3b check uses):              "
              + "  ".join(f"{s} {c2[s]}" for s in STAGES))
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
    frame = pd.DataFrame([format_row(r, fn.clock.minutes, fn.decided, fn.final) for r in fn.rows])
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

    def take(cands, hm, what, need=1, reuse=False):
        got = 0
        for r in cands:
            if (r["day"] in used and not reuse) or got >= need:
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

    if fn.final:
        # Section 2.5 item 1. (a) a bar the open-side test refused, cut at that
        # bar: the refusal must vanish with it. (b) a day that was refused and
        # then filled on a later bar, cut at the fill bar: the refusals stay,
        # the fill must vanish. (b) prefers a day that filled later ON EQ, and
        # may be the same day as (a) when only one such day exists.
        refused = [r for r in rows if r["eq_refused"]]
        ref_eq = [r for r in refused if r["fill_idx"] is not None and r["fill_zone"].kind == "eq"]
        ref_other = [r for r in refused if r["fill_idx"] is not None and r["fill_zone"].kind != "eq"]
        ref_none = [r for r in refused if r["fill_idx"] is None]
        n0 = len(cuts)
        for r in (ref_none + ref_other + ref_eq)[:1]:
            take([r], int(minutes[r["eq_refused"][0]]),
                 "at a bar the open-side test refused: the refusal is not yet in the frame and must vanish")
        for r in (ref_eq + ref_other)[:1]:
            take([r], int(minutes[r["fill_idx"]]),
                 f"at the bar that filled after {len(r['eq_refused'])} refused bar(s) "
                 f"({after_refusal(r)}): the refusals stay, the fill must vanish", reuse=True)
        # (c) the zone-in-play rule: a day that filled on a bar which itself
        # completed a fresher zone, cut at the bar AFTER the fill bar: the
        # fill, on the older zone, must stay exactly as the full run has it.
        same_bar = [r for r in rows if r["fill_idx"] is not None and r.get("fill_bar_new_zone")]
        for r in same_bar[:1]:
            if r["fill_idx"] + 1 < len(minutes) and day[r["fill_idx"] + 1] == day[r["fill_idx"]]:
                take([r], int(minutes[r["fill_idx"] + 1]),
                     f"after a fill bar that itself completed a fresher {r['fill_bar_new_zone']}: the fill on the "
                     f"older {r['fill_zone'].kind} must stay", reuse=True)
        if not same_bar:
            print("   no fill on this frame sits on a bar that itself completed a fresher zone, so no such cut exists")
        want += len(cuts) - n0
        if not refused:
            print("   no day on this frame has a bar refused by the open-side test, so no refusal cut exists")
        elif not (ref_eq + ref_other):
            print("   no refused day on this frame filled on a later bar, so no later-fill cut exists")

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
    full = [format_row(r, fn.clock.minutes, fn.decided, fn.final) for r in fn.rows]
    all_ok = True
    cuts = pick_cuts(fn)
    n_cuts = len(cuts)
    for cut, what in cuts:
        part = WideFunnel(fn.df.iloc[:cut], fn.bin_ticks, fn.hvn_frac, fn.hvn_max, mode=fn.mode,
                          eq_open_side=fn.eq_open_side, zone_same_bar=fn.zone_same_bar)
        prows = [format_row(r, part.clock.minutes, fn.decided, fn.final) for r in part.rows]
        cut_day = fn.clock.day[cut]
        k = next(j for j, r in enumerate(fn.rows) if r["day"] == pd.Timestamp(cut_day))
        earlier_ok = prows[:k] == full[:k] and len(prows) == k + 1
        expect = format_row(as_of(fn.rows[k], cut), fn.clock.minutes, fn.decided, fn.final)
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
    words = {6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}.get(n_cuts, str(n_cuts))
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
    ap.add_argument("--mode", choices=MODES, default="final",
                    help="final: DESIGN-tjr-human.md section 2.5 (the default); decided: section 2.3, which reproduces "
                         "results/tjr_intraday/part2_decided_funnel.txt byte for byte; literal: section 2.1 as built, "
                         "which reproduces results/tjr_intraday/part2_wide_funnel.txt byte for byte")
    ap.add_argument("--poc-bin-ticks", type=int, default=4)
    ap.add_argument("--hvn-frac", type=float, default=0.5)
    ap.add_argument("--hvn-max", type=int, default=6)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-proof", action="store_true", help="skip the truncation proof")
    args = ap.parse_args(argv)
    if args.mode == "literal":
        return run_literal(args)
    # Decided / final mode: the printout goes to stdout AND to <out-dir>/part2_<mode>_funnel.txt.
    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"part2_{args.mode}_funnel.txt")
    real = sys.stdout
    with open(report_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(real, fh)
        try:
            rc = run_final(args) if args.mode == "final" else run_decided(args)
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


def print_definitions_final() -> None:
    """Every place section 2.5 is silent, and the reading this build took. Printed
    after the decided block, which still holds wherever it is not superseded
    below. The orchestrator writes section 2.6; this block is the source for it."""
    print("definitions as built (final mode) -- section 2.5 on top of the block above; where 2.5 is silent, the choice made here")
    for line in (
        "scope: final = decided with three changes and nothing else. Everything in the block above stands EXCEPT the",
        "      `sweep/classes` line's five classes (now three) and the `targets` lines (now opposing-named). The ifvg rule,",
        "      the eq midpoint, ATR, excursion, formed/usable stamps, the zone rank, the fill rule for fvg/ob/breaker and",
        "      the first-one-sided-bar rule are the same code. One more difference, found in review and not in 2.5: the",
        "      zone in play on a bar (`zone in play` below).",
        "zone in play: on entry-window bar i the zone in play is the freshest (same rank) among zones with usable < i, i.e.",
        "      known at the close of bar i-1. Decided / literal / round 1 take the freshest of ALL zones including one bar i",
        "      itself completes, and then skip bar i because a fill needs i > usable: that reads bar i's close to withhold a",
        "      fill that an order resting on the older zone takes inside bar i (an eq limit rests at EQ from the open). It is",
        "      intrabar lookahead no truncation test can detect, so final mode does not do it; decided and literal keep it",
        "      because they must reproduce their committed printouts. A zone completed on bar i is in play from bar i+1, as",
        "      before. The count under round 1's convention is printed in 3f and in the summary (zone_same_bar=True, a",
        "      diagnostic run). The 5-minute funnel cannot order events inside a bar; tjr_human reads 1-minute bars (section 9.2).",
        "eq/open-side: judged on the bar that would otherwise be read for a fill: an entry-window bar (09:50..10:05),",
        "      strictly after the confirmation bar and after eq's usable bar, the day not yet filled, eq the freshest zone.",
        "      Far side is STRICT: long open > EQ, short open < EQ; open == EQ is refused. A bar that opens at or through",
        "      EQ touches it by construction (low <= open), so `refused` = such a bar opened at or through EQ; a bar that",
        "      opens on the far side and does not reach EQ is neither a fill nor a refusal. The open is the open of the",
        "      completed bar being read. An eq fill is therefore always at EQ exactly (min(EQ, open) = EQ).",
        "eq/in play: eq is in play while it is the freshest zone by the unchanged rank (formed, then ob<eq<breaker<fvg).",
        "      A refused bar fills nothing: no older zone (leg fvg, ob, earlier breaker) is consulted on it. A refusal does",
        "      not spend the setup and refusals are not capped: every later entry-window bar is judged afresh. A gap that",
        "      completes, or a breaker violated, on bar i out-ranks eq from bar i+1 (on bar i itself eq is still in play: it",
        "      fills or is refused there) and can fill from bar i+1; that is `filled on another zone that became freshest`,",
        "      not a fall-through. The open-side test applies to eq ONLY: fvg/ob/breaker still fill on a bar that opens inside or",
        "      through the zone (entry = the open), as in decided mode and round 1. While eq is still waiting for its",
        "      excursion it is not in the zone list, so another zone is in play then, as in decided mode (never happens here).",
        "eq/log: per setup `eq_refused_bars`, `eq_refused_n` (0 on a confirmed day with none) and `eq_after_refusal` in",
        "      {filled_later_on_eq, filled_on_other_zone:<kind>, no_fill}. `eq_fill_open_far_side` is 1 on every eq fill now.",
        "classes: SESSION = every level whose type is ASIA, LON or PD; H1 = H1_SH/SL; H4 = H4_SH/SL. Inside SESSION the test",
        "      is round 1's sweep_of_levels on the six: any session low AND any session high in one wick -> SESSION abstains,",
        "      whichever sessions they belong to; otherwise the deepest breached session level, closed back inside, returns.",
        "      An abstaining SESSION does not veto a swing class (section 2.3's combination rule, unchanged): H1 or H4 returning",
        "      alone sweeps the bar. Those bars are counted in 3c, split by whether SESSION abstained (round 1 skipped the bar)",
        "      or returned nothing (round 1 did not sweep it), and by whether round 1 swept that day at all.",
        "classes/level: the setup's level is still the deepest breached level of the sweep side among the RETURNING classes,",
        "      so on a bar round 1 also swept, a deeper H1/H4 level of the same side that returned is the setup's level and",
        "      the sweep extreme is unchanged (it is the bar's low/high either way). `class_states` records each class on the",
        "      sweep bar: returned / abstained / breached_no_close_back / untouched / absent (no level of the class that day).",
        "targets: candidates are the day's levels frozen at 09:25, by NAME for the directional ones (long: ASIA_H LON_H PDH",
        "      H1_SH H4_SH; short: ASIA_L LON_L PDL H1_SL H4_SL) plus POC_PREV and every HVN_* whichever side of the 09:25",
        "      close it rested on; a candidate qualifies iff strictly beyond the entry (long: price > entry). Equal price =",
        "      equal after rounding to 1e-6; the names are joined with + in the order ASIA_H .. H4_SL, POC_PREV, HVN_1..6 and",
        "      count once, under `tie`. T2 is the next distinct PRICE beyond T1 from the same set. No candidate -> no T1.",
        "      No `already taken` bookkeeping: a level the day traded through before the fill is still a target; how often",
        "      that is so for T1 is reported (`final_t1_already_traded`: bars 09:30 .. the bar before the fill bar).",
        "      The decided CSV columns t1 / t2 / t1_dir / t1_vol keep section 2.3's by-price reading; the final ones are final_*.",
        "geometry: ATR = the confirmation-bar ATR(14) already frozen for eq (`eq_atr`). wick stop = sweep extreme - side x",
        f"      {WICK_STOP_ATR} x ATR; R = side x (entry - stop); reward-to-risk = side x (target - entry) / R. Round 1's own engine",
        "      read the ATR at the FILL bar (tjr_intraday: cb.atr[ctx.last[i]]); the task fixed the confirmation bar, and the",
        "      stop distance under round 1's reading is printed next to it. Nothing after the fill bar is read: no stop is",
        "      simulated, no target is tested, no outcome exists. The funnel has NO invalidation rule (section 2 has none):",
        "      a setup whose sweep extreme or wick stop traded after the sweep still fills if a zone is touched. Such fills",
        "      are counted (`wick_stop_traded_before_fill`), and a fill whose entry is already beyond the stop has no R and",
        "      is left out of the reward-to-risk rows (round 1's engine books that as `no_risk`).",
        "side-by-side: the literal, fresh and decided columns are runs of this script in the same process; they are the",
        "      committed part2_wide_funnel.txt / part2_decided_funnel.txt numbers (--mode literal / decided reproduce them).",
        "      `attribution` rows are final mode with the open-side test and / or the zone-in-play rule switched off",
        "      (WideFunnel(eq_open_side=False), WideFunnel(zone_same_bar=True)): diagnostic runs, not modes, not specs.",
        "eq/co-occurrence: at an eq fill, the other zone types among the zones known before the fill bar (usable < i).",
        "      `fill_bar_new_zone` names the fresher zone the fill bar itself completed, when it did.",
        "proof: the seven decided cuts, plus (a) a cut at a bar the open-side test refused, (b) a cut at the later bar",
        "      that then filled, and (c) a cut at the bar after a fill bar that itself completed a fresher zone (the fill on",
        "      the older zone must stay), when such days exist on the frame. (c) shows the fill does not read bar i+1; it",
        "      cannot show the intrabar point, which is an argument about bar i's close, not a truncation.",
    ):
        print("   " + line)
    print()


def run_final(args) -> int:
    """Section 2.5: the last re-run of the wide funnel, under the three corrections."""
    print("TJR wide funnel, FINAL (DESIGN-tjr-human.md section 2.5: eq only on a bar that opens on the far side of EQ; "
          "level classes SESSION / H1 / H4; targets = nearest opposing-named level or POC/HVN beyond the entry; "
          "the zone in play on a bar is read from the zones known before it). "
          "Funnel only: no stop simulated, no target tested, no exit, Sharpe, gate or verdict.")
    print(f"params: poc_bin_ticks={args.poc_bin_ticks} (bin ${TICK * args.poc_bin_ticks:.2f})  hvn_frac={args.hvn_frac}  "
          f"hvn_max={args.hvn_max}  composite={COMPOSITE_SESSIONS} sessions  swings left/right={SWING_LEFT}/{SWING_RIGHT}  "
          f"ote={OTE}  eq excursion={EQ_EXCURSION_ATR} x ATR(14)  wick stop buffer={WICK_STOP_ATR} x ATR(14) (geometry only)")
    print()
    summary = {}
    proofs = {}
    funnels = []
    for path in args.csv:
        label = os.path.basename(path).split("_")[0].upper()
        df = data.load_csv(path)
        base = dict(bin_ticks=args.poc_bin_ticks, hvn_frac=args.hvn_frac, hvn_max=args.hvn_max)
        fn = WideFunnel(df, mode="final", **base)
        if fn.clock is None:
            print(f"=== {label}  {path}: no 09:30 bar in the frame, nothing to do ===")
            continue
        fn6 = WideFunnel(df, level_types=("ASIA", "LON", "PD"), mode="literal", **base)
        fn_lit = WideFunnel(df, mode="literal", **base)
        fn_fresh = WideFunnel(df, ifvg_fresh=True, mode="literal", **base)
        fn_dec = WideFunnel(df, mode="decided", **base)
        fn_cls = WideFunnel(df, mode="final", eq_open_side=False, **base)
        fn_cls_r1 = WideFunnel(df, mode="final", eq_open_side=False, zone_same_bar=True, **base)
        fn_r1z = WideFunnel(df, mode="final", zone_same_bar=True, **base)
        summary[label] = report_final(label, path, fn, fn6, M.day_log(df), fn_lit, fn_fresh, fn_dec, fn_cls,
                                      fn_cls_r1, fn_r1z)
        out = os.path.join(args.out_dir, f"part2_final_days_{label}.csv")
        write_days_csv(fn, out)
        print(f"   day-level rows written: {out} ({len(fn.rows)} rows)")
        print()
        funnels.append((label, fn))
    if not args.no_proof:
        for label, fn in funnels:
            proofs[label] = prove_causality(label, fn)

    print("6. synthetic daily data (quantlab.data.synthetic): the no-09:30 path")
    syn = WideFunnel(data.synthetic(), args.poc_bin_ticks, args.hvn_frac, args.hvn_max, mode="final")
    print(f"   clock is None: {syn.clock is None}; rows: {len(syn.rows)} -> all-flat, no exception")
    print()
    print_definitions_as_built()
    print_definitions_final()
    print("summary")
    total = 0
    for label, c in summary.items():
        r1 = ROUND1.get(label, (None,) * 4)
        r1c = dict(sessions=c["sessions"], sweeps=r1[0], confirmations=r1[1], zones=r1[2], fills=r1[3])
        for name, cc in (("round 1", r1c), ("literal wide", c["literal"]), ("literal fresh-ifvg", c["fresh"]),
                         ("decided", c["decided"]), ("final", c)):
            print(f"   {label} ({name}): sessions {cc['sessions']} -> sweeps {cc['sweeps']} -> confirmations {cc['confirmations']} "
                  f"-> zones {cc['zones']} -> fills {cc['fills']}"
                  + (f"   causality: {'ok' if proofs[label] else 'MISMATCH'}" if name == "final" and label in proofs else ""))
        e, k, t = c["eq"], c["classes"], c["targets"]
        total += c["fills"]
        print(f"   {label} eq: {e['refused_bars']} bars refused by the open-side test on {e['refused_setups']} setups "
              f"(then filled on eq {e['refused_then_filled_on_eq']}, on another zone {e['refused_then_filled_on_other_zone']}, "
              f"no fill {e['refused_then_no_fill']}); eq fills {e['eq_fills']} of {e['fills']}")
        z = c["zone_in_play"]
        print(f"   {label} zone in play: read from before the bar; under round 1's same-bar convention the fills would be "
              f"{z['fills_same_bar_convention']} (final {z['fills_final']}); days whose fill differs: "
              f"{', '.join(z['days_fill_differs']) or 'none'}")
        print(f"   {label} classes: sweeps {k['sweeps']}; disagreement bars {k['disagreement_bars']}; abstention-only bars "
              f"{k['abstention_only_bars']}; round-1 sweep days lost {k['round1_days_lost']}, days only H1/H4 made "
              f"{k['days_gained_vs_round1_only_H1_H4']}; swing class alone with SESSION abstaining "
              f"{k['session_abstained_swing_returned']}, with SESSION returning nothing {k['session_returned_nothing_swing_returned']}")
        print(f"   {label} targets: no T1 on {t['no_t1']} of {t['fills']} fills, no T2 on {t['no_t2']}; "
              f"T1 closer than 1.0 R on {t['t1_closer_than_1R']} of {t['t1_rr_n']}")
    if summary:
        n_sess = max(c["sessions"] for c in summary.values())
        print(f"   fills, all instruments: {total} over {n_sess} sessions ({total / n_sess:.2f} a session); "
              f"100 filled trades at that rate: {100 * n_sess / total:.0f} sessions" if total else
              f"   fills, all instruments: 0 over {n_sess} sessions")
    return 0 if all(proofs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

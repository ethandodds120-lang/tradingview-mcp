"""The ten mechanical exits of DESIGN-tjr-human.md section 5, and the random stop.

Every filled entry (and every skipped one) is replayed, after its session has
closed, under each of:

    wick_fixed                                   the wick stop, never moved
    wick_be_1r  wick_atr1.0  wick_atr1.5         the five section-1 trailing rules,
    wick_swing  wick_hybrid                      each starting from the wick stop
    fixed_atr1.0  fixed_atr1.5  fixed_atr2.0     round 2's fixed widths
    fixed_session
    random_stop                                  a fixed stop at U[0.5, 2.0] x ATR, seeded by the trade id

Semantics are `tjr_trailing_benchmark.replay`'s, on the 1-minute bars with the
5-minute context (`tjr_intraday.build_context(frame, 1, 14)`) for ATR and
swings: the entry minute is tested against the initial stop only; on every
later minute j the stop is first updated from bars <= j-1 (the ATR and the
swings at the newest context bin completed by minute j-1) and only ever
tightens, then minute j is tested against the stop (a gap through it fills at
the open), then against the target (T1, the whole position), then the 15:55
flat (the close of the minute stamped 15:55). A session with no 15:55 minute
exits at its last close.

R is against each exit's OWN initial risk and NET of the section 9.2 costs:
ES 1 bp, NQ 0.5 bp, round trip, charged once on the entry notional:

    r = (side * (exit - entry) - cost_bps * 1e-4 * entry) / (side * (entry - stop0))

Causal replay, not a forecast: nothing here runs before the session is over,
and nothing here knows a future bar when it moves a stop.

NO OUTCOMES ON MARKET DATA (section 9.8): this module computes; it prints
nothing. Whoever calls it on ES/NQ files must not print R either.
"""

from __future__ import annotations

import bisect
import hashlib
import math

import numpy as np
import pandas as pd

from ..strategies.predictive import tjr_intraday as M
from ..strategies.predictive.primitives import confirmed_swings
from .detector import ATR_LEN, SWING_LEFT, SWING_RIGHT, et_clock, hhmm

#: name -> (which candidate stop of the fill it starts from, the rule that moves it)
MECHANICAL_EXITS = {
    "wick_fixed": ("wick", "fixed"),
    "wick_be_1r": ("wick", "be_1r"),
    "wick_atr1.0": ("wick", "atr1.0"),
    "wick_atr1.5": ("wick", "atr1.5"),
    "wick_swing": ("wick", "swing"),
    "wick_hybrid": ("wick", "hybrid"),
    "fixed_atr1.0": ("atr1.0", "fixed"),
    "fixed_atr1.5": ("atr1.5", "fixed"),
    "fixed_atr2.0": ("atr2.0", "fixed"),
    "fixed_session": ("session", "fixed"),
}
EXIT_NAMES = tuple(MECHANICAL_EXITS)
RULES = ("fixed", "be_1r", "atr1.0", "atr1.5", "swing", "hybrid")
RANDOM_EXIT = "random_stop"
RANDOM_RANGE = (0.5, 2.0)
FLAT_MINUTES = 15 * 60 + 55
#: round-trip cost in basis points of notional (DESIGN-tjr-intraday.md section 8)
COSTS_BPS = {"ES": 1.0, "NQ": 0.5}


def cost_bps(instrument: str) -> float:
    """ES 1 bp, NQ 0.5 bp. Accepts the root with or without a contract suffix or an
    exchange prefix (ES, ES1!, CME_MINI:NQ1!); anything else is refused, not guessed."""
    root = str(instrument).upper().split(":")[-1]
    for key, bps in COSTS_BPS.items():
        if root.startswith(key):
            return bps
    raise ValueError(f"tjr_human.exits: no cost for instrument {instrument!r}; known: {sorted(COSTS_BPS)}")


def cost_points(instrument: str, entry: float) -> float:
    """The round trip in price points: bps x 1e-4 x the entry notional."""
    return cost_bps(instrument) * 1e-4 * float(entry)


def net_r(side: int, entry: float, exit_price: float, stop0: float, cost_pts: float) -> float:
    risk = side * (entry - stop0)
    return (side * (exit_price - entry) - cost_pts) / risk


def random_multiple(trade_id: str) -> float:
    """U[0.5, 2.0], seeded by the trade id through SHA-256 — the same in every
    process and on every machine (Python's own hash() is salted per process)."""
    seed = int.from_bytes(hashlib.sha256(str(trade_id).encode("utf-8")).digest()[:8], "big")
    return float(np.random.default_rng(seed).uniform(*RANDOM_RANGE))


class Session:
    """The 1-minute bars a replay walks, with the context it may consult.

    `bars` may carry history before the session (it should: the context ATR is
    an exponential average and wants its warm-up; after a few hundred context
    bars the difference is far below a tick) and may run past it — a replay
    never leaves the fill's own trading day.
    """

    def __init__(self, bars: pd.DataFrame):
        if not isinstance(bars.index, pd.DatetimeIndex) or len(bars) == 0:
            raise ValueError("tjr_human.exits: needs a non-empty 1-minute frame with a DatetimeIndex")
        self.index = bars.index
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.minutes, self.day = et_clock(bars.index)
        self.ctx = M.build_context(bars[["open", "high", "low", "close"]], 1, ATR_LEN)
        self.atr = self.ctx.bars.atr
        highs, lows = confirmed_swings(self.ctx.bars, SWING_LEFT, SWING_RIGHT)
        self.sw_hi, self.sw_lo = highs, lows
        self.sw_hi_at = [s.confirmed_at for s in highs]
        self.sw_lo_at = [s.confirmed_at for s in lows]

    def loc(self, stamp) -> int:
        return int(self.index.get_loc(pd.Timestamp(stamp)))

    def iso(self, i: int) -> str:
        return str(np.datetime_as_string(self.index[i].to_datetime64(), unit="s"))


def replay_exit(sess: Session, side: int, entry: float, i0: int, stop0: float,
                target: float | None, rule: str = "fixed") -> dict:
    """One entry under one rule -> exit_idx, exit, reason, r_gross, armed, stop (the last one), mae_r, mfe_r.

    reason: stop (the initial stop), trail (a moved stop), target, flat (15:55),
    session_end (the day ran out first), no_risk (entry not beyond the stop).
    """
    if rule not in RULES:
        raise ValueError(f"tjr_human.exits: rule must be one of {RULES}, got {rule!r}")
    o, h, l, c = sess.o, sess.h, sess.l, sess.c
    side, entry, stop0 = int(side), float(entry), float(stop0)
    risk = side * (entry - stop0)
    if not risk > 0:
        return {"exit_idx": i0, "exit": entry, "reason": "no_risk", "r_gross": None, "armed": False,
                "stop": stop0, "mae_r": None, "mfe_r": None}
    day = sess.day[i0]
    n = len(c)

    def fav(p: float) -> float:
        return float((p - entry) * side / risk)

    def done(j: int, px: float, reason: str, stop: float, armed: bool) -> dict:
        # Excursions the trade lived through. On a stop, the exit minute's range
        # beyond the fill was not held, and its favourable side may have come
        # after the stop: only completed minutes before it count as favourable.
        lo_all = float(l[i0:j + 1].min()) if side > 0 else float(h[i0:j + 1].max())
        if reason in ("stop", "trail"):
            # the minutes BEFORE the exit minute were lived through in full (a trailed stop
            # can sit above an earlier low); the exit minute's adverse side is floored at the fill
            prior = (float(l[i0:j].min()) if side > 0 else float(h[i0:j].max())) if j > i0 else px
            worst = min(prior, px) if side > 0 else max(prior, px)
            best = (float(h[i0:j].max()) if side > 0 else float(l[i0:j].min())) if j > i0 else entry
        else:
            worst = lo_all
            best = float(h[i0:j + 1].max()) if side > 0 else float(l[i0:j + 1].min())
            if reason == "target":
                best = px
        return {"exit_idx": int(j), "exit": float(px), "reason": reason, "r_gross": fav(px), "armed": bool(armed),
                "stop": float(stop), "mae_r": min(0.0, fav(worst)), "mfe_r": max(0.0, fav(best))}

    # the entry minute: the initial stop and nothing else
    if (side > 0 and l[i0] <= stop0) or (side < 0 and h[i0] >= stop0):
        return done(i0, stop0, "stop", stop0, False)

    stop, armed = stop0, False
    best = h[i0] if side > 0 else l[i0]
    for j in range(i0 + 1, n):
        if sess.day[j] != day:
            return done(j - 1, c[j - 1], "session_end", stop, armed)
        # ---- the stop for minute j, from bars <= j-1 ----
        best = max(best, h[j - 1]) if side > 0 else min(best, l[j - 1])
        if fav(best) >= 1.0:
            armed = True
        k = int(sess.ctx.last[j - 1])
        atr = float(sess.atr[k]) if k >= 0 else math.nan
        if rule == "fixed":
            want = stop0
        elif rule == "be_1r":
            want = entry if armed else stop0
        elif rule in ("atr1.0", "atr1.5"):
            mult = 1.0 if rule == "atr1.0" else 1.5
            want = best - side * mult * atr
        elif rule == "swing":
            pool, at = (sess.sw_lo, sess.sw_lo_at) if side > 0 else (sess.sw_hi, sess.sw_hi_at)
            q = bisect.bisect_right(at, k)
            want = pool[q - 1].price if q else stop0
        else:                                           # hybrid: nothing until +1 R, then max(breakeven, 1 ATR trail)
            if armed:
                trail = best - side * 1.0 * atr
                want = max(entry, trail) if side > 0 else min(entry, trail)
            else:
                want = stop0
        if math.isnan(want):
            want = stop
        stop = max(stop, want) if side > 0 else min(stop, want)      # only ever tightens
        # ---- minute j: the stop, then the target, then the flat ----
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            px = min(stop, o[j]) if side > 0 else max(stop, o[j])     # a gap through the stop fills at the open
            return done(j, px, "stop" if stop == stop0 else "trail", stop, armed)
        if target is not None and ((side > 0 and h[j] >= target) or (side < 0 and l[j] <= target)):
            return done(j, float(target), "target", stop, armed)
        if sess.minutes[j] >= FLAT_MINUTES:
            return done(j, c[j], "flat", stop, armed)
    return done(n - 1, c[n - 1], "session_end", stop, armed)


def _first_print(sess: Session, side: int, i0: int, last: int, price: float | None) -> dict:
    """Did `price` print between the entry minute and `last` (inclusive), and when?"""
    if price is None:
        return {"price": None, "printed": None, "time": None, "et": None}
    span = sess.h[i0:last + 1] >= price if side > 0 else sess.l[i0:last + 1] <= price
    if not span.any():
        return {"price": float(price), "printed": False, "time": None, "et": None}
    j = i0 + int(np.argmax(span))
    return {"price": float(price), "printed": True, "time": sess.iso(j), "et": hhmm(sess.minutes[j])}


def _flat_index(sess: Session, i0: int) -> int:
    """The minute the day goes flat on: the first stamped >= 15:55, else the day's last."""
    day = sess.day[i0]
    j = i0
    for j in range(i0, len(sess.c)):
        if sess.day[j] != day:
            return j - 1
        if sess.minutes[j] >= FLAT_MINUTES:
            return j
    return j


def benchmark_record(bars: pd.DataFrame | Session, fill: dict, instrument: str | None = None) -> dict:
    """The `benchmarks` journal record of section 9.4 for one entry.

    `bars`: 1-minute bars containing the fill's minute and the rest of its
    session (history before it is welcome, see `Session`), or a `Session`
    already built from them. `fill`: a detector `fill` event (or a dict with
    the same keys: setup_id, instrument, side, entry, bar_time, atr, stops,
    t1, t2, targets, target_rule). Returns plain JSON-safe data; the journal
    adds its own stamp.
    """
    sess = bars if isinstance(bars, Session) else Session(bars)
    inst = instrument or fill["instrument"]
    side, entry = int(fill["side"]), float(fill["entry"])
    i0 = sess.loc(fill["bar_time"])
    atr = float(fill["atr"])
    t1 = fill.get("t1")
    target = float(t1["price"]) if t1 else None
    cpts = cost_points(inst, entry)

    def one(stop0: float, rule: str) -> dict:
        out = replay_exit(sess, side, entry, i0, stop0, target, rule)
        risk = side * (entry - stop0)
        x = out["exit_idx"]
        rec = {"stop0": float(stop0), "risk": float(risk), "rule": rule, "exit": out["exit"],
               "exit_time": sess.iso(x), "exit_et": hhmm(sess.minutes[x]), "reason": out["reason"],
               "minutes": int(x - i0), "armed": out["armed"], "last_stop": out["stop"],
               "r_gross": out["r_gross"],
               "r": None if out["r_gross"] is None else net_r(side, entry, out["exit"], stop0, cpts),
               "mae_r": out["mae_r"], "mfe_r": out["mfe_r"]}
        return rec

    exits = {name: one(float(fill["stops"][key]), rule) for name, (key, rule) in MECHANICAL_EXITS.items()}
    mult = random_multiple(fill["setup_id"])
    rnd = one(entry - side * mult * atr, "fixed")
    rnd["atr_multiple"] = mult

    flat = _flat_index(sess, i0)
    hi, lo = float(sess.h[i0:flat + 1].max()), float(sess.l[i0:flat + 1].min())
    wick_risk = side * (entry - float(fill["stops"]["wick"]))
    mfe_pts = max(0.0, side * ((hi if side > 0 else lo) - entry))
    mae_pts = min(0.0, side * ((lo if side > 0 else hi) - entry))
    printed = {}
    for rule_name, pair in (fill.get("targets") or {}).items():
        printed[rule_name] = {k: dict(_first_print(sess, side, i0, flat, (pair[k] or {}).get("price")),
                                      names=(pair[k] or {}).get("names")) for k in ("t1", "t2")}
    t2 = fill.get("t2")
    return {
        "kind": "benchmarks", "setup_id": fill["setup_id"], "instrument": inst, "day": fill.get("day"),
        "side": side, "entry": entry, "entry_time": fill["bar_time"], "atr": atr,
        "target_rule": fill.get("target_rule"), "cost_bps": cost_bps(inst), "cost_points": cpts,
        "exits": exits, RANDOM_EXIT: rnd,
        "session": {"through": sess.iso(flat), "through_et": hhmm(sess.minutes[flat]),
                    "mae_points": mae_pts, "mfe_points": mfe_pts,
                    "mae_r_wick": mae_pts / wick_risk if wick_risk > 0 else None,
                    "mfe_r_wick": mfe_pts / wick_risk if wick_risk > 0 else None},
        "t1": dict(_first_print(sess, side, i0, flat, target), names=t1["names"] if t1 else None),
        "t2": dict(_first_print(sess, side, i0, flat, float(t2["price"]) if t2 else None),
                   names=t2["names"] if t2 else None),
        "targets_printed": printed,
    }

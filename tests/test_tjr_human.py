"""Tests for quantlab.tjr_human (DESIGN-tjr-human.md section 9.9) — detector, exits, trade, commands, journal, report,
runner (replay feed, live feed against a fake tv CLI, store, ARMED), chart, and the tjr_human.py CLI.

Runs standalone (`python tests/test_tjr_human.py`) in the style of the other
test files: PASS/FAIL lines, "N passed, M failed", exit code.

NO OUTCOMES ON MARKET DATA (section 9.8). On the TradingView files these tests
assert causality, equivalence and counts of events and print nothing else: no
R, no win rate, no P&L, no per-exit totals. Exit arithmetic is tested on
hand-built and synthetic minutes only. `benchmark_record` is run on market
fills for its SHAPE; its numbers are never printed or compared.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlab.data import load_csv  # noqa: E402
from quantlab.strategies.predictive import primitives as P  # noqa: E402
from quantlab.strategies.predictive import tjr_intraday as M  # noqa: E402
from quantlab.tjr_human import detector as D  # noqa: E402
from quantlab.tjr_human import exits as X  # noqa: E402
from quantlab.tjr_human import N_TRIALS  # noqa: E402

ET = M.ET
DAY = pd.Timestamp("2026-07-15")            # a Wednesday, EDT
CSV = {s: os.path.join(ROOT, "data", f"{s}_1min_tv.csv") for s in ("NQ", "ES")}
INERT = (100.0, 100.5, 99.5, 100.0)
HIGH = (102.0, 102.1, 101.7, 102.0)

_cache: dict = {}


def market(sym: str) -> pd.DataFrame:
    if ("df", sym) not in _cache:
        _cache[("df", sym)] = load_csv(CSV[sym])
    return _cache[("df", sym)]


def market_events(sym: str) -> list[dict]:
    if ("ev", sym) not in _cache:
        _cache[("ev", sym)] = D.detect(market(sym), sym)
    return _cache[("ev", sym)]


# ────────────────────────────── hand-built 1-minute days ──────────────────────────────

def build_day(overrides: dict | None = None, extra_days: int = 0, volume: float = 0.0,
              fill_runs: dict | None = None) -> pd.DataFrame:
    """Sessions D-1 and D on 1-minute bars (plus `extra_days` inert ones after D).
    Keys are 'HH:MM' on D or '-1 HH:MM' the calendar day before; values (o, h, l, c).
    `fill_runs` maps ('HH:MM', 'HH:MM') inclusive ranges on D to one bar."""
    start = pd.Timestamp(f"{(DAY - pd.Timedelta(days=2)).date()} 18:00", tz=ET)
    end = pd.Timestamp(f"{(DAY + pd.Timedelta(days=extra_days)).date()} 16:59", tz=ET)
    ix = pd.date_range(start, end, freq="1min", tz=ET)
    ix = ix[ix.hour != 17]
    df = pd.DataFrame([INERT] * len(ix), columns=["open", "high", "low", "close"], index=ix)
    for (a, b), ohlc in (fill_runs or {}).items():
        lo = pd.Timestamp(f"{DAY.date()} {a}", tz=ET)
        hi = pd.Timestamp(f"{DAY.date()} {b}", tz=ET)
        df.loc[lo:hi, ["open", "high", "low", "close"]] = ohlc
    for key, ohlc in (overrides or {}).items():
        off, hm = (key.split(" ") if " " in key else ("0", key))
        stamp = pd.Timestamp(f"{(DAY + pd.Timedelta(days=int(off))).date()} {hm}", tz=ET)
        assert stamp in df.index, key
        df.loc[stamp, ["open", "high", "low", "close"]] = ohlc
    df["volume"] = float(volume)
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "date"
    return df


def mirror(d: dict) -> dict:
    """The same day upside down around 100: a long setup becomes the short one."""
    return {k: (200 - o, 200 - l, 200 - h, 200 - c) for k, (o, h, l, c) in d.items()}


def utc(hm: str, day_offset: int = 0) -> str:
    ts = pd.Timestamp(f"{(DAY + pd.Timedelta(days=day_offset)).date()} {hm}", tz=ET)
    return ts.tz_convert("UTC").tz_localize(None).isoformat()


# Levels: PDH 105 / PDL 95, ASIA_H 103 / ASIA_L 98, LON_H 102 / LON_L 97; on the
# hourly bins H1_SH 102 / H1_SL 98 (nearest the 09:29 close of 100), on the
# 4-hour bins H4_SH 105 / H4_SL 95 (the 22:00 bin's swing is not confirmed by 09:29).
BASE = {
    "-1 10:00": (100.0, 105.0, 95.0, 100.0),
    "-1 22:00": (100.0, 103.0, 98.0, 100.0),
    "05:00":    (100.0, 102.0, 97.0, 100.0),
    "09:17":    (100.0, 101.5, 99.8, 100.0),     # 5-minute swing high 101.5, confirmed by the 09:25 bin
    # the 09:35 bin: wick 96.5 under ASIA_L, LON_L and H1_SL, close 98.1 back above all three
    "09:35":    (100.0, 100.2, 98.9, 99.0),
    "09:36":    (99.0, 99.1, 96.5, 97.2),        # bearish 1-minute gap [99.1, 99.5]
    "09:37":    (97.2, 97.8, 97.1, 97.6),        # bearish 1-minute gap [97.8, 98.9]
    "09:38":    (97.6, 98.0, 97.5, 97.9),
    "09:39":    (97.9, 98.2, 97.8, 98.1),
    # 09:40: closes 102.0 — above the swing (bos), above both fresh gaps (ifvg), above 100.45 (ote).
    # Dealing range [96.5, 102.3], EQ 99.4. The high takes LON_H / H1_SH at 102.
    "09:40":    (98.1, 102.3, 98.0, 102.0),
}
HIGH_RUN = {("09:41", "09:51"): HIGH}
FULL_LONG = {**BASE,
             "09:52": (102.0, 102.0, 99.3, 99.35),   # opens above EQ, trades to it, closes under it: touch
             "09:53": (99.35, 99.8, 99.2, 99.7),     # closes out above EQ: trigger
             "09:54": (99.7, 100.0, 99.6, 99.9)}     # entry at the open, 99.7


def hand(df, inst, **kw):
    """The events of day D of a hand-built frame (D-1 is there only to give D its levels)."""
    return [e for e in D.detect(df, inst, **kw) if e["day"] == str(DAY.date())]


def by_kind(events, kind):
    return [e for e in events if e["kind"] == kind]


def one(events, kind):
    got = by_kind(events, kind)
    assert len(got) == 1, f"expected one {kind!r} event, got {len(got)}: {[e['kind'] for e in events]}"
    return got[0]


def expected_atr(df: pd.DataFrame, stamp: str, before: bool) -> float:
    """The context ATR by the reference implementation (tjr_intraday.build_context):
    at the newest bin completed by the minute `stamp`, or by the minute before it."""
    ctx = M.build_context(df, 1, 14)
    i = df.index.get_loc(pd.Timestamp(stamp))
    return float(ctx.bars.atr[ctx.last[i - 1 if before else i]])


# ────────────────────────────── 1. helpers agree with the funnel script ──────────────────────────────

def test_helpers_match_the_funnel_script():
    import tjr_wide_funnel as F
    assert D.OTE == F.OTE and D.EQ_EXCURSION_ATR == F.EQ_EXCURSION_ATR and D.WICK_STOP_ATR == F.WICK_STOP_ATR
    assert D.ZONE_RANK == F.ZONE_RANK and D.CLASSES == F.CLASSES_FINAL and D.LEVEL_ORDER == F.LEVEL_ORDER
    assert D.DIRECTIONAL_LOWS == F.DIRECTIONAL_LOWS and D.DIRECTIONAL_HIGHS == F.DIRECTIONAL_HIGHS
    assert D.COMPOSITE_SESSIONS == F.COMPOSITE_SESSIONS and D.TICK == F.TICK
    assert (D.SWING_LEFT, D.SWING_RIGHT) == (F.SWING_LEFT, F.SWING_RIGHT)
    assert tuple(n for n, _ in D.RESAMPLES) == tuple(F.RESAMPLES) and dict(D.RESAMPLES) == dict(F.RESAMPLES)
    names = list(F.LEVEL_ORDER) + [f"HVN_{k}" for k in range(1, 7)]
    for n in names:
        assert D.level_type(n) == F.level_type(n)
        assert D.level_sort_key(n) == F.level_sort_key(n)
        if n in F.DIRECTIONAL_LOWS + F.DIRECTIONAL_HIGHS:
            assert D.level_class(n) == F.level_class_final(n)
    assert D.TARGET_RULE == "nearest" and D.TARGET_RULES == ("nearest", "untaken")
    assert N_TRIALS == 14 and len(X.EXIT_NAMES) == 10
    assert (D.SWEEP_START, D.SWEEP_END, D.ENTRY_START, D.ENTRY_END) == (570, 590, 590, 610)
    assert (D.TRIGGER_FIRST, D.TRIGGER_LAST, D.FREEZE_MINUTE) == (589, 608, 569)


# ────────────────────────────── 2. the incremental context is tjr_intraday's ──────────────────────────────

def test_context_clock_swings_atr_match_reference_on_market_data():
    for sym in ("NQ", "ES"):
        df = market(sym)
        det = D.Detector(sym)
        det.feed(df)
        ctx = M.build_context(df, 1, 14)
        nb = det._bt.n
        assert nb == ctx.bars.n, (sym, nb, ctx.bars.n)
        for mine, ref in ((det._bo, ctx.bars.open), (det._bh, ctx.bars.high),
                          (det._bl, ctx.bars.low), (det._bc, ctx.bars.close)):
            assert np.array_equal(mine.a[:nb], ref), sym
        assert np.array_equal(det._bfirst.a[:nb], ctx.first), sym
        assert det._last == int(ctx.last[-1]), sym
        clock = M.session_clock(df.index)
        assert np.array_equal(det._m.a[:len(df)], clock.minutes) and np.array_equal(det._day.a[:len(df)], clock.day)
        highs, lows = P.confirmed_swings(ctx.bars, 2, 2)
        assert det._sw_hi == [s for s in highs if s.confirmed_at <= det._last], sym
        assert det._sw_lo == [s for s in lows if s.confirmed_at <= det._last], sym
        for k in (0, 5, 137, nb // 2, det._last):
            assert abs(det._atr_at(k) - float(ctx.bars.atr[k])) < 1e-9, (sym, k)
        print(f"      {sym}: {nb} context bins, {len(highs)} + {len(lows)} swings, clock and ATR identical")


def test_levels_and_sweeps_match_the_final_funnel_on_the_same_bars():
    """The funnel (--mode final) on the 1-minute file resampled to 5 minutes must
    give the same directional levels and the same sweeps, bar for bar. POC / HVN
    are left out: the detector's profile is built on 1-minute bars by contract."""
    import tjr_wide_funnel as F
    for sym in ("NQ", "ES"):
        df = market(sym)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        df5 = df.resample("5min", label="left", closed="left").agg(agg).dropna(subset=["close"])
        fn = F.WideFunnel(df5, mode="final")
        ev = market_events(sym)
        lv = {e["day"]: e for e in by_kind(ev, "levels")}
        sw = {e["day"]: e for e in by_kind(ev, "sweep")}
        n_lv = n_sw = 0
        for row in fn.rows:
            day = row["day"].strftime("%Y-%m-%d")
            mine = {r["name"]: r["price"] for r in lv[day]["levels"] if r["swept"]}
            ref = {k: v for k, v in row["levels"].items() if k in F.DIRECTIONAL_LOWS + F.DIRECTIONAL_HIGHS}
            assert mine == ref, (sym, day, mine, ref)
            n_lv += 1
            if row["sweep_idx"] is None:
                assert day not in sw, (sym, day, "detector swept, funnel did not")
                continue
            s = sw[day]
            assert s["sweep_bar_time"] == df5.index[row["sweep_idx"]].isoformat(), (sym, day)
            assert (s["side"], s["level"]["name"], s["extreme"]) == (row["side"], row["level"], row["extreme"]), (sym, day)
            assert s["breached"] == row["breached"] and s["breached_opposite"] == row["breached_opp"], (sym, day)
            assert s["pre_swing"] == row["pre_swing"], (sym, day)
            assert s["classes_returned"] == row["cls_ret"] and s["classes_abstained"] == row["cls_abs"], (sym, day)
            n_sw += 1
        assert n_sw == len(sw), (sym, n_sw, len(sw))
        print(f"      {sym}: {n_lv} days of levels and {n_sw} sweeps identical to the final funnel")


def test_order_blocks_and_breakers_are_the_primitives_on_the_reference_context():
    """Every `ob` a signal lists is primitives.order_block on tjr_intraday's own
    context at the sweep bin; the breaker primitives.breaker_block gives at the
    signal minute's newest completed bin is listed whenever it sits in discount."""
    for sym in ("NQ", "ES"):
        df, ev = market(sym), market_events(sym)
        ctx = M.build_context(df, 1, 14)
        clock = M.session_clock(df.index)
        highs, lows = P.confirmed_swings(ctx.bars, 2, 2)
        n_ob = n_br = 0
        for sg in by_kind(ev, "signal"):
            side = sg["side"]
            i = df.index.get_loc(pd.Timestamp(sg["bar_time"]))
            sweep_ci = int(ctx.idx[df.index.get_loc(pd.Timestamp(sg["sweep_bar_time"]))])
            start_ci = int(ctx.idx[np.flatnonzero(clock.day == clock.day[i])[0]])
            eq = sg["eq"]
            known = {(z["kind"], z["low"], z["high"]) for z in sg["zones_known"]}
            ob = P.order_block(ctx.bars, sweep_ci, side)
            in_disc = lambda b: (b.high <= eq) if side > 0 else (b.low >= eq)          # noqa: E731
            assert (("ob", ob.low, ob.high) in known) == (ob is not None and in_disc(ob)), (sym, sg["day"])
            n_ob += ("ob", ob.low, ob.high) in known
            blk = P.breaker_block(ctx.bars, start_ci, sweep_ci, int(ctx.last[i]), side, highs if side > 0 else lows)
            if blk is not None and in_disc(blk):
                assert ("breaker", blk.low, blk.high) in known, (sym, sg["day"])
                n_br += 1
            for z in sg["zones_known"]:
                assert z["formed_time"] <= z["usable_time"] <= sg["bar_time"] and z["usable_time"] >= sg["conf_time"]
        print(f"      {sym}: {n_ob} order blocks and {n_br} breakers at signals match the primitives")


# ────────────────────────────── 3. one call, or minute by minute ──────────────────────────────

def test_batch_equals_minute_by_minute_and_every_prefix():
    """Fed one row at a time, the detector emits the batch run's events, each on
    the feed of its own bar — so the events after ANY minute k are exactly the
    batch events stamped <= minute k: every truncation of the incremental run."""
    for sym in ("NQ", "ES"):
        df = market(sym)
        full = market_events(sym)
        det = D.Detector(sym)
        got, previews = [], []
        for k in range(len(df)):
            new = det.feed(df.iloc[k:k + 1])
            stamp = df.index[k].isoformat()
            for e in new:
                assert e["bar_time"] == stamp, (sym, e["kind"], e["bar_time"], stamp)
                if e["kind"] == "entry_trigger":        # what the runner would do: look at the next open, ask
                    previews.append(det.preview_fill(float(df["open"].iloc[k + 1])))
                    assert det.snapshot()["stage"] == "pending"
            if not new and k % 97 == 0:
                assert det.preview_fill(1.0) is None or det.snapshot()["stage"] == "pending"
            got.extend(new)
        assert got == full, sym
        real = [e for e in full if e["kind"] == "fill" or (e["kind"] == "invalidated" and e["reason"] == "no_risk")]
        assert len(previews) == len(real) == len(by_kind(full, "entry_trigger"))
        for pv, e in zip(previews, real):
            assert pv.pop("provisional") is True and pv == e, (sym, e["day"])
        assert det.events == full
        kinds = collections.Counter(e["kind"] for e in full)
        print(f"      {sym}: {len(df)} single-minute feeds -> {len(full)} events, identical to one call "
              f"({kinds['sweep']} sweeps, {kinds['signal']} signals, {kinds['fill']} fills)")


def test_batch_equals_ragged_chunks_and_overlapping_feeds():
    for sym in ("NQ", "ES"):
        df = market(sym)
        full = market_events(sym)
        rng = np.random.default_rng(7)
        det, k, got = D.Detector(sym), 0, []
        while k < len(df):
            step = int(rng.choice([1, 2, 3, 5, 7, 60, 391, 1380, 5000]))
            lo = max(0, k - int(rng.integers(0, 4)))            # re-read a few rows already fed, as a store would
            got.extend(det.feed(df.iloc[lo:k + step]))
            k += step
        assert got == full, sym
        assert det.dropped > 0


# ────────────────────────────── 4. truncation at adversarial minutes ──────────────────────────────

def _cuts(sym: str) -> list[tuple[int, str]]:
    """Row numbers to cut at: a frame df.iloc[:k] ends just BEFORE row k."""
    df, ev = market(sym), market_events(sym)
    cuts, seen = [], set()

    def add(row: int, what: str):
        if 0 < row < len(df) and row not in seen:
            seen.add(row)
            cuts.append((row, what))

    def first(kind, pred=lambda e: True, skip=0):
        hits = [e for e in ev if e["kind"] == kind and pred(e)]
        return hits[min(skip, len(hits) - 1)] if hits else None

    for kind, skip in (("sweep", 3), ("confirmation", 2), ("signal", 4), ("touch", 1), ("entry_trigger", 2),
                       ("fill", 1), ("invalidated", 1), ("expired", 0), ("zone", 0), ("levels", 25)):
        e = first(kind, skip=skip)
        if e is None:
            continue
        r = df.index.get_loc(pd.Timestamp(e["bar_time"]))
        add(r, f"at the {kind} minute {e['day']} {e['et']}: it must vanish")
        add(r + 1, f"just after the {kind} minute {e['day']} {e['et']}: it must stay, exactly")
    e = first("sweep", skip=6)
    if e is not None:                                   # inside the forming 5-minute sweep bar
        r = df.index.get_loc(pd.Timestamp(e["bar_time"]))
        add(r - 2, f"inside the forming sweep bar {e['day']} (two minutes before it completes)")
    e = first("fill", skip=3)
    if e is not None:                                   # mid-afternoon on a filled day
        add(df.index.get_loc(pd.Timestamp(e["bar_time"])) + 200, f"mid-session after the fill of {e['day']}")
    return cuts


def test_truncation_reproduces_every_earlier_event():
    for sym in ("NQ", "ES"):
        df, full = market(sym), market_events(sym)
        cuts = _cuts(sym)
        assert len(cuts) >= 12, (sym, len(cuts))
        for k, what in cuts:
            last_stamp = df.index[k - 1].isoformat()
            expect = [e for e in full if e["bar_time"] <= last_stamp]
            got = D.detect(df.iloc[:k], sym)
            assert got == expect, (sym, what, len(got), len(expect))
        print(f"      {sym}: {len(cuts)} truncations reproduce the full run as of the cut")


def test_garbage_after_the_cut_changes_nothing_before_it():
    rng = np.random.default_rng(11)
    for sym in ("NQ", "ES"):
        df, full = market(sym), market_events(sym)
        fills = by_kind(full, "fill")
        triggers = by_kind(full, "entry_trigger")
        for e, shift in ((fills[len(fills) // 2], 0), (triggers[0], 1)):
            k = df.index.get_loc(pd.Timestamp(e["bar_time"])) + shift
            bad = df.copy()
            n = len(df) - k
            walk = float(df["close"].iloc[k - 1]) + np.cumsum(rng.normal(0, 25.0, n))
            bad.iloc[k:, bad.columns.get_loc("open")] = walk
            bad.iloc[k:, bad.columns.get_loc("close")] = walk + rng.normal(0, 10.0, n)
            bad.iloc[k:, bad.columns.get_loc("high")] = np.maximum(bad["open"].iloc[k:], bad["close"].iloc[k:]) + 30.0
            bad.iloc[k:, bad.columns.get_loc("low")] = np.minimum(bad["open"].iloc[k:], bad["close"].iloc[k:]) - 30.0
            last_stamp = df.index[k - 1].isoformat()
            got = [x for x in D.detect(bad, sym) if x["bar_time"] <= last_stamp]
            assert got == [x for x in full if x["bar_time"] <= last_stamp], (sym, e["kind"])


# ────────────────────────────── 5. invariants of the market events (counts only) ──────────────────────────────

def test_market_event_invariants():
    lows, highs = set(D.DIRECTIONAL_LOWS), set(D.DIRECTIONAL_HIGHS)
    for sym in ("NQ", "ES"):
        ev = market_events(sym)
        json.dumps(ev, allow_nan=False)                                     # JSON-safe, no NaN anywhere
        per_day = collections.defaultdict(list)
        for e in ev:
            per_day[e["day"]].append(e)
            assert e["setup_id"] == f"{sym}-{e['day']}"
        n_targets = 0
        for day, es in per_day.items():
            kinds = [e["kind"] for e in es]
            assert kinds[0] == "levels" and kinds.count("levels") == 1, (sym, day, kinds)
            assert kinds.count("sweep") + kinds.count("no_sweep") == 1, (sym, day, kinds)
            assert kinds.count("fill") <= 1 and kinds.count("signal") <= 1
            terminal = [k for k in kinds if k in ("fill", "invalidated", "expired", "no_sweep")]
            assert len(terminal) == 1 and kinds[-1] == terminal[0], (sym, day, kinds)
            assert [e["bar_time"] for e in es] == sorted(e["bar_time"] for e in es)
            if "fill" in kinds:
                order = [kinds.index(k) for k in ("sweep", "confirmation", "signal", "touch", "entry_trigger", "fill")]
                assert order == sorted(order), (sym, day, kinds)
            for e in es:
                if e["kind"] == "levels":
                    assert e["et"] == "09:29"
                if e["kind"] == "sweep":
                    assert "09:34" <= e["et"] <= "09:49" and e["et"][-1] in "49"
                if e["kind"] == "entry_trigger":
                    assert "09:49" <= e["et"] <= "10:08"
                if e["kind"] == "fill":
                    assert "09:50" <= e["et"] <= "10:09" and e["knowable_at"] == e["bar_time"]
                    trig = [x for x in es if x["kind"] == "entry_trigger"][0]
                    assert trig["entry_at"] == e["bar_time"]
                    side, entry, st = e["side"], e["entry"], e["stops"]
                    assert all(side * (entry - st[k]) > 0 for k in D.STOP_MODES)
                    assert side * (st["atr1.0"] - st["atr1.5"]) > 0 and side * (st["atr1.5"] - st["atr2.0"]) > 0
                    assert side * (st["wick"] - st["session"]) >= 0            # the session stop is never tighter
                if e["kind"] in ("signal", "zone", "fill"):
                    ref = e["entry"] if e["kind"] == "fill" else e["ref_price"]
                    same = lows if e["side"] > 0 else highs
                    for rule in D.TARGET_RULES:
                        prev = ref
                        for slot in ("t1", "t2"):
                            t = e["targets"][rule][slot]
                            if t is None:
                                continue
                            n_targets += 1
                            assert not (set(t["names"].split("+")) & same), (sym, day, t)   # never a same-side level
                            assert e["side"] * (t["price"] - prev) > 0
                            prev = t["price"]
                    # untaken is nearest, restricted: never nearer than it
                    a, b = e["targets"]["nearest"]["t1"], e["targets"]["untaken"]["t1"]
                    if b is not None:
                        assert a is not None and e["side"] * (b["price"] - a["price"]) >= 0
                    assert e["t1"] == e["targets"][D.TARGET_RULE]["t1"] and e["target_rule"] == D.TARGET_RULE
        warm = [e["warm"] for e in by_kind(ev, "levels")]
        assert warm == [False] * 20 + [True] * (len(warm) - 20), (sym, warm)
        print(f"      {sym}: {len(per_day)} days well-formed; {n_targets} targets, none same-side; "
              f"{sum(warm)} warm days of {len(warm)}")


def test_target_rule_switch_changes_only_the_choice():
    for sym in ("NQ",):
        a = market_events(sym)
        b = D.detect(market(sym), sym, target_rule="untaken")
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if x["kind"] in ("signal", "zone", "fill"):
                assert y["target_rule"] == "untaken" and y["t1"] == y["targets"]["untaken"]["t1"]
                strip = lambda e: {k: v for k, v in e.items() if k not in ("target_rule", "t1", "t2")}   # noqa: E731
                assert strip(x) == strip(y)
            else:
                assert x == y
    try:
        D.Detector("NQ", target_rule="furthest")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown target rule must be refused")


def test_warm_check():
    df = market("NQ")
    w = D.check_warm(df)
    assert w["ok"] and w["sessions"] == 29 and w["h4_bins"] >= D.WARM_H4_BINS, w
    clock = M.session_clock(df.index)
    days = np.unique(clock.day)
    short = df[clock.day <= days[9]]
    w2 = D.check_warm(short)
    assert not w2["ok"] and w2["sessions"] == 9 and "composite" in w2["reason"], w2
    w3 = D.check_warm(short, for_day=pd.Timestamp(days[10]))
    assert w3["sessions"] == 10 and not w3["ok"]
    det = D.Detector("NQ")
    assert not det.warm_status()["ok"]
    det.feed(df)
    assert det.warm_status() == w
    assert D.check_warm(df.iloc[:0])["ok"] is False


def test_feed_hygiene():
    df = build_day(FULL_LONG, fill_runs=HIGH_RUN)
    det = D.Detector("NQ")
    a = det.feed(df.iloc[:1000])
    assert det.feed(df.iloc[:1000]) == [] and det.dropped == 1000
    det.feed(df.iloc[900:])
    assert det.events == D.detect(df, "NQ") and a == det.events[:len(a)]
    for bad in (df.iloc[::-1], df.iloc[:5].assign(close=np.nan)):
        try:
            D.Detector("NQ").feed(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a bad frame must be refused")
    snap = det.snapshot()
    assert snap["bars"] == len(df) and snap["stage"] == "done" and "r" not in snap


# ────────────────────────────── 6. hand-built days: every rule, then every refusal ──────────────────────────────

def test_full_long_day_every_rule_fires():
    df = build_day(FULL_LONG, fill_runs=HIGH_RUN)
    ev = hand(df, "NQ")
    assert [e["kind"] for e in ev] == ["levels", "sweep", "confirmation", "signal", "touch", "entry_trigger", "fill"], \
        [e["kind"] for e in ev]
    lv = one(ev, "levels")
    assert lv["bar_time"] == utc("09:29") and lv["et"] == "09:29" and lv["knowable_at"] == utc("09:30")
    got = {r["name"]: (r["price"], r["class"]) for r in lv["levels"]}
    assert got == {"ASIA_H": (103.0, "SESSION"), "ASIA_L": (98.0, "SESSION"), "LON_H": (102.0, "SESSION"),
                   "LON_L": (97.0, "SESSION"), "PDH": (105.0, "SESSION"), "PDL": (95.0, "SESSION"),
                   "H1_SH": (102.0, "H1"), "H1_SL": (98.0, "H1"), "H4_SH": (105.0, "H4"), "H4_SL": (95.0, "H4")}, got
    assert lv["warm"] is False and lv["composite_depth"] == 1 and lv["close_pre"] == 100.0

    sw = one(ev, "sweep")
    assert sw["bar_time"] == utc("09:39") and sw["sweep_bar"] == "09:35" and sw["sweep_bar_time"] == utc("09:35")
    assert sw["side"] == 1 and sw["direction"] == "long"
    assert sw["level"] == {"name": "LON_L", "price": 97.0, "class": "SESSION"} and sw["extreme"] == 96.5
    assert sw["breached"] == ["LON_L", "ASIA_L", "H1_SL"] and sw["breached_opposite"] == []
    assert sw["classes_returned"] == {"SESSION": "LON_L", "H1": "H1_SL"} and sw["classes_abstained"] == []
    assert sw["pre_swing"] == 101.5 and abs(sw["ote_level"] - (96.5 + 0.79 * 5.0)) < 1e-12

    cf = one(ev, "confirmation")
    assert cf["bar_time"] == utc("09:40") and cf["types"] == ["bos", "ifvg", "ote"]
    assert cf["range"] == [96.5, 102.3] and abs(cf["eq"] - 99.4) < 1e-12 and cf["eq_usable"] is True
    assert abs(cf["atr"] - expected_atr(df, utc("09:40"), before=False)) < 1e-9
    assert [z["kind"] for z in cf["zones"]] == ["eq"]            # the sweep candle is the order block, above EQ

    sg = one(ev, "signal")
    assert sg["bar_time"] == utc("09:40") and sg["zone"]["kind"] == "eq"
    assert abs(sg["zone"]["low"] - 99.4) < 1e-12 and sg["zone"]["low"] == sg["zone"]["high"] == sg["ref_price"]
    assert sg["targets"]["nearest"]["t1"] == {"names": "LON_H+H1_SH", "price": 102.0}
    assert sg["targets"]["nearest"]["t2"] == {"names": "ASIA_H", "price": 103.0}
    assert sg["targets"]["untaken"]["t1"] == {"names": "ASIA_H", "price": 103.0}       # 102 traded at 09:40
    assert sg["targets"]["untaken"]["t2"] == {"names": "PDH+H4_SH", "price": 105.0}
    assert sg["t1"] == sg["targets"]["nearest"]["t1"] and sg["target_rule"] == "nearest"
    assert set(sg["stops_indicative"]) == set(M.STOP_MODES) and sg["levels"]["PDH"] == 105.0

    assert one(ev, "touch")["bar_time"] == utc("09:52")
    tr = one(ev, "entry_trigger")
    assert tr["bar_time"] == utc("09:53") and tr["entry_at"] == utc("09:54") and tr["touch_time"] == utc("09:52")

    f = one(ev, "fill")
    assert f["bar_time"] == f["knowable_at"] == f["entry_time"] == utc("09:54") and f["entry"] == 99.7
    atr = expected_atr(df, utc("09:54"), before=True)
    assert abs(f["atr"] - atr) < 1e-9
    for mode in M.STOP_MODES:
        want = M.stop_price(mode, 1, 99.7, 96.5, 96.5, atr, 0.25)
        assert abs(f["stops"][mode] - want) < 1e-9 and abs(f["risk"][mode] - (99.7 - want)) < 1e-9, mode
    assert f["session_extreme"] == 96.5 and f["seen_extreme"] == 102.3 and f["nearest_t1_already_traded"] is True
    assert f["zone"]["kind"] == "eq" and f["cooccur"] == [] and f["eq_refused"] == []
    assert f["conf_types"] == ["bos", "ifvg", "ote"] and f["level"]["name"] == "LON_L"
    assert f["targets"] == sg["targets"] and f["t1"]["price"] == 102.0 and f["t2"]["price"] == 103.0
    f2 = one(hand(df, "NQ", target_rule="untaken"), "fill")
    assert f2["t1"] == {"names": "ASIA_H", "price": 103.0} and f2["t2"]["price"] == 105.0
    json.dumps(ev, allow_nan=False)


def test_full_short_day_is_the_mirror():
    df = build_day(mirror(FULL_LONG), fill_runs={k: mirror({"x": v})["x"] for k, v in HIGH_RUN.items()})
    ev = hand(df, "ES")
    assert [e["kind"] for e in ev] == ["levels", "sweep", "confirmation", "signal", "touch", "entry_trigger", "fill"]
    sw, f = one(ev, "sweep"), one(ev, "fill")
    assert sw["side"] == -1 and sw["level"]["name"] == "LON_H" and sw["extreme"] == 103.5 and sw["pre_swing"] == 98.5
    assert sw["breached"] == ["LON_H", "ASIA_H", "H1_SH"]
    assert abs(f["entry"] - 100.3) < 1e-12 and abs(f["eq"] - 100.6) < 1e-12
    assert f["targets"]["nearest"]["t1"] == {"names": "LON_L+H1_SL", "price": 98.0}
    assert f["targets"]["untaken"]["t1"] == {"names": "ASIA_L", "price": 97.0}
    assert f["targets"]["untaken"]["t2"] == {"names": "PDL+H4_SL", "price": 95.0}
    atr = expected_atr(df, utc("09:54"), before=True)
    assert abs(f["stops"]["wick"] - (103.5 + 0.25 * atr)) < 1e-9 and abs(f["stops"]["atr2.0"] - (100.3 + 2 * atr)) < 1e-9


def test_eq_touch_that_opened_through_the_midpoint_is_refused():
    day = {**BASE,
           "09:52": (99.3, 99.9, 99.2, 99.8),      # opens UNDER EQ 99.4, closes above it: not a retrace, refused
           "09:55": (100.0, 100.1, 99.4, 99.9),    # opens above EQ, trades back to it, closes out: touch and trigger
           "09:56": (99.9, 100.2, 99.8, 100.0)}
    df = build_day(day, fill_runs={("09:41", "09:51"): HIGH, ("09:53", "09:54"): (100.0, 100.3, 99.9, 100.0)})
    ev = hand(df, "NQ")
    touch, f = one(ev, "touch"), one(ev, "fill")
    assert touch["bar_time"] == utc("09:55"), touch                    # nothing at 09:52
    assert one(ev, "entry_trigger")["bar_time"] == utc("09:55")
    assert f["bar_time"] == utc("09:56") and f["entry"] == 99.9 and f["eq_refused"] == ["09:52"]
    # and with no later retrace the refused bar is the end of it
    df2 = build_day({k: v for k, v in day.items() if k not in ("09:55", "09:56")},
                    fill_runs={("09:41", "09:51"): HIGH, ("09:53", "10:30"): (100.0, 100.3, 99.9, 100.0)})
    ev2 = hand(df2, "NQ")
    ex = one(ev2, "expired")
    assert not by_kind(ev2, "fill") and ex["reason"] == "no_touch" and ex["eq_refused"] == ["09:52"]
    assert ex["bar_time"] == utc("10:10")


def test_session_abstention():
    # unit: a wick through a session low AND a session high -> SESSION abstains; H4 alone returns
    h, l, c = np.array([102.5]), np.array([97.5]), np.array([100.0])
    lows = {"ASIA_L": 98.0, "PDL": 95.0, "H4_SL": 97.8}
    highs = {"LON_H": 102.0, "PDH": 105.0}
    r = D.class_sweep(h, l, c, 0, lows, highs)
    assert r["verdict"] == "sweep" and r["side"] == 1 and r["name"] == "H4_SL", r
    assert r["abstained"] == ["SESSION"] and r["returned"] == {"H4": "H4_SL"}
    assert r["state"] == {"SESSION": "abstained", "H1": "absent", "H4": "returned"}
    # the deeper ASIA_L... is not deeper here; make it deeper: still not the setup's level (its class abstained)
    r = D.class_sweep(h, l, c, 0, {**lows, "ASIA_L": 97.6}, highs)
    assert r["name"] == "H4_SL" and set(r["below"]) == {"ASIA_L", "H4_SL"}
    # every class abstaining or silent -> no sweep; two classes disagreeing -> ambiguous
    assert D.class_sweep(h, l, c, 0, {"ASIA_L": 98.0}, {"LON_H": 102.0})["verdict"] == "abstain_only"
    r = D.class_sweep(h, l, c, 0, {"ASIA_L": 98.0}, {"H1_SH": 102.0})
    assert r["verdict"] == "ambiguous" and r["returned"] == {"SESSION": "ASIA_L", "H1": "H1_SH"}
    assert D.class_sweep(h, l, c, 0, {"ASIA_L": 97.0}, {"LON_H": 103.0})["verdict"] == "none"
    # day: the 09:30 bin wicks through ASIA_L 98 and LON_H 102 (H1_SL / H1_SH sit on the same prices, so H1
    # abstains too) and closes inside: no sweep on it; the 09:35 bin then sweeps, and says what came before
    day = {**FULL_LONG, "09:32": (100.0, 102.2, 97.9, 100.0)}
    ev = hand(build_day(day, fill_runs=HIGH_RUN), "NQ")
    sw = one(ev, "sweep")
    assert sw["sweep_bar"] == "09:35" and sw["abstain_only_bars"] == ["09:30"] and sw["ambiguous_bars"] == []
    # and if that is the only candidate, the day has no sweep
    only = {k: v for k, v in BASE.items() if not k.startswith("09:3") and k != "09:40"}
    ev = hand(build_day({**only, "09:32": (100.0, 102.2, 97.9, 100.0)}), "NQ")
    ns = one(ev, "no_sweep")
    assert [e["kind"] for e in ev] == ["levels", "no_sweep"] and ns["abstain_only_bars"] == ["09:30"]
    assert ns["bar_time"] == utc("09:50")


def test_same_side_level_is_never_a_target():
    levels = {"ASIA_L": 100.5, "LON_L": 97.0, "H1_SL": 101.0, "LON_H": 102.0, "H1_SH": 102.0, "PDH": 105.0,
              "POC_PREV": 101.5, "HVN_1": 99.0, "HVN_2": 104.0, "ASIA_H": 99.5}
    t = D.target_pairs(levels, 1, 100.0, 101.7)
    assert t["nearest"]["t1"] == {"names": "POC_PREV", "price": 101.5}          # not ASIA_L 100.5, not H1_SL 101
    assert t["nearest"]["t2"] == {"names": "LON_H+H1_SH", "price": 102.0}       # one price, both names
    assert t["untaken"]["t1"] == {"names": "LON_H+H1_SH", "price": 102.0}       # 101.5 has traded (seen 101.7)
    assert t["untaken"]["t2"] == {"names": "HVN_2", "price": 104.0}
    s = D.target_pairs(levels, -1, 101.2, 99.2)
    # a short from 101.2: the LOWS are the opposing names now, ASIA_H 99.5 is the same-side one
    assert s["nearest"]["t1"] == {"names": "H1_SL", "price": 101.0} and s["nearest"]["t2"]["names"] == "ASIA_L"
    assert s["untaken"]["t1"] == {"names": "HVN_1", "price": 99.0}              # 99.0 < the low seen, 99.2: not traded
    assert s["untaken"]["t2"] == {"names": "LON_L", "price": 97.0}              # ASIA_H 99.5 is never in either pair
    s = D.target_pairs(levels, -1, 101.2, 99.0)
    assert s["untaken"]["t1"] == {"names": "LON_L", "price": 97.0}              # traded AT 99.0 counts as taken
    assert D.target_pairs({"ASIA_L": 101.0, "PDL": 90.0}, 1, 100.0, None) == \
        {"nearest": {"t1": None, "t2": None}, "untaken": {"t1": None, "t2": None}}
    assert D.target_pairs(levels, 1, 100.0, None)["untaken"] == D.target_pairs(levels, 1, 100.0, None)["nearest"]


def test_poc_and_hvn_are_targets_not_sweep_levels():
    df = build_day(FULL_LONG, fill_runs=HIGH_RUN, volume=1.0)
    ev = hand(df, "NQ")
    lv = {r["name"]: r for r in one(ev, "levels")["levels"]}
    assert "POC_PREV" in lv and lv["POC_PREV"]["swept"] is False and lv["POC_PREV"]["class"] == "POC"
    hv = [r for n, r in lv.items() if n.startswith("HVN_")]
    assert hv and all(r["swept"] is False and r["class"] == "HVN" for r in hv)
    assert all(abs(r["price"] % 1.0 - 0.5) < 1e-9 for r in hv + [lv["POC_PREV"]])     # bin centres, $1 bins
    f = one(ev, "fill")
    assert one(ev, "sweep")["level"]["name"] == "LON_L"                               # the sweep is unchanged
    beyond = sorted(r["price"] for r in hv + [lv["POC_PREV"]] if r["price"] > f["entry"])
    if beyond and beyond[0] < 102.0:
        assert f["targets"]["nearest"]["t1"]["price"] == beyond[0]
        assert f["targets"]["nearest"]["t1"]["names"].split("+")[0].split("_")[0] in ("POC", "HVN")
    # without volume there is no point of control and no node
    names = {r["name"] for r in one(hand(build_day(FULL_LONG, fill_runs=HIGH_RUN), "NQ"), "levels")["levels"]}
    assert not any(n.startswith(("POC", "HVN")) for n in names)


def test_dead_setup():
    # after the signal, before any entry, a minute trades through the wick stop (96.5 - 0.25 ATR)
    day = {**FULL_LONG, "09:47": (102.0, 102.0, 95.9, 101.9)}
    df = build_day(day, fill_runs=HIGH_RUN)
    ev = hand(df, "NQ")
    assert [e["kind"] for e in ev] == ["levels", "sweep", "confirmation", "signal", "invalidated"], [e["kind"] for e in ev]
    inv = one(ev, "invalidated")
    atr = expected_atr(df, utc("09:47"), before=True)
    assert inv["bar_time"] == utc("09:47") and inv["reason"] == "wick_stop_traded" and inv["stage"] == "entry"
    assert abs(inv["wick_stop"] - (96.5 - 0.25 * atr)) < 1e-9 and inv["traded"] == 95.9
    # a wick that stops short of the stop kills nothing
    ok = {**FULL_LONG, "09:47": (102.0, 102.0, 96.5, 101.9)}
    ev_ok = hand(build_day(ok, fill_runs=HIGH_RUN), "NQ")
    assert by_kind(ev_ok, "fill") and not by_kind(ev_ok, "invalidated")
    # before the confirmation too: the sweep's wick is gone, the day is done
    early = {k: v for k, v in FULL_LONG.items() if k != "09:40"}
    early["09:40"] = (98.1, 98.2, 95.0, 97.0)
    ev2 = hand(build_day(early), "NQ")
    assert [e["kind"] for e in ev2] == ["levels", "sweep", "invalidated"] and ev2[-1]["stage"] == "confirm"
    # the entry minute opening at or beyond the wick stop: no risk, no trade
    gap = {**FULL_LONG, "09:54": (96.0, 96.2, 95.9, 96.1)}
    ev3 = hand(build_day(gap, fill_runs=HIGH_RUN), "NQ")
    assert ev3[-1]["kind"] == "invalidated" and ev3[-1]["reason"] == "no_risk" and not by_kind(ev3, "fill")
    assert ev3[-1]["knowable_at"] == ev3[-1]["bar_time"] == utc("09:54")


def test_zone_completed_on_a_minute_is_in_play_from_the_next():
    """Section 9.10 item 6. The 09:52 minute completes a bullish 1-minute gap
    [98.8, 99.05] in discount, touches it and closes above it. Read on its own
    bar that would trigger at 09:52; it is in play from 09:53, which triggers."""
    day = {**BASE,
           "09:49": (102.0, 102.0, 98.6, 98.7),      # retrace through EQ from above: eq touched, not closed out
           "09:50": (98.7, 98.8, 98.6, 98.7),
           "09:51": (98.8, 99.0, 98.75, 98.95),
           "09:52": (99.05, 99.3, 99.05, 99.25),     # low 99.05 > high[09:50] 98.8: the gap; close still under EQ
           "09:53": (99.25, 99.5, 99.0, 99.45),      # touches the gap, closes out of it
           "09:54": (99.45, 99.6, 99.4, 99.5)}
    df = build_day(day, fill_runs={("09:41", "09:48"): HIGH})
    ev = hand(df, "NQ")
    z = one(ev, "zone")
    assert z["bar_time"] == utc("09:52") and z["zone"]["kind"] == "fvg"
    assert (z["zone"]["low"], z["zone"]["high"]) == (98.8, 99.05) and z["ref_price"] == 99.05
    touches = by_kind(ev, "touch")
    assert [(t["bar_time"], t["zone"]["kind"]) for t in touches] == [(utc("09:49"), "eq"), (utc("09:53"), "fvg")]
    assert one(ev, "entry_trigger")["bar_time"] == utc("09:53")
    f = one(ev, "fill")
    assert f["bar_time"] == utc("09:54") and f["entry"] == 99.45 and f["zone"]["kind"] == "fvg" and f["cooccur"] == ["eq"]


def test_pattern_completed_before_the_window_is_spent():
    day = {**BASE, "09:44": (102.0, 102.0, 99.3, 99.9)}       # touches EQ and closes out at 09:44: entry would be 09:45
    df = build_day(day, fill_runs={("09:41", "09:43"): HIGH, ("09:45", "10:30"): HIGH})
    ev = hand(df, "NQ")
    assert [e["kind"] for e in ev] == ["levels", "sweep", "confirmation", "signal", "touch", "expired"]
    assert ev[-1]["reason"] == "no_close_out" and ev[-1]["patterns_before_window"] == 1
    # the last closing-out minute is 10:08 (entry stamped 10:09); 10:09 is too late
    late = {**BASE, "10:08": (102.0, 102.0, 99.3, 99.9)}
    ev = hand(build_day(late, fill_runs={("09:41", "10:07"): HIGH, ("10:09", "10:30"): HIGH}), "NQ")
    assert one(ev, "fill")["bar_time"] == utc("10:09")
    too = {**BASE, "10:09": (102.0, 102.0, 99.3, 99.9)}
    ev = hand(build_day(too, fill_runs={("09:41", "10:08"): HIGH, ("10:10", "10:30"): HIGH}), "NQ")
    assert not by_kind(ev, "fill") and not by_kind(ev, "entry_trigger") and ev[-1]["kind"] == "expired"


def test_eq_waits_for_the_excursion_when_the_range_is_narrow():
    """Section 2.3 item 2, the path the market files never take: a dealing range
    narrower than one ATR. The midpoint is frozen at the confirmation, there is
    no zone and so no signal until a completed minute has been 0.5 ATR beyond it."""
    wild = (100.0, 103.0, 97.0, 100.0)               # 09:00-09:24: loud bins, so the context ATR is about 2.4
    day = {"-1 10:00": BASE["-1 10:00"], "-1 22:00": BASE["-1 22:00"], "05:00": BASE["05:00"],
           "09:35": (98.5, 98.6, 98.3, 98.4),
           "09:36": (98.4, 98.45, 97.3, 97.35),
           "09:37": (97.35, 97.4, 96.5, 96.6),       # the wick: under LON_L 97
           "09:38": (96.6, 96.9, 96.55, 96.85),      # bearish 1-minute gap [96.9, 97.3]
           "09:39": (96.85, 97.25, 96.8, 97.2),      # the bin closes 97.2: back above LON_L, the gap still fresh
           "09:40": (97.2, 97.5, 96.9, 97.4),        # closes above the gap: ifvg. Range [96.5, 98.6], EQ 97.55
                                                     # (low 96.9 = high[09:38]: no bullish gap on the leg)
           "09:41": (97.4, 97.6, 97.3, 97.5),
           "09:42": (97.5, 97.6, 97.4, 97.5),
           "09:43": (97.5, 99.3, 97.4, 99.2)}        # the excursion: high 99.3 >= EQ + 0.5 ATR
    df = build_day(day, fill_runs={("09:00", "09:24"): wild})
    ev = hand(df, "NQ")
    assert [e["kind"] for e in ev] == ["levels", "sweep", "confirmation", "signal", "expired"], [e["kind"] for e in ev]
    cf, sg = one(ev, "confirmation"), one(ev, "signal")
    atr = expected_atr(df, utc("09:40"), before=False)
    thr = 97.55 + 0.5 * atr
    assert 98.6 < thr < 99.3, thr                                        # the scenario is what it says it is
    assert one(ev, "sweep")["classes_returned"] == {"SESSION": "LON_L"}  # H1_SL 98 was breached, not closed back over
    assert one(ev, "sweep")["class_state"]["H1"] == "breached_no_close_back"
    assert cf["bar_time"] == utc("09:40") and cf["types"] == ["ifvg"] and cf["eq_usable"] is False and cf["zones"] == []
    assert cf["range"] == [96.5, 98.6] and abs(cf["eq"] - 97.55) < 1e-12 and abs(cf["atr"] - atr) < 1e-9
    assert sg["bar_time"] == utc("09:43") and sg["zone"]["kind"] == "eq"
    assert sg["zone"]["formed_time"] == utc("09:40") and sg["zone"]["usable_time"] == utc("09:43")
    assert ev[-1]["reason"] == "no_touch"


def test_forming_context_bar_is_never_read():
    """The sweep bin's wick prints at 09:36; nothing may say so before 09:39 closes
    the bin — and with the 09:39 minute missing the bin never completes: no sweep."""
    df = build_day(FULL_LONG, fill_runs=HIGH_RUN)
    for hm in ("09:36", "09:37", "09:38"):
        k = df.index.get_loc(pd.Timestamp(utc(hm))) + 1
        assert [e["kind"] for e in hand(df.iloc[:k], "NQ")] == ["levels"], hm
    k = df.index.get_loc(pd.Timestamp(utc("09:39"))) + 1
    assert [e["kind"] for e in hand(df.iloc[:k], "NQ")] == ["levels", "sweep"]
    holed = df.drop(pd.Timestamp(utc("09:39")))
    assert not by_kind(hand(holed, "NQ"), "sweep")
    # hand-built days are causal at every minute of the morning as well
    full = D.detect(df, "NQ")
    lo, hi = df.index.get_loc(pd.Timestamp(utc("09:25"))), df.index.get_loc(pd.Timestamp(utc("10:15")))
    for k in range(lo, hi):
        stamp = df.index[k - 1].isoformat()
        assert D.detect(df.iloc[:k], "NQ") == [e for e in full if e["bar_time"] <= stamp], df.index[k]


# ────────────────────────────── 7. exits: hand arithmetic on synthetic minutes ──────────────────────────────

def hand_fill(side=1, entry=100.0, at="09:54", atr=1.2, t1=103.0, t2=104.0, inst="NQ", **stops) -> dict:
    st = {"wick": entry - side * 1.0, "atr1.0": entry - side * 1.2, "atr1.5": entry - side * 1.8,
          "atr2.0": entry - side * 2.4, "session": entry - side * 1.5}
    st.update(stops)
    mk = lambda p, n: None if p is None else {"names": n, "price": float(p)}     # noqa: E731
    pair = {"t1": mk(t1, "LON_H"), "t2": mk(t2, "PDH")}
    return {"kind": "fill", "instrument": inst, "day": str(DAY.date()), "setup_id": f"{inst}-{DAY.date()}",
            "bar_time": utc(at), "side": side, "entry": float(entry), "atr": float(atr), "stops": st,
            "t1": pair["t1"], "t2": pair["t2"], "targets": {"nearest": pair, "untaken": pair}, "target_rule": "nearest"}


def test_cost_arithmetic():
    assert X.cost_bps("NQ") == 0.5 and X.cost_bps("ES") == 1.0 and X.cost_bps("CME_MINI:NQ1!") == 0.5
    assert X.cost_bps("es1!") == 1.0
    assert abs(X.cost_points("NQ", 20000.0) - 1.0) < 1e-12          # 0.5 bp of 20,000
    assert abs(X.cost_points("ES", 5000.0) - 0.5) < 1e-12           # 1 bp of 5,000
    # long 20000 -> 20030 against a stop at 19990: gross 3 R, the round trip costs 1 point = 0.1 R
    assert abs(X.net_r(1, 20000.0, 20030.0, 19990.0, 1.0) - 2.9) < 1e-12
    assert abs(X.net_r(-1, 5000.0, 5005.0, 5005.0, 0.5) - (-1.1)) < 1e-12      # a stopped short: -1 R - 0.1 R
    try:
        X.cost_bps("CL")
    except ValueError:
        pass
    else:
        raise AssertionError("an instrument without a registered cost must be refused")


def test_exits_stop_on_the_entry_minute_and_fixed_widths():
    # entry 100 at 09:54; that minute's low is 98.9: through the wick stop (99.0), not through 98.8 / 98.5 / ...
    day = {"09:54": (100.0, 100.2, 98.9, 99.5), "10:20": (100.0, 103.2, 99.9, 103.0)}
    df = build_day(day)
    rec = X.benchmark_record(df, hand_fill())
    cost = 0.5e-4 * 100.0
    for name in ("wick_fixed", "wick_be_1r", "wick_atr1.0", "wick_atr1.5", "wick_swing", "wick_hybrid"):
        e = rec["exits"][name]
        assert (e["reason"], e["exit"], e["exit_time"], e["minutes"]) == ("stop", 99.0, utc("09:54"), 0), (name, e)
        assert e["r_gross"] == -1.0 and abs(e["r"] - (-1.0 - cost / 1.0)) < 1e-12
    # the wider fixed stops live through the entry minute and take T1 = 103 at 10:20; R against their OWN risk
    for name, risk in (("fixed_atr1.0", 1.2), ("fixed_session", 1.5), ("fixed_atr1.5", 1.8), ("fixed_atr2.0", 2.4)):
        e = rec["exits"][name]
        assert (e["reason"], e["exit"], e["exit_time"]) == ("target", 103.0, utc("10:20")), (name, e)
        assert abs(e["risk"] - risk) < 1e-12 and abs(e["r_gross"] - 3.0 / risk) < 1e-12
        assert abs(e["r"] - (3.0 - cost) / risk) < 1e-12
        assert abs(e["mfe_r"] - 3.0 / risk) < 1e-12 and abs(e["mae_r"] - (-1.1 / risk)) < 1e-12
    # the target is NOT tested on the entry minute: a 09:54 high of 103.5 exits nothing there
    df2 = build_day({"09:54": (100.0, 103.5, 99.5, 100.0)})
    e = X.benchmark_record(df2, hand_fill())["exits"]["wick_fixed"]
    assert e["reason"] == "flat" and e["exit_time"] == utc("15:55")
    assert rec["cost_bps"] == 0.5 and abs(rec["cost_points"] - cost) < 1e-15
    json.dumps(rec, allow_nan=False)


def test_exits_breakeven_gap_flat_and_prints():
    # 10:00 runs to 101.3 (+1.3 R on the wick risk of 1): be_1r arms. 10:05 dips to 99.9: breakeven stop at 100
    # fires. 10:30 OPENS at 98.5, through the wick stop at 99: fills at the open. Nothing else hits; flat 15:55.
    day = {"09:54": (100.0, 100.3, 99.8, 100.1),
           "10:00": (100.1, 101.3, 100.0, 101.0),
           "10:05": (101.0, 101.0, 99.9, 100.2),
           "10:30": (98.5, 98.9, 98.3, 98.7),
           "15:55": (98.7, 98.9, 98.6, 98.8)}
    calm = (100.5, 100.6, 100.4, 100.5)
    df = build_day(day, fill_runs={("09:55", "09:59"): calm, ("10:01", "10:04"): calm, ("10:06", "10:29"): calm,
                                   ("10:31", "15:54"): (98.7, 98.9, 98.65, 98.7),
                                   ("15:56", "16:59"): (98.8, 99.0, 98.7, 98.8)})
    fill = hand_fill(**{"atr1.0": 98.0, "session": 98.6})
    rec = X.benchmark_record(df, fill)
    cost = 0.5e-4 * 100.0
    ex = rec["exits"]
    be = ex["wick_be_1r"]
    assert (be["reason"], be["exit"], be["exit_time"], be["armed"]) == ("trail", 100.0, utc("10:05"), True), be
    assert be["r_gross"] == 0.0 and abs(be["r"] - (-cost)) < 1e-12 and abs(be["mfe_r"] - 1.3) < 1e-12
    wf = ex["wick_fixed"]
    assert (wf["reason"], wf["exit"], wf["exit_time"]) == ("stop", 98.5, utc("10:30")), wf      # gap: the open, not 99
    assert abs(wf["r_gross"] - (-1.5)) < 1e-12 and abs(wf["r"] - (-1.5 - cost)) < 1e-12
    assert abs(wf["mae_r"] - (-1.5)) < 1e-12 and abs(wf["mfe_r"] - 1.3) < 1e-12
    ses = ex["fixed_session"]                                        # stop 98.6: the 10:30 open is under it too
    assert (ses["reason"], ses["exit"]) == ("stop", 98.5) and abs(ses["r_gross"] - (-1.5 / 1.4)) < 1e-12
    a10 = ex["fixed_atr1.0"]                                         # stop 98.0 never trades: flat at the 15:55 close
    assert (a10["reason"], a10["exit"], a10["exit_time"], a10["exit_et"]) == ("flat", 98.8, utc("15:55"), "15:55"), a10
    assert abs(a10["r_gross"] - (-1.2 / 2.0)) < 1e-12 and abs(a10["r"] - (-1.2 - cost) / 2.0) < 1e-12
    assert abs(a10["mae_r"] - (-1.7 / 2.0)) < 1e-12
    # the session's excursions and the prints, entry minute to the flat
    s = rec["session"]
    assert s["through"] == utc("15:55") and abs(s["mfe_points"] - 1.3) < 1e-12 and abs(s["mae_points"] + 1.7) < 1e-12
    assert abs(s["mfe_r_wick"] - 1.3) < 1e-12 and abs(s["mae_r_wick"] + 1.7) < 1e-12
    assert rec["t1"] == {"price": 103.0, "printed": False, "time": None, "et": None, "names": "LON_H"}
    day2 = {**day, "11:00": (98.7, 104.5, 98.7, 98.7)}
    runs = {("09:55", "10:59"): calm}
    rec2 = X.benchmark_record(build_day({k: v for k, v in day2.items() if k in ("09:54", "11:00")}, fill_runs=runs), fill)
    assert rec2["t1"]["printed"] and rec2["t1"]["time"] == utc("11:00") and rec2["t2"]["et"] == "11:00"
    assert rec2["targets_printed"]["untaken"]["t2"] == {"price": 104.0, "printed": True, "time": utc("11:00"),
                                                         "et": "11:00", "names": "PDH"}
    # no T1 at all: stop or flat, never a target
    rec3 = X.benchmark_record(build_day({k: v for k, v in day2.items() if k in ("09:54", "11:00")}, fill_runs=runs),
                              hand_fill(t1=None, t2=None, **{"atr1.0": 98.0}))
    assert rec3["exits"]["fixed_atr1.0"]["reason"] == "flat" and rec3["t1"]["printed"] is None


def _trail_by_hand(df, fill, mult: float, arm_first: bool, floor_entry: bool):
    """Minute by minute, the long-side trail written out longhand: for minute j the stop is the highest of
    every `best high through j-1  -  mult x ATR(newest 5-minute bin completed by j-1)` so far, never under
    the initial stop; with `arm_first` nothing moves until a completed minute has printed +1 R, and with
    `floor_entry` the armed stop is at least the entry. Returns (exit minute, exit price)."""
    ctx = M.build_context(df, 1, 14)
    i0 = df.index.get_loc(pd.Timestamp(fill["bar_time"]))
    entry, stop0 = fill["entry"], fill["stops"]["wick"]
    hi, lo, op = df["high"].to_numpy(), df["low"].to_numpy(), df["open"].to_numpy()
    stop, best = stop0, -np.inf
    for j in range(i0 + 1, len(df)):
        best = max(best, hi[j - 1])
        armed = best - entry >= entry - stop0
        if armed or not arm_first:
            want = best - mult * float(ctx.bars.atr[ctx.last[j - 1]])
            if floor_entry:
                want = max(want, entry)
            stop = max(stop, want)
        if lo[j] <= stop:
            return df.index[j].isoformat(), min(stop, op[j])
    raise AssertionError("never stopped")


def test_exits_atr_trails_and_hybrid_by_hand():
    calm = (100.5, 100.6, 100.4, 100.5)
    day = {"09:54": (100.0, 100.3, 99.8, 100.1),
           "10:00": (100.1, 101.3, 100.0, 101.0),     # +1.3 R on the wick risk of 1
           "10:05": (101.0, 101.0, 99.9, 100.2),
           "10:30": (98.5, 98.9, 98.3, 98.7)}
    df = build_day(day, fill_runs={("09:55", "09:59"): calm, ("10:01", "10:04"): calm, ("10:06", "10:29"): calm,
                                   ("10:31", "15:54"): (98.7, 98.9, 98.65, 98.7)})
    fill = hand_fill(t1=110.0, t2=None)
    ex = X.benchmark_record(df, fill)["exits"]
    got = {}
    for name, mult, arm, floor in (("wick_atr1.0", 1.0, False, False), ("wick_atr1.5", 1.5, False, False),
                                   ("wick_hybrid", 1.0, True, True)):
        when, px = _trail_by_hand(df, fill, mult, arm, floor)
        e = ex[name]
        assert (e["exit_time"], e["reason"]) == (when, "trail") and abs(e["exit"] - px) < 1e-12, (name, e, when, px)
        assert abs(e["r_gross"] - (px - 100.0) / 1.0) < 1e-12 and abs(e["r"] - (px - 100.0 - 0.005)) < 1e-12
        got[name] = (when, px)
    # the 1.0 trail sits above the 1.5 one and cannot exit later; the hybrid, once armed, is never under the entry
    assert got["wick_atr1.0"][0] <= got["wick_atr1.5"][0] and got["wick_hybrid"][1] >= 100.0
    # before +1 R the hybrid does not move: without the 10:00 run-up it is the wick stop, filled at the 10:30 gap
    flat_day = {**{k: v for k, v in day.items() if k != "10:00"}, "10:05": (100.5, 100.6, 99.9, 100.2)}
    df2 = build_day(flat_day, fill_runs={("09:55", "10:04"): calm, ("10:06", "10:29"): calm,
                                         ("10:31", "15:54"): (98.7, 98.9, 98.65, 98.7)})
    e = X.benchmark_record(df2, fill)["exits"]["wick_hybrid"]
    assert (e["reason"], e["exit"], e["exit_time"], e["armed"]) == ("stop", 98.5, utc("10:30"), False), e


def test_exits_short_side_and_early_close():
    day = {"09:54": (100.0, 100.2, 99.7, 99.9), "10:10": (99.9, 100.0, 96.8, 97.0)}
    fill = hand_fill(side=-1, t1=97.0, t2=96.0, inst="ES")
    rec = X.benchmark_record(build_day(day), fill)
    cost = 1.0e-4 * 100.0
    e = rec["exits"]["wick_fixed"]
    assert (e["reason"], e["exit"], e["stop0"]) == ("target", 97.0, 101.0) and abs(e["r"] - (3.0 - cost)) < 1e-12
    e = rec["exits"]["fixed_atr2.0"]
    assert abs(e["r"] - (3.0 - cost) / 2.4) < 1e-12 and rec["cost_bps"] == 1.0
    # a session that ends before 15:55 exits at its last close
    df = build_day({"09:54": (100.0, 100.2, 99.7, 99.9)}, extra_days=1)
    cut = df[(df.index <= pd.Timestamp(utc("12:54"))) | (df.index >= pd.Timestamp(utc("18:00")))]
    e = X.benchmark_record(cut, hand_fill(side=-1, t1=90.0, t2=None))["exits"]["wick_fixed"]
    assert (e["reason"], e["exit_time"], e["exit"]) == ("session_end", utc("12:54"), 100.0), e


def test_exits_swing_trail_by_hand():
    """A confirmed 5-minute swing low ABOVE the wick stop becomes the stop once its
    right-hand window closes, and not a minute before."""
    # bins 10:00..10:20, lows 100.4 / 100.4 / 99.6 / 100.4 / 100.4 -> swing low 99.6 at the 10:10 bin,
    # confirmed when the 10:20 bin completes (the 10:24 minute), the stop from the 10:25 minute on. Before it
    # the newest swing low is 99.5 — the 09:50 bin (inert minutes, low 99.5) with higher bins either side,
    # confirmed at 10:04 — and before THAT the London low 97, which is under the initial stop and moves nothing.
    # (No minute can probe under 99.6 before 10:25 without un-making the swing: that is what `right=2` means.)
    up = (100.6, 100.8, 100.4, 100.6)
    day = {"09:54": (100.0, 100.3, 99.9, 100.2), "10:12": (100.6, 100.7, 99.6, 100.5),
           "10:40": (100.6, 100.7, 99.55, 100.0)}    # under 99.6: stops at 99.6, not at the low
    df = build_day(day, fill_runs={("09:55", "10:11"): up, ("10:13", "10:39"): up})
    rec = X.benchmark_record(df, hand_fill(t1=110.0, t2=None, wick=98.0))
    e = rec["exits"]["wick_swing"]
    assert (e["reason"], e["exit"], e["exit_time"], e["last_stop"]) == ("trail", 99.6, utc("10:40"), 99.6), e
    assert abs(e["r_gross"] - (-0.4 / 2.0)) < 1e-12
    assert rec["exits"]["wick_fixed"]["reason"] == "flat"


def test_exits_match_the_part1_replay_on_synthetic_walks():
    """The six rules from the wick stop against tjr_trailing_benchmark.replay —
    the reference the contract names — on seeded synthetic 1-minute sessions."""
    import tjr_trailing_benchmark as TB
    rng = np.random.default_rng(3)
    base = build_day(extra_days=1)
    n = len(base)
    close = 100.0 + np.cumsum(rng.normal(0, 0.08, n))
    opens = np.r_[100.0, close[:-1]]
    df = pd.DataFrame({"open": opens, "close": close,
                       "high": np.maximum(opens, close) + np.abs(rng.normal(0, 0.05, n)),
                       "low": np.minimum(opens, close) - np.abs(rng.normal(0, 0.05, n)),
                       "volume": 1.0}, index=base.index)[["open", "high", "low", "close", "volume"]]
    ctx = M.build_context(df, 1, 14)
    clock = M.session_clock(df.index)
    sh, sl = P.confirmed_swings(ctx.bars, 2, 2)
    sess = X.Session(df)
    checked, reasons = 0, collections.Counter()
    for hm, off in (("09:54", 0), ("10:03", 0), ("11:17", 0), ("13:40", 0), ("15:50", 0), ("09:51", 1), ("12:02", 1)):
        for side in (1, -1):
            for width, reach in ((0.3, 0.5), (0.6, 1.5), (1.5, 0.4)):
                stamp = pd.Timestamp(utc(hm, off))
                i0 = df.index.get_loc(stamp)
                entry = float(df["open"].iloc[i0])
                stop, target = entry - side * width, entry + side * reach
                trade = SimpleNamespace(side=side, entry=entry, stop=stop, target=target, entry_time=stamp)
                for rule in X.RULES:
                    ref = TB.replay(df, ctx, clock, sl, sh, trade, rule)
                    got = X.replay_exit(sess, side, entry, i0, stop, target, rule)
                    assert got["exit_idx"] == ref["exit_idx"] and abs(got["exit"] - ref["exit"]) < 1e-12, (hm, side, rule)
                    assert abs(got["r_gross"] - ref["r"]) < 1e-9 and got["armed"] == ref["armed"], (hm, side, rule)
                    assert got["reason"] == {"eod": "session_end"}.get(ref["reason"], ref["reason"]), (hm, side, rule)
                    assert got["stop"] == ref["stop_path"][-1]
                    reasons[got["reason"]] += 1
                    checked += 1
    assert {"stop", "trail", "target"} <= set(reasons), reasons
    print(f"      {checked} synthetic replays identical to tjr_trailing_benchmark.replay; reasons {dict(reasons)}")


def test_random_stop_is_seeded_by_the_trade_id():
    a, b = X.random_multiple("NQ-2026-07-15"), X.random_multiple("NQ-2026-07-15")
    assert a == b and 0.5 <= a <= 2.0
    others = {X.random_multiple(f"ES-2026-07-{d:02d}") for d in range(1, 29)}
    assert len(others) == 28 and all(0.5 <= u <= 2.0 for u in others) and max(others) - min(others) > 0.5
    day = {"09:54": (100.0, 100.2, 99.9, 100.0), "10:20": (100.0, 103.2, 99.9, 103.0)}
    fill = hand_fill()
    rec = X.benchmark_record(build_day(day), fill)
    r = rec[X.RANDOM_EXIT]
    assert r["atr_multiple"] == a and abs(r["stop0"] - (100.0 - a * 1.2)) < 1e-12 and r["rule"] == "fixed"
    assert r["reason"] == "target" and abs(r["r"] - (3.0 - 0.5e-4 * 100.0) / (a * 1.2)) < 1e-12
    assert X.benchmark_record(build_day(day), fill)[X.RANDOM_EXIT] == r
    assert set(rec["exits"]) == set(X.EXIT_NAMES) and X.RANDOM_EXIT not in rec["exits"]


def test_full_day_detector_to_benchmarks_by_hand():
    """The hand-built long day end to end: fill 99.7, T1 102 (nearest) prints at 10:30."""
    day = {**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}
    df = build_day(day, fill_runs=HIGH_RUN)
    f = one(hand(df, "NQ"), "fill")
    rec = X.benchmark_record(df, f)
    cost = 0.5e-4 * 99.7
    for name, (key, _) in X.MECHANICAL_EXITS.items():
        e = rec["exits"][name]
        risk = 99.7 - f["stops"][key]
        assert (e["reason"], e["exit"], e["exit_time"]) == ("target", 102.0, utc("10:30")), (name, e)
        assert abs(e["r"] - (102.0 - 99.7 - cost) / risk) < 1e-9, name
    assert rec["t1"]["printed"] and rec["t1"]["names"] == "LON_H+H1_SH" and rec["t2"]["printed"] is False
    # under the untaken rule the same entry aims at 103, which never prints: flat at 15:55
    f2 = one(hand(df, "NQ", target_rule="untaken"), "fill")
    rec2 = X.benchmark_record(df, f2)
    assert rec2["exits"]["wick_fixed"]["reason"] == "flat" and rec2["t1"]["price"] == 103.0
    assert rec2["targets_printed"]["nearest"]["t1"]["printed"] is True


def test_benchmark_record_shape_on_market_fills():
    """Shape only. No number from these records is printed, compared or returned."""
    allowed = {"stop", "trail", "target", "flat", "session_end"}
    for sym in ("NQ", "ES"):
        df = market(sym)
        sess = X.Session(df)
        fills = by_kind(market_events(sym), "fill")
        for f in fills:
            rec = X.benchmark_record(sess, f)
            json.dumps(rec, allow_nan=False)
            assert set(rec["exits"]) == set(X.EXIT_NAMES)
            for e in list(rec["exits"].values()) + [rec[X.RANDOM_EXIT]]:
                assert e["reason"] in allowed and e["exit_time"] >= f["bar_time"] and e["risk"] > 0
                assert e["exit_time"][:10] == f["bar_time"][:10] and e["exit_et"] <= "15:55"
        print(f"      {sym}: {len(fills)} benchmark records well-formed (numbers not shown)")


# ══════════════════════════════ STAGE 2: the trade, the human and the record ══════════════════════════════
#
# All of it on hand-built or synthetic minutes with a scripted human. Nothing below reads an
# ES/NQ file. The Telegram transport is always a fake; the "token" is a dummy string.

import shutil      # noqa: E402
import tempfile    # noqa: E402
import time as _time   # noqa: E402

from quantlab import alerts as A            # noqa: E402
from quantlab import validate as V          # noqa: E402
from quantlab.tjr_human import commands as C   # noqa: E402
from quantlab.tjr_human import journal as JN   # noqa: E402
from quantlab.tjr_human import report as R     # noqa: E402
from quantlab.tjr_human import trade as TR     # noqa: E402

FAKE_TOKEN = "000000:TEST-ONLY-NOT-A-TOKEN"
NQ_COST = 0.5e-4 * 100.0                     # 0.5 bp of an entry at 100


def epoch(stamp: str) -> float:
    return JN.bar_epoch(stamp)


def at(hms: str, day=None) -> float:
    """'HH:MM[:SS]' ET on the hand day (or `day`) as epoch seconds."""
    d = (day or DAY).date()
    return pd.Timestamp(f"{d} {hms}", tz=ET).timestamp()


def h_fill(side=1, entry=100.0, at_="09:54", inst="NQ", day=None, t1=103.0, t2=104.0, zone="eq",
           level="LON_L", conf=("bos", "ote"), cooccur=(), **stops) -> dict:
    """A detector-shaped fill: wick 1.0, 1.0/1.5/2.0 ATR = 1.2/1.8/2.4, session 1.5 from the entry."""
    day = day or DAY
    f = hand_fill(side=side, entry=entry, at=at_, t1=t1, t2=t2, inst=inst, **stops)
    bt = pd.Timestamp(f"{day.date()} {at_}", tz=ET).tz_convert("UTC").tz_localize(None).isoformat()
    f.update({"day": str(day.date()), "setup_id": f"{inst}-{day.date()}", "bar_time": bt, "knowable_at": bt,
              "entry_time": bt, "et": at_, "direction": "long" if side > 0 else "short",
              "level": {"name": level, "price": entry - side * 0.8, "class": D.level_class(level)},
              "extreme": entry - side * 0.9, "conf_types": list(conf),
              "zone": {"kind": zone, "low": entry - 0.2, "high": entry + 0.2, "formed_time": bt, "usable_time": bt},
              "cooccur": list(cooccur)})
    return f


def h_signal(f: dict, at_="09:40") -> dict:
    d = pd.Timestamp(f["day"])
    bt = pd.Timestamp(f"{d.date()} {at_}", tz=ET).tz_convert("UTC").tz_localize(None)
    s = {k: f[k] for k in ("instrument", "day", "setup_id", "side", "direction", "level", "extreme", "conf_types",
                           "zone", "targets", "target_rule", "t1", "t2", "atr")}
    s.update({"kind": "signal", "bar_time": bt.isoformat(), "et": at_,
              "knowable_at": (bt + pd.Timedelta(minutes=1)).isoformat(), "ref_price": f["entry"],
              "stops_indicative": dict(f["stops"])})
    return s


def h_trigger(f: dict) -> dict:
    bt = pd.Timestamp(f["bar_time"]) - pd.Timedelta(minutes=1)
    return {"kind": "entry_trigger", "instrument": f["instrument"], "day": f["day"], "setup_id": f["setup_id"],
            "bar_time": bt.isoformat(), "knowable_at": f["bar_time"], "entry_at": f["bar_time"],
            "side": f["side"], "wick_stop": f["stops"]["wick"], "atr": f["atr"]}


class Sim:
    """A scripted day: hand events + the bars of a frame -> TradeManager, in clock order.

    For each bar b closing at T: first the script items timed before T (they happened
    while b was forming), then the detector events knowable at T, then the bar itself.
    With `preview`, the fill is known at the entry instant (Detector.preview_fill's
    job); without, only when the entry minute has closed — the closed-bars-only feed.
    Script items: ("HH:MM:SS", "cmd", text) | ("HH:MM:SS", "price", p).
    """

    def __init__(self, df, fill, armed=True, preview=False, base=None, journal=None, manager=None, signal_at="09:40"):
        self.df, self.fill, self.preview = df, fill, preview
        self.inst = fill["instrument"]
        self.day = pd.Timestamp(fill["day"])
        self.own = base is None and journal is None
        self.base = base or (journal.base if journal else tempfile.mkdtemp(prefix="tjrh_"))
        self.j = journal or JN.Journal(self.base, fsync=False)
        self.m = manager or TR.TradeManager(self.j, armed=armed)
        self.signal, self.trigger = h_signal(fill, signal_at), h_trigger(fill)
        self.notices: list[dict] = []
        self.replies: list[str] = []

    def close(self):
        if self.own:
            shutil.rmtree(self.base, ignore_errors=True)

    def run(self, script=(), start="09:30", until="16:05", benchmarks=True, events=True):
        script = sorted(((at(t, self.day), k, v) for t, k, v in script), key=lambda x: x[0])
        lo = pd.Timestamp(f"{self.day.date()} {start}", tz=ET).tz_convert("UTC").tz_localize(None)
        hi = pd.Timestamp(f"{self.day.date()} {until}", tz=ET).tz_convert("UTC").tz_localize(None)
        bars = self.df.loc[lo:hi]
        q = 0
        for stamp, row in bars.iterrows():
            bt = stamp.isoformat()
            T = epoch(bt) + 60.0
            while q < len(script) and script[q][0] < T:
                self._item(*script[q])
                q += 1
            if events:
                for ev in (self.signal, self.trigger):
                    if epoch(ev["knowable_at"]) == T:
                        self._note(self.m.on_event(ev, T))
                if self.preview and epoch(self.fill["bar_time"]) == T:
                    self._note(self.m.on_preview({**self.fill, "provisional": True}, T))
                if epoch(self.fill["bar_time"]) + 60.0 == T:
                    self._note(self.m.on_event(self.fill, T))
            self._note(self.m.on_bar(self.inst, bt, row.open, row.high, row.low, row.close, T))
        while q < len(script):
            self._item(*script[q])
            q += 1
        if benchmarks:
            self.m.close_session(self.inst, self.df, epoch(bars.index[-1].isoformat()) + 120.0)
        return self

    def _item(self, t, kind, val):
        if kind == "cmd":
            self._note(self.m.on_command(C.parse(val), t))
        elif kind == "price":
            self._note(self.m.on_price(self.inst, val, t))
        else:
            raise AssertionError(kind)

    def _note(self, notices):
        self.notices += notices
        self.replies += [n["text"] for n in notices if n["kind"] == "reply"]

    def recs(self, kind=None, **match):
        out = [r for r in self.j.records() if kind is None or r["kind"] == kind]
        return [r for r in out if all(r.get(k) == v for k, v in match.items())]

    def state(self):
        return JN.rebuild(self.j.records())

    def the_trade(self):
        tr = self.state()["trades"]
        assert len(tr) == 1, [t["setup_id"] for t in tr]
        return tr[0]


def near(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) < tol


# ────────────────────────────── 8. commands: parsing ──────────────────────────────

def test_commands_parse_the_four_controls_and_the_reason_codes():
    cases = {
        "skip": ("SKIP", None, None, "unspecified"),
        "  /Skip@tjr_bot   GUT ": ("SKIP", None, None, "gut"),
        "nq  STOP   1.5   noise": ("STOP", "NQ", "atr1.5", "noise"),
        "Stop wick": ("STOP", None, "wick", "unspecified"),
        "STOP 2": ("STOP", None, "atr2.0", "unspecified"),
        "stop 1.0 atr": ("STOP", None, "atr1.0", "unspecified"),
        "stop session ES1! news": ("STOP", "ES", "session", "news"),
        "stop 20,012.5": ("STOP", None, 20012.5, "unspecified"),
        "STOP 19987.25 structure_changed": ("STOP", None, 19987.25, "structure_changed"),
        "move  stop 20010 structure changed": ("MOVE_STOP", None, 20010.0, "structure_changed"),
        "MOVE_STOP 5012.75 es": ("MOVE_STOP", "ES", 5012.75, "unspecified"),
        "exit now News": ("EXIT_NOW", None, None, "news"),
        "EXIT": ("EXIT_NOW", None, None, "unspecified"),
        "CME_MINI:NQ1! exit  now": ("EXIT_NOW", "NQ", None, "unspecified"),
        "gut": ("REASON", None, None, "gut"),
        " Structure_Changed ": ("REASON", None, None, "structure_changed"),
    }
    for text, want in cases.items():
        c = C.parse(text)
        assert c.ok and (c.action, c.instrument, c.arg, c.reason) == want, (text, c)
    for text in ("", "   ", "hello", "stop", "stop wide", "STOP wick 1.5", "move stop", "move stop wick", "skip 100",
                 "exit now 100", "NQ ES skip", "stop -5", "stop 1e5", "buy", "STOP 20012,5", None):
        c = C.parse(text)
        assert not c.ok and c.action is None and c.error, (text, c)
    assert set(C.REASONS) == {"noise", "structure_changed", "news", "gut"}


# ────────────────────────────── 9. the trade: every acceptance, every rejection ──────────────────────────────

def test_trade_no_human_is_the_wick_fixed_exit():
    """Nothing valid by the fill -> the wick stop from the first tick; a human who does
    nothing IS the wick_fixed benchmark: same exit, same minute, same net R."""
    for day, want_reason, want_r in (
            ({"10:20": (100.0, 103.2, 99.9, 103.0)}, "target", (3.0 - NQ_COST) / 1.0),
            ({"09:54": (100.0, 100.2, 98.9, 99.5)}, "stop", (-1.0 - NQ_COST) / 1.0),
            ({"11:00": (99.5, 99.6, 98.0, 98.2)}, "stop", (-1.0 - NQ_COST) / 1.0),
            ({"11:00": (98.7, 98.9, 98.0, 98.2)}, "stop", (98.7 - 100.0 - NQ_COST) / 1.0),    # a gap through: the open
            ({}, "flat", (0.0 - NQ_COST) / 1.0)):
        s = Sim(build_day(day), h_fill()).run()
        try:
            t = s.the_trade()
            assert t["exit_reason"] == want_reason and near(t["r"], want_r) and near(t["r_journaled"], want_r), (day, t["r"])
            assert t["stop_mode"] == "wick" and t["stop_when"] == "default" and t["stop0"] == 99.0
            wf = t["benchmarks"]["exits"]["wick_fixed"]
            assert near(wf["r"], t["r"], 1e-12) and wf["exit_time"] == t["exit"]["exit_bar"] and wf["exit"] == t["exit"]["exit"]
            assert t["benchmarks"]["human"]["held"]["r"] == wf["r"] and t["complete"]
            assert JN.integrity(s.base)["ok"]
        finally:
            s.close()


def test_trade_short_side_mirror():
    day = {"10:20": (100.0, 100.1, 96.8, 97.0)}
    s = Sim(build_day(day), h_fill(side=-1, t1=97.0, t2=96.0)).run([("09:45:00", "cmd", "stop 1.5 gut")])
    try:
        t = s.the_trade()
        assert t["exit_reason"] == "target" and t["stop0"] == 101.8 and near(t["r"], (3.0 - NQ_COST) / 1.8)
        assert near(t["benchmarks"]["exits"]["fixed_atr1.5"]["r"], t["r"], 1e-12)
    finally:
        s.close()


def test_stop_named_width_before_the_fill_resolves_at_the_fill():
    # the entry minute trades 98.9: through the wick (99.0), not through the 2.0 ATR stop (97.6)
    day = {"09:54": (100.0, 100.2, 98.9, 99.5), "10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill()).run([("09:45:10", "cmd", "stop 2.0 gut")])
    try:
        assert "noted" in s.replies[0] and "resolved at the fill" in s.replies[0]
        pend = s.recs("command", stage="stop_pending")
        assert len(pend) == 1 and pend[0]["arg"] == "atr2.0" and pend[0]["seconds_since_entry"] is None
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["when"] == "before_fill" and ss[0]["price"] == 97.6 and ss[0]["mode"] == "atr2.0"
        assert ss[0]["human_reason"] == "gut" and ss[0]["seconds_from_fill"] < 0 and ss[0]["candidates"]["wick"] == 99.0
        t = s.the_trade()
        assert t["exit_reason"] == "target" and t["stop0"] == 97.6 and near(t["r"], (3.0 - NQ_COST) / 2.4)
        assert near(t["benchmarks"]["exits"]["fixed_atr2.0"]["r"], t["r"], 1e-12)       # the human IS fixed_atr2.0 here
        assert t["benchmarks"]["exits"]["wick_fixed"]["reason"] == "stop"
        assert t["stop_mode"] == "atr2.0" and t["stop_when"] == "before_fill" and near(t["stop_atr_multiple"], 2.0)
    finally:
        s.close()


def test_stop_price_is_validated_at_the_fill_inside_the_band():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    # inside [97.6, 99.0]
    s = Sim(build_day(day), h_fill()).run([("09:46:00", "cmd", "STOP 98.0 noise")])
    try:
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["mode"] == "price" and ss[0]["price"] == 98.0 and ss[0]["band"] == [97.6, 99.0]
        assert near(s.the_trade()["r"], (3.0 - NQ_COST) / 2.0) and not s.recs("reject")
    finally:
        s.close()
    # wider than 2.0 ATR, and tighter than the wick: both rejected at the fill, with the requested price
    for text, px in (("stop 97.0", 97.0), ("stop 99.5", 99.5), ("stop 101", 101.0)):
        s = Sim(build_day(day), h_fill()).run([("09:46:00", "cmd", text)])
        try:
            rj = s.recs("reject")
            assert len(rj) == 1 and rj[0]["why"] == "stop_outside_band" and rj[0]["requested"] == px, rj
            assert rj[0]["band"] == [97.6, 99.0] and not s.recs("stop_set")
            t = s.the_trade()
            assert t["stop0"] == 99.0 and t["stop_mode"] == "wick" and near(t["r"], (3.0 - NQ_COST) / 1.0)
            assert any("outside the band" in x for x in s.replies)
        finally:
            s.close()
    # the band edges themselves are inside
    for px in (97.6, 99.0):
        s = Sim(build_day(day), h_fill()).run([("09:46:00", "cmd", f"stop {px}")])
        try:
            assert s.the_trade()["stop0"] == px and not s.recs("reject")
        finally:
            s.close()


def test_a_rejected_stop_is_not_the_one_choice_but_a_second_stop_is_rejected():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    # a price rejected at the fill leaves the human the 60 s: with the fill known at the entry instant
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:46:00", "cmd", "stop 97.0"),
                                                         ("09:54:20", "cmd", "stop 1.5 noise")])
    try:
        assert [r["why"] for r in s.recs("reject")] == ["stop_outside_band"]
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["when"] == "after_fill" and ss[0]["price"] == 98.2 and ss[0]["seconds_from_fill"] == 20.0
        assert s.the_trade()["stop0"] == 98.2
    finally:
        s.close()
    # once per trade: a second STOP, before or after the fill, is rejected and journaled with what was asked
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:45:00", "cmd", "stop 1.5"), ("09:47:00", "cmd", "stop 2.0"),
                                                         ("09:54:10", "cmd", "stop 98.5 gut")])
    try:
        rj = s.recs("reject")
        assert [r["why"] for r in rj] == ["stop_already_used"] * 2 and [r["requested"] for r in rj] == ["atr2.0", 98.5]
        assert rj[1]["human_reason"] == "gut" and rj[1]["seconds_since_entry"] == 10.0 and rj[1]["unrealised_r"] is not None
        assert len(s.recs("stop_set")) == 1 and s.the_trade()["stop0"] == 98.2
    finally:
        s.close()


def test_grace_window_the_wick_stop_is_live_and_a_trade_it_stops_stays_stopped():
    """THE grace case (9.3): the fill is known at 09:54:00; STOP 2.0 arrives at 09:54:30 and is
    accepted — but the entry minute trades through the wick. The trade is stopped at the wick,
    R is against the wick, the choice never took effect, and nothing reopens it."""
    day = {"09:54": (100.0, 100.2, 98.9, 99.5), "10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:54:30", "cmd", "stop 2.0"),
                                                         ("09:55:10", "cmd", "stop 1.5"),
                                                         ("09:55:20", "cmd", "move stop 98")])
    try:
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["when"] == "after_fill" and ss[0]["eff_bar"] == at("09:55")
        x = s.recs("exit")
        assert len(x) == 1 and x[0]["reason"] == "stop" and x[0]["exit"] == 99.0 and x[0]["stop0"] == 99.0
        assert x[0]["grace"] is True and x[0]["chosen_stop_void"] is True and x[0]["stop_mode"] == "wick"
        assert near(x[0]["r"], -1.0 - NQ_COST)
        assert [r["why"] for r in s.recs("reject")] == ["not_in_trade", "not_in_trade"]
        t = s.the_trade()
        assert t["stop_mode"] == "wick" and t["stop_when"] == "default" and near(t["r"], -1.0 - NQ_COST)
        assert near(t["benchmarks"]["exits"]["wick_fixed"]["r"], t["r"], 1e-12)
        assert any("never took effect" in n["text"] for n in s.notices if n["kind"] == "trade")
    finally:
        s.close()
    # the same with an observed price: 98.95 prints at 09:54:20, BEFORE the message -> stopped, and STOP is refused
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:54:20", "price", 98.95), ("09:54:30", "cmd", "stop 2.0")])
    try:
        x = s.recs("exit")
        assert len(x) == 1 and x[0]["via"] == "tick" and x[0]["exit"] == 99.0 and near(x[0]["r"], -1.0 - NQ_COST)
        assert [r["why"] for r in s.recs("reject")] == ["not_in_trade"] and not s.recs("stop_set")
    finally:
        s.close()
    # an observed price through the wick AFTER the accepted message, still inside the entry minute: the wick is live
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:54:30", "cmd", "stop 2.0"), ("09:54:45", "price", 98.95)])
    try:
        x = s.recs("exit")
        assert len(x) == 1 and x[0]["via"] == "tick" and x[0]["stop0"] == 99.0 and x[0]["chosen_stop_void"] is True
    finally:
        s.close()
    # and when the entry minute does NOT trade the wick, the chosen stop is the initial stop from 09:55 on
    day2 = {"09:55": (100.0, 100.1, 98.5, 99.0), "10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day2), h_fill(), preview=True).run([("09:54:30", "cmd", "stop 2.0 news")])
    try:
        t = s.the_trade()
        assert t["exit_reason"] == "target" and t["stop0"] == 97.6 and near(t["r"], (3.0 - NQ_COST) / 2.4)
        assert t["stop_when"] == "after_fill" and t["stop_seconds_from_fill"] == 30.0 and t["exit"]["grace"] is False
    finally:
        s.close()


def test_stop_window_closes_60_seconds_after_the_fill():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:55:00", "cmd", "stop 1.5")])      # exactly +60 s: inside
    try:
        assert len(s.recs("stop_set")) == 1 and not s.recs("reject")
    finally:
        s.close()
    s = Sim(build_day(day), h_fill(), preview=True).run([("09:55:01", "cmd", "stop 1.5")])
    try:
        rj = s.recs("reject")
        assert [r["why"] for r in rj] == ["stop_window_closed"] and rj[0]["requested"] == "atr1.5"
        assert rj[0]["seconds_since_entry"] == 61.0 and s.the_trade()["stop0"] == 99.0
    finally:
        s.close()
    # closed bars only: the fill is learnt at 09:55:00. A STOP typed at 09:54:20 (nobody knew of the fill yet)
    # is resolved when the fill arrives, as a choice made AFTER the entry instant
    s = Sim(build_day(day), h_fill()).run([("09:54:20", "cmd", "stop 2.0")])
    try:
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["when"] == "after_fill" and ss[0]["seconds_from_fill"] == 20.0
        assert s.the_trade()["stop0"] == 97.6
    finally:
        s.close()


def test_skip_from_signal_until_the_fill():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill()).run([("09:45:00", "cmd", "skip news"), ("09:46:00", "cmd", "skip"),
                                           ("09:47:00", "cmd", "stop 1.5"), ("10:00:00", "cmd", "exit now")])
    try:
        sk = s.recs("skip")
        assert [r["stage"] for r in sk] == ["command", "would_fill"]
        assert sk[0]["human_reason"] == "news" and sk[0]["ev"]["kind"] == "signal" and sk[0]["ev"]["zone"]["kind"] == "eq"
        assert sk[1]["ev"]["entry"] == 100.0
        assert [r["why"] for r in s.recs("reject")] == ["already_skipped", "setup_skipped", "not_in_trade"]
        assert not s.recs("fill") and not s.recs("exit")
        st = s.state()
        assert st["filled"] == 0 and len(st["skipped"]) == 1 and JN.filled_count(s.j.records()) == 0
        k = st["skipped"][0]
        assert k["would_fill"] and k["ended"] == "would_fill" and k["reason"] == "news" and k["benchmarks"]["skipped"] is True
        assert set(k["benchmarks"]["exits"]) == set(X.EXIT_NAMES) and "human" not in k["benchmarks"]
        assert near(k["benchmarks"]["exits"]["wick_fixed"]["r"], (3.0 - NQ_COST) / 1.0)      # what it would have done
    finally:
        s.close()
    # after the fill: rejected, with the fill known at the entry instant or not
    for preview, when in ((True, "09:54:05"), (False, "09:54:05"), (False, "09:56:00")):
        s = Sim(build_day(day), h_fill(), preview=preview).run([(when, "cmd", "skip gut")])
        try:
            rj = s.recs("reject")
            assert [r["why"] for r in rj] == ["skip_after_fill"] and s.state()["filled"] == 1, (preview, when)
            assert not s.recs("skip")
        finally:
            s.close()


def test_move_stop_only_in_the_trades_favour_after_the_stop_window():
    day = {"10:00": (100.0, 100.3, 99.4, 100.0),          # trades 99.4 in the minute the stop is moved to 99.5
           "10:01": (100.0, 100.1, 99.45, 99.6)}          # the first bar the moved stop owns: through it
    script = [("09:45:00", "cmd", "move stop 99.5"),      # before the fill
              ("09:54:30", "cmd", "move stop 99.5"),      # inside the STOP window
              ("09:58:00", "cmd", "move stop 98.0 gut"),  # a widen
              ("09:58:10", "cmd", "move stop 99.0"),      # no tighter than the stop
              ("09:58:20", "cmd", "move stop 100.2"),     # through the market (last price 100)
              ("10:00:10", "cmd", "move stop 99.5 structure_changed")]
    s = Sim(build_day(day), h_fill(), preview=True).run(script)
    try:
        rj = s.recs("reject")
        assert [r["why"] for r in rj] == ["move_before_fill", "move_in_stop_window", "move_widens", "move_widens",
                                          "move_through_market"]
        assert rj[2]["requested"] == 98.0 and rj[2]["human_reason"] == "gut" and rj[2]["stop"] == 99.0
        assert rj[2]["seconds_since_entry"] == 240.0 and near(rj[2]["unrealised_r"], 0.0) and rj[2]["market_price"] == 100.0
        mv = s.recs("stop_move")
        assert len(mv) == 1 and (mv[0]["from"], mv[0]["price"]) == (99.0, 99.5) and mv[0]["eff_bar"] == at("10:01")
        assert mv[0]["human_reason"] == "structure_changed" and mv[0]["seconds_since_entry"] == 370.0
        t = s.the_trade()
        # 10:00's low (99.4) is not evidence against a stop set at 10:00:10; 10:01 is: a trailed stop at 99.5,
        # R against the INITIAL stop (99.0)
        assert t["exit_reason"] == "trail" and t["exit"]["exit"] == 99.5 and t["exit"]["exit_bar"] == utc("10:01")
        assert near(t["r"], (-0.5 - NQ_COST) / 1.0) and t["stop0"] == 99.0 and t["exit"]["last_stop"] == 99.5
        assert len(t["moves"]) == 1 and len(t["rejects"]) == 5
    finally:
        s.close()
    # an observed price tests a moved stop at once
    s = Sim(build_day(day), h_fill(), preview=True).run([("10:00:10", "cmd", "move stop 99.5"),
                                                         ("10:00:40", "price", 99.48)])
    try:
        x = s.recs("exit")[0]
        assert x["via"] == "tick" and x["reason"] == "trail" and x["exit"] == 99.5 and x["exit_bar"] == utc("10:00")
    finally:
        s.close()
    # a stop moved beyond the entry locks the trade in: R is still against the initial stop
    day3 = {"10:10": (100.0, 101.6, 100.0, 101.5), "10:12": (101.5, 101.5, 100.9, 101.0)}
    s = Sim(build_day(day3, fill_runs={("10:11", "10:11"): (101.5, 101.6, 101.4, 101.5)}), h_fill(), preview=True)
    s.run([("10:11:30", "cmd", "move stop 101.0 noise")])
    try:
        t = s.the_trade()
        assert t["exit_reason"] == "trail" and near(t["r"], (1.0 - NQ_COST) / 1.0)
    finally:
        s.close()


def test_mae_keeps_the_drawdown_a_trailed_stop_later_sits_above():
    """-0.8 R first, then +1.1 R, then stopped at breakeven: the trade LIVED through -0.8 R. The
    human who moves the stop to the entry and the be_1r rule report the same excursions."""
    day = {"10:00": (100.0, 100.1, 99.2, 99.8), "10:10": (99.8, 101.1, 99.8, 101.0), "10:20": (100.6, 100.7, 99.9, 100.0)}
    runs = {("10:11", "10:19"): (101.0, 101.05, 100.6, 100.8)}
    s = Sim(build_day(day, fill_runs=runs), h_fill(), preview=True).run([("10:12:00", "cmd", "move stop 100.0 noise")])
    try:
        t = s.the_trade()
        be = t["benchmarks"]["exits"]["wick_be_1r"]
        assert (be["reason"], be["exit"], be["exit_time"]) == ("trail", 100.0, utc("10:20"))
        assert near(be["mae_r"], -0.8) and near(be["mfe_r"], 1.1)
        x = t["exit"]
        assert (x["reason"], x["exit"], x["exit_bar"]) == ("trail", 100.0, utc("10:20"))
        assert near(x["mae_r"], -0.8) and near(x["mfe_r"], 1.1) and near(t["r"], be["r"], 1e-12)
    finally:
        s.close()


def test_exit_now_fills_at_the_next_observed_price_reason_attached_after():
    day = {"10:05": (100.0, 100.4, 99.9, 100.35)}
    # with an observed price: executed first, the reason asked, the answer attached
    s = Sim(build_day(day), h_fill(), preview=True).run([("10:05:20", "cmd", "exit now"), ("10:05:25", "price", 100.3),
                                                         ("10:06:30", "cmd", "noise"), ("10:07:00", "cmd", "exit now")])
    try:
        assert "reason?" in s.replies[0] and "attached" in s.replies[1]
        x = s.recs("exit")
        assert len(x) == 1 and x[0]["reason"] == "exit_now" and x[0]["via"] == "tick" and x[0]["exit"] == 100.3
        assert x[0]["human_reason"] == "unspecified" and x[0]["seconds_since_entry"] == 685.0
        assert near(x[0]["r"], (0.3 - NQ_COST) / 1.0)
        rs = s.recs("command", stage="reason")
        assert len(rs) == 1 and rs[0]["reason"] == "noise" and rs[0]["attaches_to"] == x[0]["command_seq"]
        t = s.the_trade()
        assert t["human_reason"] == "noise" and near(t["r"], 0.3 - NQ_COST)
        assert [r["why"] for r in s.recs("reject")] == ["not_in_trade"]
        assert t["benchmarks"]["human"]["held"]["reason"] == "flat"
    finally:
        s.close()
    # closed bars only: the next observed price is the close of the bar that closes after the command
    s = Sim(build_day(day), h_fill()).run([("10:05:20", "cmd", "exit now gut"), ("10:05:40", "cmd", "exit now")])
    try:
        x = s.recs("exit")
        assert len(x) == 1 and x[0]["via"] == "bar" and x[0]["exit"] == 100.35 and x[0]["exit_bar"] == utc("10:05")
        assert x[0]["human_reason"] == "gut" and [r["why"] for r in s.recs("reject")] == ["exit_already_requested"]
    finally:
        s.close()
    # the stop comes first: the bar that would have carried the EXIT NOW trades through the stop
    s = Sim(build_day({"10:05": (100.0, 100.1, 98.9, 99.2)}), h_fill()).run([("10:05:20", "cmd", "exit now")])
    try:
        assert s.recs("exit")[0]["reason"] == "stop" and s.recs("exit")[0]["exit"] == 99.0
    finally:
        s.close()
    # before the fill, and a reason with nothing to attach to / too late
    s = Sim(build_day(day), h_fill()).run([("09:45:00", "cmd", "exit now"), ("09:45:30", "cmd", "news"),
                                           ("10:20:00", "cmd", "move stop 99.2"), ("10:31:00", "cmd", "gut")])
    try:
        assert [r["why"] for r in s.recs("reject")] == ["not_in_trade", "no_reason_target"]
        assert s.recs("command", stage="reason")[0]["reason"] == "news"              # a rejected action can be explained too
        assert len(s.recs("stop_move")) == 1                                         # and 11 minutes is too late for the move
    finally:
        s.close()


def test_unparsable_stale_and_ambiguous_commands_are_rejected_and_journaled():
    day = {}
    s = Sim(build_day(day), h_fill())
    es = h_fill(inst="ES")
    try:
        s.m.on_event(h_signal(es), at("09:41"))
        s.run([("09:45:00", "cmd", "hello there"), ("09:45:10", "cmd", "skip"), ("09:45:20", "cmd", "es skip gut")])
        rj = s.recs("reject")
        assert [r["why"] for r in rj] == ["unparsable", "ambiguous_instrument"] and rj[0]["text"] == "hello there"
        st = s.state()
        assert [k["setup_id"] for k in st["skipped"]] == [es["setup_id"]] and st["filled"] == 1
        # stale: Telegram stamped the message 5 minutes before it arrived
        c = C.parse("exit now")
        c.sent_ts = at("10:00:00")
        out = s.m.on_command(c, at("10:05:00"))
        assert "too old" in out[0]["text"] and s.recs("reject")[-1]["why"] == "stale"
        # every command, accepted or not, has its inbound record
        assert len(s.recs("command", stage="inbound")) == 4
    finally:
        s.close()


def test_observe_mode_journals_and_takes_no_entry():
    s = Sim(build_day({"10:20": (100.0, 103.2, 99.9, 103.0)}), h_fill(), armed=False, preview=True)
    s.run([("09:45:00", "cmd", "skip"), ("09:56:00", "cmd", "exit now")])
    try:
        kinds = collections.Counter(r["kind"] for r in s.j.records())
        assert kinds["observe"] == 3 and kinds["fill"] == 0 and kinds["exit"] == 0 and kinds["benchmarks"] == 0, kinds
        assert {r["event"] for r in s.recs("observe")} == {"signal", "entry_trigger", "fill"}
        assert [r["why"] for r in s.recs("reject")] == ["observe_mode"] * 2
        assert s.state()["filled"] == 0 and s.m.open_trades() == []
        sig = [n for n in s.notices if n["kind"] == "signal"]
        assert len(sig) == 1 and sig[0]["text"].startswith("OBSERVE") and sig[0]["fields"]["armed"] is False
        # arming later in the day starts with the NEXT signal, not this setup
        s.m.set_armed(True)
        assert s.m.trade("NQ").phase == "observe"
    finally:
        s.close()


def test_signal_alert_carries_what_the_chart_draws_and_notices_are_the_three_kinds():
    assert {"signal", "trade", "reply"} <= A.PUSH_KINDS and {"fill", "stale", "halt", "test"} <= A.PUSH_KINDS
    s = Sim(build_day({"10:20": (100.0, 103.2, 99.9, 103.0)}), h_fill(), preview=True).run([("09:45:00", "cmd", "stop 1.5 gut")])
    try:
        assert {n["kind"] for n in s.notices} == {"signal", "trade", "reply"}
        sig = [n for n in s.notices if n["kind"] == "signal"][0]
        for k in ("level", "conf_types", "zone", "stops_indicative", "t1", "t2", "side", "target_rule"):
            assert k in sig["fields"], k
        json.dumps(s.notices, allow_nan=False)
        for word in ("wick", "1.0 ATR", "2.0 ATR", "session", "T1", "T2", "SKIP"):
            assert word in sig["text"], word
        sent = []
        out = TR.send_notices(s.base, s.notices, notify=lambda base, kind, text, fields=None: sent.append(kind) or {"kind": kind})
        assert sent == [n["kind"] for n in s.notices] and len(out) == len(sent)
        assert TR.send_notices(s.base, s.notices[:1], notify=lambda *a, **k: 1 / 0)[0]["pushed"] is False   # never raises
    finally:
        s.close()


def test_provisional_fill_is_confirmed_corrected_or_voided_by_the_closed_bar():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill(), preview=True).run()
    try:
        f = s.recs("fill")
        assert [r["provisional"] for r in f] == [True, False] and f[1]["preview_match"] is True
        assert s.state()["filled"] == 1
    finally:
        s.close()
    # the closed bar opens elsewhere: the closed bar is the record, a named width is re-resolved
    s = Sim(build_day(day), h_fill(), preview=True)
    try:
        f = s.fill
        s.m.on_event(s.signal, at("09:41"))
        s.m.on_command(C.parse("stop 1.5"), at("09:45"))
        s.m.on_preview({**h_fill(entry=100.25), "provisional": True}, at("09:54"))
        assert s.m.trade("NQ").newest_stop()["price"] == 100.25 - 1.8
        s.m.on_event(f, at("09:55"))
        tr = s.m.trade("NQ")
        assert tr.entry == 100.0 and tr.newest_stop()["price"] == 98.2 and not tr.provisional
        assert s.recs("fill")[-1]["preview_match"] is False
        # and one the closed bar refuses altogether
        s2 = Sim(build_day(day), h_fill(), preview=True)
        s2.m.on_event(s2.signal, at("09:41"))
        s2.m.on_preview({**s2.fill, "provisional": True}, at("09:54"))
        inv = {k: s2.fill[k] for k in ("instrument", "day", "setup_id", "bar_time", "knowable_at", "side")}
        inv.update(kind="invalidated", reason="no_risk", traded=98.9, wick_stop=99.0)
        s2.m.on_event(inv, at("09:55"))
        assert s2.state()["filled"] == 0 and s2.m.open_trades() == [] and s2.recs("invalidated")[0]["void"] is True
        s2.close()
    finally:
        s.close()


def test_full_hand_day_detector_to_manager_to_journal():
    """The real detector on the hand-built long day, minute by minute, with preview_fill at the
    entry instant: signal 09:40, fill 99.7 at 09:54, T1 102 prints at 10:30."""
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
    base = tempfile.mkdtemp(prefix="tjrh_")
    try:
        j = JN.Journal(base, fsync=False)
        m = TR.TradeManager(j, armed=True)
        det = D.Detector("NQ")
        cut = df.index.get_loc(pd.Timestamp(utc("09:00")))
        det.feed(df.iloc[:cut])                                        # history: not routed to the manager
        script = {utc("09:45"): "stop session gut"}
        for k in range(cut, len(df)):
            row, bt = df.iloc[k], df.index[k].isoformat()
            T = epoch(bt) + 60.0
            for ev in det.feed(df.iloc[k:k + 1]):
                m.on_event(ev, T)
                if ev["kind"] == "entry_trigger":
                    m.on_preview(det.preview_fill(float(df.iloc[k + 1].open)), T)
            m.on_bar("NQ", bt, row.open, row.high, row.low, row.close, T)
            if bt in script:
                m.on_command(C.parse(script[bt]), T + 1.0)
        m.close_session("NQ", df, epoch(df.index[-1].isoformat()) + 60.0)
        st = JN.rebuild(j.records())
        t = st["trades"][0]
        f = one(hand(df, "NQ"), "fill")
        assert st["filled"] == 1 and t["entry"] == 99.7 and t["stop0"] == f["stops"]["session"] and t["stop_mode"] == "session"
        assert t["exit_reason"] == "target" and t["exit"]["exit"] == 102.0 and t["exit"]["exit_bar"] == utc("10:30")
        assert near(t["r"], (102.0 - 99.7 - 0.5e-4 * 99.7) / (99.7 - f["stops"]["session"]))
        assert near(t["benchmarks"]["exits"]["fixed_session"]["r"], t["r"], 1e-12)
        assert (t["level_type"], t["level_class"], t["zone"], t["conf"]) == (D.level_type(f["level"]["name"]),
                                                                            f["level"]["class"], "eq", "bos+ifvg+ote")
        kinds = collections.Counter(r["kind"] for r in j.records())
        assert kinds["signal"] == 1 and kinds["fill"] == 2 and kinds["exit"] == 1 and kinds["benchmarks"] == 1
        assert {r["event"] for r in j.records() if r["kind"] == "event"} >= {"levels", "sweep", "confirmation", "touch", "entry_trigger"}
        assert JN.integrity(base)["ok"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 10. the inbound transport (always a fake) ──────────────────────────────

class FakeTelegram:
    def __init__(self, batches):
        self.batches, self.calls = list(batches), []

    def __call__(self, url, body, timeout):
        self.calls.append({"url": url, "body": dict(body), "timeout": timeout})
        result = self.batches.pop(0) if self.batches else []
        if isinstance(result, Exception):
            raise result
        return 200, json.dumps({"ok": True, "result": result})


def _upd(uid, chat, text, date=None):
    date = int(at("09:44:50")) if date is None else date
    msg = {"message_id": uid, "date": date, "chat": {"id": chat, "type": "private"}}
    if text is not None:
        msg["text"] = text
    return {"update_id": uid, "message": msg}


def _with_env(fn, token=FAKE_TOKEN, chat="42"):
    saved = {k: os.environ.get(k) for k in (A.TOKEN_VAR, A.CHAT_VAR)}
    real = C._fetch
    try:
        for k, v in ((A.TOKEN_VAR, token), (A.CHAT_VAR, chat)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn()
    finally:
        C._fetch = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_inbound_accepts_only_the_configured_chat_and_journals_everything():
    base = tempfile.mkdtemp(prefix="tjrh_")

    def body():
        j = JN.Journal(base, fsync=False)
        fake = FakeTelegram([[_upd(11, 42, "nq skip gut"), _upd(12, 99, "EXIT NOW"), _upd(13, 42, None),
                              _upd(14, 42, "what is this")], []])
        C._fetch = fake
        tg = C.TelegramInbound(j, long_poll_s=0)
        got = tg.poll()
        assert [(c.action, c.instrument, c.reason, c.update_id) for c in got if c.ok] == [("SKIP", "NQ", "gut", 11)]
        assert len(got) == 3 and [c.ok for c in got] == [True, False, False]          # unreadable ones go to the manager too
        assert all(c.seq is not None and c.sent_ts == at("09:44:50") for c in got)
        recs = j.records()
        cm = [r for r in recs if r["kind"] == "command"]
        assert [r["update_id"] for r in cm] == [11, 12, 13, 14] and [r["authorized"] for r in cm] == [True, False, True, True]
        assert cm[1]["chat_id"] == "99" and cm[1]["text"] == "EXIT NOW" and cm[1]["instrument"] == "RUN"
        rj = [r for r in recs if r["kind"] == "reject"]
        assert len(rj) == 1 and rj[0]["why"] == "unauthorized_chat" and rj[0]["command_seq"] == cm[1]["seq"]
        # the offset confirms what was read, and goes out with the next request
        assert tg.offset == 15 and "offset" not in fake.calls[0]["body"]
        assert tg.poll() == [] and fake.calls[1]["body"]["offset"] == 15
        assert fake.calls[0]["url"].endswith("/getUpdates") and fake.calls[0]["body"]["timeout"] == 0
        # the stranger's EXIT NOW reached nobody: a manager with a live trade never saw it
        m = TR.TradeManager(j, armed=True)
        f = h_fill()
        m.on_event(h_signal(f), at("09:41"))
        for c in got:
            m.on_command(c, at("09:45"))
        assert m.trade("NQ").phase == "skipped"
        assert [r["why"] for r in j.records() if r["kind"] == "reject"] == ["unauthorized_chat", "unparsable", "unparsable"]
        assert len([r for r in j.records() if r["kind"] == "command" and r.get("stage") == "inbound"]) == 4   # none journaled twice
        # a restart does not act twice: the offset comes back from the journal
        assert C.TelegramInbound(JN.Journal(base, fsync=False)).offset == 15
        fake2 = FakeTelegram([[_upd(14, 42, "skip"), _upd(15, 42, "es skip")]])
        C._fetch = fake2
        again = C.TelegramInbound(JN.Journal(base, fsync=False), long_poll_s=0).poll()
        assert [c.update_id for c in again] == [15] and fake2.calls[0]["body"]["offset"] == 15
        # the token is in no record and no file
        for p in JN.journal_files(base):
            assert FAKE_TOKEN not in p.read_text(encoding="utf-8")

    try:
        _with_env(body)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_inbound_never_raises_never_hangs_never_leaks_the_token():
    base = tempfile.mkdtemp(prefix="tjrh_")

    def body():
        j = JN.Journal(base, fsync=False)
        url_echo = RuntimeError(f"cannot reach https://api.telegram.org/bot{FAKE_TOKEN}/getUpdates")
        C._fetch = FakeTelegram([url_echo])
        tg = C.TelegramInbound(j, long_poll_s=0)
        assert tg.poll() == [] and tg.last_error and FAKE_TOKEN not in tg.last_error and "<token>" in tg.last_error
        C._fetch = lambda url, b, timeout: (502, f"<html>bad gateway\nbot{FAKE_TOKEN}</html>")
        assert tg.poll() == [] and FAKE_TOKEN not in tg.last_error and "\n" not in tg.last_error
        C._fetch = lambda url, b, timeout: (200, "not json")
        assert tg.poll() == [] and tg.last_error
        C._fetch = lambda url, b, timeout: (200, json.dumps({"ok": True, "result": ["garbage", {"update_id": 7}]}))
        assert tg.poll() == [] and tg.offset == 8                       # malformed updates are skipped, not fatal
        # a server that never answers: abandoned by the clock, like alerts.py
        margin = C.ATTEMPT_MARGIN_S
        C.ATTEMPT_MARGIN_S = 0.3
        try:
            C._fetch = lambda url, b, timeout: _time.sleep(3.0) or (200, "{}")
            t0 = _time.monotonic()
            assert tg.poll(0) == [] and _time.monotonic() - t0 < 1.5 and "abandoned" in tg.last_error
        finally:
            C.ATTEMPT_MARGIN_S = margin
        # the background loop hands over what arrives
        C._fetch = FakeTelegram([[_upd(21, 42, "exit now")]])
        tg2 = C.TelegramInbound(j, long_poll_s=0)
        assert tg2.start() is True
        deadline = _time.monotonic() + 5.0
        got = []
        while not got and _time.monotonic() < deadline:
            got = tg2.drain()
            _time.sleep(0.02)
        tg2.stop()
        assert [c.action for c in got] == ["EXIT_NOW"]
        # the journal itself refuses a record that carries the token
        try:
            j.write("command", {"text": f"my token is {FAKE_TOKEN}"})
        except ValueError:
            pass
        else:
            raise AssertionError("a record carrying the token must be refused")
        for p in JN.journal_files(base):
            assert FAKE_TOKEN not in p.read_text(encoding="utf-8")

    def unset():
        calls = FakeTelegram([])
        C._fetch = calls
        j = JN.Journal(base, fsync=False)
        tg = C.TelegramInbound(j)
        assert tg.poll() == [] and tg.start() is False and calls.calls == [] and not C.configured()
        # and the manager still journals with nothing configured
        m = TR.TradeManager(j, armed=True)
        m.on_event(h_signal(h_fill()), at("09:41"))
        assert [r["kind"] for r in j.records()][-1] == "signal"

    try:
        _with_env(body)
        _with_env(unset, token=None, chat=None)
        _with_env(unset, token=FAKE_TOKEN, chat=None)
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 11. the journal ──────────────────────────────

def test_journal_round_trip_files_sequence_and_stamps():
    base = tempfile.mkdtemp(prefix="tjrh_")
    try:
        clock = {"t": pd.Timestamp("2026-07-31 23:59:58.250", tz="UTC").timestamp()}
        j = JN.Journal(base, clock=lambda: clock["t"], fsync=True)
        a = j.write("signal", {"setup_id": "NQ-2026-07-31", "x": np.float64(1.5), "nan": float("nan"),
                               "arr": (1, 2), "flag": np.bool_(True)}, instrument="NQ")
        clock["t"] += 0.5
        b = j.write("signal", {"setup_id": "ES-2026-07-31"}, instrument="CME_MINI:ES1!")
        clock["t"] += 2.0                                               # into August
        c = j.write("exit", {"setup_id": "NQ-2026-07-31", "exit": 1.0}, instrument="NQ")
        d = j.write("armed", {"reason": "test"})
        clock["t"] -= 30.0                                              # a clock that steps back never un-orders the journal
        e = j.write("reject", {"why": "x"}, instrument="NQ")
        assert [r["seq"] for r in (a, b, c, d, e)] == [1, 2, 3, 4, 5]
        assert a["ts"] == "2026-07-31T23:59:58.250Z" and a["ts_ms"] == 1785542398250 and e["ts_ms"] == d["ts_ms"]
        assert sorted(p.name for p in JN.journal_files(base)) == ["ES-2026-07.jsonl", "NQ-2026-07.jsonl",
                                                                  "NQ-2026-08.jsonl", "RUN-2026-08.jsonl"]
        back = JN.read_journal(base)
        assert back == [a, b, c, d, e] and back[0]["x"] == 1.5 and back[0]["nan"] is None and back[0]["arr"] == [1, 2]
        assert back[0]["flag"] is True and back[1]["instrument"] == "ES" and back[3]["instrument"] == "RUN"
        assert JN.integrity(base) == {"ok": True, "records": 5, "files": 4, "bad_lines": 0, "first_seq": 1,
                                      "last_seq": 5, "problems": []}
        try:
            j.write("order", {})
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown kind must be refused")
        assert set(JN.CONTRACT_KINDS) == {"signal", "command", "fill", "stop_set", "stop_move", "reject", "exit", "skip",
                                          "invalidated", "benchmarks", "review", "observe"}
        # a second writer (the `arm` command while the runner is up): the sequence carries on from the disk
        j2 = JN.Journal(base, clock=lambda: clock["t"] + 60.0, fsync=False)
        assert j2.write("armed", {"reason": "cli"})["seq"] == 6
        assert j.write("reject", {"why": "y"}, instrument="NQ")["seq"] == 7
        # a torn last line (the process died inside a write) is skipped, counted, and the next record is whole
        path = os.path.join(base, "NQ-2026-08.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"seq": 8, "kind": "exi')
        j3 = JN.Journal(base, clock=lambda: clock["t"] + 120.0, fsync=False)
        g = j3.write("exit", {"setup_id": "NQ-2026-08-03"}, instrument="NQ")
        assert g["seq"] == 8 and JN.read_journal(base)[-1] == g
        # ARMED: the last record decides; a disarm is an `armed` record that says false
        assert JN.armed_record(JN.read_journal(base))["reason"] == "cli"
        j3.write("armed", {"armed": False, "reason": "changed my mind before trade one"})
        assert JN.armed_record(JN.read_journal(base)) is None and JN.rebuild(JN.read_journal(base))["armed"] is None
        chk = JN.integrity(base)
        assert chk["bad_lines"] == 1 and chk["records"] == 9 and not chk["problems"] and chk["ok"] is False
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_journal_gap_record_on_restart():
    day = {"10:20": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day), h_fill(), preview=True)
    try:
        assert s.j.mark_start(now=at("09:00")) is None                  # an empty journal has no gap
        s.run([("09:45:00", "cmd", "stop 1.5")], until="10:00", benchmarks=False)
        last = s.j.records()[-1]
        t_last = last["ts_ms"] / 1000.0
        # restarted with a trade on: always a gap record, and it names the trade
        j2 = JN.Journal(s.base, fsync=False)
        g = j2.mark_start(now=t_last + 5.0)
        assert g["kind"] == "gap" and g["open_trades"] == [s.fill["setup_id"]] and g["last_seq"] == last["seq"]
        assert g["seconds"] == 5.0 and g["instrument"] == "RUN" and g["last_ts"] == last["ts"]
        s.run(start="10:01", events=False)                              # the trade ends
        t_end = s.j.records()[-1]["ts_ms"] / 1000.0
        assert JN.Journal(s.base, fsync=False).mark_start(now=t_end + 10.0) is None     # nothing open, no time lost
        g2 = JN.Journal(s.base, fsync=False).mark_start(now=t_end + 3600.0)
        assert g2["seconds"] == 3600.0 and g2["open_trades"] == []
        assert [x["seq"] for x in JN.rebuild(s.j.records())["gaps"]] == [g["seq"], g2["seq"]] and JN.integrity(s.base)["ok"]
    finally:
        s.close()


def test_restart_resumes_the_trade_from_the_journal():
    """Stop the process at 10:02 with a chosen and a moved stop; a new manager, `resume`, the store
    re-fed from 09:30: no event is journaled twice, the moved stop is back, the gap's bars are applied."""
    day = {"10:30": (100.0, 100.1, 99.3, 99.35)}                         # through the moved stop (99.4), not the chosen (98.2)
    script = [("09:45:00", "cmd", "stop 1.5 gut"), ("10:00:10", "cmd", "move stop 99.4 noise")]
    whole = Sim(build_day(day), h_fill(), preview=True).run(script)
    s = Sim(build_day(day), h_fill(), preview=True)
    try:
        s.run(script, until="10:02", benchmarks=False)
        n_before = len(s.j.records())
        j2 = JN.Journal(s.base, fsync=False)
        assert j2.mark_start(now=at("10:20"))["open_trades"] == [s.fill["setup_id"]]
        m2 = TR.TradeManager(j2, armed=False)
        info = m2.resume()
        assert info["armed"] is False and info["seen_events"] == 3        # no ARMED record here: the flag rides on the signal
        s2 = Sim(s.df, s.fill, preview=False, journal=j2, manager=m2).run()
        recs = j2.records()
        assert collections.Counter(r["kind"] for r in recs)["signal"] == 1
        assert len([r for r in recs if r["kind"] == "fill"]) == 2 and len([r for r in recs if r["kind"] == "stop_set"]) == 1
        assert len(recs) == n_before + 3                                  # gap, exit, benchmarks
        a, b = JN.rebuild(recs)["trades"][0], whole.the_trade()
        for k in ("r", "stop0", "stop_mode", "exit_reason", "entry"):
            assert a[k] == b[k], (k, a[k], b[k])
        assert a["exit"]["exit"] == 99.4 and a["exit"]["exit_bar"] == utc("10:30") and a["exit_reason"] == "trail"
        assert near(a["r"], (-0.6 - NQ_COST) / 1.8)
        s2.close()
    finally:
        s.close()
        whole.close()


# ────────────────────────────── 12. the reports ──────────────────────────────

def synth_frame(day: pd.Timestamp, seed: int) -> pd.DataFrame:
    """One synthetic session of 1-minute bars: a seeded random walk around 100."""
    ix = pd.date_range(pd.Timestamp(f"{(day - pd.Timedelta(days=1)).date()} 18:00", tz=ET),
                       pd.Timestamp(f"{day.date()} 16:59", tz=ET), freq="1min")
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.normal(0.0, 0.09, len(ix)))
    o = np.concatenate([[100.0], c[:-1]])
    wig = np.abs(rng.normal(0.0, 0.04, (2, len(ix))))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + wig[0], "low": np.minimum(o, c) - wig[1], "close": c,
                       "volume": 0.0}, index=ix.tz_convert("UTC").tz_localize(None))
    df.index.name = "date"
    return df


def scripted_human(i: int) -> list:
    """Deterministic, varied, and blind: it never looks at a price."""
    out = []
    k = i % 7
    if k == 1:
        out.append(("09:45:00", "cmd", "stop 1.5 gut"))
    elif k == 2:
        out.append(("09:46:00", "cmd", "stop 2.0"))
        out.append(("09:46:30", "cmd", "noise"))
    elif k == 3:
        out.append(("09:54:20", "cmd", "stop session news"))
    if i % 5 == 0:
        out.append(("10:15:10", "cmd", "exit now structure_changed"))
    if i % 6 == 1:
        out.append(("10:05:00", "cmd", "move stop 50 gut"))              # a widen attempt, rejected
    if i % 11 == 4:
        out.append(("09:50:00", "cmd", "skip gut"))
    return out


def _run_synthetic(n_fills: int, base: str, hooks=None) -> dict:
    j = JN.Journal(base, fsync=False)
    m = TR.TradeManager(j, armed=True)
    days = pd.bdate_range("2026-01-05", periods=400)
    i = 0
    while JN_filled(j) < n_fills:
        day = days[i]
        inst = "NQ" if i % 2 == 0 else "ES"
        df = synth_frame(day, 1000 + i)
        side = 1 if i % 3 else -1
        bt = pd.Timestamp(f"{day.date()} 09:54", tz=ET).tz_convert("UTC").tz_localize(None)
        entry = float(df.loc[bt, "open"])
        zone = ("eq", "fvg", "ob")[i % 3]
        f = h_fill(side=side, entry=entry, inst=inst, day=day, t1=entry + side * 2.0, t2=entry + side * 4.0, zone=zone,
                   level=("LON_L", "ASIA_L", "H1_SL", "PDL")[i % 4] if side > 0 else ("LON_H", "ASIA_H", "H4_SH", "PDH")[i % 4],
                   conf=(("bos",), ("bos", "ote"), ("ifvg",))[i % 3], cooccur=(("fvg",) if i % 2 else ()) if zone == "eq" else ())
        Sim(df, f, preview=True, journal=j, manager=m).run(scripted_human(i), until="16:00")
        if hooks:
            hooks(j, JN_filled(j))
        i += 1
    return {"journal": j, "days": i}


def JN_filled(j) -> int:
    return JN.filled_count(j.records())


def _independent_r(records: list[dict]) -> list[dict]:
    """The human's R and the ten exits' R per filled trade, straight from the records — no `rebuild`."""
    fills = {r["setup_id"]: r for r in records if r["kind"] == "fill" and not r.get("provisional")}
    exits = {r["setup_id"]: r for r in records if r["kind"] == "exit"}
    bench = {r["setup_id"]: r for r in records if r["kind"] == "benchmarks" and not r.get("skipped")}
    out = []
    for sid, f in sorted(fills.items(), key=lambda kv: kv[1]["seq"]):
        ev, x = f["ev"], exits[sid]
        cost = X.cost_points(f["instrument"], ev["entry"])
        r = (ev["side"] * (x["exit"] - ev["entry"]) - cost) / (ev["side"] * (ev["entry"] - x["stop0"]))
        out.append({"setup_id": sid, "human": r, **{n: bench[sid]["exits"][n]["r"] for n in X.EXIT_NAMES}})
    return out


def test_reports_reviews_once_at_25_and_50_final_refuses_at_99_runs_at_100():
    assert N_TRIALS == 14 and R.N_TRIALS == 14 and R.SAMPLE == 100 and R.PASS_BAR == 0.95 and R.REVIEW_AT == (25, 50)
    base = tempfile.mkdtemp(prefix="tjrh_")
    seen = {"reviews": [], "final99": None, "weekly40": None}

    def hooks(j, filled):
        wrote = R.maybe_review(j)
        seen["reviews"] += [(r["n"], filled) for r in wrote]
        assert R.maybe_review(j) == []                                   # asked again at once: nothing
        if filled == 99 and seen["final99"] is None:
            seen["final99"] = R.final(j)
        if filled == 40 and seen["weekly40"] is None:
            seen["weekly40"] = R.weekly(j.records())

    try:
        run = _run_synthetic(103, base, hooks)
        j = run["journal"]
        recs = j.records()
        st = JN.rebuild(recs)
        assert st["filled"] == 103 and len(st["skipped"]) >= 5 and JN.integrity(base)["ok"]
        # ---- reviews: once each, at the 25th and the 50th, on the first N trades, journaled ----
        assert seen["reviews"] == [(25, 25), (50, 50)], seen["reviews"]
        rv = [r for r in recs if r["kind"] == "review"]
        assert [r["n"] for r in rv] == [25, 50] and rv[0]["ts_ms"] < rv[1]["ts_ms"]
        ind = _independent_r(recs)
        for r in rv:
            n = r["n"]
            assert r["trades"] == [x["setup_id"] for x in ind[:n]]
            d = r["data"]
            assert d["n"] == n and near(d["table"]["human"]["mean"], float(np.mean([x["human"] for x in ind[:n]])))
            means = {k: float(np.mean([x[k] for x in ind[:n]])) for k in X.EXIT_NAMES}
            best = max(X.EXIT_NAMES, key=lambda k: means[k])
            assert d["best_so_far"] == best and all(near(d["table"][k]["mean"], means[k]) for k in X.EXIT_NAMES)
            dd = np.array([x["human"] - x[best] for x in ind[:n]])
            p = d["paired_vs_best"]
            assert near(p["mean"], float(dd.mean())) and near(p["se"], float(dd.std(ddof=1) / np.sqrt(n)))
            assert p["beaten"] == int((dd > 0).sum()) and p["n"] == n
            blob = json.dumps(d).lower()
            assert "dsr" not in blob and "deflated" not in blob and "pass" not in blob         # declares nothing
            assert "no deflated statistic" in r["text"] and f"interim review at {n} filled trades" in r["text"]
            for key in ("stop_width", "moves_within_2min", "widen_attempts", "underwater_exits_before_any_rule", "skipped"):
                assert key in d["behaviour"], key
            assert set(d["branches"]) >= {"level", "confirmation", "zone", "eq_by_cooccurring_zone"}
            assert os.path.exists(os.path.join(base, f"review-{n}.txt"))
        assert R.review_due(recs) == [] and R.maybe_review(j) == []
        b50 = rv[1]["data"]["behaviour"]
        assert b50["widen_attempts"] > 0 and b50["stop_width"].keys() >= {"wick", "atr1.5", "atr2.0"}
        assert b50["skipped"]["count"] >= 1 and sum(v["n"] for v in b50["stop_width"].values()) == 50
        # ---- final: refuses at 99, says how many to go, n_trials in the printout ----
        f99 = seen["final99"]
        assert f99["ran"] is False and f99["filled"] == 99 and f99["to_go"] == 1
        assert "REFUSED" in f99["text"] and "1 to go" in f99["text"] and "n_trials = 14" in f99["text"]
        assert not [r for r in recs if r["kind"] == "final"]
        # ---- final at 100 (103 filled: the sample is the FIRST 100) ----
        fin = R.final(j)
        assert fin["ran"] and fin["n_trials"] == 14 and "n_trials = 14" in fin["text"] and fin["sample"] == 100
        assert fin["trades"] == [x["setup_id"] for x in ind[:100]]
        means = {k: float(np.mean([x[k] for x in ind[:100]])) for k in X.EXIT_NAMES}
        best = max(X.EXIT_NAMES, key=lambda k: means[k])
        d = np.array([x["human"] - x[best] for x in ind[:100]])
        ts = [float((h - e).mean() / (h - e).std(ddof=1)) if (h - e).std(ddof=1) > 0 else 0.0
              for h, e in ((np.array([x["human"] for x in ind[:100]]), np.array([x[k] for x in ind[:100]])) for k in X.EXIT_NAMES)]
        want = V.deflated_sharpe(pd.Series(d), 1.0, np.array(ts), n_trials=14)
        assert fin["best_of_ten"] == best and near(fin["mean_d"], float(d.mean())) and want["trials"] == 14
        assert (fin["dsr"] == want["dsr"]) or (np.isnan(fin["dsr"]) and np.isnan(want["dsr"]))
        assert fin["passed"] == bool(np.isfinite(want["dsr"]) and want["dsr"] >= 0.95 and d.mean() > 0)
        assert ("RESULT: PASS" in fin["text"]) == fin["passed"] and ("RESULT: FAIL" in fin["text"]) != fin["passed"]
        assert len([r for r in j.records() if r["kind"] == "final"]) == 1
        R.final(j)
        assert len([r for r in j.records() if r["kind"] == "final"]) == 1            # journaled once
        # ---- weekly and status ----
        wk = seen["weekly40"]
        assert wk["filled"] == 40 and wk["to_go"] == 60 and "to go 60" in wk["text"] and "n_trials = 14" in wk["text"]
        for name in X.EXIT_NAMES + (X.RANDOM_EXIT, "human"):
            assert name in wk["text"], name
        assert "dsr" not in json.dumps(wk["data"]).lower()
        stt = R.status(j.records(), base)
        assert stt["filled"] == 103 and stt["to_go"] == 0 and stt["mode"] == "observe" and stt["integrity"]["ok"]
        assert list(stt["reviews"]) == [25, 50] and stt["final_written"] is True and stt["open_trades"] == []
        flat = json.dumps({k: v for k, v in stt.items() if k != "text"}).lower()
        for word in ('"r"', "mean", "r_gross", "total", "beaten"):
            assert word not in flat and word not in stt["text"].lower(), word      # where the run is, not how it is going
        print(f"      {run['days']} synthetic sessions -> 103 fills, {len(st['skipped'])} skips, {len(recs)} records; "
              f"reviews at {[n for n, _ in seen['reviews']]}; final ran once with n_trials = {fin['n_trials']}")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _fabricated(n: int, human_edge: float, seed: int = 7) -> list[dict]:
    """A journal's worth of records written by hand: n trades whose ten exits are noise around
    zero and whose human is `wick_hybrid + human_edge + small noise`."""
    rng = np.random.default_rng(seed)
    recs, seq = [], 0

    def rec(kind, **body):
        nonlocal seq
        seq += 1
        recs.append({"seq": seq, "ts": JN.iso_ms(1_790_000_000 + seq), "ts_ms": (1_790_000_000 + seq) * 1000, "v": 1,
                     "kind": kind, "instrument": "NQ", **body})

    for i in range(n):
        day = str((pd.Timestamp("2026-01-05") + pd.Timedelta(days=i)).date())
        sid = f"NQ-{day}"
        ev = h_fill(day=pd.Timestamp(day))
        rec("signal", setup_id=sid, day=day, armed=True, event="signal", ev=h_signal(ev))
        rec("fill", setup_id=sid, day=day, provisional=False, armed=True, ev=ev)
        rs = {k: float(rng.normal(0.0, 1.0)) for k in X.EXIT_NAMES}
        rs["wick_hybrid"] += 0.3                                                       # a clear best of ten
        human = rs["wick_hybrid"] + human_edge + float(rng.normal(0.0, 0.2))
        px = 100.0 + human * 1.0 + NQ_COST                                            # so that the recomputed R is `human`
        rec("exit", setup_id=sid, day=day, exit=px, reason="flat", stop0=99.0, stop_mode="wick", r=human,
            exit_bar=ev["bar_time"], r_gross=human)
        rec("benchmarks", setup_id=sid, day=day, skipped=False,
            exits={k: {"r": v, "exit_time": ev["bar_time"], "reason": "flat"} for k, v in rs.items()},
            random_stop={"r": 0.0})
    return recs


def test_final_pass_and_fail_on_fabricated_journals():
    assert R.final(_fabricated(99, 1.0))["ran"] is False and R.final(_fabricated(99, 1.0))["to_go"] == 1
    assert R.final(_fabricated(0, 1.0))["to_go"] == 100
    good = R.final(_fabricated(100, 0.5))
    assert good["ran"] and good["best_of_ten"] == "wick_hybrid" and good["mean_d"] > 0.4 and good["dsr"] >= 0.95
    assert good["passed"] is True and "RESULT: PASS" in good["text"] and "n_trials = 14" in good["text"]
    flat = R.final(_fabricated(100, 0.0))
    assert flat["ran"] and flat["passed"] is False and "RESULT: FAIL" in flat["text"]       # no edge: not distinguishable
    bad = R.final(_fabricated(100, -0.5))
    assert bad["passed"] is False and bad["mean_d"] < 0
    more = R.final(_fabricated(120, 0.5))
    assert more["trades"] == good["trades"] and more["dsr"] == good["dsr"]                  # the first 100, whatever came after
    # the 100th trade still open: refuses rather than scoring 99
    recs = _fabricated(100, 0.5)
    open_ = [r for r in recs if not (r["kind"] in ("exit", "benchmarks") and r["setup_id"] == recs[-1]["setup_id"])]
    out = R.final(open_)
    assert out["ran"] is False and out["waiting"] == [recs[-1]["setup_id"]] and R.review_due(open_) == [25, 50]


def test_underwater_exit_statistic():
    """EXIT NOW while underwater, before any of the ten exits would have closed the trade — and what the
    trade then did under the human's own initial stop."""
    # a quiet entry and ten quiet minutes, so that no trailing rule has anything to trail from
    quiet = (100.0, 100.05, 99.95, 100.0)
    day = {"09:54": quiet, "10:05": (100.0, 100.0, 99.55, 99.6), "10:40": (100.0, 103.2, 99.9, 103.0)}
    s = Sim(build_day(day, fill_runs={("09:55", "10:04"): quiet, ("10:06", "10:07"): (99.6, 99.7, 99.52, 99.6)}),
            h_fill(), preview=True)
    s.run([("10:06:30", "cmd", "exit now gut")])
    try:
        b = R.behaviour(R.complete(s.state()["trades"]), [])
        u = b["underwater_exits_before_any_rule"]
        assert u["count"] == 1 and near(u["trades"][0]["r_at_exit"], (-0.4 - NQ_COST) / 1.0)
        assert u["trades"][0]["held_reason"] == "target" and near(u["trades"][0]["held_r"], (3.0 - NQ_COST) / 1.0)
        assert u["trades"][0]["reason_code"] == "gut" and b["exit_reasons"] == {"exit_now": 1}
    finally:
        s.close()
    # in profit, or after a rule has already closed the trade: not counted
    s = Sim(build_day({"10:05": (100.0, 100.5, 100.0, 100.4)}, fill_runs={("10:06", "10:07"): (100.4, 100.5, 100.3, 100.4)}),
            h_fill(), preview=True).run([("10:06:30", "cmd", "exit now")])
    try:
        assert R.behaviour(R.complete(s.state()["trades"]), [])["underwater_exits_before_any_rule"]["count"] == 0
    finally:
        s.close()
    # the human sits on a 2.0 ATR stop, the wick rules stop out at 10:05, the human bails out at 10:20 underwater
    s = Sim(build_day({"10:05": (100.0, 100.0, 98.9, 99.2)}, fill_runs={("10:06", "10:25"): (99.2, 99.3, 99.1, 99.2)}),
            h_fill(), preview=True).run([("09:45:00", "cmd", "stop 2.0"), ("10:20:30", "cmd", "exit now")])
    try:
        t = s.the_trade()
        assert t["exit_reason"] == "exit_now" and t["r"] < 0
        assert R.behaviour(R.complete([t]), [])["underwater_exits_before_any_rule"]["count"] == 0
    finally:
        s.close()


# ══════════════════════════════ STAGE 3: the loop, the feeds, the chart, the CLI ══════════════════════════════
#
# The replay feed runs end to end on hand-built (synthetic) days with a scripted human. The LIVE feed
# cannot be exercised here — there is no TradingView session — so it is tested against a FAKE `tv` CLI:
# a small python script that answers `replay status`, `pane list`, `pane focus`, `state`, `ohlcv`,
# `ui eval` and `draw` with canned JSON from a state file. On the market files the replay asserts
# counts of events and the ABSENCE of outcome fields, nothing else (section 9.8).

import contextlib   # noqa: E402
import io           # noqa: E402
import re           # noqa: E402

from quantlab.tjr_human import chart as CH     # noqa: E402
from quantlab.tjr_human import runner as RUN   # noqa: E402

FAKE_TV = r'''
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "state.json")
with open(path, encoding="utf-8") as fh:
    st = json.load(fh)
args = sys.argv[1:]
with open(os.path.join(here, "calls.log"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(args[:3] if args[:2] != ["draw", "shape"] else args) + "\n")
key = " ".join(args[:2])
time.sleep(float(st.get("sleep", {}).get(key, 0)))
def out(d, code=0):
    (sys.stdout if code == 0 else sys.stderr).write(json.dumps(d))
    sys.exit(code)
if key in st.get("fail", {}):
    out({"success": False, "error": st["fail"][key]}, 1)
panes = st["panes"]
def bars(sym, n):
    return st["bars"].get(sym, [])[-n:]
if key == "replay status":
    out({"success": True, "is_replay_started": bool(st.get("replay"))})
if key == "pane list":
    out({"success": True, "panes": [{"index": i, "symbol": p["symbol"], "resolution": p["resolution"]} for i, p in enumerate(panes)]})
if key == "pane focus":
    if not st.get("focus_broken"):
        st["active"] = int(args[2])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
    out({"success": True, "focused_index": int(args[2])})
if args[:1] == ["state"]:
    p = panes[st.get("active", 0)]
    out({"success": True, "symbol": p["symbol"], "resolution": p["resolution"]})
if args[:1] == ["ohlcv"]:
    n = int(args[args.index("-n") + 1])
    rows = bars(panes[st.get("active", 0)]["symbol"], n)
    out({"success": True, "bars": [dict(zip(("time", "open", "high", "low", "close", "volume"), r)) for r in rows]})
if key == "ui eval":
    assert "/*tjr_human:panes*/" in args[2]
    n = int(args[2].split("var n=")[1].split(";")[0])
    out({"success": True, "result": {"panes": [{"index": i, "symbol": p["symbol"], "resolution": p["resolution"],
                                                "bars": bars(p["symbol"], n)} for i, p in enumerate(panes)]}})
if key == "draw shape":
    out({"success": True, "entity_id": "shape%d" % int(time.time() * 1e6)})
if key == "draw remove":
    out({"success": True})
out({"success": False, "error": "unknown command " + key}, 1)
'''

PANES = [{"symbol": "CME_MINI:NQ1!", "resolution": "1"}, {"symbol": "CME_MINI:ES1!", "resolution": "1"}]


def _rows(frame: pd.DataFrame, forming_as_open: bool = True) -> list:
    """Bars as the chart returns them; the newest one is still FORMING (only its open is real)."""
    rows = [[int(ts.value // 10**9), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)]
            for ts, r in zip(frame.index, frame.itertuples(index=False))]
    if rows and forming_as_open:
        o = rows[-1][1]
        rows[-1] = [rows[-1][0], o, o, o, o + 0.0123, 1.0]          # a half-made bar: NOT what the closed bar will say
    return rows


class FakeTv:
    """A directory holding the fake `tv` script, its state file and its call log."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="tjrh_tv_")
        self.script = os.path.join(self.dir, "fake_tv.py")
        with open(self.script, "w", encoding="utf-8") as fh:
            fh.write(FAKE_TV)
        self.state = {"panes": PANES, "active": 0, "replay": False, "bars": {}, "sleep": {}, "fail": {}}
        self.save()

    def save(self, **kw):
        self.state.update(kw)
        with open(os.path.join(self.dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(self.state, fh)

    def serve(self, **frames):
        self.state["bars"].update({PANES[0 if k == "NQ" else 1]["symbol"]: _rows(f) for k, f in frames.items()})
        self.save()

    def cli(self, timeout=20.0) -> "RUN.TvCli":
        return RUN.TvCli([sys.executable, self.script], timeout=timeout)

    def calls(self) -> list:
        p = os.path.join(self.dir, "calls.log")
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


def warm_frames() -> dict:
    """22 synthetic sessions per instrument (a seeded random walk; NQ near 20000, ES near 5000)."""
    if "warm" not in _cache:
        _cache["warm"] = {"NQ": RUN.synthetic_walk(days=22, seed=11, price=20000.0, sd=4.0),
                          "ES": RUN.synthetic_walk(days=22, seed=12, price=5000.0, sd=1.0)}
    return _cache["warm"]


def live_setup(frames: dict, cut: int, method="eval", timeout=20.0, **kw):
    """A runner on the fake CLI: stores hold frames[:cut]; the clock stands 5 s into the minute after the newest served bar."""
    tv = FakeTv()
    base = tempfile.mkdtemp(prefix="tjrh_live_")
    stores = {}
    for i, f in frames.items():
        path = RUN.Path(base) / "store" / f"{i}_1min.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        RUN._write_frame(path, f.iloc[:cut])
        stores[i] = RUN.BarStore(path)
    cli = tv.cli(timeout)
    feed = RUN.LiveTvFeed(cli, method=method)
    clock = Clock(0.0)
    sent = []
    runner = RUN.Runner(base, feed, instruments=tuple(frames), clock=clock, sleep=clock.sleep, stores=stores,
                        notify=lambda b, kind, text, fields=None: sent.append((kind, text, fields)) or {"kind": kind},
                        background=False, inbound=None, log=lambda line: None, **kw)
    return SimpleNamespace(tv=tv, base=base, runner=runner, clock=clock, sent=sent, stores=stores, feed=feed, cli=cli)


def live_close(ls):
    ls.tv.close()
    shutil.rmtree(ls.base, ignore_errors=True)


def serve_upto(ls, frames: dict, upto: int, back: int = 80):
    """The chart shows rows [upto-back, upto]; row `upto` is forming. The clock: 5 s into that minute."""
    ls.tv.serve(**{i: f.iloc[max(0, upto - back):upto + 1] for i, f in frames.items()})
    ls.clock.t = max(float(f.index[upto].value) / 1e9 for f in frames.values()) + 5.0


# ────────────────────────────── 13. small pure pieces ──────────────────────────────

def test_poll_interval_and_trading_minutes():
    assert RUN.poll_interval(at("09:25:00"), False) == 5.0 and RUN.poll_interval(at("10:14:59"), False) == 5.0
    assert RUN.poll_interval(at("09:24:59"), False) == 60.0 and RUN.poll_interval(at("10:15:00"), False) == 60.0
    assert RUN.poll_interval(at("14:00:00"), True) == 5.0                       # a setup or a trade is on
    t = lambda s: pd.Timestamp(s, tz=ET).tz_convert("UTC").tz_localize(None)    # noqa: E731
    assert RUN.trading_minutes_between(t("2026-07-15 10:00"), t("2026-07-15 10:01")) == 0
    assert RUN.trading_minutes_between(t("2026-07-15 10:00"), t("2026-07-15 10:31")) == 30
    assert RUN.trading_minutes_between(t("2026-07-15 16:59"), t("2026-07-15 18:00")) == 0      # the daily halt
    assert RUN.trading_minutes_between(t("2026-07-17 16:59"), t("2026-07-19 18:00")) == 0      # the weekend
    assert RUN.trading_minutes_between(t("2026-07-17 16:00"), t("2026-07-19 18:30")) == 59 + 30
    df = build_day()
    assert RUN.recent_holes(df) == []
    holed = df.drop(df.loc[utc("03:00"):utc("03:20")].index)
    h = RUN.recent_holes(holed)
    assert len(h) == 1 and h[0]["trading_minutes"] == 21 and h[0]["after"] == utc("02:59")
    assert RUN.recent_holes(df.drop(df.loc[utc("03:00"):utc("03:05")].index)) == []            # six quiet minutes are not a hole
    # the market files: one early close each (time stamps only — no price is read)
    for sym in ("NQ", "ES"):
        assert RUN.recent_holes(market(sym), sessions=30) and len(RUN.recent_holes(market(sym), sessions=30)) == 1


def test_script_loader():
    items = RUN.load_script([{"at": "2026-07-15 09:45:00", "text": "stop 1.5"},
                             {"at": "2026-07-15T13:44:00Z", "price": 99.5, "instrument": "NQ"}])
    assert items[0] == (at("09:44:00"), "price", 99.5, "NQ") and items[1][:3] == (at("09:45:00"), "cmd", "stop 1.5")
    assert RUN.load_script(None) == [] and RUN.load_script("none") == []
    for bad in ([{"text": "skip"}], [{"at": "2026-07-15 09:45", "text": "a", "price": 1.0}],
                [{"at": "2026-07-15 09:45", "price": 1.0}]):
        try:
            RUN.load_script(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_store_is_append_only_and_refuses_another_series():
    base = tempfile.mkdtemp(prefix="tjrh_store_")
    try:
        df = build_day(volume=5.0)
        seed = os.path.join(base, "seed.csv")
        RUN._write_frame(RUN.Path(seed), df.iloc[:1000])
        path = os.path.join(base, "store", "NQ_1min.csv")
        st = RUN.BarStore(path, seed=seed)
        assert len(st.frame) == 1000 and st.seeded_from == seed and os.path.exists(path)
        revised = df.iloc[990:1010].copy()
        revised.iloc[0, revised.columns.get_loc("close")] += 0.01           # upstream revised a bar we already hold
        new, info = st.merge(revised)
        assert len(new) == 10 and info["overlap"] == 10 and info["revised_upstream"] == 1 and info["hole"] is None
        assert st.frame.loc[df.index[990], "close"] == df["close"].iloc[990]                  # ours is never rewritten
        again = RUN.BarStore(path)
        assert len(again.frame) == 1010 and again.frame.equals(st.frame.astype(float))
        assert st.merge(df.iloc[500:600])[1]["new"] == 0                                        # nothing older goes in
        new, info = st.merge(df.iloc[1100:1110])
        assert len(new) == 10 and info["hole"]["trading_minutes"] == 90
        for wrong in (df.iloc[1105:1120] * 4.0, df.iloc[1200:1210] * 4.0):                     # overlap / no overlap
            try:
                st.merge(wrong)
            except RUN.FeedError as exc:
                assert exc.kind == "series"
            else:
                raise AssertionError("another instrument's bars were accepted")
        assert len(RUN.BarStore(path).frame) == 1020
        b = st.backfill(df.iloc[1000:1100])
        assert b["added"] == 90 and len(RUN.BarStore(path).frame) == 1110 and RUN.recent_holes(st.frame) == []
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 14. the synthetic marker and --show-outcomes ──────────────────────────────

def test_synthetic_is_a_hash_bound_sidecar_never_a_file_name():
    base = tempfile.mkdtemp(prefix="tjrh_syn_")
    try:
        p = RUN.write_synthetic_csv(os.path.join(base, "NQ_1min_tv.csv"), build_day(), "test")
        assert RUN.synthetic_status(p)["synthetic"] is True
        with open(p, "a", encoding="utf-8") as fh:                          # the bytes change: the marker no longer holds
            fh.write("2030-01-01 00:00:00,1,1,1,1,0\n")
        assert RUN.synthetic_status(p)["synthetic"] is False
        named = os.path.join(base, "synthetic_NQ.csv")                       # a name proves nothing
        shutil.copy(p, named)
        assert RUN.synthetic_status(named)["synthetic"] is False
        copy = os.path.join(base, "copy.csv")                                # a market file with a forged, correct sidecar
        shutil.copy(CSV["NQ"], copy)
        with open(copy + RUN.MARKER_SUFFIX, "w", encoding="utf-8") as fh:
            json.dump({"synthetic": True, "generator": "forged", "sha256": RUN.file_sha256(copy)}, fh)
        assert RUN.synthetic_status(copy) == {"synthetic": False, "why": "byte-identical to a file under data/"}
        for target in (copy, CSV["NQ"]):
            try:
                RUN.mark_synthetic(target, "x")
            except RUN.Refused:
                continue
            raise AssertionError(target)
        assert not os.path.exists(CSV["NQ"] + RUN.MARKER_SUFFIX)
        assert RUN.synthetic_status(CSV["NQ"])["synthetic"] is False and RUN.synthetic_status(CSV["ES"])["synthetic"] is False
    finally:
        shutil.rmtree(base, ignore_errors=True)


OUTCOME_KEYS = {"r", "r_gross", "r_net", "mae_r", "mfe_r", "pnl", "win_rate", "wins", "losses", "exits", "entry",
                "stop0", "price", "reason", "exit_reason", "unrealised_r", "mean_d", "dsr", "best_of_ten", "held", "t1", "t2"}
OUTCOME_TEXT = re.compile(r"\bR\b|r_gross|mae|mfe|win|loss|p&l|pnl|profit|sharpe|target|stopped|\bbest\b", re.I)


def _keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v, path)


def _floats(obj) -> list:
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, dict):
        return [f for v in obj.values() for f in _floats(v)]
    if isinstance(obj, (list, tuple)):
        return [f for v in obj for f in _floats(v)]
    return []


def test_replay_on_a_market_file_prints_machinery_facts_only_and_refuses_outcomes():
    """Section 9.8 on data/NQ_1min_tv.csv: counts of events, journal integrity, alert and draw calls —
    and no outcome field anywhere in what is returned or printed. `--show-outcomes` and a kept journal are refused."""
    import tjr_human as CLI
    before = set(os.listdir(tempfile.gettempdir()))
    for kw in ({"show_outcomes": True}, {"base_dir": os.path.join(tempfile.gettempdir(), "tjrh_never")}):
        try:
            RUN.replay({"NQ": CSV["NQ"]}, **kw)
        except RUN.Refused as exc:
            assert "REFUSED" in str(exc)
        else:
            raise AssertionError(kw)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        assert CLI.main(["replay", "--csv", f"NQ={CSV['NQ']}", "--show-outcomes"]) == 2
    assert "REFUSED" in err.getvalue() and "not flagged synthetic" in err.getvalue()

    out = RUN.replay({"NQ": CSV["NQ"]}, human=[{"at": "2026-09-01 09:40:00", "text": "hello"}])
    facts = out["facts"]
    assert out["synthetic"] is False and "outcomes" not in out
    bad = OUTCOME_KEYS & set(_keys(facts))
    assert not bad, bad
    assert not _floats(facts)                                               # counts, names and flags: not one price, not one R
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        CLI._print_facts(facts)
    head, body = text.getvalue().split("\n", 1)
    assert head.startswith("MACHINERY FACTS") and len(body) > 300           # the header names what is absent; the body has none
    assert not OUTCOME_TEXT.search(body), OUTCOME_TEXT.search(body)
    assert not re.search(r"\d\.\d", body), re.search(r"\d\.\d", body)          # and not one decimal number
    # the same detector, the same events: counts equal the batch run's, and only warm days were routed
    evs = market_events("NQ")
    assert facts["instruments"]["NQ"]["detector_events"] == dict(sorted(collections.Counter(e["kind"] for e in evs).items()))
    warm_days = {e["day"] for e in evs if e["kind"] == "levels" and e["warm"]}
    f = facts["instruments"]["NQ"]
    assert f["routed_days"] == len(warm_days) == 10 and f["cold_days_not_routed"] == 20 and f["bars"] == len(market("NQ"))
    warm_kinds = collections.Counter(e["kind"] for e in evs if e["day"] in warm_days)
    assert facts["setups"]["signals"] == warm_kinds["signal"] and facts["setups"]["filled"] == warm_kinds["fill"]
    t = facts["trades"]
    assert t["filled"] == t["exit_records"] == t["benchmarks_records"] == warm_kinds["fill"] and t["open_at_end"] == 0
    assert warm_kinds["fill"] <= t["previews"] <= warm_kinds["entry_trigger"]
    assert facts["journal"]["integrity_ok"] and facts["commands"]["rejected_by_rule"] == {"unparsable": 1}
    assert facts["alerts"]["sent_to_notify"] == 0 and facts["draws"]["enabled"] is False and facts["draws"]["drawn"] == 0
    assert facts["draws"]["asked"] > 0 and facts["alerts"]["by_kind"]["signal"] >= warm_kinds["signal"]
    left = {d for d in set(os.listdir(tempfile.gettempdir())) - before if d.startswith("tjr_human_replay_")}
    assert not left, left                                                   # the journal that held R is gone


# ────────────────────────────── 15. the replay feed, end to end, with a scripted human ──────────────────────────────

def weeks(days: list[dict], volume: float = 0.0) -> pd.DataFrame:
    """Hand-built days a week apart in one frame (D-1 and D of each)."""
    parts = []
    for k, (over, runs) in enumerate(days):
        df = build_day(over, fill_runs=runs, volume=volume)
        df.index = df.index + pd.Timedelta(days=7 * k)
        parts.append(df)
    return pd.concat(parts)


def _wk(k: int, hms: str) -> str:
    return f"{(DAY + pd.Timedelta(days=7 * k)).date()} {hms}"


def test_replay_end_to_end_with_a_scripted_human_exercising_every_control():
    target_day = ({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, HIGH_RUN)
    quiet_day = (dict(FULL_LONG), HIGH_RUN)
    df = weeks([target_day, quiet_day, quiet_day, target_day])
    fills = [e for e in D.detect(df, "NQ") if e["kind"] == "fill"]
    assert len(fills) == 4 and all(f["entry"] == 99.7 for f in fills)
    s2 = fills[2]["stops"]
    inside = round((s2["wick"] + s2["atr2.0"]) / 2.0, 2)
    script = [
        # week 0: SKIP from the signal; the setup is journaled to the end as if traded
        {"at": _wk(0, "09:45:00"), "text": "skip news"}, {"at": _wk(0, "09:46:00"), "text": "skip"},
        {"at": _wk(0, "09:47:00"), "text": "stop 2.0"}, {"at": _wk(0, "10:00:00"), "text": "exit now"},
        # week 1: a named width before the fill, MOVE STOP (too early, a widen, accepted), EXIT NOW, the reason after
        {"at": _wk(1, "09:45:00"), "text": "STOP 2.0 gut"}, {"at": _wk(1, "09:54:30"), "text": "move stop 99.0"},
        {"at": _wk(1, "10:00:00"), "text": "move stop 50"}, {"at": _wk(1, "10:01:00"), "text": "move stop 99.0 structure changed"},
        {"at": _wk(1, "10:15:10"), "text": "exit now"}, {"at": _wk(1, "10:15:20"), "price": 99.9, "instrument": "NQ"},
        {"at": _wk(1, "10:15:40"), "text": "noise"}, {"at": _wk(1, "10:16:00"), "text": "exit now"},
        # week 2: a price outside the band is rejected at the fill and is not the one choice; a valid one inside the
        # 60 s; a second STOP; something unparsable; a SKIP after the fill
        {"at": _wk(2, "09:45:00"), "text": "stop 50"}, {"at": _wk(2, "09:54:20"), "text": f"stop {inside} noise"},
        {"at": _wk(2, "09:54:40"), "text": "stop wick"}, {"at": _wk(2, "09:55:00"), "text": "what now"},
        {"at": _wk(2, "09:56:00"), "text": "skip"},
        # week 3: nobody home
    ]
    base = tempfile.mkdtemp(prefix="tjrh_rep_")
    try:
        csv = RUN.write_synthetic_csv(os.path.join(base, "hand_NQ.csv"), df, "test weeks")
        keep = os.path.join(base, "journal")
        out = RUN.replay({"NQ": str(csv)}, human=script, base_dir=keep, show_outcomes=True, allow_cold=True)
        assert out["synthetic"] and "weekly report" in out["outcomes"]
        recs = JN.read_journal(keep)
        st = JN.rebuild(recs)
        facts = out["facts"]
        assert JN.integrity(keep)["ok"] and facts["journal"]["integrity_ok"]
        assert st["filled"] == 3 and len(st["skipped"]) == 1 and facts["setups"]["signals"] == 4
        assert facts["commands"]["scripted"] == 17 and facts["commands"]["received"] == 16        # the price is not a message
        assert facts["commands"]["price_observations"] == 1
        why = facts["commands"]["rejected_by_rule"]
        assert why == {"already_skipped": 1, "setup_skipped": 1, "not_in_trade": 2, "move_in_stop_window": 1,
                       "move_widens": 1, "stop_outside_band": 1, "stop_already_used": 1, "unparsable": 1,
                       "skip_after_fill": 1}, why
        assert facts["commands"]["rejected"] == 10 and facts["commands"]["accepted"] == 6
        # week 0: skipped, would have filled, benchmarked as skipped
        sk = st["skipped"][0]
        b0 = [r for r in recs if r["kind"] == "benchmarks" and r["setup_id"] == sk["setup_id"]]
        assert len(b0) == 1 and b0[0]["skipped"] is True and [r for r in recs if r["kind"] == "skip" and r.get("stage") == "would_fill"]
        # week 1: the 2.0 ATR stop from the fill, moved to 99.0, EXIT NOW filled at the next observed price, reason attached
        t1, t2, t3 = st["trades"]
        assert t1["stop_mode"] == "atr2.0" and t1["stop_when"] == "before_fill" and near(t1["stop0"], fills[1]["stops"]["atr2.0"])
        assert [m["price"] for m in t1["moves"]] == [99.0] and t1["exit_reason"] == "exit_now"
        assert t1["exit"]["exit"] == 99.9 and t1["exit"]["via"] == "tick" and t1["exit"]["reason_code"] == "noise"
        assert near(t1["r"], (99.9 - 99.7 - 0.5e-4 * 99.7) / (99.7 - t1["stop0"]))
        assert [r.get("requested") for r in t1["rejects"] if r["why"] == "move_widens"] == [50.0]
        # week 2: 50 rejected with the band, the inside price accepted after the fill, R against it
        rej = [r for r in t2["rejects"] if r["why"] == "stop_outside_band"][0]
        assert rej["requested"] == 50.0 and rej["band"] == sorted([s2["wick"], s2["atr2.0"]])
        assert t2["stop_mode"] == "price" and t2["stop_when"] == "after_fill" and t2["stop0"] == inside
        assert t2["stop_seconds_from_fill"] == 20.0 and t2["exit_reason"] == "flat"
        # week 3: no human is the wick fixed stop, to the cent
        assert t3["stop_mode"] == "wick" and t3["stop_when"] == "default" and t3["exit_reason"] == "target"
        assert near(t3["r"], t3["benchmarks"]["exits"]["wick_fixed"]["r"], 1e-12)
        for t in (t1, t2, t3):
            assert t["complete"] and t["benchmarks"]["setup_id"] == t["setup_id"]
        # every notice was recorded, none was sent; the chart was asked, nothing was drawn
        assert facts["alerts"]["by_kind"].keys() == {"signal", "trade", "reply"} and facts["alerts"]["sent_to_notify"] == 0
        assert not os.path.exists(os.path.join(keep, "alerts.log"))
        assert facts["draws"]["asked"] > 40 and facts["draws"]["drawn"] == 0 and facts["draws"]["enabled"] is False
        runs = [r for r in recs if r["kind"] == "run"]
        assert [r["what"] for r in runs] == ["start", "stop"] and runs[0]["synthetic"] is True and runs[0]["mode"] == "replay"
        assert RUN.journal_is_reportable(recs) == (True, "")

        # closed bars only (no preview): the same trades, the same numbers — and no human at all: wick_fixed on every trade
        plain = os.path.join(base, "plain")
        RUN.replay({"NQ": str(csv)}, human=None, base_dir=plain, allow_cold=True, preview=False)
        tp = JN.rebuild(JN.read_journal(plain))["trades"]
        assert len(tp) == 4 and all(near(t["r"], t["benchmarks"]["exits"]["wick_fixed"]["r"], 1e-12) for t in tp)
        assert all(r.get("provisional") is False for r in JN.read_journal(plain) if r["kind"] == "fill")
        assert near(tp[3]["r"], t3["r"], 1e-12)

        # observe mode: detects, alerts, journals `observe`, takes no entry and accepts no command
        obs = os.path.join(base, "observe")
        fo = RUN.replay({"NQ": str(csv)}, human=script[:1], base_dir=obs, allow_cold=True, armed=False)["facts"]
        ro = JN.read_journal(obs)
        kinds = collections.Counter(r["kind"] for r in ro)
        assert kinds["observe"] > 20 and not (set(kinds) & {"fill", "exit", "benchmarks", "signal", "skip", "stop_set"})
        assert fo["setups"]["filled"] == 0 and fo["commands"]["rejected_by_rule"] == {"observe_mode": 1}
        assert fo["alerts"]["by_kind"]["signal"] >= 4 and JN.rebuild(ro)["filled"] == 0

        # a journal a replay wrote over files NOT flagged synthetic is refused by report / review / final
        import tjr_human as CLI
        bad = os.path.join(base, "bad")
        JN.Journal(bad, fsync=False).write("run", {"what": "start", "mode": "replay", "synthetic": False}, now=at("09:00"))
        for cmd in ("report", "review", "final"):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    CLI.main(["--dir", bad, cmd])
            except SystemExit as exc:
                assert "REFUSED" in str(exc)
            else:
                raise AssertionError(cmd)
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            assert CLI.main(["--dir", keep, "status"]) == 0 and CLI.main(["--dir", keep, "final"]) == 2
            assert CLI.main(["--dir", keep, "report"]) == 0 and CLI.main(["--dir", keep, "review"]) == 0
        assert "97 to go" in text.getvalue() and f"n_trials = {N_TRIALS}" in text.getvalue() and N_TRIALS == 14
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 16. ARMED ──────────────────────────────

def _git(commit="c0ffee" * 6 + "abcd", dirty=()):
    return lambda: {"commit": commit, "dirty": list(dirty)}


def test_arm_writes_the_hashes_and_refuses_to_disarm_after_a_fill():
    base = tempfile.mkdtemp(prefix="tjrh_arm_")
    try:
        # the hash is of section 5 and of nothing else, whatever the line endings
        text = open(os.path.join(ROOT, "DESIGN-tjr-human.md"), encoding="utf-8").read().replace("\r\n", "\n")
        a = text.index("\n## 5.") + 1
        b = text.index("\n## ", a) + 1
        import hashlib
        want = hashlib.sha256((text[a:b].rstrip() + "\n").encode("utf-8")).hexdigest()
        assert "n_trials` = 14" in text[a:b] and RUN.prereg_sha256() == want
        crlf = os.path.join(base, "crlf.md")
        with open(crlf, "w", encoding="utf-8", newline="") as fh:
            fh.write((text[:b] + text[b:].replace("nine-way", "fourteen-way")).replace("\n", "\r\n"))
        assert RUN.prereg_sha256(crlf) == want
        moved = os.path.join(base, "moved.md")
        with open(moved, "w", encoding="utf-8") as fh:
            fh.write(text.replace("**Sample: 100 filled trades.**", "**Sample: 60 filled trades.**"))
        assert RUN.prereg_sha256(moved) != want

        jd = os.path.join(base, "j")
        for kw in ({"reason": "  "}, {"reason": "go", "git": lambda: {"commit": None, "dirty": []}},
                   {"reason": "go", "git": _git(dirty=[" M quantlab/tjr_human/trade.py"])}):
            try:
                RUN.arm(jd, **{"git": _git(), **kw})
            except RUN.Refused:
                continue
            raise AssertionError(kw)
        assert JN.read_journal(jd) == [] and JN.armed_record(JN.read_journal(jd)) is None
        rec = RUN.arm(jd, "observed for two weeks; the feed held", now=at("08:00"), git=_git())
        assert rec["kind"] == "armed" and rec["commit"] == "c0ffee" * 6 + "abcd" and rec["prereg_sha256"] == want
        assert rec["target_rule"] == D.TARGET_RULE and rec["n_trials"] == 14 and rec["dirty"] == [] and rec["supersedes"] is None
        assert R.status(JN.read_journal(jd))["mode"] == "armed"
        # before a fill: disarm is allowed, and so is arming again (it supersedes, with --allow-dirty journaled)
        assert RUN.disarm(jd, "feed problem", now=at("08:05"))["armed"] is False and JN.armed_record(JN.read_journal(jd)) is None
        try:
            RUN.disarm(jd, "again")
        except RUN.Refused:
            pass
        else:
            raise AssertionError("disarmed twice")
        r2 = RUN.arm(jd, "fixed", now=at("08:10"), git=_git(dirty=[" M tjr_human.py"]), allow_dirty=True)
        assert r2["dirty"] == [" M tjr_human.py"]
        r3 = RUN.arm(jd, "committed", now=at("08:20"), git=_git())
        assert r3["supersedes"] == r2["seq"]
        # trade one fills
        j = JN.Journal(jd, fsync=False)
        s = Sim(build_day({"10:20": (100.0, 103.2, 99.9, 103.0)}), h_fill(), journal=j,
                manager=TR.TradeManager(j, armed=JN.armed_record(j.records()) is not None)).run()
        assert JN.filled_count(j.records()) == 1 and s.the_trade()["closed"]
        n = len(j.records())
        for fn, kw in ((RUN.disarm, {"reason": "it is going badly"}), (RUN.arm, {"reason": "new rule", "git": _git()})):
            try:
                fn(jd, **kw)
            except RUN.Refused as exc:
                assert "trade one" in str(exc)
            else:
                raise AssertionError(fn.__name__)
        assert len(JN.read_journal(jd)) == n and JN.armed_record(JN.read_journal(jd))["seq"] == r3["seq"]
        import tjr_human as CLI
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert CLI.main(["--dir", jd, "arm", "--disarm", "--reason", "please"]) == 2
        assert "REFUSED" in err.getvalue() and len(JN.read_journal(jd)) == n
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 17. the live feed, against a fake tv CLI ──────────────────────────────

def test_live_feed_closed_bar_rule_store_merge_both_methods():
    frames = warm_frames()
    n = len(frames["NQ"])
    for method in ("eval", "focus"):
        ls = live_setup(frames, n - 40, method=method)
        try:
            serve_upto(ls, frames, n - 30)
            ls.runner.start_live()
            ls.runner.cycle_live()
            for i, f in frames.items():
                st = ls.stores[i]
                assert st.last == f.index[n - 31], (method, i, st.last)             # the forming bar is NOT in the store
                assert len(st.frame) == n - 30 and st.frame.iloc[-10:].equals(f.iloc[n - 40:n - 30].astype(float))
                assert len(RUN.BarStore(st.path).frame) == n - 30                     # and it is on disk
                assert ls.runner.det[i].snapshot()["last_bar"] == f.index[n - 31].isoformat()
            serve_upto(ls, frames, n - 28)                                           # two minutes later
            ls.runner.cycle_live()
            for i, f in frames.items():
                st = ls.stores[i]
                # the bar that was forming is now closed and stored with its FINAL values, not the half-made ones
                assert st.last == f.index[n - 29] and st.frame.loc[f.index[n - 30]].equals(f.loc[f.index[n - 30]].astype(float))
            ls.runner.cycle_live()                                                   # nothing new: nothing appended
            assert all(len(ls.stores[i].frame) == n - 28 for i in frames)
            calls = [" ".join(c[:2]) for c in ls.tv.calls()]
            assert calls.count("replay status") == 3
            if method == "eval":
                assert set(calls) == {"replay status", "ui eval"}
            else:
                assert set(calls) == {"replay status", "pane list", "pane focus", "state", "ohlcv -n"} and calls.count("pane focus") == 6
            kinds = collections.Counter(r["kind"] for r in JN.read_journal(ls.base))
            assert kinds["run"] == 1 and kinds.get("feed", 0) == 0
            start = [r for r in JN.read_journal(ls.base) if r["kind"] == "run"][0]
            assert start["mode"] == "observe" and start["feed"] == "tradingview" and start["method"] == method
            assert start["warm"]["NQ"]["ok"] and start["telegram_configured"] is False and start["inbound_thread"] is False
        finally:
            live_close(ls)


def test_live_feed_refuses_replay_mode_cold_store_and_holes():
    frames = warm_frames()
    n = len(frames["NQ"])
    # the chart is in replay mode at the start: refused, nothing ingested
    ls = live_setup(frames, n - 40)
    try:
        serve_upto(ls, frames, n - 30)
        ls.tv.save(replay=True)
        try:
            ls.runner.run_live(max_cycles=1)
        except RUN.Refused as exc:
            assert "REPLAY" in str(exc)
        else:
            raise AssertionError("replay mode accepted")
        assert all(len(ls.stores[i].frame) == n - 40 for i in frames)
        runs = [r for r in JN.read_journal(ls.base) if r["kind"] == "run"]
        assert [r["what"] for r in runs] == ["start", "refused", "stop"]
    finally:
        live_close(ls)
    # replay mode switched on in the middle of a run: journaled, retried, never fatal, nothing ingested
    ls = live_setup(frames, n - 40)
    try:
        serve_upto(ls, frames, n - 30)
        ls.runner.start_live()
        ls.runner.cycle_live()
        serve_upto(ls, frames, n - 25)
        ls.tv.save(replay=True)
        ls.runner.cycle_live()
        ls.runner.cycle_live()
        assert all(len(ls.stores[i].frame) == n - 30 for i in frames)
        ls.tv.save(replay=False)
        ls.runner.cycle_live()
        assert all(len(ls.stores[i].frame) == n - 25 for i in frames)
        feed = [(r["what"], r.get("error_kind")) for r in JN.read_journal(ls.base) if r["kind"] == "feed"]
        assert feed == [("failure", "replay"), ("recovered", None)], feed
    finally:
        live_close(ls)
    # a store without the warm history: refused before anything is read
    short = {i: f.iloc[-5 * 1380:] for i, f in frames.items()}
    ls = live_setup(short, len(short["NQ"]) - 40)
    try:
        serve_upto(ls, short, len(short["NQ"]) - 30)
        try:
            ls.runner.start_live()
        except RUN.Refused as exc:
            assert "20 needed" in str(exc) and "4-hour" in str(exc)
        else:
            raise AssertionError("a cold store was accepted")
        assert ls.tv.calls() == [] and [r["what"] for r in JN.read_journal(ls.base)] == ["refused"]
    finally:
        live_close(ls)
    # the PC was off for three hours of the newest session: refused unless --accept-holes (then journaled)
    for accept in (False, True):
        ls = live_setup(frames, n - 300, accept_holes=accept)
        try:
            serve_upto(ls, frames, n - 30, back=80)
            ls.runner.start_live()
            if accept:
                ls.runner.cycle_live()
                whats = [r["what"] for r in JN.read_journal(ls.base) if r["kind"] == "feed"]
                assert whats == ["hole", "holes_accepted"] * 2 and len(ls.stores["NQ"].frame) == n - 300 + 80
            else:
                try:
                    ls.runner.cycle_live()
                except RUN.Refused as exc:
                    assert "hole" in str(exc) and "--backfill" in str(exc)
                else:
                    raise AssertionError("a hole was accepted")
        finally:
            live_close(ls)


def test_live_feed_cli_timeout_wrong_pane_and_garbage_are_journaled_and_retried():
    frames = warm_frames()
    n = len(frames["NQ"])
    ls = live_setup(frames, n - 40, timeout=1.5)
    try:
        serve_upto(ls, frames, n - 30)
        ls.runner.start_live()
        ls.tv.save(sleep={"ui eval": 30})
        t0 = _time.monotonic()
        ls.runner.cycle_live()                                                       # the CLI hangs: bounded, not fatal
        assert _time.monotonic() - t0 < 10.0
        ls.runner.cycle_live()
        ls.tv.save(sleep={}, fail={"ui eval": "CDP connection refused"})
        ls.runner.cycle_live()
        assert all(len(ls.stores[i].frame) == n - 40 for i in frames)
        ls.tv.save(fail={})
        ls.runner.cycle_live()
        assert all(len(ls.stores[i].frame) == n - 30 for i in frames)
        feed = [r for r in JN.read_journal(ls.base) if r["kind"] == "feed"]
        assert [(r["what"], r.get("error_kind")) for r in feed] == [("failure", "timeout"), ("recovered", None)]
        assert feed[1]["failures"] == 3 and "no answer within" in feed[0]["error"]
        assert ls.runner.facts["feed_failures"] == 3
        # the ES pane shows NQ's prices (a symbol switched under us is caught by the layout check; this is the
        # other guard: bars that do not continue the series are refused by the store)
        serve_upto(ls, frames, n - 20)
        ls.tv.state["bars"][PANES[1]["symbol"]] = ls.tv.state["bars"][PANES[0]["symbol"]]
        ls.tv.save()
        ls.runner.cycle_live()
        assert len(ls.stores["NQ"].frame) == n - 20 and len(ls.stores["ES"].frame) == n - 30
        last = [r for r in JN.read_journal(ls.base) if r["kind"] == "feed"][-1]
        assert last["what"] == "failure" and last["error_kind"] == "series" and last["scope"] == "ES"
        # a pane at the wrong resolution, a missing pane
        ls.tv.save(panes=[PANES[0], {"symbol": "CME_MINI:ES1!", "resolution": "5"}])
        try:
            ls.feed.cycle(ls.clock(), {"NQ": 30, "ES": 30})
        except RUN.FeedError as exc:
            assert exc.kind == "layout"
        else:
            raise AssertionError("a 5-minute pane was read")
        ls.tv.save(panes=[PANES[0]])
        try:
            ls.feed.cycle(ls.clock(), {"NQ": 30, "ES": 30})
        except RUN.FeedError as exc:
            assert exc.kind == "layout" and "ES1!" in str(exc)
        else:
            raise AssertionError("a missing pane was read")
    finally:
        live_close(ls)
    # method "focus": the focus click did not take — `state` says so and ES is not filed under NQ's bars
    ls = live_setup(frames, n - 40, method="focus")
    try:
        serve_upto(ls, frames, n - 30)
        ls.tv.save(focus_broken=True)
        ls.runner.start_live()
        ls.runner.cycle_live()
        assert len(ls.stores["NQ"].frame) == n - 30 and len(ls.stores["ES"].frame) == n - 40
        f = [r for r in JN.read_journal(ls.base) if r["kind"] == "feed"]
        assert len(f) == 1 and f[0]["scope"] == "ES" and f[0]["error_kind"] == "layout"
    finally:
        live_close(ls)


class ScriptedInbound:
    """Stands in for TelegramInbound: the test drops parsed commands in, the runner drains them."""

    def __init__(self):
        self.box, self.started = [], 0

    def start(self):
        self.started += 1
        return True

    def stop(self):
        pass

    def drain(self):
        out, self.box = self.box, []
        return out


def test_live_loop_observe_then_arm_signal_fill_commands_chart_and_no_network():
    """The hand-built long day through the LIVE path, minute by minute, on the fake CLI: observe mode
    until `arm` is written by another Journal; the signal push carries what section 3.2 draws plus the
    indicative stops; the fill (known at the entry minute's OPEN through the forming bar) carries the
    exact stops and the 60 s deadline; commands are answered; the chart draws through `tv draw` and a
    failing draw changes nothing. TELEGRAM_* are unset and both transports are booby-trapped."""
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
    frames = {"NQ": df}
    cut = df.index.get_loc(pd.Timestamp(utc("09:20")))
    stop_at = df.index.get_loc(pd.Timestamp(utc("16:05")))
    ls = live_setup(frames, cut, allow_cold=True)
    saved = {k: os.environ.pop(k, None) for k in (A.TOKEN_VAR, A.CHAT_VAR)}
    real_post, real_fetch = A._post, C._fetch

    def boom(*a, **k):
        raise AssertionError("the network was touched")

    A._post, C._fetch = boom, boom
    try:
        r = ls.runner
        inbound = ScriptedInbound()
        r.inbound = inbound
        r.chart = CH.Chart(executor=CH.TvDraw(ls.cli, ls.feed.pane_of), log_dir=ls.base, background=False)
        serve_upto(ls, frames, cut)
        r.start_live()
        arm_at = df.index.get_loc(pd.Timestamp(utc("09:31")))
        say = {utc("09:45"): "stop session gut", utc("10:00"): "move stop 50", utc("10:01"): "move stop 99.2 noise"}
        fill_known_at = None
        for k in range(cut, stop_at + 1, 1):
            stamp = df.index[k].isoformat()
            if k == arm_at:
                RUN.arm(ls.base, "test: armed from another process", now=ls.clock(), git=_git())
            if 570 <= D.et_clock(df.index[k:k + 1])[0][0] <= 650 or k % 30 == 0 or k > stop_at - 12:
                serve_upto(ls, frames, k)
                if stamp in say:
                    inbound.box.append(C.parse(say[stamp]))
                if stamp == utc("09:50"):
                    ls.tv.save(fail={"draw shape": "TradingView said no"})          # the chart breaks for a while
                if stamp == utc("09:58"):
                    ls.tv.save(fail={})
                r.cycle_live()
                if fill_known_at is None and any(x["kind"] == "fill" for x in JN.read_journal(ls.base)):
                    fill_known_at = stamp
        recs = JN.read_journal(ls.base)
        st = JN.rebuild(recs)
        f = one(hand(df, "NQ"), "fill")
        # observe until armed: the 09:29 levels were journaled as observe, the signal (09:40) was armed
        assert [x["event"] for x in recs if x["kind"] == "observe"] == ["levels"]
        assert st["filled"] == 1 and st["armed"]["commit"].startswith("c0ffee")
        t = st["trades"][0]
        assert fill_known_at == utc("09:54")                                       # at the OPEN of the entry minute
        fills = [x for x in recs if x["kind"] == "fill"]
        assert [x["provisional"] for x in fills] == [True, False] and fills[1]["preview_match"] is True
        assert t["entry"] == 99.7 and t["stop_mode"] == "session" and t["stop0"] == f["stops"]["session"]
        assert [m["price"] for m in t["moves"]] == [99.2] and [x["why"] for x in t["rejects"]] == ["move_widens"]
        # 99.2 is above the 09:59.. lows? the inert minutes trade 99.5: the moved stop holds until the target prints
        assert t["exit_reason"] == "target" and t["exit"]["exit_bar"] == utc("10:30") and t["complete"]
        assert near(t["r"], (102.0 - 99.7 - 0.5e-4 * 99.7) / (99.7 - f["stops"]["session"]))
        # the pushes: signal with everything 3.2 draws + indicative stops; fill with exact stops + the deadline; replies
        kinds = [k for k, _, _ in ls.sent]
        assert set(kinds) == {"signal", "trade", "reply"}
        sig = [(txt, fl) for k, txt, fl in ls.sent if k == "signal"][0]
        for key in ("levels", "level", "extreme", "sweep_bar_time", "conf_types", "conf_time", "zone", "stops_indicative",
                    "t1", "t2", "targets", "target_rule", "ref_price", "atr", "side"):
            assert key in sig[1], key
        assert set(sig[1]["stops_indicative"]) == set(D.STOP_MODES) and "PDH" in sig[1]["levels"]
        assert "OBSERVE" not in sig[0] and "LATE" not in sig[0]
        json.dumps([fl for _, _, fl in ls.sent], allow_nan=False)
        fill_push = [(txt, fl) for k, txt, fl in ls.sent if k == "trade" and "FILL" in txt][0]
        assert fill_push[1]["stops"] == f["stops"] and "STOP accepted until" not in fill_push[0]   # the choice was made
        assert fill_push[1]["band"] == sorted([f["stops"]["wick"], f["stops"]["atr2.0"]])
        replies = [txt for k, txt, _ in ls.sent if k == "reply"]
        assert len(replies) == 3 and "noted" in replies[0] and "not tighter" in replies[1] and "STOP MOVED" in replies[2]
        # the chart: drawn through `tv draw shape` on the NQ pane; the failures were logged and swallowed
        draws = [c for c in ls.tv.calls() if c[:2] == ["draw", "shape"]]
        calls = [" ".join(c[:2]) for c in ls.tv.calls()]
        assert draws and calls[calls.index("draw shape") - 1] == "pane focus"
        texts = " | ".join(c[c.index("--text") + 1] for c in draws if "--text" in c)
        for word in ("PDH [PD / SESSION]", "LON_L [LON / SESSION]", "H4_SH [H4 / H4]", "SWEEP", "CONF bos+ifvg+ote",
                     "zone eq", "STOP wick (indicative)", "T1 LON_H+H1_SH", "T2 ", "ENTRY long", "STOP session", "STOP price"):
            assert word in texts, word
        assert any("rectangle" in c for c in draws) and "draw remove" in calls                # the stop line was replaced
        assert r.chart.counts()["failed"] == 5 and r.chart.counts()["drawn"] >= 15, r.chart.counts()   # the fill's four + its stop
        assert "TradingView said no" in open(os.path.join(ls.base, CH.CHART_LOG), encoding="utf-8").read()
        assert JN.integrity(ls.base)["ok"] and inbound.started == 1
        assert not os.path.exists(os.path.join(ls.base, "alerts.log"))               # the fake notify was used: nothing else wrote
    finally:
        A._post, C._fetch = real_post, real_fetch
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        live_close(ls)


def test_with_telegram_unset_nothing_tries_the_network_and_the_runner_still_journals():
    """The REAL notify and the REAL TelegramInbound, the two variables unset, both transports booby-trapped."""
    df = build_day(dict(FULL_LONG), fill_runs=HIGH_RUN)
    frames = {"NQ": df}
    cut = df.index.get_loc(pd.Timestamp(utc("09:25")))
    saved = {k: os.environ.pop(k, None) for k in (A.TOKEN_VAR, A.CHAT_VAR)}
    real_post, real_fetch = A._post, C._fetch
    hits = []
    A._post = lambda *a, **k: hits.append("post") or (200, "{}")
    C._fetch = lambda *a, **k: hits.append("fetch") or (200, "{}")
    tv = FakeTv()
    base = tempfile.mkdtemp(prefix="tjrh_nonet_")
    err = io.StringIO()
    try:
        clock = Clock(0.0)
        stores = {"NQ": RUN.BarStore(os.path.join(base, "store", "NQ_1min.csv"), frame=df.iloc[:cut])}
        RUN.arm(base, "test", now=float(df.index[cut].value) / 1e9, git=_git())
        r = RUN.Runner(base, RUN.LiveTvFeed(tv.cli()), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                       stores=stores, allow_cold=True, background=False, log=lambda line: None)
        assert isinstance(r.inbound, C.TelegramInbound) and r.mgr.armed
        ls = SimpleNamespace(tv=tv, clock=clock)
        with contextlib.redirect_stderr(err):
            serve_upto(ls, frames, cut)
            r.start_live()
            for k in range(cut, df.index.get_loc(pd.Timestamp(utc("09:56"))) + 1):
                serve_upto(ls, frames, k)
                r.cycle_live()
            r.stop()
        assert hits == []
        recs = JN.read_journal(base)
        kinds = collections.Counter(x["kind"] for x in recs)
        assert kinds["signal"] == 1 and kinds["fill"] == 2 and kinds["run"] == 2 and JN.integrity(base)["ok"]
        start = [x for x in recs if x["kind"] == "run"][0]
        assert start["telegram_configured"] is False and start["inbound_thread"] is False and start["mode"] == "armed"
        log = open(os.path.join(base, "alerts.log"), encoding="utf-8").read()
        assert "[signal]" in log and "[trade]" in log and "SIGNAL long" in err.getvalue()   # logged, not pushed
        assert "STOP accepted until" in log and "13:55:00 UTC" in log                        # the fill carries the 60 s deadline
    finally:
        A._post, C._fetch = real_post, real_fetch
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        tv.close()
        shutil.rmtree(base, ignore_errors=True)


def test_restart_mid_trade_restores_from_the_store_and_never_enters_an_old_day():
    """The process dies and comes back the NEXT morning. (a) With a trade on: the gap is journaled, the
    old day is re-fed silently (no second signal, no second fill, no second alert), the bars it missed
    close the trade, the benchmarks are written. (b) With a signal waiting and no fill yet: the fill it
    missed is NOT taken after the fact — an old day is re-fed to restore, never to enter."""
    day1 = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
    day2 = build_day().loc[utc("18:00", -1):]
    day2.index = day2.index + pd.Timedelta(days=1)
    df = pd.concat([day1, day2])
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    frames = {"NQ": df}
    cut = df.index.get_loc(pd.Timestamp(utc("09:20")))
    back = df.index.get_loc(pd.Timestamp(utc("11:00")) + pd.Timedelta(days=1))

    def run(died_hm: str):
        died = df.index.get_loc(pd.Timestamp(utc(died_hm)))
        ls = live_setup(frames, cut, allow_cold=True)
        note = lambda b, kind, text, fields=None: ls.sent.append((kind, text, fields)) or {}       # noqa: E731
        RUN.arm(ls.base, "test", now=float(df.index[cut].value) / 1e9 - 600, git=_git())
        mk = lambda stores: RUN.Runner(ls.base, ls.feed, instruments=("NQ",), clock=ls.clock, sleep=ls.clock.sleep,   # noqa: E731
                                       stores=stores, allow_cold=True, background=False, inbound=None,
                                       log=lambda line: None, notify=note)
        r = mk(ls.stores)
        serve_upto(ls, frames, cut)
        r.start_live()
        for k in range(cut, died + 1):
            serve_upto(ls, frames, k)
            r.cycle_live()
        before = JN.read_journal(ls.base)
        n_sent = len(ls.sent)
        # ... the process is gone (the OS drops its runner.lock with it). A new one starts at 11:00 the
        # next day; the chart still shows yesterday
        r._lock.release()
        r2 = mk({"NQ": RUN.BarStore(ls.stores["NQ"].path)})
        serve_upto(ls, frames, back, back=back - died + 20)
        r2.start_live()
        r2.cycle_live()
        return ls, before, n_sent

    ls, before, n_sent = run("10:05")
    try:
        assert JN.rebuild(before)["trades"][0]["closed"] is False
        recs = JN.read_journal(ls.base)
        st = JN.rebuild(recs)
        assert st["gaps"][-1]["open_trades"] == ["NQ-2026-07-15"] and st["gaps"][-1]["seconds"] > 86000
        t = st["trades"][0]
        assert st["filled"] == 1 and t["closed"] and t["exit_reason"] == "target" and t["exit"]["exit_bar"] == utc("10:30")
        assert t["complete"] and near(t["r"], t["benchmarks"]["exits"]["wick_fixed"]["r"], 1e-12)
        kinds = collections.Counter(x["kind"] for x in recs)
        assert kinds["signal"] == 1 and kinds["fill"] == 2 and kinds["benchmarks"] == 1 and kinds["exit"] == 1
        start2 = [x for x in recs if x["kind"] == "run" and x["what"] == "start"][1]
        assert start2["re_fed_from"] == "2026-07-15" and start2["today"] == "2026-07-16" and start2["mode"] == "armed"
        new = [txt for _, txt, _ in ls.sent[n_sent:]]
        assert len(new) == 1 and "EXIT target" in new[0] and "LATE" in new[0], new      # told once, and told it is late
        assert JN.integrity(ls.base)["ok"]
    finally:
        live_close(ls)

    ls, before, n_sent = run("09:45")
    try:
        assert JN.rebuild(before)["filled"] == 0 and [x["kind"] for x in before].count("signal") == 1
        recs = JN.read_journal(ls.base)
        st = JN.rebuild(recs)
        assert st["gaps"][-1]["setups_waiting"] == ["NQ-2026-07-15"] and st["gaps"][-1]["open_trades"] == []
        assert st["filled"] == 0 and not [x for x in recs if x["kind"] in ("fill", "exit", "benchmarks")]
        assert ls.sent[n_sent:] == []
        assert len(RUN.BarStore(ls.stores["NQ"].path).frame) == back              # the store has the bars all the same
    finally:
        live_close(ls)


def test_pushes_and_the_chart_cannot_delay_or_break_the_loop():
    base = tempfile.mkdtemp(prefix="tjrh_push_")
    try:
        slow = []

        def notify(b, kind, text, fields=None):
            _time.sleep(0.6)
            slow.append(kind)
            if kind == "reply":
                raise RuntimeError("boom")
            return {"kind": kind}

        p = RUN.Pusher(base, notify=notify, background=True)
        t0 = _time.monotonic()
        p.push([{"kind": "signal", "text": "a", "fields": {}}, {"kind": "reply", "text": "b", "fields": {}}])
        assert _time.monotonic() - t0 < 0.2 and p.by_kind == {"signal": 1, "reply": 1}
        p.close(5.0)
        assert slow == ["signal", "reply"] and len(p.sent) == 2
        rec = RUN.Pusher(base, notify=False)
        rec.push([{"kind": "trade", "text": "x", "fields": {}}])
        assert rec.by_kind == {"trade": 1} and rec.sent == []

        class Broken:
            def draw(self, inst, shape):
                raise RuntimeError("no chart")

            def remove(self, inst, eid):
                raise RuntimeError("no chart")

        ch = CH.Chart(executor=Broken(), log_dir=base, background=True)
        sig = one(hand(build_day(FULL_LONG, fill_runs=HIGH_RUN), "NQ"), "signal")
        t0 = _time.monotonic()
        n = ch.on_event(sig)
        assert _time.monotonic() - t0 < 0.2 and n >= 15
        assert ch.on_event({"kind": "signal"}) == 0 and ch.on_event({"kind": "signal", "bar_time": "x"}) == 0
        assert ch.on_event(None) == 0 and ch.on_stop("NQ", "2026-07-15", float("nan"), "wick", 0.0) in (0, 1)
        end = _time.monotonic() + 5.0
        while ch.failed < n and _time.monotonic() < end:
            _time.sleep(0.05)
        ch.close()
        assert ch.failed >= n and ch.drawn == 0
        assert "no chart" in open(os.path.join(base, CH.CHART_LOG), encoding="utf-8").read()
        # shapes: every level labelled by type, the sweep, the confirmation and its type, the zone and its type, stop, T1, T2
        tags = [s["tag"] for s in CH.shapes_for(sig)]
        assert {"sweep", "confirmation", "zone", "stop", "t1", "t2"} <= set(tags)
        assert {f"level:{k}" for k in sig["levels"]} <= set(tags) and len(sig["levels"]) >= 10
        args = CH.draw_args([s for s in CH.shapes_for(sig) if s["tag"] == "zone"][0])
        assert args[:4] == ["draw", "shape", "--type", "rectangle"] and "--price2" in args and "--time2" in args
        off = CH.Chart(executor=None)
        assert off.on_event(sig) == n and off.counts()["drawn"] == 0 and off.enabled is False
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_run_live_waits_the_poll_interval_answers_commands_between_polls_and_survives_a_broken_cycle():
    class StubFeed:
        name, live, method = "stub", True, None

        def __init__(self):
            self.at, self.boom = [], False

        def cycle(self, now, want):
            self.at.append(now)
            if self.boom:
                raise ZeroDivisionError("a bug in a feed")
            return {}

    base = tempfile.mkdtemp(prefix="tjrh_wait_")
    try:
        df = build_day()
        clock = Clock(at("09:23:00"))
        feed, inbound, sent = StubFeed(), ScriptedInbound(), []
        r = RUN.Runner(base, feed, instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True, inbound=inbound,
                       stores={"NQ": RUN.BarStore(None, frame=df.loc[:utc("09:22")])}, background=False,
                       notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {}, log=lambda line: None)
        real_sleep = clock.sleep

        def sleep(s):                                                         # a message arrives while the loop sleeps
            real_sleep(s)
            if not sent and clock() >= at("09:24:10"):
                inbound.box.append(C.parse("exit now"))
        r.sleep = sleep
        r.run_live(max_cycles=5)
        gaps = [round(b - a, 1) for a, b in zip(feed.at, feed.at[1:])]
        assert gaps == [60.0, 60.0, 5.0, 5.0], gaps                           # slow until 09:25 ET, then every 5 s
        assert [k for k, _ in sent] == ["reply"] and "no" in sent[0][1].lower()
        recs = JN.read_journal(base)
        assert [x["why"] for x in recs if x["kind"] == "reject"] == ["no_setup"]
        cmd_ts = [x["ts_ms"] for x in recs if x["kind"] == "command"][0] / 1000.0
        assert at("09:24:10") <= cmd_ts < at("09:24:11")                     # answered when it came, not at the next poll
        feed.boom = True                                                     # a bug in a feed is a feed failure, not the end
        r2 = RUN.Runner(base, feed, instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True, inbound=None,
                        stores={"NQ": RUN.BarStore(None, frame=df.loc[:utc("09:22")])}, background=False, notify=False,
                        log=lambda line: None)
        r2.run_live(max_cycles=3)
        feeds = [x for x in JN.read_journal(base) if x["kind"] == "feed"]
        assert len(feeds) == 1 and "a bug in a feed" in feeds[0]["error"] and feeds[0]["what"] == "failure"
        assert r2.facts["cycles"] == 3 and r2.facts["feed_failures"] == 3
        assert [x["what"] for x in JN.read_journal(base) if x["kind"] == "run"] == ["start", "stop", "start", "stop"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 20. after the three adversarial reviews ──────────────────────────────
# Hand-built days, fakes and fabricated records only. Nothing here reads an ES / NQ file.

def _forming_sim(day: dict, script, **kw):
    """A Sim whose script may carry ("HH:MM:SS", "forming", (o, h, l, last)): the forming bar as a live feed shows it."""
    s = Sim(build_day(day), h_fill(**kw.pop("fill", {})), preview=True, **kw)
    real = s._item

    def item(t, kind, val):
        if kind == "forming":
            o, h, l, last = val                                               # noqa: E741
            s._note(s.m.on_forming(s.inst, JN.epoch_bar(t), o, h, l, last, t))
        else:
            real(t, kind, val)
    s._item = item
    return s.run(script)


def test_named_stop_is_held_to_the_band_like_a_price():
    """Sections 3.3 / 5 / 9.3: inside [wick, 2.0 ATR] or rejected and logged — a width given by NAME too."""
    # (a) the session stop is wider than the 2.0 ATR stop: `STOP session` is rejected at the fill, with the price
    #     it resolved to; the wick stays; the one choice is not consumed
    s = _forming_sim({"10:20": (100.0, 103.2, 99.9, 103.0)},
                     [("09:45:00", "cmd", "stop session gut"), ("09:54:20", "cmd", "stop 2.0")], fill={"session": 96.0})
    try:
        rj = s.recs("reject")
        assert [r["why"] for r in rj] == ["stop_outside_band"] and rj[0]["requested"] == 96.0
        assert rj[0]["requested_name"] == "session" and rj[0]["band"] == [97.6, 99.0] and rj[0]["human_reason"] == "gut"
        ss = s.recs("stop_set")
        assert len(ss) == 1 and ss[0]["mode"] == "atr2.0" and ss[0]["price"] == 97.6 and ss[0]["when"] == "after_fill"
        assert s.the_trade()["stop0"] == 97.6
    finally:
        s.close()
    # (b) THE WICK IS THE WIDE END (wick 97.0 beyond 2.0 ATR 97.6): the band is still the two prices, [97.6, 97.0]
    #     sorted; 1.5 ATR (98.2) and 1.0 ATR (98.8) are tighter than both -> rejected; session (96.5) wider -> rejected
    wide = {"wick": 97.0, "session": 96.5}
    for text, ok, price in (("stop 1.5", False, 98.2), ("stop 1.0", False, 98.8), ("stop session", False, 96.5),
                            ("stop 2.0", True, 97.6), ("stop wick", True, 97.0), ("stop 97.3", True, 97.3),
                            ("stop 96.9", False, 96.9), ("stop 97.7", False, 97.7)):
        s = _forming_sim({}, [("09:45:00", "cmd", text)], fill=wide)
        try:
            rj, ss = s.recs("reject"), s.recs("stop_set")
            if ok:
                assert not rj and len(ss) == 1 and ss[0]["price"] == price and ss[0]["band"] == [97.0, 97.6], text
                assert s.the_trade()["stop0"] == price
            else:
                assert not ss and [r["why"] for r in rj] == ["stop_outside_band"] and rj[0]["requested"] == price, text
                assert rj[0]["band"] == [97.0, 97.6] and rj[0]["candidates"]["wick"] == 97.0
                assert s.the_trade()["stop0"] == 97.0 and s.the_trade()["stop_mode"] == "wick"
        finally:
            s.close()


def test_exit_now_cannot_undo_a_stop_that_traded_earlier_in_the_minute():
    """EXIT NOW fills at the next observed price — after that minute's stop and target tests. The forming bar's
    running low had already traded the wick: the exit is the wick and the human's R is wick_fixed's."""
    s = _forming_sim({"10:00": (100.0, 100.2, 98.9, 99.7)},
                     [("10:00:30", "cmd", "exit now"), ("10:00:35", "forming", (100.0, 100.2, 98.9, 99.6))])
    try:
        x, t = s.recs("exit")[0], s.the_trade()
        assert x["reason"] == "stop" and x["exit"] == 99.0 and x["via"] == "tick" and x["exit_bar"] == utc("10:00")
        wf = t["benchmarks"]["exits"]["wick_fixed"]
        assert near(t["r"], wf["r"], 1e-12) and near(t["r"], (-1.0 - NQ_COST) / 1.0) and wf["exit_time"] == x["exit_bar"]
    finally:
        s.close()
    # the running range is clean: the fill is the last price
    s = _forming_sim({"10:00": (100.0, 100.2, 99.3, 99.7)},
                     [("10:00:30", "cmd", "exit now noise"), ("10:00:35", "forming", (100.0, 100.2, 99.3, 99.6))])
    try:
        x = s.recs("exit")[0]
        assert x["reason"] == "exit_now" and x["exit"] == 99.6 and x["via"] == "tick" and near(x["r"], (-0.4 - NQ_COST) / 1.0)
        assert near(x["mae_r"], -0.7) and x["human_reason"] == "noise"
    finally:
        s.close()
    # T1 printed earlier in the minute: the trade was already out at T1
    s = _forming_sim({"10:00": (100.0, 103.1, 99.9, 102.5)},
                     [("10:00:30", "cmd", "exit now"), ("10:00:35", "forming", (100.0, 103.1, 99.9, 102.5))])
    try:
        x = s.recs("exit")[0]
        assert x["reason"] == "target" and x["exit"] == 103.0 and near(x["r"], (3.0 - NQ_COST) / 1.0)
    finally:
        s.close()
    # a feed that cannot know the running range (the replay): the old reading, the next observed price
    s = _forming_sim({"10:00": (100.0, 100.2, 98.9, 99.7)},
                     [("10:00:30", "cmd", "exit now"), ("10:00:35", "forming", (100.0, None, None, 99.6))])
    try:
        assert s.recs("exit")[0]["reason"] == "exit_now" and s.recs("exit")[0]["exit"] == 99.6
    finally:
        s.close()


def test_moved_stop_is_live_inside_its_own_minute_on_new_extremes_only():
    day = {"10:00": (100.0, 100.3, 99.4, 100.0)}
    first = ("10:00:05", "forming", (100.0, 100.1, 99.9, 100.0))              # the low so far: 99.9
    move = ("10:00:10", "cmd", "move stop 99.5 noise")
    # (a) a NEW low after the move trades through it, seen on the forming bar: stopped at the moved stop, that minute
    s = _forming_sim(day, [first, move, ("10:00:40", "forming", (100.0, 100.3, 99.4, 99.8))])
    try:
        mv, x = s.recs("stop_move")[0], s.recs("exit")[0]
        assert mv["base_adv"] == 99.9 and mv["own_bar"] == at("10:00") and mv["eff_bar"] == at("10:01")
        assert x["reason"] == "trail" and x["exit"] == 99.5 and x["via"] == "tick" and x["exit_bar"] == utc("10:00")
        assert near(x["r"], (-0.5 - NQ_COST) / 1.0) and x["stop0"] == 99.0
    finally:
        s.close()
    # (b) the dip and the recovery fall between two polls: the CLOSED bar of that minute shows the new low
    s = _forming_sim(day, [first, move])
    try:
        x = s.recs("exit")[0]
        assert x["reason"] == "trail" and x["exit"] == 99.5 and x["via"] == "bar" and x["exit_bar"] == utc("10:00")
    finally:
        s.close()
    # (c) the 99.4 had printed BEFORE the move (it is the low the forming bar showed at the move): no evidence
    #     against a stop that did not exist yet; the next minute's low (the inert 99.5) is
    s = _forming_sim(day, [("10:00:05", "forming", (100.0, 100.1, 99.4, 100.0)), move])
    try:
        x = s.recs("exit")[0]
        assert s.recs("stop_move")[0]["base_adv"] == 99.4 and x["exit_bar"] == utc("10:01") and x["exit"] == 99.5
    finally:
        s.close()
    # (d) no forming range at all (closed bars only, the replay): the move's own minute cannot be split
    s = _forming_sim(day, [move])
    try:
        assert s.recs("stop_move")[0]["base_adv"] is None and s.recs("exit")[0]["exit_bar"] == utc("10:01")
    finally:
        s.close()
    # (e) a later minute, which the moved stop owns whole: the forming bar says what the closed bar will, sooner
    s = _forming_sim({"10:02": (100.0, 100.1, 99.45, 99.6)},
                     [("09:58:10", "cmd", "move stop 99.48"), ("10:02:20", "forming", (100.0, 100.1, 99.45, 99.5))],
                     )
    try:
        x = s.recs("exit")
        # the inert minutes before 10:02 have low 99.5 > 99.48: untouched until 10:02
        assert len(x) == 1 and x[0]["exit"] == 99.48 and x[0]["via"] == "tick" and x[0]["exit_bar"] == utc("10:02")
    finally:
        s.close()
    # (f) A HUMAN WHO DOES NOTHING IS STILL wick_fixed: a forming range through the wick decides nothing — the
    #     closed bar does (here it never confirms that low)
    s = _forming_sim({}, [("10:00:30", "forming", (100.0, 100.2, 98.5, 99.0))])
    try:
        t = s.the_trade()
        assert t["exit_reason"] == "flat" and near(t["r"], t["benchmarks"]["exits"]["wick_fixed"]["r"], 1e-12)
    finally:
        s.close()


def test_malformed_token_turns_telegram_off_and_is_never_echoed():
    """A token with a trailing CR / LF (a CRLF launcher) makes http.client refuse the URL with a message that
    spells the token out, escaped. Such a token is 'not configured': no request is built from it, anywhere."""
    base = tempfile.mkdtemp(prefix="tjrh_tok_")
    calls = []
    real_post = A._post

    def body():
        for bad in (FAKE_TOKEN + "\r", FAKE_TOKEN + "\n", " " + FAKE_TOKEN, FAKE_TOKEN + "\r\n"):
            os.environ[A.TOKEN_VAR] = bad
            assert C.telegram_problem() and "malformed" in C.telegram_problem() and FAKE_TOKEN not in C.telegram_problem()
            assert C.malformed() and not C.configured() and A.configured()   # alerts.py itself would try to push
        os.environ[A.TOKEN_VAR] = FAKE_TOKEN + "\r"
        C._fetch = lambda *a, **k: calls.append("fetch") or (200, "{}")
        A._post = lambda *a, **k: calls.append("post") or (200, "{}")
        j = JN.Journal(base, fsync=False)
        inbound = C.TelegramInbound(j)
        assert inbound.start() is False and inbound.poll(0) == [] and inbound.requests == 0
        assert FAKE_TOKEN not in (inbound.last_error or "") and "malformed" in inbound.last_error
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = TR.send_notices(base, [{"kind": "signal", "text": "NQ SIGNAL", "fields": {"a": 1}}])
        assert calls == [] and out[0]["logged"] and not out[0]["pushed"]
        log = open(os.path.join(base, "alerts.log"), encoding="utf-8").read()
        assert "[signal] NQ SIGNAL" in log and FAKE_TOKEN not in log and FAKE_TOKEN not in err.getvalue()
        # the message http.client would have produced, escaped as repr() writes it
        leak = "InvalidURL: URL can't contain control characters. '/bot" + FAKE_TOKEN + "\\r/sendMessage'"
        assert FAKE_TOKEN not in C._mask(leak) and "<token>" in C._mask(leak)
        for text in (FAKE_TOKEN, leak):
            try:
                j.write("reject", {"why": "x", "text": text})
            except ValueError:
                continue
            raise AssertionError("the journal took a record carrying the token")
        # the runner: Telegram OFF is in the start record, in words
        clock = Clock(at("09:23:00"))

        class Quiet:
            name, live, method = "stub", True, None

            def cycle(self, now, want):
                return {}
        r = RUN.Runner(base, Quiet(), instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True,
                       stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])}, background=False,
                       log=lambda line: None)
        with contextlib.redirect_stderr(io.StringIO()):
            rec = r.start_live()
            r.cycle_live()
            r.stop()
        assert rec["telegram_configured"] is False and rec["inbound_thread"] is False
        assert any("malformed" in w for w in rec["warnings"]) and calls == []
        for p in JN.journal_files(base):
            assert FAKE_TOKEN not in p.read_text(encoding="utf-8")

    try:
        _with_env(body, token=FAKE_TOKEN + "\r")
    finally:
        A._post = real_post
        shutil.rmtree(base, ignore_errors=True)


class _FlakyJournal(JN.Journal):
    """Raises PermissionError the first `n` times a record of `kind` is written (the file held by another program)."""

    def __init__(self, *a, fail=None, **k):
        super().__init__(*a, **k)
        self.fail = dict(fail or {})
        self.raised = 0

    def write(self, kind, body=None, instrument=None, now=None):
        if self.fail.get(kind, 0) > 0:
            self.fail[kind] -= 1
            self.raised += 1
            raise PermissionError(13, "the journal is held by another program")
        return super().write(kind, body, instrument, now)


def test_failing_getupdates_is_never_silent_and_an_unjournaled_update_is_kept():
    base = tempfile.mkdtemp(prefix="tjrh_inb_")

    def body():
        # (a) the transport: consecutive failures are counted, a success clears them
        two = [_upd(11, 42, "skip"), _upd(12, 42, "exit now")]
        fake = FakeTelegram([[], two, two])                                   # Telegram serves them again: not acknowledged
        C._fetch = lambda url, body, timeout: (409, '{"ok":false,"description":"Conflict: terminated by other getUpdates"}')
        inbound = C.TelegramInbound(JN.Journal(base, fsync=False))
        for _ in range(3):
            assert inbound.poll(0) == []
        assert inbound.error_count == 3 and "409" in inbound.last_error and FAKE_TOKEN not in inbound.last_error
        C._fetch = fake
        assert inbound.poll(0) == [] and inbound.error_count == 0 and inbound.last_error is None
        # (b) an update whose journal write failed is NOT consumed: the offset stays, it comes again, once
        j = _FlakyJournal(base, fsync=False, fail={"command": 1})
        inbound = C.TelegramInbound(j)
        assert inbound.poll(0) == [] and inbound.offset is None and "journal" in inbound.last_error
        got = inbound.poll(0)
        assert [c.action for c in got] == ["SKIP", "EXIT_NOW"] and inbound.offset == 13 and inbound.last_error is None
        assert [r["update_id"] for r in j.records() if r["kind"] == "command"] == [11, 12]
        assert all("offset" not in c["body"] for c in fake.calls)             # neither request acknowledged anything
    try:
        _with_env(body)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # (c) the runner watches it: journaled on the transition, pushed once in the window, once on recovery
    class Inbound(ScriptedInbound):
        last_error, error_count = None, 0

    class Quiet:
        name, live, method = "stub", True, None

        def cycle(self, now, want):
            return {}

    for start, pushes in (("09:30:00", 2), ("12:00:00", 0)):
        base = tempfile.mkdtemp(prefix="tjrh_inb_")
        try:
            clock, sent, inbound = Clock(at(start)), [], Inbound()
            r = RUN.Runner(base, Quiet(), instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True,
                           inbound=inbound, stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])},
                           background=False, log=lambda line: None,
                           notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {})
            r.start_live()
            inbound.last_error, inbound.error_count = "HTTP 409: Conflict", 1
            r.cycle_live()                                                    # one failed request is weather
            assert not [x for x in JN.read_journal(base) if x["kind"] == "feed"]
            inbound.error_count = 2
            for _ in range(4):
                clock.sleep(5)
                r.cycle_live()
            feeds = [x for x in JN.read_journal(base) if x["kind"] == "feed"]
            assert [(x["what"], x["scope"]) for x in feeds] == [("failure", "telegram_inbound")] and "409" in feeds[0]["error"]
            inbound.last_error, inbound.error_count = None, 0
            clock.sleep(5)
            r.cycle_live()
            r.stop()
            feeds = [x for x in JN.read_journal(base) if x["kind"] == "feed"]
            assert [x["what"] for x in feeds] == ["failure", "recovered"] and feeds[1]["failures"] == 4
            assert len(sent) == pushes and r.facts["feed_failures"] == 0
            if pushes:
                assert sent[0][0] == "trade" and "NOT being received" in sent[0][1] and "RECOVERED" in sent[1][1]
        finally:
            shutil.rmtree(base, ignore_errors=True)


class MemFeed:
    """A live feed out of a frame and the runner's clock: a bar is closed once its minute is over; of the
    forming bar only the open is real (its range is the open)."""

    name, live, method = "mem", True, None

    def __init__(self, frames: dict, clock):
        self.frames, self.clock, self.fail = frames, clock, None

    def cycle(self, now, want):
        if self.fail is not None:
            raise self.fail
        out = {}
        for inst, df in self.frames.items():
            ep = df.index.values.astype("datetime64[s]").astype(np.int64)
            j = int(np.searchsorted(ep, now - 60.0, side="right"))
            closed = df.iloc[max(0, j - 40):j]
            forming = None
            if j < len(ep) and ep[j] <= now:
                o = float(df["open"].iloc[j])
                forming = {"time": df.index[j].isoformat(), "open": o, "high": o, "low": o, "last": o}
            out[inst] = RUN.Poll(closed, forming)
        return out


def _mem_run(base, journal=None, until="10:40", df=None, hook=None):
    """The hand-built long day through the LIVE path in memory, two polls a minute from 09:21. Returns
    (runner, sent, exceptions that came out of cycle_live)."""
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN) if df is None else df
    clock, sent, errors = Clock(at("09:21:05")), [], []
    RUN.arm(base, "test", now=at("08:00"), git=_git())
    r = RUN.Runner(base, MemFeed({"NQ": df}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                   journal=journal, allow_cold=True, inbound=None, background=False, log=lambda line: None,
                   stores={"NQ": RUN.BarStore(None, frame=df.loc[:utc("09:19")])},
                   notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {})
    r.start_live()
    while clock() < at(until):
        try:
            if hook is not None:
                hook(r, clock())
            r.cycle_live()
        except RUN.Refused:
            raise
        except Exception as exc:
            errors.append(type(exc).__name__)
        clock.sleep(30)
    r.stop()
    return r, sent, errors


def _shape(records):
    return sorted((x["kind"], x.get("event") or x.get("what") or x.get("stage") or "", bool(x.get("provisional")))
                  for x in records if x["kind"] not in ("run", "gap"))


def test_events_are_delivered_at_least_once_when_a_journal_write_fails():
    """One transient PermissionError on the journal must not lose the signal (never journaled, never pushed), the
    fill record, the exit or the human's pre-fill STOP: what a bar produced stays queued until it is taken."""
    base0 = tempfile.mkdtemp(prefix="tjrh_alo_")
    try:
        r0, sent0, err0 = _mem_run(base0)
        want = JN.read_journal(base0)
        kinds0 = [k for k, _ in sent0]
        assert not err0 and JN.rebuild(want)["trades"][0]["exit_reason"] == "target" and kinds0.count("signal") == 1
        for kind in ("signal", "fill", "exit", "event"):
            base = tempfile.mkdtemp(prefix="tjrh_alo_")
            try:
                j = _FlakyJournal(base, fsync=False, fail={kind: 1})
                r, sent, errors = _mem_run(base, journal=j)
                got = [x for x in JN.read_journal(base)]
                assert j.raised == 1 and errors == ["PermissionError"], (kind, errors)
                assert _shape(got) == _shape(want), kind
                t, t0 = JN.rebuild(got)["trades"][0], JN.rebuild(want)["trades"][0]
                assert (t["entry"], t["exit"]["exit"], t["exit_reason"], t["r"]) == \
                       (t0["entry"], t0["exit"]["exit"], t0["exit_reason"], t0["r"]), kind
                assert sorted(k for k, _ in sent) == sorted(kinds0), kind     # the push is not lost either
                assert JN.integrity(base)["ok"]
            finally:
                shutil.rmtree(base, ignore_errors=True)
    finally:
        shutil.rmtree(base0, ignore_errors=True)
    # the journal itself retries a sharing violation before it gives up
    base = tempfile.mkdtemp(prefix="tjrh_alo_")
    try:
        j = JN.Journal(base, fsync=False)
        j.write("run", {"what": "start"})
        real_open, state = RUN.Path.open, {"n": 0}

        def flaky_open(self, *a, **k):
            if self.suffix == ".jsonl" and a and a[0] == "a" and state["n"] < 2:
                state["n"] += 1
                raise PermissionError(13, "sharing violation")
            return real_open(self, *a, **k)
        RUN.Path.open = flaky_open
        try:
            rec = j.write("run", {"what": "stop"})
        finally:
            RUN.Path.open = real_open
        assert state["n"] == 2 and rec["seq"] == 2 and [x["seq"] for x in JN.read_journal(base)] == [1, 2]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_journal_torn_inside_a_multibyte_character_is_skipped_and_said():
    base = tempfile.mkdtemp(prefix="tjrh_torn_")
    try:
        j = JN.Journal(base, fsync=False)
        j.write("reject", {"why": "x", "text": "97.00 is outside the band \u2014 the wick stop stays"}, instrument="NQ")
        j.write("reject", {"why": "y", "text": "second \u2014 torn"}, instrument="NQ")
        path = JN.journal_files(base)[0]
        raw = path.read_bytes()
        cut = raw.rindex("\u2014".encode("utf-8")) + 1                       # one byte into the em dash
        path.write_bytes(raw[:cut])
        j2 = JN.Journal(base, fsync=False)                                   # used to raise UnicodeDecodeError
        assert [x["why"] for x in j2.records()] == ["x"] and JN.integrity(base)["bad_lines"] == 1
        rec = j2.write("reject", {"why": "z"}, instrument="NQ")
        assert rec["seq"] == 2 and [x["why"] for x in JN.read_journal(base)] == ["x", "z"]
        st = R.status(JN.read_journal(base), base)
        assert "1 torn line(s) skipped" in st["text"] and "BROKEN" not in st["text"] and not st["integrity"]["problems"]
        clock = Clock(at("09:23:00"))

        class Quiet:
            name, live, method = "stub", True, None

            def cycle(self, now, want):
                return {}
        lines = []
        r = RUN.Runner(base, Quiet(), instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True, inbound=None,
                       stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])}, background=False,
                       notify=False, log=lines.append)
        rec = r.start_live()
        r.stop()
        assert rec["journal_bad_lines"] == 1 and any("torn line" in w for w in rec["warnings"])
        assert any("torn line" in ln for ln in lines)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_store_torn_last_row_is_cut_and_never_a_bar():
    base = tempfile.mkdtemp(prefix="tjrh_store_")
    try:
        df = build_day().iloc[:300]
        whole, more = df.iloc[:200], df.iloc[200:]
        for k, tear in enumerate(("row cut inside the close", "row cut after the low", "a short row with its newline",
                                  "NULs")):
            path = RUN.Path(base) / f"tear{k}.csv"
            RUN._write_frame(path, whole.iloc[:199])
            line = RUN._lines(RUN._norm(whole.iloc[199:200]))
            with open(path, "ab") as fh:
                if tear == "row cut inside the close":
                    assert line.count(",") == 5                               # '...,99.5,100.0,0.0' torn to '...,99.5,10':
                    fh.write((",".join(line.split(",")[:4]) + ",10").encode())  # pandas reads it as a bar closing at 10
                elif tear == "row cut after the low":
                    fh.write(",".join(line.split(",")[:4]).encode())
                elif tear == "NULs":
                    fh.write(b"\x00" * 64)
                else:
                    fh.write((",".join(line.split(",")[:4]) + "\n").encode())
            st = RUN.BarStore(path)
            assert len(st.repairs) == 1 and st.repairs[0]["lines_dropped"] == 1, tear
            assert len(st.frame) == 199 and st.frame.equals(RUN._norm(whole.iloc[:199])), tear   # never a bar with a wrong close
            assert path.read_bytes().endswith(b"\n")
            new, _ = st.merge(df.iloc[190:])                                 # the feed brings the minute back
            assert len(new) == 101 and RUN.BarStore(path).frame.equals(RUN._norm(df)), tear
        # a whole file is left alone
        path = RUN.Path(base) / "whole.csv"
        RUN._write_frame(path, whole)
        before = path.read_bytes()
        assert RUN.BarStore(path).repairs == [] and path.read_bytes() == before
        # a torn append that merge() meets (the file changed under it): it never glues two rows together
        with open(path, "ab") as fh:
            fh.write(b"2026-07-15 01:00:00,100.0,100.5,99")
        st = RUN.BarStore(path, frame=whole)
        st.merge(more)
        assert len(st.repairs) == 1 and RUN.BarStore(path).frame.equals(RUN._norm(df))
        # a store pandas cannot parse is a refusal in words, not a traceback out of Runner()
        jd = os.path.join(base, "run")
        bad = RUN.Path(jd) / "store" / "NQ_1min.csv"
        bad.parent.mkdir(parents=True)
        text = path.read_text(encoding="utf-8").splitlines()
        text[50] = text[50] + ",1,2,3"
        bad.write_text("\n".join(text) + "\n", encoding="utf-8")
        try:
            RUN.Runner(jd, MemFeed({}, Clock(0.0)), instruments=("NQ",), seeds={}, inbound=None, notify=False)
        except RUN.Refused as exc:
            assert "cannot be read" in str(exc) and "NQ" in str(exc)
        else:
            raise AssertionError("a broken store was loaded")
        # the runner journals a repair
        good = RUN.Path(jd) / "store" / "NQ_1min.csv"
        RUN._write_frame(good, build_day().loc[:utc("09:22")])
        with open(good, "ab") as fh:
            fh.write(b"2026-07-15 13:23:00,100.0,100.5")
        clock = Clock(at("09:23:30"))
        r = RUN.Runner(jd, MemFeed({"NQ": build_day()}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                       seeds={}, inbound=None, notify=False, allow_cold=True, background=False, log=lambda line: None)
        r.start_live()
        r.cycle_live()
        r.stop()
        fixes = [x for x in JN.read_journal(jd) if x["kind"] == "feed" and x["what"] == "store_repaired"]
        assert len(fixes) == 1 and fixes[0]["instrument"] == "NQ" and fixes[0]["lines_dropped"] == 1
        assert RUN.BarStore(good).frame.index[-1] == pd.Timestamp(utc("09:22"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_second_detect_on_the_same_directory_is_refused():
    base = tempfile.mkdtemp(prefix="tjrh_lock_")
    try:
        clock = Clock(at("09:23:00"))

        def mk():
            return RUN.Runner(base, MemFeed({"NQ": build_day()}, clock), instruments=("NQ",), clock=clock,
                              sleep=clock.sleep, allow_cold=True, inbound=None, notify=False, background=False,
                              stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])},
                              log=lambda line: None)
        r1, r2 = mk(), mk()
        r1.start_live()
        r1.cycle_live()
        n = len(JN.read_journal(base))
        try:
            r2.start_live()
        except RUN.Refused as exc:
            assert "another `detect`" in str(exc)
        else:
            raise AssertionError("two runners on one directory")
        recs = JN.read_journal(base)
        assert len(recs) == n + 1 and recs[-1]["kind"] == "run" and recs[-1]["what"] == "refused"   # no gap, no start
        assert RUN.arm(base, "arming needs no lock", now=at("09:24"), git=_git())["kind"] == "armed"
        r1.stop()
        r3 = mk()
        assert r3.start_live()["what"] == "start"                            # the lock went with the first runner
        r3.stop()
        # the same through run_live: refused, exit path clean, and the first holder is not disturbed
        r4, r5 = mk(), mk()
        r4.start_live()
        try:
            r5.run_live(max_cycles=1)
        except RUN.Refused:
            pass
        else:
            raise AssertionError("run_live started a second runner")
        r4.cycle_live()
        r4.stop()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_feed_failures_reach_the_phone_once_and_cycle_errors_are_not_per_poll():
    for start, want_push in (("09:30:00", True), ("12:00:00", False)):
        base = tempfile.mkdtemp(prefix="tjrh_tell_")
        try:
            clock, sent = Clock(at(start)), []
            feed = MemFeed({"NQ": build_day()}, clock)
            r = RUN.Runner(base, feed, instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True, inbound=None,
                           stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])}, background=False,
                           notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {},
                           log=lambda line: None, accept_holes=True)
            r.start_live()
            feed.fail = RUN.FeedError("no pane shows NQ1!: the layout is ['AAPL']", "layout")
            for _ in range(5):
                r.cycle_live()
                clock.sleep(5)
            feed.fail = None
            r.cycle_live()
            feeds = [(x["what"], x["scope"]) for x in JN.read_journal(base) if x["kind"] == "feed" and x.get("scope")]
            assert feeds == [("failure", "feed"), ("recovered", "feed")], feeds
            told = [t for k, t in sent if "FAILING" in t or "RECOVERED" in t]
            if want_push:
                assert len(told) == 2 and "no pane shows NQ1!" in told[0] and "no signal can be detected" in told[0]
                assert all(k == "trade" for k, t in sent if t in told)
            else:
                assert told == []
            # the store csv held by another program: a feed failure of that instrument, not an exception
            real = r.stores["NQ"].merge
            r.stores["NQ"].merge = lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "held by Excel"))
            clock.sleep(60)
            r.cycle_live()
            r.stores["NQ"].merge = real
            last = [x for x in JN.read_journal(base) if x["kind"] == "feed"][-1]
            assert (last["what"], last["scope"]) == ("failure", "NQ") and "Excel" in last["error"]
            r.stop()
        finally:
            shutil.rmtree(base, ignore_errors=True)
    # an exception inside every cycle: ONE cycle_error record, one push, then the recovery — not one per poll
    base = tempfile.mkdtemp(prefix="tjrh_tell_")
    try:
        clock, sent = Clock(at("09:30:00")), []
        r = RUN.Runner(base, MemFeed({"NQ": build_day()}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                       allow_cold=True, inbound=None, background=False, log=lambda line: None, accept_holes=True,
                       stores={"NQ": RUN.BarStore(None, frame=build_day().loc[:utc("09:22")])},
                       notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {})
        good = r._close_sessions
        state = {"n": 0}

        def broken(*a, **k):
            state["n"] += 1
            if state["n"] <= 6:
                raise RuntimeError("PermissionError on the store while it is open elsewhere")
            return good(*a, **k)
        r._close_sessions = broken
        r.run_live(max_cycles=9)
        runs = [x["what"] for x in JN.read_journal(base) if x["kind"] == "run"]
        assert runs == ["start", "cycle_error", "cycle_recovered", "stop"], runs
        told = [t for k, t in sent if "FAILING" in t or "RECOVERED" in t]
        assert len(told) == 2 and "inside a cycle" in told[0]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_store_start_is_in_armed_and_cannot_move_after_a_fill():
    """H1 / H4 swing levels are the nearest admissible swing over ALL the history the store holds, so where the
    store starts is part of their definition: ARMED records it, a start on another store is refused once a
    trade has filled, and so is a backfill of older history."""
    base = tempfile.mkdtemp(prefix="tjrh_depth_")
    try:
        df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
        store = RUN.Path(base) / "store" / "NQ_1min.csv"
        store.parent.mkdir(parents=True)
        RUN._write_frame(store, df.loc[:utc("09:19")].iloc[60:])
        first = df.index[60].isoformat()
        rec = RUN.arm(base, "test", now=at("08:00"), git=_git())
        assert rec["store_first_bar"] == {"NQ": first, "ES": None} and rec["late_fill_s"] is None
        r, sent, errors = _mem_run_on_store(base, df)
        assert not errors and JN.filled_count(JN.read_journal(base)) == 1, (errors, R.status(JN.read_journal(base))["counts"])
        start = [x for x in JN.read_journal(base) if x["kind"] == "run" and x["what"] == "start"][-1]
        assert start["store_first_bar"] == {"NQ": first} and any("completed sessions" in w for w in start["warnings"])
        # the same store: starts. Older history backfilled now: refused, in words
        older = RUN.Path(base) / "older.csv"
        RUN._write_frame(older, df.iloc[:700])                               # starts an hour before the store does
        try:
            RUN.live_runner(base, draw=False, backfill={"NQ": str(older)})
        except RUN.Refused as exc:
            assert "before the store's first bar" in str(exc)
        else:
            raise AssertionError("older history was backfilled after a fill")
        assert RUN.BarStore(store).frame.index[0] == df.index[60]
        # a store that starts elsewhere: refused at start, journaled
        RUN._write_frame(store, df.loc[:utc("09:19")].iloc[300:])
        clock = Clock(at("09:21:05") + 86400.0)
        r2 = RUN.Runner(base, MemFeed({"NQ": df}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                        allow_cold=True, inbound=None, notify=False, background=False, log=lambda line: None)
        try:
            r2.start_live()
        except RUN.Refused as exc:
            assert "ARMED" in str(exc) and "level definition" in str(exc)
        else:
            raise AssertionError("started on a store with another first bar after a fill")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _mem_run_on_store(base, df):
    clock, sent, errors = Clock(at("09:21:05")), [], []
    r = RUN.Runner(base, MemFeed({"NQ": df}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                   allow_cold=True, inbound=None, background=False, log=lambda line: None, seeds={}, accept_holes=True,
                   notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {})
    r.start_live()
    while clock() < at("10:40"):
        try:
            r.cycle_live()
        except Exception as exc:
            errors.append(type(exc).__name__)
        clock.sleep(30)
    r.stop()
    return r, sent, errors


def test_late_fill_rule_is_off_by_default_and_journals_observe_when_set():
    """Section 9.7 says only that a stop is journaled as a gap; whether a fill the human never had a chance to
    manage counts is the user's decision before `arm`. Default None: it is taken. Set: `observe`, not a trade."""
    assert TR.LATE_FILL_S is None
    f = h_fill()
    late = epoch(f["bar_time"]) + 60.0 + 3600.0                              # the process learns of it an hour later
    for limit, filled in ((None, 1), (300.0, 0)):
        base = tempfile.mkdtemp(prefix="tjrh_late_")
        try:
            j = JN.Journal(base, fsync=False)
            m = TR.TradeManager(j, armed=True, late_fill_s=limit)
            m.on_event(h_signal(f), epoch(f["bar_time"]) - 800.0)
            notes = m.on_event(f, late)
            assert JN.filled_count(j.records()) == filled
            if limit is not None:
                ob = [x for x in j.records() if x["kind"] == "observe"]
                assert len(ob) == 1 and ob[0]["why"] == "late_fill" and ob[0]["lag_s"] == 3660.0
                assert m.trade("NQ").phase == "dead" and "NOT TAKEN" in notes[0]["text"]
                m2 = TR.TradeManager(j, armed=True, late_fill_s=limit)       # a restart keeps it not taken
                m2.resume()
                m2.on_event(h_signal(f), late + 10)
                assert m2.on_event(f, late + 10) == [] and m2.trade("NQ").phase == "dead"
                assert JN.filled_count(j.records()) == 0 and len(j.records()) == 2
        finally:
            shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── 20. third review: entered = trade one, replay journals, stalls, clock, holes ──────────────────────────────

def _refused(fn, *a, **k) -> str:
    try:
        fn(*a, **k)
    except RUN.Refused as exc:
        return str(exc)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} was not refused")


def test_armed_cannot_be_undone_or_rewritten_in_the_entry_minute_of_trade_one():
    """The fill is known at the OPEN of the entry minute (a provisional fill record); the closed bar confirms it a
    minute later. In between `filled_count` is still 0 — `arm` / `disarm` must ask `entered_count`."""
    base = tempfile.mkdtemp(prefix="tjrh_entered_")
    try:
        seen = {}

        def hook(r, now):
            if abs(now - at("09:54:35")) < 1e-6:                        # 30 s into the entry minute of trade one
                recs = JN.read_journal(base)
                seen["filled"], seen["entered"], seen["n"] = JN.filled_count(recs), JN.entered_count(recs), len(recs)
                assert [x["provisional"] for x in recs if x["kind"] == "fill"] == [True]
                seen["disarm"] = _refused(RUN.disarm, base, "it opened badly", now=now)
                seen["arm"] = _refused(RUN.arm, base, "again", now=now, git=_git("d00d" * 10))
                seen["after"] = len(JN.read_journal(base))

        r, sent, errors = _mem_run(base, hook=hook)
        assert errors == [] and seen["filled"] == 0 and seen["entered"] == 1, seen
        assert "trade one" in seen["disarm"] and "trade one" in seen["arm"] and seen["after"] == seen["n"]
        recs = JN.read_journal(base)
        armed = [x for x in recs if x["kind"] == "armed"]
        assert len(armed) == 1 and armed[0]["commit"] == "c0ffee" * 6 + "abcd" and JN.filled_count(recs) == 1
        # disarmed BEFORE the entry, entered while... no: a disarmed run takes no entry. But a journal whose ARMED
        # was superseded by a disarm record written by hand must still refuse a new arm once a trade was entered
        j = JN.Journal(base, fsync=False)
        j.write("armed", {"armed": False, "reason": "written around the guard"})
        assert JN.armed_record(j.records()) is None
        assert "trade one" in _refused(RUN.arm, base, "new rule", git=_git("d00d" * 10))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    # a provisional fill the closed bar voids, and a skipped setup, are not entered trades
    s = Sim(build_day(), h_fill(), preview=True)
    try:
        s.m.on_event(s.signal, at("09:41"))
        assert JN.entered_count(s.j.records()) == 0
        s.m.on_preview({**s.fill, "provisional": True}, at("09:54"))
        assert JN.entered_count(s.j.records()) == 1 and JN.filled_count(s.j.records()) == 0
        inv = {k: s.fill[k] for k in ("instrument", "day", "setup_id", "bar_time", "knowable_at", "side")}
        inv.update(kind="invalidated", reason="no_risk", traded=98.9, wick_stop=99.0)
        s.m.on_event(inv, at("09:55"))
        assert JN.entered_count(s.j.records()) == 0
    finally:
        s.close()
    s = Sim(build_day(), h_fill(), armed=False, preview=True)             # observe mode: nothing is entered
    try:
        s.m.on_event(s.signal, at("09:41"))
        s.m.on_preview({**s.fill, "provisional": True}, at("09:54"))
        assert JN.entered_count(s.j.records()) == 0
    finally:
        s.close()


def test_a_replay_never_writes_into_the_experiments_journal_and_live_never_into_a_replays():
    base = tempfile.mkdtemp(prefix="tjrh_mix_")
    try:
        df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
        csv = RUN.write_synthetic_csv(os.path.join(base, "hand_NQ.csv"), df, "test day")
        real = os.path.join(base, "tjr_human_runs")
        RUN.arm(real, "the real thing", now=at("08:00"), git=_git())
        n = len(JN.read_journal(real))
        why = _refused(RUN.replay, {"NQ": str(csv)}, base_dir=real, allow_cold=True)
        assert "did not write" in why and "count toward the 100" in why
        assert len(JN.read_journal(real)) == n and JN.entered_count(JN.read_journal(real)) == 0
        # ... also a journal that was only ever observed (a run start, no ARMED)
        obs = os.path.join(base, "observed")
        JN.Journal(obs, fsync=False).write("run", {"what": "start", "mode": "observe"})
        _refused(RUN.replay, {"NQ": str(csv)}, base_dir=obs, allow_cold=True)
        assert len(JN.read_journal(obs)) == 1
        # a directory of its own is fine, and a second replay into it too
        keep = os.path.join(base, "keep")
        out = RUN.replay({"NQ": str(csv)}, base_dir=keep, allow_cold=True)
        assert out["facts"]["trades"]["filled"] == 1
        who = JN.origin(JN.read_journal(keep))
        assert who["live"] == 0 and who["replay"] == len(JN.read_journal(keep))
        n = len(JN.read_journal(keep))
        # the other way round: no ARMED, no live run and no backfill record on top of replayed trades
        assert "replay's journal" in _refused(RUN.arm, keep, "go", git=_git())
        clock = Clock(at("09:21:05"))
        r = RUN.Runner(keep, MemFeed({"NQ": df}, clock), instruments=("NQ",), clock=clock, sleep=clock.sleep,
                       allow_cold=True, inbound=None, background=False, log=lambda line: None,
                       stores={"NQ": RUN.BarStore(None, frame=df.loc[:utc("09:19")])}, notify=False)
        assert "replay's journal" in _refused(r.start_live)
        assert "replay's journal" in _refused(RUN.live_runner, keep)
        assert len(JN.read_journal(keep)) == n                         # not even a `refused` record: it is not its journal
        assert RUN.journal_is_reportable(JN.read_journal(keep))[0]
        # a journal that mixes both (written before this guard existed) is not reportable
        mixed = JN.read_journal(keep) + [{"kind": "run", "what": "start", "mode": "armed", "seq": 10 ** 6}]
        ok, why = RUN.journal_is_reportable(mixed)
        assert not ok and "mixes" in why
    finally:
        shutil.rmtree(base, ignore_errors=True)


class ShiftedFeed(MemFeed):
    """MemFeed whose chart can FREEZE (it still answers, its data stops advancing) and whose true time can differ
    from the runner's PC clock (`offset` = PC clock minus true time)."""

    def __init__(self, frames, clock, freeze_at=None, offset=0.0):
        super().__init__(frames, clock)
        self.freeze_at, self.offset = freeze_at, float(offset)

    def cycle(self, now, want):
        t = now - self.offset
        return super().cycle(t if self.freeze_at is None else min(t, self.freeze_at), want)


def _live(base, feed_cls, clock, df, store_to, sent, **kw):
    feed = feed_cls({"NQ": df}, clock, **kw.pop("feed", {}))
    r = RUN.Runner(base, feed, instruments=("NQ",), clock=clock, sleep=clock.sleep, allow_cold=True, inbound=None,
                   background=False, log=lambda line: None,
                   stores=kw.pop("stores", None) or {"NQ": RUN.BarStore(None, frame=df.loc[:utc(store_to)])},
                   notify=lambda b, kind, text, fields=None: sent.append((kind, text)) or {}, **kw)
    return r, feed


def test_a_frozen_chart_is_a_feed_failure_journaled_and_pushed_not_silence():
    assert RUN.stalled_minutes("2026-07-15T13:40:00", at("09:40:50")) == 0
    assert RUN.stalled_minutes("2026-07-15T13:40:00", at("09:43:59")) == 2
    assert RUN.stalled_minutes("2026-07-15T13:40:00", at("09:44:00")) == 3
    assert RUN.stalled_minutes("2026-07-17T20:59:00", pd.Timestamp("2026-07-19 22:02:30", tz="UTC").timestamp()) == 2
    assert RUN.stalled_minutes("2026-07-15T20:59:00", pd.Timestamp("2026-07-15 22:01:10", tz="UTC").timestamp()) == 1
    assert RUN.stalled_minutes(None, at("09:44:00")) == 0
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
    base = tempfile.mkdtemp(prefix="tjrh_stall_")
    try:                                                                  # frozen 09:40:10 .. 09:47, in the window
        clock, sent = Clock(at("09:21:05")), []
        RUN.arm(base, "test", now=at("08:00"), git=_git())
        r, feed = _live(base, ShiftedFeed, clock, df, "09:19", sent)
        r.start_live()
        feed.freeze_at = at("09:40:10")
        told_at = None
        while clock() < at("10:40"):
            if clock() >= at("09:47:00"):
                feed.freeze_at = None                                     # the chart re-syncs
            r.cycle_live()
            if told_at is None and any("FAILING" in t for k, t in sent):
                told_at = clock()
            clock.sleep(30)
        r.stop()
        recs = JN.read_journal(base)
        feeds = [(x["what"], x.get("scope"), x.get("error_kind")) for x in recs if x["kind"] == "feed"]
        assert feeds == [("failure", "NQ", "stalled"), ("recovered", "NQ", None)], feeds
        assert told_at == at("09:44:05"), told_at                        # three trading minutes without a bar
        fail = [t for k, t in sent if "FAILING" in t]
        assert len(fail) == 1 and "STALLED" in fail[0] and "PC clock is fast" in fail[0] and "stops and targets" in fail[0]
        assert len([t for k, t in sent if "RECOVERED" in t]) == 1
        assert all(k == "trade" for k, t in sent if "FAILING" in t or "RECOVERED" in t)
        sig = [t for k, t in sent if k == "signal"]
        assert len(sig) == 1 and sig[0].startswith("[LATE")               # the 09:40 signal, told late and said so
        assert JN.filled_count(recs) == 1 and r.facts["feed_failures"] >= 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
    base = tempfile.mkdtemp(prefix="tjrh_stall_")
    try:                                                                  # at noon with nothing on: journaled, not pushed
        clock, sent = Clock(at("12:00:05")), []
        r, feed = _live(base, ShiftedFeed, clock, build_day(), "11:58", sent)
        r.start_live()
        r.cycle_live()
        feed.freeze_at = at("12:00:30")
        for _ in range(30):
            clock.sleep(60)
            r.cycle_live()
        recs = [x for x in JN.read_journal(base) if x["kind"] == "feed"]
        assert [(x["what"], x["scope"]) for x in recs] == [("failure", "NQ"), ("still_failing", "NQ"), ("still_failing", "NQ")]
        assert sent == []
        # while it is stalled the stale forming bar is not a price: a pending entry is not previewed on it
        r._pending["NQ"] = r.feed.cycle(clock(), {})["NQ"].forming["time"]
        r.cycle_live()
        assert r.facts["previews"] == 0 and "NQ" in r._pending
        r.stop()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _dated(text: str, sent_ts: float):
    cmd = C.parse(text)
    cmd.sent_ts = float(sent_ts)                                          # Telegram's own message date
    return cmd


def test_a_wrong_pc_clock_is_refused_at_the_start_and_told_when_it_drifts():
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN, extra_days=1)
    for offset, words in ((86400.0, "older than the PC clock"), (-600.0, "BEHIND")):
        base = tempfile.mkdtemp(prefix="tjrh_clock_")
        try:
            clock, sent = Clock(at("09:21:05") + offset), []
            r, feed = _live(base, ShiftedFeed, clock, df, "09:19", sent, feed={"offset": offset})
            r.start_live()
            why = _refused(r.cycle_live)
            assert words in why and "clock" in why, why
            recs = JN.read_journal(base)
            ref = [x for x in recs if x["kind"] == "run" and x["what"] == "refused"]
            assert len(ref) == 1 and ref[0]["skew_s"] == (86405.0 if offset > 0 else -595.0)
            assert not [x for x in recs if x.get("setup_id")] and r.facts["bars"]["NQ"] == 0
            r.stop()
        finally:
            shutil.rmtree(base, ignore_errors=True)
    base = tempfile.mkdtemp(prefix="tjrh_clock_")
    try:                                                                  # the clock falls behind while running, at noon
        clock, sent = Clock(at("12:00:05")), []
        r, feed = _live(base, ShiftedFeed, clock, build_day(), "11:58", sent)      # an inert day: nothing is on
        r.start_live()
        r.cycle_live()
        assert sent == [] and not r._failing
        feed.offset = -200.0                                              # PC clock 200 s slow: the chart is "in the future"
        for _ in range(3):
            clock.sleep(60)
            r.cycle_live()
        runs = [x for x in JN.read_journal(base) if x["kind"] == "run" and str(x["what"]).startswith("clock")]
        assert [x["what"] for x in runs] == ["clock_suspect"] and -200.0 <= runs[0]["skew_s"] <= -140.0, runs
        assert len(sent) == 1 and sent[0][0] == "trade" and "PC CLOCK is suspect" in sent[0][1] and "BEHIND" in sent[0][1]
        feed.offset = 0.0
        clock.sleep(60)
        r.cycle_live()
        runs = [x["what"] for x in JN.read_journal(base) if x["kind"] == "run" and str(x["what"]).startswith("clock")]
        assert runs == ["clock_suspect", "clock_recovered"] and "RECOVERED: clock" in sent[-1][1] and len(sent) == 2
        # Telegram's own message date is the second witness: dated after its receipt = the PC clock is slow
        now = clock()
        r._command(_dated("skip", now + 40.0), now)
        runs = [x for x in JN.read_journal(base) if x["kind"] == "run" and str(x["what"]).startswith("clock")]
        assert runs[-1]["what"] == "clock_suspect" and runs[-1]["skew_s"] == -40.0 and "AFTER" in runs[-1]["error"]
        r._command(_dated("skip", now - 2.0), now)
        assert "clock" not in r._failing
        for k in range(2):                                                # two stale commands in a row: fast clock or slow delivery
            r._command(_dated("skip", now - 500.0), now)
            assert ("clock" in r._failing) == (k == 1)
        rej = [x["why"] for x in JN.read_journal(base) if x["kind"] == "reject"]
        assert rej.count("stale") == 2
        r.stop()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_a_hole_that_opens_mid_run_is_pushed_and_the_day_is_not_traded_until_a_person_accepts_it():
    df = build_day({**FULL_LONG, "10:30": (100.0, 102.4, 99.9, 102.2)}, fill_runs=HIGH_RUN)
    for slept_from, woke, hole_pushed in (("08:30:05", "09:26:05", True), ("03:00:05", "04:30:05", False)):
        base = tempfile.mkdtemp(prefix="tjrh_hole_")
        try:
            clock, sent = Clock(at(slept_from)), []
            RUN.arm(base, "test", now=at("02:00"), git=_git())
            r, feed = _live(base, MemFeed, clock, df, {"08": "08:28", "03": "02:58"}[slept_from[:2]], sent)
            r.start_live()
            r.cycle_live()
            clock.t = at(woke)                                            # the PC slept; the chart serves the last 40 bars only
            while clock() < at("10:40"):
                r.cycle_live()
                clock.sleep(30 if clock() >= at("09:20") else 60)
            recs = JN.read_journal(base)
            feeds = [x["what"] for x in recs if x["kind"] == "feed"]
            assert feeds == ["hole", "day_not_routed"], feeds
            nr = [x for x in recs if x["kind"] == "feed" and x["what"] == "day_not_routed"][0]
            assert nr["day"] == str(DAY.date()) and nr["holes"][0]["trading_minutes"] >= 10
            holes = [t for k, t in sent if "HOLE in the 1-minute store" in t]
            days = [t for k, t in sent if "NOT traded" in t]
            assert len(holes) == (1 if hole_pushed else 0) and len(days) == 1 and "--accept-holes" in days[0]
            assert all(k == "trade" for k, t in sent)
            assert not [x for x in recs if x["kind"] in ("signal", "fill", "observe", "event")]   # the day was not routed
            assert JN.entered_count(recs) == 0 and r.facts["events"]["NQ"].get("signal") == 1    # the detector saw it
            r.stop()
            if not hole_pushed:
                continue
            # a person looks, decides it may stand (an early close) and restarts with --accept-holes: the day is routed
            clock2, sent2 = Clock(at("09:27:05")), []
            r2, _ = _live(base, MemFeed, clock2, df, "09:25", sent2, accept_holes=True,
                          stores={"NQ": RUN.BarStore(None, frame=r.stores["NQ"].frame.loc[:utc("09:25")])})
            r2.start_live()
            while clock2() < at("09:45"):
                r2.cycle_live()
                clock2.sleep(30)
            r2.stop()
            recs = JN.read_journal(base)
            assert [x["what"] for x in recs if x["kind"] == "feed"][-1] == "holes_accepted"
            assert len([x for x in recs if x["kind"] == "signal"]) == 1 and not [t for k, t in sent2 if "NOT traded" in t]
        finally:
            shutil.rmtree(base, ignore_errors=True)


def test_the_live_loop_asks_windows_to_stay_awake_and_says_so():
    calls = []
    real = RUN.keep_awake
    assert isinstance(real(True), bool) and isinstance(real(False), bool)   # never raises, on any platform
    RUN.keep_awake = lambda on: calls.append(on) or True
    base = tempfile.mkdtemp(prefix="tjrh_awake_")
    try:
        for want in (True, False):
            clock, sent = Clock(at("12:00:05")), []
            r, _ = _live(os.path.join(base, str(want)), MemFeed, clock, build_day(), "11:58", sent, keep_awake=want)
            r.run_live(max_cycles=1)
            start = [x for x in JN.read_journal(os.path.join(base, str(want))) if x["kind"] == "run" and x["what"] == "start"]
            assert start[0]["keep_awake"] is want
        assert calls == [True, False]                                       # asked, and undone; never asked when told not to
    finally:
        RUN.keep_awake = real
        shutil.rmtree(base, ignore_errors=True)


# ────────────────────────────── the test runner ──────────────────────────────

def main() -> int:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    only = [a for a in sys.argv[1:] if not a.startswith("-")]          # optional: name substrings to run
    if only:
        tests = [(k, v) for k, v in tests if any(a in k for a in only)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {name}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Unit tests for quantlab.strategies.predictive.tjr_intraday (DESIGN-tjr-intraday.md §9).

Runs standalone (`python tests/test_tjr_intraday.py`) and under pytest. The
hand-built days are 5-minute bars laid out on the New York clock and converted
to the tz-naive UTC index the codebase uses, so the timezone path is exercised by
every test, not mocked around.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlab.data import load_csv, synthetic  # noqa: E402
from quantlab.strategies.predictive import tjr_intraday as m  # noqa: E402
from quantlab.strategies.predictive.primitives import atr  # noqa: E402

ET = m.ET
DAY = pd.Timestamp("2026-07-15")            # a Wednesday, EDT: 09:30 ET is 13:30 UTC
NQ_CSV = os.path.join(ROOT, "data", "NQ_5min_60d.csv")
ES_CSV = os.path.join(ROOT, "data", "ES_5min_60d.csv")

INERT = (100.0, 100.5, 99.5, 100.0)         # o h l c: no swing forms on a run of these


def et(hhmm: str, day_offset: int = 0) -> pd.Timestamp:
    """An ET wall-clock time on day D (+offset) as the naive-UTC stamp the frame uses."""
    ts = pd.Timestamp(f"{(DAY + pd.Timedelta(days=day_offset)).date()} {hhmm}", tz=ET)
    return ts.tz_convert("UTC").tz_localize(None)


def build_day(overrides: dict | None = None, night=INERT, day=INERT, extra_days: int = 0,
              freq: str = "5min") -> pd.DataFrame:
    """Two full CME sessions, D-1 and D, on `freq` bars (5-minute unless a test asks
    for 1-minute), plus `extra_days` inert sessions after D. Keys of `overrides`
    are 'HH:MM' on D or '-1 HH:MM' on the calendar day before, values are
    (o, h, l, c). `night` is the inert bar for everything stamped before 08:30 ET
    on D (session D-1, Asia, London), `day` for the rest."""
    start = pd.Timestamp(f"{(DAY - pd.Timedelta(days=2)).date()} 18:00", tz=ET)
    end = pd.Timestamp(f"{(DAY + pd.Timedelta(days=extra_days)).date()} 17:00", tz=ET) - pd.Timedelta(freq)
    ix = pd.date_range(start, end, freq=freq, tz=ET)
    ix = ix[~((ix.hour == 17))]                          # the daily maintenance hour
    cutoff = pd.Timestamp(f"{DAY.date()} 08:30", tz=ET)
    rows = [night if t < cutoff else day for t in ix]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=ix)
    for key, ohlc in (overrides or {}).items():
        off, hhmm = (key.split(" ") if " " in key else ("0", key))
        stamp = pd.Timestamp(f"{(DAY + pd.Timedelta(days=int(off))).date()} {hhmm}", tz=ET)
        assert stamp in df.index, key
        df.loc[stamp, ["open", "high", "low", "close"]] = ohlc
    df["volume"] = 0.0
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "date"
    return df


# The day where every rule fires, long side. Levels: PDH 105 / PDL 95,
# ASIA_H 103 / ASIA_L 98, LON_H 102 / LON_L 97.
FULL_SETUP = {
    "-1 10:00": (100.0, 105.0, 95.0, 100.0),      # previous session's extremes
    "-1 22:00": (100.0, 103.0, 98.0, 100.0),      # Asia of D
    "05:00":    (100.0, 102.0, 97.0, 100.0),      # London of D
    "09:15":    (100.0, 101.5, 99.5, 100.0),      # swing high 101.5, confirmed on the 09:25 bar
    "09:35":    (98.5, 98.6, 96.5, 98.0),         # sweep: wick under ASIA_L and LON_L, close back above LON_L
    "09:40":    (98.0, 98.9, 98.0, 98.8),         # displacement leg
    "09:45":    (98.9, 101.9, 98.8, 101.8),       # BOS close > 101.5; bullish FVG [98.6, 98.8]; EQ = 99.0
    "09:50":    (101.0, 101.2, 98.7, 99.5),       # retrace into the gap -> entry 98.8
    "10:10":    (99.5, 102.5, 99.3, 102.2),       # takes LON_H 102
}


def day_row(log: pd.DataFrame) -> pd.Series:
    """The log row for day D. Session D-1 has a 09:30 bar too and logs as no_sweep."""
    assert list(log["session_day"]) == [DAY - pd.Timedelta(days=1), DAY], log
    return log.iloc[-1]


def variant(**changes) -> dict:
    """FULL_SETUP with some bars moved or replaced. A value of None makes the bar inert."""
    d = dict(FULL_SETUP)
    for key, v in changes.items():
        if v is None:
            d.pop(key, None)
        else:
            d[key] = v
    return d


# ────────────────────────────── the day where everything fires ──────────────────────────────

def test_full_setup_produces_the_expected_trade():
    df = build_day(FULL_SETUP)
    res = m.simulate(df, stop_buffer_atr=0.0)
    t = res.trades
    assert len(t) == 1, t
    row = t.iloc[0]
    assert row.side == 1
    assert row.entry_time == et("09:50") and row.exit_time == et("10:10")
    assert row.entry == 98.8 and row.stop == 96.5 and row.target == 102.0 and row.exit == 102.0
    assert row.reason == "target" and row.bars == 4
    assert abs(row.r - (102.0 - 98.8) / (98.8 - 96.5)) < 1e-12
    assert row.sweep_level == "LON_L" and row.target_level == "LON_H" and row.zone_kind == "fvg"
    assert row.smt is False or row.smt == False  # noqa: E712  (numpy bool)
    assert pd.Timestamp(row.session_day) == DAY
    # position: entry bar through the bar before the exit bar
    held = res.position[res.position != 0]
    assert list(held.index) == [et("09:50"), et("09:55"), et("10:00"), et("10:05")], held
    assert (held == 1).all()
    d = day_row(m.day_log(df, stop_buffer_atr=0.0))
    assert d.outcome == "traded" and d.sweep_time == et("09:35") and d.bos_time == et("09:45")
    # cut the frame inside the trade: the position is the same, the open trade reads eod
    full = res.position
    for cut in ("09:50", "09:55", "10:00", "10:05", "10:10"):
        k = df.index.get_loc(et(cut)) + 1
        part = m.simulate(df.iloc[:k], stop_buffer_atr=0.0)
        assert float((full.iloc[:k] - part.position).abs().max()) == 0.0, cut
        if cut == "09:50":
            assert part.trades.empty and part.position.iloc[-1] == 1   # opened on the last bar, still on
        elif cut != "10:10":
            assert len(part.trades) == 1 and part.trades.iloc[0].reason == "eod"


def test_default_stop_buffer_sits_under_the_wick_by_a_quarter_atr():
    df = build_day(FULL_SETUP)
    t = m.simulate(df).trades
    assert len(t) == 1
    i = df.index.get_loc(et("09:50"))
    expected = 96.5 - 0.25 * atr(df, 14)[i]
    assert abs(t.iloc[0].stop - expected) < 1e-12 and t.iloc[0].stop < 96.5


# ────────────────────────────── one violated rule each: no trade ──────────────────────────────

def _no_trade(overrides, outcome, **params):
    df = build_day(overrides)
    res = m.simulate(df, stop_buffer_atr=0.0, **params)
    assert res.trades.empty, res.trades
    assert (res.position == 0).all()
    d = day_row(m.day_log(df, stop_buffer_atr=0.0, **params))
    assert d.outcome == outcome, d


def test_sweep_outside_the_window():
    # the sweep bar moves to 09:25: same shape, wrong clock
    _no_trade(variant(**{"09:35": None, "09:25": FULL_SETUP["09:35"]}), "no_sweep")


def test_bos_after_1005():
    _no_trade(variant(**{"09:45": None, "10:10": FULL_SETUP["09:45"]}), "no_bos")


def test_bos_inside_the_sweep_window_still_fills_at_0950():
    """The whole setup five minutes earlier: sweep 09:30, displacement 09:35, BOS
    on the 09:40 bar. The 09:45 bar is then a sweep-window bar with a live
    'entry' setup on it. The contract lets a BOS land on any bar after the sweep
    and the fill on a later entry-window bar, so this must trade at 09:50.
    Review caught the module killing the setup on 09:45 instead; on the
    seven-month NQ frame that was most of the days that would have traded."""
    d = variant(**{"09:35": None, "09:40": None, "09:45": None})
    d["09:30"], d["09:35"], d["09:40"] = FULL_SETUP["09:35"], FULL_SETUP["09:40"], FULL_SETUP["09:45"]
    df = build_day(d)
    res = m.simulate(df, stop_buffer_atr=0.0)
    assert len(res.trades) == 1, res.trades
    row = res.trades.iloc[0]
    assert row.entry_time == et("09:50") and row.entry == 98.8 and row.exit == 102.0
    assert row.reason == "target" and row.zone_kind == "fvg"
    log = day_row(m.day_log(df, stop_buffer_atr=0.0))
    assert log.sweep_time == et("09:30") and log.bos_time == et("09:40") and log.outcome == "traded"


def test_zone_above_equilibrium():
    # same leg, lifted: the gap becomes [99.1, 99.3], top above EQ 99.0
    _no_trade(variant(**{"09:35": (98.9, 99.1, 96.5, 98.0),
                         "09:40": (98.0, 99.2, 98.0, 99.2),
                         "09:45": (99.4, 101.9, 99.3, 101.8)}), "no_zone")


def test_fill_outside_the_entry_window():
    _no_trade(variant(**{"09:50": None, "10:10": FULL_SETUP["09:50"]}), "no_fill")


def _no_target_day(pdh: float) -> dict:
    # Everything overnight lives around 96 so the three highs sit under the entry.
    return {
        "-1 10:00": (96.0, pdh, 95.0, 96.0),
        "-1 22:00": (96.0, 96.4, 95.5, 96.0),
        "05:00":    (96.0, 96.4, 95.6, 96.0),
        "09:15":    (96.0, 98.0, 95.7, 96.0),      # swing high 98
        "09:35":    (96.2, 96.3, 95.3, 96.0),      # sweeps ASIA_L 95.5 (lowest breached), no high touched
        "09:40":    (96.0, 96.5, 96.0, 96.4),
        "09:45":    (96.5, 98.4, 96.6, 98.3),      # BOS > 98; FVG [96.3, 96.6]; EQ = 96.65
        "09:50":    (97.5, 97.6, 96.5, 96.9),      # entry 96.6
    }


def test_no_liquidity_above_entry():
    night = (96.0, 96.3, 95.7, 96.0)
    df = build_day(_no_target_day(pdh=96.5), night=night, day=night)
    res = m.simulate(df, stop_buffer_atr=0.0)
    assert res.trades.empty and (res.position == 0).all()
    d = day_row(m.day_log(df, stop_buffer_atr=0.0))
    assert d.outcome == "no_target" and d.sweep_level == "ASIA_L"
    # control: raise PDH above the entry and the same day trades at it
    df2 = build_day(_no_target_day(pdh=97.0), night=night, day=night)
    t = m.simulate(df2, stop_buffer_atr=0.0).trades
    assert len(t) == 1 and t.iloc[0].target_level == "PDH" and t.iloc[0].target == 97.0
    assert t.iloc[0].entry == 96.6 and t.iloc[0].stop == 95.3


def test_ambiguous_sweep_bar_is_ignored():
    # the same sweep bar also wicks through LON_H 102 and closes back under it
    _no_trade(variant(**{"09:35": (98.5, 102.5, 96.5, 98.0)}), "no_sweep")


# ────────────────────────────── SMT ──────────────────────────────

def _with_pair(df: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[f"pair_{c}"] = pair[c].values
    return out


def test_smt_pair_also_swept_means_no_setup():
    df = build_day(FULL_SETUP)
    both = _with_pair(df, df)                      # the pair took its London low too
    res = m.simulate(both, stop_buffer_atr=0.0, smt=True)
    assert res.trades.empty and (res.position == 0).all()
    assert day_row(m.day_log(both, stop_buffer_atr=0.0, smt=True)).outcome == "smt"


def test_smt_divergence_trades_and_is_tagged():
    df = build_day(FULL_SETUP)
    pair = df.copy()
    pair.loc[et("09:35"), "low"] = 97.5           # the pair held its London low (97)
    res = m.simulate(_with_pair(df, pair), stop_buffer_atr=0.0, smt=True)
    assert len(res.trades) == 1 and bool(res.trades.iloc[0].smt) is True
    assert res.trades.iloc[0].entry == 98.8


def test_smt_needs_the_pair_columns():
    df = build_day(FULL_SETUP)
    try:
        m.simulate(df, smt=True)
    except ValueError as e:
        assert "pair" in str(e)
    else:
        raise AssertionError("smt=True without pair columns must raise")


# ────────────────────────────── inverted FVG ──────────────────────────────

def test_inverted_fvg_zone_and_the_allow_ifvg_switch():
    # no bullish gap on the leg (BOS low 98.5 < sweep high 98.6); a bearish gap
    # [98.9, 98.95] from 09:40 is closed above on the BOS bar and becomes the zone.
    day = variant(**{"09:30": (100.0, 100.5, 98.95, 100.0),
                     "09:45": (98.9, 101.9, 98.5, 101.8)})
    df = build_day(day)
    t = m.simulate(df, stop_buffer_atr=0.0).trades
    assert len(t) == 1 and t.iloc[0].zone_kind == "ifvg"
    assert t.iloc[0].entry == 98.95 and t.iloc[0].target_level == "LON_H"
    off = m.simulate(df, stop_buffer_atr=0.0, allow_ifvg=False)
    assert off.trades.empty and (off.position == 0).all()
    assert day_row(m.day_log(df, stop_buffer_atr=0.0, allow_ifvg=False)).outcome == "no_zone"


# ────────────────────────────── frames with nothing to do ──────────────────────────────

def _daily_frame() -> pd.DataFrame:
    if not os.path.exists(NQ_CSV):
        return synthetic(n=300, seed=1)
    df = load_csv(NQ_CSV)
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


def test_all_flat_without_a_0930_bar():
    for frame in (synthetic(n=400), _daily_frame()):
        for smt in (False, True):
            s = m.signal(frame, smt=smt)
            assert isinstance(s, pd.Series) and len(s) == len(frame)
            assert s.index.equals(frame.index) and (s == 0).all()
            res = m.simulate(frame, smt=smt)
            assert res.trades.empty and list(res.trades.columns) == m._TRADE_COLS
            assert m.day_log(frame, smt=smt).empty


# ────────────────────────────── real data ──────────────────────────────

def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print(f"  skipped: {msg}")
        return True


def _minutes_et(ts) -> int:
    t = pd.Timestamp(ts).tz_localize("UTC").tz_convert(ET)
    return t.hour * 60 + t.minute


def _truncation_probe(df: pd.DataFrame, ks: list[int], **params) -> float:
    full = m.signal(df, **params)
    worst = 0.0
    for k in ks:
        part = m.signal(df.iloc[:k], **params)
        assert len(part) == k
        worst = max(worst, float((full.iloc[:k] - part).abs().max()))
    return worst


def test_causality_on_the_real_nq_frame():
    if not os.path.exists(NQ_CSV):
        return _skip("data/NQ_5min_60d.csv not present")
    df = load_csv(NQ_CSV)
    log = m.day_log(df)
    loc = df.index.get_loc
    ks = []
    # a setup that is alive at the cut, waiting for its break
    d = log[log.outcome.isin(["no_bos", "no_zone", "no_fill", "traded"])].iloc[0]
    ks.append(loc(d.sweep_time) + 1)
    # a zone that is alive at the cut, waiting for its fill
    d = log[log.outcome.isin(["no_fill", "traded"])].iloc[0]
    ks.append(loc(d.bos_time) + 1)
    # midday after the setup has expired, and midday on a day that never swept
    d = log[log.outcome == "no_zone"].iloc[0]
    ks.append(loc(d.session_day.tz_localize(ET).replace(hour=12).tz_convert("UTC").tz_localize(None)))
    d = log[log.outcome == "no_sweep"].iloc[-1]
    ks.append(loc(d.session_day.tz_localize(ET).replace(hour=14, minute=30).tz_convert("UTC").tz_localize(None)))
    assert len(set(ks)) == 4
    for k in ks:                                   # every cut is inside a trading day
        assert 9 * 60 + 30 <= _minutes_et(df.index[k - 1]) < 16 * 60, df.index[k - 1]
    worst = _truncation_probe(df, ks)
    assert worst == 0.0, worst


def test_causality_inside_an_open_trade():
    """NQ has no trade under the defaults, so the open-trade cut comes from whichever
    frame has one; the truncated run must report that trade as held (`eod`)."""
    for path in (NQ_CSV, ES_CSV):
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        trades = m.simulate(df).trades
        if trades.empty:
            continue
        ks = []
        for _, t in trades.iterrows():
            e = df.index.get_loc(t.entry_time)
            ks += [e + 1, e + max(1, int(t.bars) // 2)]
        ks = sorted(set(ks))
        assert _truncation_probe(df, ks) == 0.0
        t = trades.iloc[0]
        k = df.index.get_loc(t.entry_time) + 1
        part = m.simulate(df.iloc[:k])
        assert part.trades.empty and part.position.iloc[-1] == t.side   # opened on the last bar
        if int(t.bars) >= 2:
            part = m.simulate(df.iloc[:k + 1]).trades
            assert len(part) == 1 and part.iloc[0].reason == "eod"
        return
    return _skip("no real frame with a trade under the defaults")


def test_every_real_trade_satisfies_the_acceptance_list():
    for path in (NQ_CSV, ES_CSV):
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        clock = m.session_clock(df.index)
        levels = m.session_levels(clock, df["high"].to_numpy(float), df["low"].to_numpy(float))
        res = m.simulate(df)
        log = m.day_log(df).set_index("session_day")
        assert res.trades["session_day"].is_unique             # at most one trade a day
        assert set(res.trades["reason"]) <= {"stop", "target", "flat", "eod"}
        for _, t in res.trades.iterrows():
            d = log.loc[t.session_day]
            s, b, e = (df.index.get_loc(x) for x in (d.sweep_time, d.bos_time, t.entry_time))
            assert clock.in_sweep[s] and clock.in_entry[e] and s < b < e
            wick = df["low"].iloc[s] if t.side > 0 else df["high"].iloc[s]
            assert t.side * (wick - t.stop) >= 0                # stop beyond the sweep wick
            six = {k: levels[k][e] for k in m.LOWS + m.HIGHS}
            assert six[t.target_level] == t.target and t.side * (t.target - t.entry) > 0
            assert t.zone_kind in ("fvg", "ifvg") and d.sweep_level == t.sweep_level
        # and the frame itself has the expected shape
        assert (res.position.index == df.index).all()


# ────────────────────────────── round 2: the stop-width variable (§11.4) ──────────────────────────────

def test_each_stop_mode_puts_the_stop_where_the_table_says():
    """FULL_SETUP under the defaults, one mode at a time. ATR is the context's — on
    the 5-minute frame that is atr(df, 14) at the entry bar, the same call the
    module makes. The sweep bar printed the day's low, so `session` lands on
    `wick`. Every mode still fills at 98.8 and runs to LON_H."""
    df = build_day(FULL_SETUP)
    i = df.index.get_loc(et("09:50"))
    a = atr(df, 14)[i]
    table = {"wick": 96.5 - 0.25 * a, "atr1.0": 98.8 - 1.0 * a, "atr1.5": 98.8 - 1.5 * a,
             "atr2.0": 98.8 - 2.0 * a, "session": 96.5 - 0.25 * a}
    assert tuple(table) == m.STOP_MODES
    for mode, stop in table.items():
        t = m.simulate(df, stop_mode=mode).trades
        assert len(t) == 1, (mode, t)
        row = t.iloc[0]
        assert abs(row.stop - stop) < 1e-12, (mode, row.stop, stop)
        assert row.entry == 98.8 and row.target == 102.0 and row.reason == "target", mode
        assert row.stop_mode == mode and int(row.entry_tf) == 5
        assert abs(row.r - (102.0 - 98.8) / (98.8 - row.stop)) < 1e-12
    assert list(m.simulate(df).trades.columns) == m._TRADE_COLS
    assert m._TRADE_COLS[-2:] == ["stop_mode", "entry_tf"]
    for bad in ({"stop_mode": "atr3.0"}, {"entry_tf": 15}):
        try:
            m.simulate(df, **bad)
        except ValueError as e:
            assert next(iter(bad)) in str(e)
        else:
            raise AssertionError(f"{bad} must raise")


def test_session_stop_is_the_wick_stop_unless_an_earlier_bar_went_lower():
    df = build_day(FULL_SETUP)
    w = m.simulate(df, stop_mode="wick").trades
    s = m.simulate(df, stop_mode="session").trades
    assert len(w) == 1 and len(s) == 1 and w.iloc[0].stop == s.iloc[0].stop
    # An Asia bar of D printed 96.0, under the sweep's 96.5. ASIA_L moves to 96.0
    # so the 09:35 wick now only takes LON_L; the sweep and the trade are the
    # same, the session extreme is not.
    lower = build_day(variant(**{"01:00": (100.0, 100.5, 96.0, 100.0)}))
    w2 = m.simulate(lower, stop_mode="wick").trades
    s2 = m.simulate(lower, stop_mode="session").trades
    assert len(w2) == 1 and len(s2) == 1
    assert w2.iloc[0].sweep_level == "LON_L" and s2.iloc[0].sweep_level == "LON_L"
    assert w2.iloc[0].entry == s2.iloc[0].entry == 98.8
    assert s2.iloc[0].stop < w2.iloc[0].stop
    assert abs((w2.iloc[0].stop - s2.iloc[0].stop) - 0.5) < 1e-9      # same ATR buffer, 0.5 lower base
    assert s2.iloc[0].stop < 96.0                                     # under the session low by the buffer


# ────────────────────────────── round 2: the two-timeframe model (§11.3) ──────────────────────────────

# The 1-minute day where every rule fires, long side. Same six levels as
# FULL_SETUP. Structure lives on the 5-minute bins: a swing high 101.5 on the
# 09:15 bin, the sweep on the 09:30 bin (wick 96.5 printed at 09:31, close
# 98.0 back above LON_L 97), no BOS on the 09:35 bin, BOS on the 09:40 bin
# (close 101.8 > 101.5, completing at 09:44). The bars into the open overlap so
# no 1-minute gap forms before the leg; the leg leaves two bullish gaps under
# EQ 99.0, [98.4, 98.5] at 09:37 and [98.7, 98.8] at 09:38, and three above
# it that the discount rule must reject. The 09:43 minute alone would close
# the partial 09:40 bin above 101.5: a model reading partial bins breaks
# structure a minute early, and the truncation test below checks it does not.
FULL_SETUP_1M = {
    "-1 10:00": (100.0, 105.0, 95.0, 100.0),      # previous session's extremes
    "-1 22:00": (100.0, 103.0, 98.0, 100.0),      # Asia of D
    "05:00":    (100.0, 102.0, 97.0, 100.0),      # London of D
    "09:17":    (100.0, 101.5, 99.5, 100.0),      # swing high 101.5 on the 09:15 bin, confirmed with the 09:25 bin (09:29)
    "09:27":    (100.0, 100.5, 99.2, 99.3),       # the drift into the open
    "09:28":    (99.3, 99.4, 98.7, 98.8),
    "09:29":    (98.8, 99.3, 98.4, 98.5),
    "09:30":    (98.5, 98.6, 98.2, 98.4),         # the 09:30 bin: o 98.5 h 98.6 l 96.5 c 98.0
    "09:31":    (98.4, 98.5, 96.5, 97.2),         #   the wick, under ASIA_L 98 and LON_L 97
    "09:32":    (97.2, 98.3, 97.1, 97.8),
    "09:33":    (97.8, 98.3, 97.7, 98.2),
    "09:34":    (98.2, 98.6, 98.1, 98.0),         #   completes the bin: the sweep bar
    "09:35":    (98.0, 98.4, 97.9, 98.3),         # the 09:35 bin closes 99.5: no break yet
    "09:36":    (98.3, 98.7, 98.2, 98.6),
    "09:37":    (98.6, 99.0, 98.5, 98.9),         #   bullish gap [98.4, 98.5]
    "09:38":    (98.9, 99.3, 98.8, 99.2),         #   bullish gap [98.7, 98.8]: the freshest under EQ
    "09:39":    (99.2, 99.6, 99.0, 99.5),
    "09:40":    (99.5, 100.0, 99.3, 99.9),        # the 09:40 bin: o 99.5 h 101.9 l 99.3 c 101.8
    "09:41":    (99.9, 100.6, 99.8, 100.5),       #   gap [99.6, 99.8], premium
    "09:42":    (100.5, 101.2, 100.4, 101.1),     #   gap [100.0, 100.4], premium
    "09:43":    (101.1, 101.7, 101.0, 101.6),     #   the partial bin already closes above 101.5 here
    "09:44":    (101.6, 101.9, 101.4, 101.8),     #   completes the bin: the BOS bar
    "09:45":    (101.7, 101.9, 101.5, 101.7),     # holding up through the rest of the sweep window
    "09:46":    (101.7, 101.9, 101.5, 101.7),
    "09:47":    (101.7, 101.9, 101.5, 101.7),
    "09:48":    (101.7, 101.9, 101.5, 101.7),
    "09:49":    (101.7, 101.9, 101.5, 101.7),
    "09:50":    (101.7, 101.8, 101.0, 101.2),     # the retrace
    "09:51":    (101.2, 101.3, 100.0, 100.3),
    "09:52":    (100.3, 100.4, 98.75, 99.6),      # into the gap: entry 98.8
    "09:53":    (99.6, 100.4, 99.5, 100.3),
    "09:54":    (100.3, 101.0, 100.2, 100.9),
    "09:55":    (100.9, 101.6, 100.8, 101.5),
    "09:56":    (101.5, 102.3, 101.4, 102.2),     # takes LON_H 102
}


def _context_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR(14) of the 5-minute bins, computed here rather than by the module."""
    bins = (df[["open", "high", "low", "close"]]
            .resample("5min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    return pd.Series(atr(bins, length), index=bins.index)


def test_one_minute_day_produces_the_expected_trade():
    df = build_day(FULL_SETUP_1M, freq="1min")
    assert len(df) == 2 * 23 * 60
    res = m.simulate(df, stop_buffer_atr=0.0, entry_tf=1)
    t = res.trades
    assert len(t) == 1, t
    row = t.iloc[0]
    assert row.side == 1
    assert row.entry_time == et("09:52") and row.exit_time == et("09:56")
    assert row.entry == 98.8 and row.stop == 96.5 and row.target == 102.0 and row.exit == 102.0
    assert row.reason == "target" and row.bars == 4
    assert abs(row.r - (102.0 - 98.8) / (98.8 - 96.5)) < 1e-12
    assert row.sweep_level == "LON_L" and row.target_level == "LON_H" and row.zone_kind == "fvg"
    assert row.stop_mode == "wick" and int(row.entry_tf) == 1
    assert pd.Timestamp(row.session_day) == DAY
    held = res.position[res.position != 0]
    assert list(held.index) == [et("09:52"), et("09:53"), et("09:54"), et("09:55")], held
    assert (held == 1).all()
    # sweep and BOS are dated by the minute that completed their 5-minute bin
    d = day_row(m.day_log(df, stop_buffer_atr=0.0, entry_tf=1))
    assert d.outcome == "traded" and d.sweep_time == et("09:34") and d.bos_time == et("09:44")
    assert d.fill_time == et("09:52") and d.zone_kind == "fvg"
    clock = m.session_clock(df.index)
    s, b, e = (df.index.get_loc(x) for x in (d.sweep_time, d.bos_time, row.entry_time))
    assert clock.in_sweep[s] and clock.in_entry[e] and s < b < e
    # the default buffer reads the context ATR at the newest COMPLETE bin, the
    # 09:45 one (done at 09:49) — not the 09:50 bin the entry minute sits in
    ctx = m.build_context(df, 1, 14)
    i = df.index.get_loc(et("09:52"))
    assert ctx.idx[i] == ctx.idx[df.index.get_loc(et("09:50"))] and not ctx.done[i]
    assert ctx.last[i] == ctx.idx[df.index.get_loc(et("09:49"))] == ctx.idx[i] - 1
    a = _context_atr(df)[et("09:45")]
    assert abs(ctx.bars.atr[ctx.last[i]] - a) < 1e-12
    dflt = m.simulate(df, entry_tf=1).trades
    assert len(dflt) == 1 and abs(dflt.iloc[0].stop - (96.5 - 0.25 * a)) < 1e-12
    for mode, stop in {"atr1.0": 98.8 - a, "atr1.5": 98.8 - 1.5 * a, "atr2.0": 98.8 - 2.0 * a,
                       "session": 96.5 - 0.25 * a}.items():
        tm = m.simulate(df, entry_tf=1, stop_mode=mode).trades
        assert len(tm) == 1 and abs(tm.iloc[0].stop - stop) < 1e-12, mode
    # SMT on 1-minute bars reads the pair's 5-minute bin: the same frame as pair
    # swept too; a pair that held 97 on every minute of the 09:30 bin trades.
    both = _with_pair(df, df)
    assert m.simulate(both, stop_buffer_atr=0.0, entry_tf=1, smt=True).trades.empty
    assert day_row(m.day_log(both, stop_buffer_atr=0.0, entry_tf=1, smt=True)).outcome == "smt"
    pair = df.copy()
    pair.loc[et("09:31"), "low"] = 97.5
    tp = m.simulate(_with_pair(df, pair), stop_buffer_atr=0.0, entry_tf=1, smt=True).trades
    assert len(tp) == 1 and bool(tp.iloc[0].smt) is True and tp.iloc[0].entry == 98.8


def test_one_minute_truncation_inside_a_bin_never_reads_the_partial_bin():
    """Cut the day at minutes that are not 5-minute boundaries. The position over
    the overlap must not move, and the log must not know anything the completed
    bins did not say: at 09:31 the partial 09:30 bin already shows the wick, at
    09:43 the partial 09:40 bin already closes above the swing — neither counts."""
    df = build_day(FULL_SETUP_1M, freq="1min")
    kw = dict(stop_buffer_atr=0.0, entry_tf=1)
    full = m.simulate(df, **kw)
    full_log = day_row(m.day_log(df, **kw))
    expect = {
        "09:31": ("no_sweep", pd.NaT, pd.NaT),
        "09:37": ("no_bos", et("09:34"), pd.NaT),
        "09:43": ("no_bos", et("09:34"), pd.NaT),
        "09:52": ("traded", et("09:34"), et("09:44")),
        "09:58": ("traded", et("09:34"), et("09:44")),
    }
    for cut, (outcome, sweep, bos) in expect.items():
        k = df.index.get_loc(et(cut)) + 1
        assert df.index[k - 1].minute % 5 != 4, cut                   # a mid-bin cut, by construction
        part = m.simulate(df.iloc[:k], **kw)
        assert float((full.position.iloc[:k] - part.position).abs().max()) == 0.0, cut
        d = day_row(m.day_log(df.iloc[:k], **kw))
        assert d.outcome == outcome, (cut, d)
        assert (d.sweep_time is pd.NaT and sweep is pd.NaT) or d.sweep_time == sweep, (cut, d)
        assert (d.bos_time is pd.NaT and bos is pd.NaT) or d.bos_time == bos, (cut, d)
        for col in ("sweep_time", "bos_time", "fill_time", "sweep_level", "zone_kind"):
            v = d[col]
            if v is not None and v is not pd.NaT:
                assert v == full_log[col], (cut, col, v, full_log[col])
        if cut == "09:52":
            assert part.trades.empty and part.position.iloc[-1] == 1   # opened on the last bar
        elif cut == "09:58":
            assert len(part.trades) == 1 and part.trades.iloc[0].reason == "target"


def test_entry_tf_1_is_flat_with_no_0930_bar_and_on_five_minute_bars():
    for frame in (synthetic(n=400), _daily_frame()):
        s = m.signal(frame, entry_tf=1)
        assert isinstance(s, pd.Series) and len(s) == len(frame) and (s == 0).all()
        res = m.simulate(frame, entry_tf=1)
        assert res.trades.empty and list(res.trades.columns) == m._TRADE_COLS
        assert m.day_log(frame, entry_tf=1).empty
    # a 5-minute frame has no :04 minute, so no context bar ever completes
    five = build_day(FULL_SETUP)
    res = m.simulate(five, stop_buffer_atr=0.0, entry_tf=1)
    assert res.trades.empty and (res.position == 0).all()
    assert set(m.day_log(five, stop_buffer_atr=0.0, entry_tf=1).outcome) == {"no_sweep"}


# ────────────────────────────── runner ──────────────────────────────

def _main() -> int:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

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


def build_day(overrides: dict | None = None, night=INERT, day=INERT, extra_days: int = 0) -> pd.DataFrame:
    """Two full CME sessions, D-1 and D, on 5-minute bars, plus `extra_days` inert
    sessions after D. Keys of `overrides` are 'HH:MM' on D or '-1 HH:MM' on the
    calendar day before, values are (o, h, l, c). `night` is the inert bar for
    everything stamped before 08:30 ET on D (session D-1, Asia, London), `day`
    for the rest."""
    start = pd.Timestamp(f"{(DAY - pd.Timedelta(days=2)).date()} 18:00", tz=ET)
    end = pd.Timestamp(f"{(DAY + pd.Timedelta(days=extra_days)).date()} 16:55", tz=ET)
    ix = pd.date_range(start, end, freq="5min", tz=ET)
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

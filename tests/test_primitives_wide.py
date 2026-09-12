"""Unit tests for the wider-spec primitives (DESIGN-tjr-human.md §2):
resample_context, volume_profile, poc, hvn, order_block, breaker_block.

Runs standalone (`python tests/test_primitives_wide.py`) and under pytest. The
hand-built sessions are 5-minute bars laid out on the New York clock and
converted to the tz-naive UTC index the codebase uses, so the timezone path is
exercised, not mocked around.
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

from quantlab.strategies.predictive import primitives as P  # noqa: E402

ET = "America/New_York"
DAY = pd.Timestamp("2026-07-15")            # a Wednesday, EDT: 09:30 ET is 13:30 UTC
NQ_CSV = os.path.join(ROOT, "data", "NQ_5min_tv.csv")


def et(hhmm: str, day_offset: int = 0) -> pd.Timestamp:
    """An ET wall-clock time on day D (+offset) as the naive-UTC stamp the frame uses."""
    ts = pd.Timestamp(f"{(DAY + pd.Timedelta(days=day_offset)).date()} {hhmm}", tz=ET)
    return ts.tz_convert("UTC").tz_localize(None)


def build_sessions(freq: str = "5min") -> pd.DataFrame:
    """Two full CME sessions, D-1 and D, on `freq` bars, every bar distinct: bar
    number i has open i, high i+0.75, low i-0.75, close i+0.25, volume i+1, so
    any aggregation can be checked by hand."""
    start = pd.Timestamp(f"{(DAY - pd.Timedelta(days=2)).date()} 18:00", tz=ET)
    end = pd.Timestamp(f"{DAY.date()} 17:00", tz=ET) - pd.Timedelta(freq)
    ix = pd.date_range(start, end, freq=freq, tz=ET)
    ix = ix[ix.hour != 17]                                # the daily maintenance hour
    i = np.arange(len(ix), dtype=float)
    df = pd.DataFrame({"open": i, "high": i + 0.75, "low": i - 0.75, "close": i + 0.25,
                       "volume": i + 1}, index=ix)
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "date"
    return df


def frame(rows: list[tuple]) -> pd.DataFrame:
    """Bars from (o, h, l, c) tuples on an arbitrary 5-minute index."""
    ix = pd.date_range("2026-07-15 13:30", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=ix)


# ────────────────────────────── volume at price ──────────────────────────────

def test_volume_profile_spreads_each_bar_over_the_bins_it_covers():
    # $1 bins (tick 0.25 x 4). A: 100-102 / 20 -> 10 + 10. B: 101.5-102.5 / 10 -> 5 + 5.
    # C: a doji at 103 / 7 -> all of it in [103, 104).
    prof = P.volume_profile(high=[102.0, 102.5, 103.0], low=[100.0, 101.5, 103.0],
                            volume=[20.0, 10.0, 7.0], tick=0.25, bin_ticks=4)
    assert list(prof.edges) == [100.0, 101.0, 102.0, 103.0, 104.0], prof.edges
    assert np.allclose(prof.volume, [10.0, 15.0, 5.0, 7.0]), prof.volume
    assert np.allclose(prof.centers, [100.5, 101.5, 102.5, 103.5])
    assert prof.width == 1.0 and abs(prof.volume.sum() - 37.0) < 1e-12
    # the same bars on a shared grid land in the same bins, and a composite is a sum
    grid = 98.0 + np.arange(9.0)
    on_grid = P.volume_profile([102.0, 102.5, 103.0], [100.0, 101.5, 103.0], [20.0, 10.0, 7.0], edges=grid)
    assert np.allclose(on_grid.volume, [0, 0, 10, 15, 5, 7, 0, 0])
    twice = on_grid.volume + P.volume_profile([102.0], [100.0], [20.0], edges=grid).volume
    assert np.allclose(twice, [0, 0, 20, 25, 5, 7, 0, 0])
    # a bar whose range does not sit on the grid is split by overlap, not by bin count
    prof = P.volume_profile([101.5], [100.0], [30.0], tick=0.25, bin_ticks=4)
    assert np.allclose(prof.volume, [20.0, 10.0]), prof.volume
    # bars with no volume or NaNs are skipped; an empty set is an empty profile
    prof = P.volume_profile([102.0, np.nan], [100.0, np.nan], [0.0, 5.0])
    assert prof.volume.sum() == 0.0
    assert P.volume_profile([], [], []).volume.sum() == 0.0


def test_poc_is_the_max_bin_and_ties_break_toward_vwap():
    prof = P.Profile(edges=np.array([100.0, 101, 102, 103, 104]), volume=np.array([10.0, 15, 5, 7]))
    assert P.poc(prof) == 101.5
    tied = P.Profile(edges=np.array([100.0, 101, 102, 103, 104]), volume=np.array([10.0, 5, 5, 10]))
    assert P.poc(tied, near=103.2) == 103.5
    assert P.poc(tied, near=100.9) == 100.5
    assert P.poc(tied) == 100.5                       # nothing to break the tie with: lowest
    assert P.poc(tied, near=float("nan")) == 100.5
    # vwap is the typical-price weight the funnel passes as `near`
    v = P.vwap(high=[101.0, 103.0], low=[99.0, 101.0], close=[100.0, 102.0], volume=[1.0, 3.0])
    assert abs(v - (100.0 * 1 + 102.0 * 3) / 4) < 1e-12
    assert np.isnan(P.vwap([1.0], [1.0], [1.0], [0.0]))


def test_hvn_returns_both_peaks_and_drops_the_bump_under_threshold():
    vols = np.zeros(30)
    vols[3:8] = [10, 20, 30, 20, 10]         # peak at bin 5, smoothed 18 (the max)
    vols[10:15] = [1, 2, 4, 2, 1]            # bump at bin 12, smoothed 2.0: under 0.5 x 18
    vols[18:23] = [10, 20, 28, 20, 10]       # peak at bin 20, smoothed 17.6
    prof = P.Profile(edges=100.0 + np.arange(31.0), volume=vols)
    assert P.hvn(prof) == [105.5, 120.5]                          # by volume when nothing is near
    assert P.hvn(prof, near=118.0) == [120.5, 105.5]              # nearest first
    assert P.hvn(prof, near=118.0, max_n=1) == [120.5]
    assert P.hvn(prof, frac=0.99) == [105.5]                      # the second peak is 17.6 / 18
    assert P.hvn(prof, frac=0.05) == [105.5, 120.5, 112.5]        # the bump is a real local max
    # a flat plateau is one node, at its top (>= below, > above)
    flat = P.Profile(edges=100.0 + np.arange(8.0), volume=np.array([0, 0, 9, 9, 9, 0, 0, 0.0]))
    assert P.hvn(flat, smooth=1) == [104.5]
    assert P.hvn(P.Profile(edges=np.array([0.0, 1.0]), volume=np.array([0.0]))) == []


# ────────────────────────────── blocks ──────────────────────────────

def test_order_block_is_the_last_opposing_candle_at_or_before_the_sweep():
    bars = frame([
        (100.0, 100.8, 99.6, 100.5),      # 0 bullish
        (100.5, 100.9, 99.0, 99.4),       # 1 bearish  <- the long OB when the sweep is bullish
        (99.4, 100.2, 99.2, 100.1),       # 2 bullish
        (100.1, 100.3, 97.5, 100.0),      # 3 bearish sweep bar (close < open)
        (100.0, 101.0, 99.9, 100.9),      # 4
    ])
    ob = P.order_block(bars, sweep_idx=2, side=1)
    assert ob == P.Block(1, 99.0, 100.9, 1), ob
    # a bearish sweep bar is itself the last bearish candle
    assert P.order_block(bars, sweep_idx=3, side=1) == P.Block(3, 97.5, 100.3, 3)
    # shorts mirror on bullish candles; `start` bounds the search
    assert P.order_block(bars, sweep_idx=3, side=-1) == P.Block(2, 99.2, 100.2, 2)
    assert P.order_block(bars, sweep_idx=1, side=-1, start=1) is None
    assert P.order_block(bars, sweep_idx=0, side=1) is None


def _breaker_day() -> pd.DataFrame:
    return frame([
        (100.0, 100.6, 99.5, 100.4),      # 0 bullish
        (100.4, 101.0, 100.0, 100.9),     # 1 bullish  <- the breaker candle [100.0, 101.0]
        (100.9, 101.3, 100.7, 100.8),     # 2 bearish
        (100.8, 103.0, 100.6, 102.5),     # 3 the swing high bar (103.0), needs bars 4-5 to confirm
        (102.5, 102.6, 101.0, 101.2),     # 4
        (101.2, 101.4, 99.5, 99.8),       # 5 confirms the swing (right=2)
        (99.8, 99.9, 97.0, 99.6),         # 6 the sweep bar
        (99.6, 100.9, 99.4, 100.5),       # 7 close 100.5 <= 101.0: not yet violated
        (100.5, 101.8, 100.4, 101.5),     # 8 close 101.5 > 101.0: the breaker is born here
    ])


def test_breaker_needs_the_violation_close_and_an_admissible_swing():
    df = _breaker_day()
    highs, lows = P.confirmed_swings(df, 2, 2)
    assert [(s.index, s.confirmed_at) for s in highs] == [(3, 5)], highs
    # through bar 7 nothing has closed above the candle: no breaker
    assert P.breaker_block(df, 0, 6, 7, 1, highs) is None
    # bar 8 closes above 101.0: the zone is the candle, usable from bar 8
    assert P.breaker_block(df, 0, 6, 8, 1, highs) == P.Block(8, 100.0, 101.0, 1)
    # the guard: a swing not yet confirmed at i cannot supply a breaker
    early = [P.Swing(3, 103.0, 9)]
    assert P.breaker_block(df, 0, 6, 8, 1, early) is None
    # the swing has to be on the day and before the sweep
    assert P.breaker_block(df, 4, 6, 8, 1, highs) is None
    assert P.breaker_block(df, 0, 3, 8, 1, highs) is None
    # shorts mirror: no swing lows confirm on this frame, so nothing
    assert P.breaker_block(df, 0, 6, 8, -1, lows) is None


def test_breaker_short_side_mirrors():
    # Mirror of the long day around 100.
    df = _breaker_day()
    m = df.copy()
    m["open"], m["close"] = 200.0 - df["open"], 200.0 - df["close"]
    m["high"], m["low"] = 200.0 - df["low"], 200.0 - df["high"]
    highs, lows = P.confirmed_swings(m, 2, 2)
    assert [(s.index, s.confirmed_at) for s in lows] == [(3, 5)]
    assert P.breaker_block(m, 0, 6, 7, -1, lows) is None
    assert P.breaker_block(m, 0, 6, 8, -1, lows) == P.Block(8, 99.0, 100.0, 1)


# ────────────────────────────── the resampled context ──────────────────────────────

def _bin_at(ctx: P.Context, df: pd.DataFrame, stamp: pd.Timestamp) -> int:
    return int(ctx.idx[df.index.get_loc(stamp)])


def test_one_hour_bins_complete_only_at_the_55_bar():
    df = build_sessions()
    ctx = P.resample_context(df, 60)
    labels = ctx.frame.index
    # bins are labelled by their nominal open on the ET clock, 18:00 first
    assert labels[0] == et("18:00", -2) and labels[1] == et("19:00", -2)
    b = _bin_at(ctx, df, et("09:30"))
    assert labels[b] == et("09:00")
    for mm in ("00", "05", "30", "50"):
        k = df.index.get_loc(et(f"09:{mm}"))
        assert ctx.idx[k] == b and not ctx.done[k], mm
        assert ctx.last[k] == b - 1                       # may read the 08:00 bin, not this one
    k = df.index.get_loc(et("09:55"))
    assert ctx.done[k] and ctx.last[k] == b
    assert ctx.first[b] == df.index.get_loc(et("09:00"))
    # the aggregation is first/max/min/last/sum over the twelve bars
    lo, hi = ctx.first[b], ctx.first[b] + 12
    src = df.iloc[lo:hi]
    row = ctx.frame.iloc[b]
    assert row.open == src.open.iloc[0] and row.close == src.close.iloc[-1]
    assert row.high == src.high.max() and row.low == src.low.min() and row.volume == src.volume.sum()
    assert ctx.bars.n == len(ctx.frame) == 2 * 23           # 23 hourly bins a session, none empty
    assert ctx.done.sum() == 2 * 23
    # the last bin of a session ends at the close: 16:00 completes at 16:55
    k = df.index.get_loc(et("16:55"))
    assert ctx.done[k] and labels[ctx.idx[k]] == et("16:00")


def test_four_hour_bins_complete_at_the_right_bar_and_the_last_is_clipped():
    df = build_sessions()
    ctx = P.resample_context(df, 240)
    labels = ctx.frame.index
    stamps = [et("18:00", -1), et("22:00", -1), et("02:00"), et("06:00"), et("10:00"), et("14:00")]
    day_labels = [labels[_bin_at(ctx, df, t)] for t in stamps]
    assert day_labels == [et("18:00", -1), et("22:00", -1), et("02:00"), et("06:00"), et("10:00"), et("14:00")], day_labels
    assert ctx.bars.n == 2 * 6 and ctx.done.sum() == 2 * 6
    done_at = [df.index[k] for k in np.flatnonzero(ctx.done)]
    assert done_at[-6:] == [et("21:55", -1), et("01:55"), et("05:55"), et("09:55"), et("13:55"), et("16:55")], done_at[-6:]
    # 09:30 sits in the 06:00 bin and may only read the 02:00 bin
    k = df.index.get_loc(et("09:30"))
    assert labels[ctx.idx[k]] == et("06:00") and labels[ctx.last[k]] == et("02:00")
    # the clipped 14:00 bin holds 14:00 .. 16:55, thirty-six bars
    b = ctx.idx[df.index.get_loc(et("16:55"))]
    assert ctx.first[b] == df.index.get_loc(et("14:00"))
    assert ctx.frame.iloc[b].volume == df.loc[et("14:00"):et("16:55")].volume.sum()


def test_truncation_mid_bin_changes_no_completed_bin():
    df = build_sessions()
    for minutes in (60, 240):
        full = P.resample_context(df, minutes)
        for cut in ("09:30", "09:55", "10:00", "13:50", "16:55"):
            k = df.index.get_loc(et(cut)) + 1
            part = P.resample_context(df.iloc[:k], minutes)
            n = full.last[k - 1] + 1                       # completed bins as of the cut
            assert part.last[-1] + 1 == n, (minutes, cut)
            assert float((part.frame.iloc[:n] - full.frame.iloc[:n]).abs().to_numpy().max()) == 0.0, (minutes, cut)
            assert (part.idx == full.idx[:k]).all() and (part.done == full.done[:k]).all()
            assert (part.last == full.last[:k]).all()
            # the partial bin, when there is one, is a row in `frame` but not a readable bar
            assert len(part.frame) == n + (0 if full.done[k - 1] else 1), (minutes, cut)


def test_swings_on_the_resample_are_admitted_by_confirmed_at_on_completed_bins():
    # The rule the funnel gates on: a resampled swing is usable at input bar i
    # only if ctx.last[i] >= confirmed_at. Truncating inside the confirming bin
    # must not change the set of admissible swings.
    df = build_sessions()
    ctx = P.resample_context(df, 60)
    highs, lows = P.confirmed_swings(ctx.bars, 2, 2)
    # the hand-built series is monotone, so there are no pivots at all — the
    # point is the gate itself, on a bin that is still printing
    assert highs == [] and lows == []
    k = df.index.get_loc(et("09:30"))
    assert ctx.last[k] < ctx.idx[k]


def _skip(msg):
    print(f"SKIP  {msg}")


def test_causality_of_the_resample_on_the_real_nq_frame():
    if not os.path.exists(NQ_CSV):
        return _skip("no NQ TradingView file")
    from quantlab.data import load_csv
    df = load_csv(NQ_CSV)
    rng = np.random.default_rng(7)
    cuts = sorted(rng.integers(len(df) // 2, len(df), 6).tolist())
    for minutes in (60, 240):
        full = P.resample_context(df, minutes)
        # every completing bar is the last slot of its bin, by minute arithmetic
        assert (full.last <= full.idx).all()
        for k in cuts:
            part = P.resample_context(df.iloc[:k], minutes)
            n = full.last[k - 1] + 1
            assert part.last[-1] + 1 == n
            assert float((part.frame.iloc[:n] - full.frame.iloc[:n]).abs().to_numpy().max()) == 0.0, (minutes, k)
            ph, pl = P.confirmed_swings(part.bars, 2, 2)
            fh, fl = P.confirmed_swings(full.bars, 2, 2)
            admit = lambda sw: [s for s in sw if s.confirmed_at < n]  # noqa: E731
            assert admit(ph) == admit(fh) and admit(pl) == admit(fl), (minutes, k)


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

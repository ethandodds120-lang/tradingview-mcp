#!/usr/bin/env python3
"""The search cost of the whole TJR-intraday test, not one run of it.

run.py deflates each run's best Sharpe against that run's own grid. But the test
tried two timeframes, two instruments and two variants over two rounds, so the
number of things that got a chance to look good is 192 (DESIGN-tjr-intraday.md
§11.5), and the honest question is whether the best of *those* clears what 192
tries would produce on noise. This pools every trial at real costs and asks that
question once.

The pool, per §11.5:

    round 1   the 5-minute model, stop_mode=wick, over round 1's grid
              (stop_buffer_atr x min_fvg_atr x allow_ifvg = 8) on the four
              (instrument, variant) combos                              =  32
    round 2   the §11.5 grid (stop_mode x min_fvg_atr x allow_ifvg = 20) on
              the eight (timeframe, instrument, variant) combos          = 160

Round 1's 32 are recomputed here rather than read from a file, on a copy of the
registry entry carrying the old grid; eight of them recur in round 2 and are
counted twice, as the contract says. If a 1-minute file is missing, whatever
exists is run, the skipped combos are named and the DSR is labelled partial.

PBO is run across the pooled matrix too: the combinatorial cross-validation does
not care whether a column is a parameter setting, a variant or an instrument —
it asks whether picking the in-sample winner out of everything that was tried
would have chosen something that held up. It does need every column on one
timeline, and the 1-minute frames are a different timeline from the 5-minute
ones (a different span, at a different grain), so there is one pooled PBO per
timeframe rather than one across both.

Session counts are printed before any result, because §11.2 says a 1-minute
frame under ~100 sessions is run but is not evidence either way.

Read-only. Prints; writes nothing.

    python tjr_intraday_search.py
    python tjr_intraday_search.py data/ES_5min_60d.csv data/NQ_5min_60d.csv
    python tjr_intraday_search.py --es-1m data/ES_1min_tv.csv --nq-1m data/NQ_1min_tv.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd

from quantlab import data, engine, validate
from quantlab.strategies import REGISTRY
from quantlab.strategies.predictive import tjr_intraday

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BT = dict(vol_target=0.15)

#: Round 1's grid (§6). stop_buffer_atr left the grid in round 2, so these
#: trials are only reproducible on a copy of the entry carrying it.
ROUND1_GRID = {"stop_buffer_atr": [0.0, 0.25],
               "min_fvg_atr": [0.0, 0.25],
               "allow_ifvg": [True, False]}

#: The cumulative count the contract fixes (§11.5). The DSR is reported against
#: whatever was actually pooled, and the two are compared so a partial run says so.
NOMINAL_TRIALS = 32 + 160

#: Below this many sessions with a 09:30 bar the frame is labelled (§11.2).
MIN_SESSIONS = 100

#: Real round-trip cost, bps, per instrument (§8).
REAL_BPS = {"NQ": 0.5, "ES": 1.0}

#: The parameters every trial is listed under, whichever grid it came from.
SHOWN = ("stop_mode", "stop_buffer_atr", "min_fvg_atr", "allow_ifvg")


def round1_strategy(name: str):
    """The 5-minute entry as round 1 searched it: same params, old grid.

    The params already carry stop_mode='wick', entry_tf=5, which is the round-1
    model bar for bar (§11.7), so only the grid has to be put back.
    """
    return dataclasses.replace(REGISTRY[name], grid=dict(ROUND1_GRID))


def runs(es5: str, nq5: str, es1: str, nq1: str) -> list:
    # round, label, timeframe, strategy, traded csv, pair csv, real cost in bps
    out = []
    for inst, csv, pair in (("NQ", nq5, es5), ("ES", es5, nq5)):
        for variant, name in (("base", "tjr_intraday"), ("smt", "tjr_intraday_smt")):
            out.append((1, f"{inst} {variant} 5m", 5, round1_strategy(name),
                        csv, pair, REAL_BPS[inst]))
    for tf, es, nq, suffix in ((5, es5, nq5, ""), (1, es1, nq1, "_1m")):
        for inst, csv, pair in (("NQ", nq, es), ("ES", es, nq)):
            for variant, name in (("base", f"tjr_intraday{suffix}"),
                                  ("smt", f"tjr_intraday{suffix}_smt")):
                out.append((2, f"{inst} {variant} {tf}m", tf, REGISTRY[name],
                            csv, pair, REAL_BPS[inst]))
    return out


def sessions_with_open(index: pd.Index) -> int:
    """Trading days on which a 09:30 ET bar exists — the unit the test is sized in."""
    clock = tjr_intraday.session_clock(index)
    if clock is None:
        return 0
    return int(len(np.unique(clock.day[clock.minutes == 9 * 60 + 30])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # default to the TradingView dumps (~7 months); pass the two Yahoo files to
    # rerun the 5-minute half on the 60-day window
    ap.add_argument("es_csv", nargs="?", default="data/ES_5min_tv.csv")
    ap.add_argument("nq_csv", nargs="?", default="data/NQ_5min_tv.csv")
    ap.add_argument("--es-1m", default="data/ES_1min_tv.csv", metavar="PATH")
    ap.add_argument("--nq-1m", default="data/NQ_1min_tv.csv", metavar="PATH")
    args = ap.parse_args()
    RUNS = runs(args.es_csv, args.nq_csv, args.es_1m, args.nq_1m)
    print(f"  5-minute: {args.es_csv}  +  {args.nq_csv}")
    print(f"  1-minute: {args.es_1m}  +  {args.nq_1m}")

    # ── data first: the session count is reported before anything is read (§11.2) ──
    frames, skipped = {}, set()
    for _, label, tf, _, csv, pair, _ in RUNS:
        if (csv, pair) in frames or (csv, pair) in skipped:
            continue
        missing = [p for p in (csv, pair) if not os.path.exists(p)]
        if missing:
            skipped.add((csv, pair))
            print(f"  ! {tf}m {csv}: missing {', '.join(missing)} — its combos are skipped")
            continue
        df = data.load_futures_pair(csv, pair)
        n_sess = sessions_with_open(df.index)
        flag = "" if n_sess >= MIN_SESSIONS else f"   <- under {MIN_SESSIONS}: not evidence either way"
        print(f"  {tf}m {csv:<24} {len(df):>6} bars  {df.index[0].date()} → {df.index[-1].date()}"
              f"  {n_sess:>3} sessions{flag}")
        frames[(csv, pair)] = df

    # ── every trial ──
    trials, pools, run_skipped = [], {}, []
    print()
    for rnd, label, tf, strat, csv, pair, bps in RUNS:
        if (csv, pair) not in frames:
            run_skipped.append((rnd, label))
            continue
        df = frames[(csv, pair)]
        costs = engine.CostModel(commission_bps=bps / 2, slippage_bps=bps / 2)
        rows, matrix, ppy = validate.trial_matrix(df, strat, costs, **BT)
        if rows.empty:
            print(f"  round {rnd}  {label}: no grid rows"); continue
        # one PBO pool per timeframe; the timeline has to match within it
        pool = pools.setdefault(tf, {"index": df.index, "cols": [], "ok": True})
        if not pool["index"].equals(df.index):
            # the pair join makes both instruments the same length; if not, PBO
            # across them is meaningless and we say so rather than silently align
            print(f"  ! {label}: index differs from the first {tf}m run — that PBO pool is skipped")
            pool["ok"] = False
        for j, r in rows.iterrows():
            params = {**strat.params, **{k: r[k] for k in strat.grid}}
            trials.append({"round": rnd, "run": label, "tf": tf, "strategy": strat.name,
                           "bps": bps, "sharpe": float(r["sharpe"]),
                           "max_dd": float(r["max_dd"]), "trades": bool(matrix[:, j].any()),
                           "col": matrix[:, j], "ppy": ppy, "df": df,
                           "grid": {k: params[k] for k in strat.grid},
                           **{k: params[k] for k in SHOWN}})
        pool["cols"].append(matrix)
        print(f"  round {rnd}  {label:<12} N={len(rows):>2}  best {rows['sharpe'].max():+.2f}  "
              f"median {rows['sharpe'].median():+.2f}  worst {rows['sharpe'].min():+.2f}")

    if not trials:
        print("\n  nothing ran"); return 1
    tbl = pd.DataFrame([{k: v for k, v in t.items() if k not in ("col", "df", "grid")}
                        for t in trials])
    n_total = len(tbl)
    partial = n_total != NOMINAL_TRIALS
    best_i = int(tbl["sharpe"].values.argmax())
    best = trials[best_i]
    n_dead = int((~tbl["trades"]).sum())

    print(f"\n  everything tried: {n_total} trials across {len(RUNS) - len(run_skipped)} runs"
          + (f"  — PARTIAL, the contract's count is {NOMINAL_TRIALS}" if partial
             else "  (= the contract's 192)"))
    for rnd, label in run_skipped:
        print(f"    skipped: round {rnd}  {label}")
    print(f"  trials that never trade: {n_dead} of {n_total}"
          + ("   (a Sharpe of exactly 0 is one of these)" if n_dead else ""))
    print(f"  best of all: round {best['round']}  {best['run']}  sharpe {best['sharpe']:+.2f}  "
          + "  ".join(f"{k}={v}" for k, v in best["grid"].items())
          + ("" if best["trades"] else "   <- never trades"))
    sim = tjr_intraday.simulate(best["df"], **{**REGISTRY[best["strategy"]].params, **best["grid"]})
    t = sim.trades
    if len(t):
        print(f"    that trial: {len(t)} trades, {int((t['r'] > 0).sum())} wins, "
              f"{t['r'].sum():+.2f}R total")
    else:
        print("    that trial: no trades")

    all_sharpes = tbl["sharpe"].values
    best_returns = pd.Series(best["col"], index=best["df"].index)
    own = tbl[(tbl["run"] == best["run"]) & (tbl["round"] == best["round"])]["sharpe"].values
    ds_own = validate.deflated_sharpe(best_returns, best["ppy"], own)
    ds_all = validate.deflated_sharpe(best_returns, best["ppy"], all_sharpes, n_trials=n_total)

    def fmt(ds: dict) -> str:
        return (f"expected max from noise {ds['sr0']:+.2f}   DSR "
                + ("undefined" if np.isnan(ds["dsr"]) else f"{ds['dsr']:.3f}"))

    print("\n  DEFLATED SHARPE of the best trial")
    print(f"    against its own grid  (N={ds_own['trials']:>3}):  {fmt(ds_own)}")
    print(f"    against everything    (N={ds_all['trials']:>3}):  {fmt(ds_all)}   <- the honest one"
          + ("  [PARTIAL]" if partial else ""))
    if np.isnan(ds_all["dsr"]):
        print("    gate: DSR >= 0.95  ->  FAIL (undefined: the best trial has no return "
              "dispersion, so nothing beat noise)")
    else:
        print(f"    gate: DSR >= 0.95  ->  {'PASS' if ds_all['dsr'] >= 0.95 else 'FAIL'}")

    print("\n  PBO, pooled per timeline (the two timelines cannot share a matrix)")
    for tf, pool in sorted(pools.items(), reverse=True):
        if not pool["ok"] or not pool["cols"]:
            print(f"    {tf}m: skipped"); continue
        pooled = np.column_stack(pool["cols"])
        pb = validate.probability_of_backtest_overfitting(pooled)
        if np.isnan(pb["pbo"]):
            print(f"    {tf}m: {pooled.shape[1]} trials — undefined (too few bars, or no "
                  f"finite in-sample Sharpe on any split)"); continue
        print(f"    {tf}m: {pb['pbo']:.0%} across {pooled.shape[1]} trials over {pb['splits']} splits"
              f"  (gate <= 60%: {'PASS' if pb['pbo'] <= 0.6 else 'FAIL'})")
    print("    read loosely — sd ~0.19 on pure noise (rule 8)")

    print("\n  every trial, sorted:")
    show = tbl.sort_values(["sharpe", "round", "run"], ascending=[False, True, True])
    cols = ["round", "run", *SHOWN, "trades", "sharpe", "max_dd"]
    print("   " + show[cols].to_string(index=False,
          float_format=lambda v: f"{v:.3f}").replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    sys.exit(main())

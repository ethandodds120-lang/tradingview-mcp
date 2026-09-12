#!/usr/bin/env python3
"""The search cost of the whole TJR-intraday test, not one run of it.

run.py deflates each run's best Sharpe against that run's own grid (N=8). But the
test tried two instruments and two variants, so the number of things that got a
chance to look good is 32, and the honest question is whether the best of *those*
clears what 32 tries would produce on noise. This pools every trial across the
four real-cost runs and asks that question once.

Also runs PBO across the pooled 32-column matrix: the combinatorial cross-
validation does not care whether a column is a parameter setting, a variant or
an instrument — it asks whether picking the in-sample winner out of everything
that was tried would have chosen something that held up.

Read-only. Prints; writes nothing.

    python tjr_intraday_search.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantlab import data, engine, metrics, validate
from quantlab.strategies import REGISTRY

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BT = dict(vol_target=0.15)


def runs(es_csv: str, nq_csv: str) -> list:
    # label, strategy, traded csv, pair csv, real round-trip cost in bps
    return [
        ("NQ base", "tjr_intraday",     nq_csv, es_csv, 0.5),
        ("NQ smt",  "tjr_intraday_smt", nq_csv, es_csv, 0.5),
        ("ES base", "tjr_intraday",     es_csv, nq_csv, 1.0),
        ("ES smt",  "tjr_intraday_smt", es_csv, nq_csv, 1.0),
    ]


def main() -> int:
    # default to the TradingView dumps (~7 months); pass the two Yahoo files to
    # rerun on the 60-day window:  python tjr_intraday_search.py data/ES_5min_60d.csv data/NQ_5min_60d.csv
    es_csv = sys.argv[1] if len(sys.argv) > 1 else "data/ES_5min_tv.csv"
    nq_csv = sys.argv[2] if len(sys.argv) > 2 else "data/NQ_5min_tv.csv"
    RUNS = runs(es_csv, nq_csv)
    print(f"  data: {es_csv}  +  {nq_csv}")
    trials, cols, index = [], [], None
    for label, name, csv, pair, bps in RUNS:
        df = data.load_futures_pair(csv, pair)
        strat = REGISTRY[name]
        costs = engine.CostModel(commission_bps=bps / 2, slippage_bps=bps / 2)
        rows, matrix, ppy = validate.trial_matrix(df, strat, costs, **BT)
        if rows.empty:
            print(f"  {label}: no grid rows"); continue
        if index is None:
            index = df.index
        elif not index.equals(df.index):
            # the pair join makes both frames the same length; if not, PBO across
            # instruments is meaningless and we say so rather than silently align
            print(f"  ! {label}: index differs from the first run — pooled PBO skipped")
            index = False
        for j, r in rows.iterrows():
            params = {k: r[k] for k in strat.grid}
            trials.append({"run": label, "strategy": name, "bps": bps, **params,
                           "sharpe": float(r["sharpe"]), "max_dd": float(r["max_dd"]),
                           "col": matrix[:, j], "ppy": ppy})
        cols.append(matrix)
        print(f"  {label:<8} N={len(rows)}  best {rows['sharpe'].max():+.2f}  "
              f"median {rows['sharpe'].median():+.2f}  worst {rows['sharpe'].min():+.2f}")

    tbl = pd.DataFrame([{k: v for k, v in t.items() if k != "col"} for t in trials])
    n_total = len(tbl)
    best_i = int(tbl["sharpe"].values.argmax())
    best = trials[best_i]
    print(f"\n  everything tried: {n_total} trials across {len(RUNS)} runs")
    print(f"  best of all: {best['run']}  sharpe {best['sharpe']:+.2f}  "
          + "  ".join(f"{k}={best[k]}" for k in REGISTRY[best['strategy']].grid))

    all_sharpes = tbl["sharpe"].values
    best_returns = pd.Series(best["col"], index=index if index is not False else None)
    ds_own = validate.deflated_sharpe(best_returns, best["ppy"],
                                      tbl[tbl["run"] == best["run"]]["sharpe"].values)
    ds_all = validate.deflated_sharpe(best_returns, best["ppy"], all_sharpes, n_trials=n_total)

    print("\n  DEFLATED SHARPE of the best trial")
    print(f"    against its own grid  (N={ds_own['trials']:>2}):  expected max from noise "
          f"{ds_own['sr0']:+.2f}   DSR {ds_own['dsr']:.3f}")
    print(f"    against everything    (N={ds_all['trials']:>2}):  expected max from noise "
          f"{ds_all['sr0']:+.2f}   DSR {ds_all['dsr']:.3f}   <- the honest one")
    print(f"    gate: DSR >= 0.95  ->  {'PASS' if ds_all['dsr'] >= 0.95 else 'FAIL'}")

    if index is not False and cols:
        pooled = np.column_stack(cols)
        pb = validate.probability_of_backtest_overfitting(pooled)
        print(f"\n  PBO across all {pooled.shape[1]} trials pooled: {pb['pbo']:.0%} "
              f"over {pb['splits']} splits  (gate <= 60%: {'PASS' if pb['pbo'] <= 0.6 else 'FAIL'})")

    print("\n  every trial, sorted:")
    show = tbl.sort_values("sharpe", ascending=False)
    grid_cols = [c for c in show.columns if c not in ("strategy", "bps", "sharpe", "max_dd", "ppy")]
    print("   " + show[grid_cols + ["sharpe", "max_dd"]].to_string(index=False,
          float_format=lambda v: f"{v:.3f}").replace("\n", "\n   "))
    return 0


if __name__ == "__main__":
    sys.exit(main())

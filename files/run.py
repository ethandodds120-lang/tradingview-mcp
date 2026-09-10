#!/usr/bin/env python3
"""quantlab — run a strategy through the full validation gauntlet.

Examples:
    python run.py --strategy tsmom --synthetic
    python run.py --strategy fvg --csv NQ_5min.csv --cost-bps 4
    python run.py --strategy donchian --yahoo SPY --start 2005-01-01
    python run.py --compare --synthetic
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from quantlab import data, engine, metrics, validate
from quantlab.strategies import REGISTRY

pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 60)


def load(args) -> tuple[pd.DataFrame, str]:
    if args.csv:
        return data.load_csv(args.csv), args.csv
    if args.yahoo:
        return data.load_yahoo(args.yahoo, start=args.start), args.yahoo
    return data.synthetic(n=args.synthetic_bars, seed=args.seed), "SYNTHETIC RANDOM WALK"


def rule(char="─", n=78):
    print(char * n)


def main():
    ap = argparse.ArgumentParser(description="Quantitative strategy validation harness")
    src = ap.add_argument_group("data")
    src.add_argument("--csv", help="OHLCV csv (TradingView export works)")
    src.add_argument("--yahoo", help="ticker via yfinance")
    src.add_argument("--start", default="2005-01-01")
    src.add_argument("--synthetic", action="store_true", help="use a random walk (the null hypothesis)")
    src.add_argument("--synthetic-bars", type=int, default=4000)
    src.add_argument("--seed", type=int, default=0)

    cfg = ap.add_argument_group("config")
    cfg.add_argument("--strategy", default="tsmom", choices=list(REGISTRY))
    cfg.add_argument("--compare", action="store_true", help="run every strategy side by side")
    cfg.add_argument("--cost-bps", type=float, default=3.0, help="round-trip cost in bps")
    cfg.add_argument("--vol-target", type=float, default=0.15)
    cfg.add_argument("--folds", type=int, default=5)
    cfg.add_argument("--trials", type=int, default=300)
    cfg.add_argument("--quick", action="store_true", help="skip the slow tests")
    args = ap.parse_args()

    df, label = load(args)
    costs = engine.CostModel(commission_bps=args.cost_bps / 2, slippage_bps=args.cost_bps / 2)
    bt = dict(vol_target=args.vol_target)

    print()
    rule("═")
    print(f"  DATA   {label}")
    print(f"         {len(df)} bars   {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  COSTS  {args.cost_bps:.1f} bps round trip    VOL TARGET {args.vol_target:.0%}")
    rule("═")

    if args.synthetic:
        print("  NOTE: synthetic data has no predictable structure by construction.")
        print("        Any strategy scoring well here is measuring overfitting.\n")

    # ── headline comparison ──
    names = list(REGISTRY) if args.compare else [args.strategy, "buy_hold"]
    print("\nIN-SAMPLE (the number that lies to you)\n")
    for name in names:
        s = REGISTRY[name]
        r = engine.run(df, s.signal(df), costs, **bt)
        print("  " + metrics.format_summary(name, metrics.summary(r)))

    if args.compare:
        rule()
        print("\nRun a single strategy without --compare for the full gauntlet.\n")
        return

    strat = REGISTRY[args.strategy]
    if args.strategy == "buy_hold":
        return

    base = engine.run(df, strat.signal(df), costs, **bt)
    base_summary = metrics.summary(base)

    # ── 1. walk-forward ──
    rule()
    print("\n[1] WALK-FORWARD  — parameters chosen in-sample, applied out-of-sample\n")
    wf = validate.walk_forward(df, strat, costs, n_folds=args.folds, **bt)
    if wf.folds.empty:
        print("  insufficient data for walk-forward")
        wf_sum = {"sharpe": 0.0, "folds_positive": 0.0}
    else:
        print(wf.folds.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        wf_sum = wf.summary(base.ppy)
        print(f"\n  Combined OOS Sharpe: {wf_sum['sharpe']:.2f}   "
              f"(in-sample was {base_summary['sharpe']:.2f})")
        decay = 1 - (wf_sum["sharpe"] / base_summary["sharpe"]) if base_summary["sharpe"] > 0 else 1.0
        print(f"  Decay from in-sample to out-of-sample: {decay:.0%}")

    # ── 2. parameter sensitivity ──
    rule()
    print("\n[2] PARAMETER SENSITIVITY — is it a plateau or a spike?\n")
    sens = validate.parameter_sensitivity(df, strat, costs, **bt)
    if sens.empty:
        print("  no parameter grid defined")
    else:
        print(sens.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  best {sens['sharpe'].max():.2f} / median {sens['sharpe'].median():.2f} "
              f"/ worst {sens['sharpe'].min():.2f}")
        print(f"  {(sens['sharpe'] > 0).mean():.0%} of settings profitable")

    # ── 3. random benchmark ──
    if not args.quick:
        rule()
        print("\n[3] RANDOM-ENTRY BENCHMARK — can coin flips do this too?\n")
        pos = base.position
        trade_rate = max(float((pos.diff().abs() > 1e-9).mean()), 0.001)
        rb = validate.random_benchmark(df, base_summary["sharpe"], trade_rate, 10,
                                       costs, n_trials=args.trials, **bt)
        print(f"  strategy Sharpe      {rb['strategy']:.2f}")
        print(f"  random mean          {rb['random_mean']:.2f} (sd {rb['random_std']:.2f})")
        print(f"  random 95th pct      {rb['random_p95']:.2f}")
        print(f"  random best of {rb['trials']:<4d}  {rb['random_max']:.2f}")
        print(f"\n  → your strategy beats {rb['percentile']:.0f}% of random entries")
        rand_pct = rb["percentile"]
    else:
        rand_pct = 0.0

    # ── 4. cost sweep ──
    rule()
    print("\n[4] COST SWEEP — where does the edge die?\n")
    cs = validate.cost_sweep(df, strat, **bt)
    print(cs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    positive = cs[cs["sharpe"] > 0]
    breakeven = float(positive["round_trip_bps"].max()) if not positive.empty else 0.0
    print(f"\n  Edge survives to roughly {breakeven:.0f} bps round trip")

    # ── verdict ──
    rule("═")
    print("\nVERDICT\n")
    for line in validate.verdict(wf_sum.get("sharpe", 0.0), wf_sum.get("folds_positive", 0.0),
                                 sens, rand_pct, breakeven, args.cost_bps):
        print("  " + line)
    print()
    rule("═")
    print("\n  Passing all five is necessary, not sufficient. It means the strategy")
    print("  is not obviously fake. Paper trade it for months before risking money.\n")


if __name__ == "__main__":
    sys.exit(main() or 0)

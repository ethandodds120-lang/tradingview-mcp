#!/usr/bin/env python3
"""quantlab — run a strategy through the full validation gauntlet.

Examples:
    python run.py --strategy tsmom --synthetic
    python run.py --strategy tjr --csv NQ_5min.csv --cost-bps 4
    python run.py --strategy donchian --yahoo SPY --start 2005-01-01
    python run.py --compare --synthetic
    python run.py --family predictive --csv NQ_5min.csv
    python run.py --head-to-head --csv NQ_5min.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from quantlab import data, engine, metrics, tjr as tjr_model, validate
from quantlab.strategies import FAMILIES, REGISTRY, by_family, grouped

pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 60)

# Windows consoles default to cp1252 and choke on the box-drawing characters
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def load(args) -> tuple[pd.DataFrame, str]:
    if args.csv:
        return data.load_csv(args.csv), args.csv
    if args.yahoo:
        return data.load_yahoo(args.yahoo, start=args.start), args.yahoo
    return data.synthetic(n=args.synthetic_bars, seed=args.seed), "SYNTHETIC RANDOM WALK"


def rule(char="─", n=78):
    print(char * n)


#: What each family claims. Printed above its block so the grouping means
#: something to a reader who has not read strategies/__init__.py.
FAMILY_BLURB = {
    "predictive": "a pattern implies what price does next",
    "systematic": "a statistical property persists across the sample",
    "benchmark": "no claim; what both families have to beat",
}


def family_line(name: str, s: dict) -> str:
    """format_summary with the strategy's evidence tag alongside the name.

    Padded to a fixed width so the metric columns still line up underneath each
    other once the tag is in front of them.
    """
    tag = f"[{REGISTRY[name].evidence}]"
    return metrics.format_summary(f"{name:<14}{tag:<12}", s)


def head_to_head(df, costs, bt, args):
    """Run every strategy in both families and compare the families, not the names.

    The interesting column is cost drag. Predictive strategies trade on setups and
    therefore trade far more often, so they need a much larger gross edge to clear
    the same cost model — that is a structural difference between the families, not
    a fact about any one strategy.

    The random-entry gate is not run here (it is 300 backtests per strategy); gates
    are scored out of the four that remain.
    """
    rows = []
    for fam in ("predictive", "systematic", "benchmark"):
        for name in by_family(fam):
            strat = REGISTRY[name]
            print(f"  running {name} ...", flush=True)
            base = engine.run(df, strat.signal(df), costs, **bt)
            summ = metrics.summary(base)

            wf = validate.walk_forward(df, strat, costs, n_folds=args.folds, **bt)
            wf_sum = wf.summary(base.ppy) if not wf.folds.empty else {
                "sharpe": 0.0, "folds_positive": 0.0}
            sens = validate.parameter_sensitivity(df, strat, costs, **bt)
            cs = validate.cost_sweep(df, strat, **bt)
            positive = cs[cs["sharpe"] > 0]
            breakeven = float(positive["round_trip_bps"].max()) if not positive.empty else 0.0

            gates = [
                wf_sum["sharpe"] > 0.3,
                wf_sum["folds_positive"] >= 0.6,
                # a strategy with no grid has nothing to be sensitive to; score the
                # gate as failed rather than silently dropping it, so the
                # denominator stays the same for every row
                bool(not sens.empty and (sens["sharpe"] > 0).mean() >= 0.6),
                breakeven > args.cost_bps * 3,
            ]
            rows.append({
                "family": fam,
                "name": name,
                "evidence": strat.evidence,
                "oos_sharpe": wf_sum["sharpe"],
                "gates": sum(gates),
                "trades": summ["trades"],
                "drag_sharpe": summ["cost_drag_sharpe"],
                "cost_pct": summ["total_costs"],
            })

    tbl = pd.DataFrame(rows)

    rule()
    print("\nHEAD TO HEAD — the two families on identical data\n")
    hdr = (f"  {'strategy':<14}{'evidence':<11}{'OOS Sharpe':>11}{'gates':>7}"
           f"{'trades':>8}{'drag(Sh)':>10}{'cost%':>8}")
    for fam in ("predictive", "systematic", "benchmark"):
        block = tbl[tbl["family"] == fam]
        if block.empty:
            continue
        print(f"  {fam.upper()} — {FAMILY_BLURB[fam]}")
        print(hdr)
        for r in block.to_dict("records"):
            print(f"  {r['name']:<14}{r['evidence']:<11}{r['oos_sharpe']:>11.2f}"
                  f"{str(r['gates']) + '/4':>7}{r['trades']:>8d}"
                  f"{r['drag_sharpe']:>10.2f}{r['cost_pct']:>8.2%}")
        print()

    rule()
    print("\nFAMILY SUMMARY\n")
    # Per-strategy means, not sums: the families do not have the same number of
    # members, so a total would measure the size of the registry rather than
    # anything about how the two kinds of strategy behave.
    print(f"  {'family':<14}{'best OOS':>10}{'median OOS':>12}{'gates':>9}"
          f"{'trades/strat':>14}{'drag(Sh)':>10}{'cost%':>8}")
    for fam in ("predictive", "systematic"):
        block = tbl[tbl["family"] == fam]
        if block.empty:
            continue
        print(f"  {fam:<14}{block['oos_sharpe'].max():>10.2f}"
              f"{block['oos_sharpe'].median():>12.2f}"
              f"{str(int(block['gates'].sum())) + '/' + str(4 * len(block)):>9}"
              f"{block['trades'].mean():>14.0f}"
              f"{block['drag_sharpe'].mean():>10.2f}"
              f"{block['cost_pct'].mean():>8.2%}")

    pred = tbl[tbl["family"] == "predictive"]
    syst = tbl[tbl["family"] == "systematic"]
    if not pred.empty and not syst.empty:
        p_tr, s_tr = pred["trades"].mean(), syst["trades"].mean()
        p_dg, s_dg = pred["drag_sharpe"].mean(), syst["drag_sharpe"].mean()
        print(f"\n  Trades per strategy: {p_tr:.0f} predictive vs {s_tr:.0f} "
              f"systematic" + (f" ({p_tr / s_tr:.1f}x)." if s_tr else "."))
        print(f"  Mean cost drag: {p_dg:.2f} Sharpe vs {s_dg:.2f}, on "
              f"{pred['cost_pct'].mean():.1%} of equity paid away vs "
              f"{syst['cost_pct'].mean():.1%}.")
        print("  Cost drag is the column to read. It is charged on every setup taken,"
              "\n  whether or not the setup worked, so a family that trades more needs"
              "\n  a proportionally larger gross edge just to finish level.")
    print("\n  Gates here are out of 4 — the random-entry benchmark is skipped."
          "\n  Run a single strategy without --head-to-head for the full gauntlet.\n")


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
    cfg.add_argument("--family", choices=list(FAMILIES) + ["all"], default="all",
                     help="limit which strategies run. Filters --compare, and on "
                          "its own implies --compare over that family. Ignored by "
                          "--head-to-head, which needs both families by definition.")
    cfg.add_argument("--head-to-head", action="store_true",
                     help="run both families on the same data and compare them: "
                          "OOS Sharpe, gate pass rates, trade counts and cost drag")
    cfg.add_argument("--cost-bps", type=float, default=3.0, help="round-trip cost in bps")
    cfg.add_argument("--vol-target", type=float, default=0.15)
    cfg.add_argument("--folds", type=int, default=5)
    cfg.add_argument("--trials", type=int, default=300)
    cfg.add_argument("--quick", action="store_true", help="skip the slow tests")
    cfg.add_argument("--param", action="append", metavar="K=V",
                     help="override a strategy param, e.g. --param long_only=true. "
                          "Applies to the defaults only — walk-forward and "
                          "sensitivity still search the strategy's own grid.")
    args = ap.parse_args()

    if args.param:
        strat_obj = REGISTRY[args.strategy]
        for kv in args.param:
            key, _, val = kv.partition("=")
            if key not in strat_obj.params:
                raise SystemExit(f"{args.strategy} has no param {key!r}; "
                                 f"have {list(strat_obj.params)}")
            if val.lower() in ("true", "false"):
                parsed = val.lower() == "true"
            else:
                try:
                    parsed = float(val) if "." in val else int(val)
                except ValueError:
                    parsed = val
            strat_obj.params[key] = parsed

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

    if args.head_to_head:
        head_to_head(df, costs, bt, args)
        return

    # ── headline comparison ──
    # --family on its own means "compare that family"; there is nothing else a
    # filter could usefully do to a single-strategy run.
    compare = args.compare or args.family != "all"
    print("\nIN-SAMPLE (the number that lies to you)\n")
    if compare:
        for fam, names in grouped(args.family).items():
            print(f"  {fam.upper()} — {FAMILY_BLURB[fam]}")
            for name in names:
                r = engine.run(df, REGISTRY[name].signal(df), costs, **bt)
                print("    " + family_line(name, metrics.summary(r)))
            print()
        rule()
        print("\nRun a single strategy without --compare for the full gauntlet.")
        print("Run --head-to-head to compare the families rather than the names.\n")
        return

    for name in [args.strategy, "buy_hold"]:
        s = REGISTRY[name]
        r = engine.run(df, s.signal(df), costs, **bt)
        print("  " + metrics.format_summary(name, metrics.summary(r)))

    strat = REGISTRY[args.strategy]
    if args.strategy == "buy_hold":
        return

    base = engine.run(df, strat.signal(df), costs, **bt)
    base_summary = metrics.summary(base)

    # ── path-dependent models report at the trade level too ──
    if args.strategy == "tjr":
        rule()
        print("\nTRADE DETAIL (real fill prices, before costs and vol targeting)\n")
        sim = tjr_model.simulate(df, **strat.params)
        print(tjr_model.format_trade_stats(tjr_model.trade_stats(sim.trades)))
        if len(sim.trades) < 30:
            print("\n  Fewer than 30 trades. Nothing below this line means anything"
                  "\n  at that sample size, whatever the Sharpe says.")

    # ── 0. causality ──
    rule()
    print("\n[0] CAUSALITY — recompute the signal on truncated history\n")
    cc = validate.causality_check(df, strat)
    if cc["clean"]:
        print(f"  clean across {cc['probes']} truncations — the signal at bar t "
              f"does not move when later bars are removed")
    else:
        print(f"  LOOKAHEAD: signal changed by {cc['max_diff']:.4f} when future bars "
              f"were removed\n  first mismatch at {cc['first_mismatch']}")
        print("  Everything below this line is fiction. Fix the strategy first.")

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
        if base_summary["sharpe"] > 0:
            decay = 1 - (wf_sum["sharpe"] / base_summary["sharpe"])
            print(f"  Decay from in-sample to out-of-sample: {decay:.0%}")
        else:
            print("  In-sample Sharpe was not positive, so there is nothing to decay from.")

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
        rand_pct = None

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

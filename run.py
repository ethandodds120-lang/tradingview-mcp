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


def load(args) -> tuple[pd.DataFrame, str, bool]:
    """Returns (data, label, is_panel).

    A panel is a dates x tickers price frame; everything else is a single-
    instrument OHLCV frame. They are not interchangeable, and the third element
    is what stops a cross-sectional strategy being handed one close column.
    """
    if args.panel_csv:
        return data.load_panel_csv(args.panel_csv), args.panel_csv, True
    if args.panel_synthetic:
        return (data.synthetic_panel(n=args.synthetic_bars, seed=args.seed),
                "SYNTHETIC CORRELATED PANEL", True)
    if args.csv:
        return data.load_csv(args.csv), args.csv, False
    if args.yahoo:
        return data.load_yahoo(args.yahoo, start=args.start), args.yahoo, False
    return data.synthetic(n=args.synthetic_bars, seed=args.seed), "SYNTHETIC RANDOM WALK", False


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


def head_to_head(df, costs, bt, args, kind="single"):
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
        for name in by_family(fam, kind):
            strat = REGISTRY[name]
            print(f"  running {name} ...", flush=True)
            base = strat.backtest(df, strat.signal(df), costs, **bt)
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
    src.add_argument("--panel-csv", help="dates x tickers price panel (wide or long) "
                                         "for cross-sectional strategies")
    src.add_argument("--panel-synthetic", action="store_true",
                     help="correlated random-walk panel — the null hypothesis for "
                          "a cross-sectional test")
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
    cfg.add_argument("--ruin-dd", type=float, default=0.50,
                     help="the drawdown that would end the experiment, as a "
                          "fraction. 0.50 is 'capital halved'; set it to your prop "
                          "firm's max-drawdown rule if you have one (default 0.50)")
    cfg.add_argument("--paths", type=int, default=2000,
                     help="bootstrap paths for the drawdown distribution")
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

    df, label, is_panel = load(args)
    costs = engine.CostModel(commission_bps=args.cost_bps / 2, slippage_bps=args.cost_bps / 2)
    bt = dict(vol_target=args.vol_target)

    print()
    rule("═")
    print(f"  DATA   {label}")
    width = f"{df.shape[1]} instruments x " if is_panel else ""
    print(f"         {width}{len(df)} bars   {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  COSTS  {args.cost_bps:.1f} bps round trip    VOL TARGET {args.vol_target:.0%}")
    rule("═")

    if args.synthetic:
        print("  NOTE: synthetic data has no predictable structure by construction.")
        print("        Any strategy scoring well here is measuring overfitting.\n")

    if args.head_to_head:
        head_to_head(df, costs, bt, args, "panel" if is_panel else "single")
        return

    # ── headline comparison ──
    # --family on its own means "compare that family"; there is nothing else a
    # filter could usefully do to a single-strategy run.
    compare = args.compare or args.family != "all"
    kind = "panel" if is_panel else "single"
    print("\nIN-SAMPLE (the number that lies to you)\n")
    if compare:
        for fam, names in grouped(args.family, kind).items():
            print(f"  {fam.upper()} — {FAMILY_BLURB[fam]}")
            for name in names:
                st = REGISTRY[name]
                r = st.backtest(df, st.signal(df), costs, **bt)
                print("    " + family_line(name, metrics.summary(r)))
            print()
        rule()
        print("\nRun a single strategy without --compare for the full gauntlet.")
        print("Run --head-to-head to compare the families rather than the names.\n")
        return

    # the benchmark has to match the shape of the thing it benchmarks: owning one
    # instrument means nothing next to a book that ranks twenty
    bench = "equal_weight" if is_panel else "buy_hold"
    for name in [args.strategy, bench]:
        s = REGISTRY[name]
        r = s.backtest(df, s.signal(df), costs, **bt)
        print("  " + metrics.format_summary(name, metrics.summary(r)))

    strat = REGISTRY[args.strategy]
    if args.strategy in ("buy_hold", "equal_weight"):
        return

    base = strat.backtest(df, strat.signal(df), costs, **bt)
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
    # One pass over the grid, reused by the deflation tests below. On tjr the grid
    # is 36 path-dependent simulations, so running it twice would double the wait
    # for numbers derived from the same trials.
    rule()
    print("\n[2] PARAMETER SENSITIVITY — is it a plateau or a spike?\n")
    raw, matrix, grid_ppy = validate.trial_matrix(df, strat, costs, **bt)
    sens = raw.sort_values("sharpe", ascending=False) if not raw.empty else raw
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
                                       costs, n_trials=args.trials,
                                       strat=strat, **bt)
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

    # ── 5. what the search itself cost ──
    dsr_val = pbo_val = None
    if not sens.empty and matrix.shape[1] > 1:
        rule()
        print("\n[5] SEARCH COST — how much of the best result was luck?\n")
        best_col = int(raw["sharpe"].values.argmax())
        best_returns = pd.Series(matrix[:, best_col], index=df.index)
        ds = validate.deflated_sharpe(best_returns, grid_ppy, raw["sharpe"].values)
        pb = validate.probability_of_backtest_overfitting(matrix)
        dsr_val, pbo_val = ds["dsr"], pb["pbo"]

        print(f"  grid searched            {ds['trials']} parameter combinations")
        print(f"  best Sharpe found        {raw['sharpe'].max():.2f}")
        print(f"  expected best from noise {ds['sr0']:.2f}   "
              f"← what {ds['trials']} tries would give on data with no edge")
        print(f"  deflated Sharpe          {ds['dsr']:.2f}   "
              f"← P(the edge is real, given the search)")
        print(f"  overfitting probability  {pb['pbo']:.0%}   "
              f"← how often the in-sample winner lands below the")
        print(f"                                    out-of-sample median, over "
              f"{pb['splits']} splits")
        print(f"                                    (noisy: sd ~19% on a single "
              f"run — read it loosely)")
        if np.isfinite(ds["dsr"]) and ds["dsr"] < 0.95:
            print(f"\n  The best setting does not clear what {ds['trials']} attempts "
                  f"produce on noise.\n  That is a statement about the search, not "
                  f"about the market.")

    # ── 6. drawdown distribution ──
    rule()
    print("\n[6] DRAWDOWN DISTRIBUTION — can the sizing survive its own bad luck?\n")
    ruin = validate.drawdown_distribution(base.returns, base.ppy, n_paths=args.paths,
                                          ruin_threshold=args.ruin_dd)
    if not ruin.get("paths"):
        print("  too few bars to resample")
        ruin = None
    else:
        print(f"  realized max drawdown    {ruin['realized_dd']:>7.1%}   "
              f"← the one number a backtest gives you")
        print(f"  median of {ruin['paths']} paths     {ruin['dd_median']:>7.1%}")
        print(f"  95th percentile          {ruin['dd_p95']:>7.1%}   "
              f"← plan for this one")
        print(f"  99th percentile          {ruin['dd_p99']:>7.1%}")
        print(f"\n  the realized drawdown was luckier than "
              f"{ruin['realized_pct']:.0f}% of resampled paths")
        print(f"  longest time under water {ruin['tuw_p95_bars']:.0f} bars "
              f"({ruin['tuw_p95_years']:.1f} years) at the 95th percentile")
        print(f"\n  P(breaching -{ruin['ruin_threshold']:.0%}) = "
              f"{ruin['p_ruin']:.1%} of paths")

    # ── verdict ──
    rule("═")
    print("\nVERDICT\n")
    for line in validate.verdict(wf_sum.get("sharpe", 0.0), wf_sum.get("folds_positive", 0.0),
                                 sens, rand_pct, breakeven, args.cost_bps,
                                 dsr=dsr_val, pbo=pbo_val,
                                 t_stat=base_summary.get("t_stat"), ruin=ruin):
        print("  " + line)
    print()
    rule("═")
    print("\n  Passing every gate is necessary, not sufficient. It means the strategy")
    print("  is not obviously fake. Paper trade it for months before risking money.\n")


if __name__ == "__main__":
    sys.exit(main() or 0)

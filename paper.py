#!/usr/bin/env python3
"""quantlab — paper trading, i.e. forward testing.

Watches bars arrive, runs the strategy, and writes down what it would have done,
when it would have done it. Fills are simulated by default; pass --broker alpaca
and they are sent to an Alpaca paper account instead — see quantlab/broker.py.

    # start a run against the live TradingView chart
    python paper.py start --strategy tjr --tv-symbol NQ1! --tv-timeframe 5

    # same decisions, but route the orders to the Alpaca paper account.
    # the bars still come from the chart; --broker-symbol is what actually trades
    python paper.py start --strategy tjr --tv-symbol NQ1! --tv-timeframe 5 \
        --broker alpaca --broker-symbol QQQ --allow-short

    # what the account looks like right now
    python paper.py broker --symbol QQQ

    # one iteration (this is the cron-friendly form)
    python paper.py poll --id tjr-NQ1-20260905

    # or stay in the foreground and poll on an interval
    python paper.py run --id tjr-NQ1-20260905 --interval 60

    # where it stands
    python paper.py report --id tjr-NQ1-20260905
    python paper.py list

    # rehearse the machinery on a stored file — NOT a forward test
    python paper.py start --strategy tjr --replay NQ_5min.csv --id rehearsal
    python paper.py run --id rehearsal --interval 0 --max-polls 500
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from quantlab import broker as broker_mod
from quantlab import feeds, paper
from quantlab.strategies import REGISTRY

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BANNER = """
  SIMULATED FILLS. No order leaves this process. The record this writes is a
  decision log, and its only value over a backtest is that the data did not exist
  when the strategy did. Treat it accordingly: months, not days.
"""

ROUTED_BANNER = """
  ORDERS ARE BEING SENT to the Alpaca {mode} account for {symbol}. The decisions are
  the same as a simulated run; only the fill price is real. A paper account is
  still a simulator — no queue, no market impact, no borrow — so this buys you a
  better fill model and the failures a simulation cannot have, not a live result.
  The clock still starts today: months, not days.
"""


def feed_spec(args) -> dict:
    if args.tv_symbol or args.tv_timeframe or args.tv:
        return {"kind": "tradingview", "symbol": args.tv_symbol,
                "timeframe": args.tv_timeframe, "count": args.bars,
                "allow_replay": args.allow_replay}
    if args.alpaca:
        return {"kind": "alpaca", "symbol": args.alpaca,
                "timeframe": args.alpaca_timeframe, "count": args.bars}
    if args.yahoo:
        return {"kind": "yahoo", "ticker": args.yahoo, "interval": args.interval_yf,
                "period": args.period}
    if args.csv:
        return {"kind": "csv", "path": args.csv}
    if args.replay:
        return {"kind": "replay", "path": args.replay, "start": args.replay_start,
                "step": args.replay_step}
    raise SystemExit("pick a feed: --tv-symbol, --yahoo, --csv or --replay")


def broker_spec(args) -> dict | None:
    """None means the simulated ledger, which stays the default."""
    if not getattr(args, "broker", None) or args.broker == "none":
        return None
    if not args.broker_symbol:
        raise SystemExit(
            "--broker needs --broker-symbol. The chart symbol is not it: the bars can\n"
            "come from NQ1! while the thing you actually trade is QQQ, and Alpaca has\n"
            "no idea what NQ1! is. Say which instrument the orders are for.")
    return {"kind": args.broker, "symbol": args.broker_symbol,
            "paper": not args.live, "allow_live": args.live,
            "fractional": not args.whole_units, "allow_short": args.allow_short,
            "require_open": not args.queue_when_closed,
            "fill_timeout": args.fill_timeout}


def banner_for(spec: dict | None) -> str:
    if not spec:
        return BANNER
    return ROUTED_BANNER.format(mode="LIVE" if not spec["paper"] else "paper",
                                symbol=spec["symbol"])


def default_id(args, spec: dict) -> str:
    tag = (spec.get("symbol") or spec.get("ticker") or spec.get("path") or "feed")
    tag = "".join(ch for ch in str(tag).split("/")[-1] if ch.isalnum() or ch in "-_")
    return f"{args.strategy}-{tag}-{date.today():%Y%m%d}"


def default_interval(spec: dict) -> int:
    tf = str(spec.get("timeframe") or spec.get("interval") or "")
    if tf.isdigit():
        return max(20, int(tf) * 60 // 4)      # poll a few times per bar
    return 300


def cmd_start(args) -> int:
    spec = feed_spec(args)
    bspec = broker_spec(args)
    run_id = args.id or default_id(args, spec)
    overrides = {}
    for kv in args.param or []:
        key, _, val = kv.partition("=")
        if val.lower() in ("true", "false"):
            overrides[key] = val.lower() == "true"
            continue
        try:
            overrides[key] = float(val) if "." in val else int(val)
        except ValueError:
            overrides[key] = val

    run = paper.PaperRun.create(
        args.dir, run_id, args.strategy, spec, params=overrides,
        cost_bps=args.cost_bps, vol_target=args.vol_target,
        start_equity=args.equity, min_history=args.min_history,
        note=args.note or "", broker_spec=bspec)
    print(banner_for(bspec))
    print(f"  created {run.root}")
    print(f"  strategy {args.strategy}  {run.config['params']}")
    print(f"  fingerprint {run.config['fingerprint']} — editing config.json breaks the run")
    if bspec:
        print(f"\n  account {run.config['broker_account']}  "
              f"equity {run.config['broker_equity_at_start']:,.2f} — "
              f"the run starts from what the account actually has, not --equity")
        for warn in run.config.get("broker_warnings") or []:
            print(f"  ! {warn}")
    print("\n  Nothing has been recorded yet. The first poll seeds history; the bars")
    print("  that are already on the chart are marked as history, not as results.")
    print("\n  If this strategy failed the gauntlet in run.py, forward testing it mostly")
    print("  buys a slower, more expensive version of the same answer.\n")
    result = run.poll()
    print(f"  first poll: {result}\n")
    return 0


def cmd_poll(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    result = run.poll()
    print(result)
    return 0


def cmd_run(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    interval = args.interval if args.interval is not None else default_interval(run.config["feed"])
    print(banner_for(run.config.get("broker")))
    print(f"  polling {run.config['run_id']} every {interval}s — ctrl-c to stop\n")
    try:
        paper.poll_forever(run, interval, max_polls=args.max_polls)
    except KeyboardInterrupt:
        print("\n  stopped. state is on disk; start again whenever.\n")
    print(run.report())
    return 0


def cmd_report(args) -> int:
    print(paper.PaperRun.load(args.dir, args.id).report())
    return 0


def cmd_list(args) -> int:
    runs = paper.PaperRun.list_runs(args.dir)
    if not runs:
        print(f"  no runs under {args.dir}/")
        return 0
    for run_id in runs:
        try:
            run = paper.PaperRun.load(args.dir, run_id)
            st, cfg = run.state, run.config
            print(f"  {run_id:<28} {cfg['strategy']:<12} "
                  f"{st.get('bars_forward', 0):>5} fwd bars  "
                  f"{st.get('fills', 0):>4} fills  "
                  f"equity {st.get('equity', 0):>12,.2f}")
        except paper.ParamDrift:
            print(f"  {run_id:<28} PARAM DRIFT — config edited after creation")
    return 0


def cmd_portfolio(args) -> int:
    """What the runs add up to. Reads only — changes nothing.

    Every run sizes itself to its own volatility target as though it were the
    only thing you own. It is not, and correlated runs stack: two trend models
    on gold and silver are close to the same position twice. Nothing in the
    engine knows that, because nothing in the engine owns the portfolio. This
    command is the missing view, not the missing allocator.
    """
    import numpy as np

    from quantlab import feeds

    rows = []
    for run_id in paper.PaperRun.list_runs(args.dir):
        try:
            run = paper.PaperRun.load(args.dir, run_id)
        except paper.ParamDrift:
            print(f"  skipping {run_id} — PARAM DRIFT")
            continue
        cfg, st = run.config, run.state
        symbol = (cfg.get("broker") or {}).get("symbol") or cfg["feed"].get("symbol")
        if not symbol:
            continue
        equity = float(st.get("equity") or 0.0)
        units = float(st.get("units") or 0.0)
        cash = float(st.get("cash") or 0.0)
        # what the run is actually holding, as a fraction of its own equity
        frac = (equity - cash) / equity if equity else 0.0
        # ... and what it intends to hold once its next order fills. A run that is
        # flat today because its market is shut is not a run with no exposure; it
        # is a run with exposure arriving at the next open, and a book-level view
        # that ignores that is reassuring at exactly the wrong moment.
        pending = st.get("pending") or {}
        want = float(pending.get("target", frac))
        rows.append({"run": run_id, "symbol": symbol, "strategy": cfg["strategy"],
                     "equity": equity, "units": units, "frac": frac, "want": want,
                     "vol_target": float(cfg.get("vol_target") or 0.0),
                     "routed": bool(cfg.get("broker"))})

    if not rows:
        print(f"  no runs under {args.dir}/")
        return 0

    total = sum(r["equity"] for r in rows)
    print(f"\n  {len(rows)} runs, {total:,.2f} total equity\n")
    print(f"  {'run':<28}{'symbol':<10}{'held':>9}{'wants':>9}{'of book':>10}"
          f"{'vol tgt':>9}  routed")
    for r in rows:
        weight = r["equity"] / total if total else 0.0
        print(f"  {r['run']:<28}{r['symbol']:<10}{r['frac']:>8.1%}{r['want']:>9.1%}"
              f"{weight * r['want']:>10.1%}{r['vol_target']:>9.0%}"
              f"   {'yes' if r['routed'] else 'no'}")

    symbols = sorted({r["symbol"] for r in rows})
    if len(symbols) < 2:
        print("\n  Only one instrument — nothing to diversify or double up on.\n")
        return 0

    # Price what can be priced. A run fed from a TradingView chart can carry a
    # symbol Alpaca has never heard of (NQ1! and friends), and letting one of
    # those abort the whole calculation makes this command useless exactly when
    # the book is most mixed. Skip them, say so, and report the covariance over
    # the rest — a partial answer that names what it left out.
    px, skipped = {}, []
    for s in symbols:
        try:
            close = feeds.AlpacaFeed(symbol=s, timeframe="1D",
                                     count=args.bars).fetch()["close"]
            # Crypto daily bars are stamped 00:00 UTC and equity bars 05:00, so a
            # raw join aligns on nothing and dropna() empties the frame. Collapse
            # both to the calendar date before lining them up.
            close.index = close.index.normalize()
            px[s] = close[~close.index.duplicated(keep="last")]
        except Exception as exc:
            skipped.append((s, str(exc).split("\n")[0][:60]))

    if skipped:
        print()
        for s, why in skipped:
            print(f"  ! {s} left out of the covariance — {why}")
        symbols = [s for s in symbols if s in px]
        rows = [r for r in rows if r["symbol"] in px]

    if len(px) < 2:
        print("\n  need at least two priceable symbols for a covariance\n")
        return 1

    import pandas as pd

    ret = pd.DataFrame(px).pct_change().dropna()
    if len(ret) < 30:
        print("\n  not enough overlapping history to estimate a covariance\n")
        return 1

    print(f"\n  correlation of daily returns ({len(ret)} bars)\n")
    print("   " + ret.corr().round(2).to_string().replace("\n", "\n   "))

    # exposure of the whole book to each instrument: a run's own position scaled
    # by its share of total capital, summed over runs that hold the same thing
    cov = ret[symbols].cov().values * 252.0

    def book_vol(key: str) -> tuple[np.ndarray, float, float]:
        w = np.zeros(len(symbols))
        for r in rows:
            w[symbols.index(r["symbol"])] += (r["equity"] / total) * r[key]
        return (w, float(np.sqrt(w @ cov @ w)),
                float(np.sqrt((w ** 2 * np.diag(cov)).sum())))

    w_now, vol_now, naive_now = book_vol("frac")
    w_next, vol_next, naive_next = book_vol("want")
    targets = {r["vol_target"] for r in rows if r["vol_target"]}

    print(f"\n  {'':<22}{'held now':>12}{'once filled':>14}")
    for i, s in enumerate(symbols):
        print(f"  {s:<22}{w_now[i]:>11.1%}{w_next[i]:>14.1%}")
    print(f"  {'combined book vol':<22}{vol_now:>11.1%}{vol_next:>14.1%}")
    print(f"  {'if uncorrelated':<22}{naive_now:>11.1%}{naive_next:>14.1%}")

    if targets:
        tgt = max(targets)
        print(f"\n  Each run targets {tgt:.0%} vol, sized as though it were the only "
              f"thing you own.")
        if vol_next > tgt * 1.15:
            print(f"  Once the pending orders fill the book runs at {vol_next:.1%} — "
                  f"{vol_next / tgt:.1f}x that.")
            print(f"  Correlated positions stack and the runs cannot see each other."
                  f"\n  Size the book, not the strategies.")
        elif vol_now <= tgt:
            print(f"  Currently within it, but only because some runs are still flat.")
    print()
    return 0


def cmd_unfilled(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    df = run.unfilled_frame()
    if df.empty:
        print("  nothing unfilled"
              + ("" if run.routes_orders else " — this run simulates its fills, so"
                                              " there is nothing that could be"))
        return 0
    cols = [c for c in ("bar", "decided_at", "side", "wanted_units", "ref_price",
                        "why_not") if c in df.columns]
    print(f"\n  {len(df)} order(s) the broker did not fill — these are trades the "
          f"record is missing\n")
    print(df.tail(args.tail)[cols].to_string(index=False))
    print()
    return 0


def cmd_broker(args) -> int:
    """What the account looks like right now. Reads only — places nothing."""
    b = broker_mod.AlpacaBroker(symbol=args.symbol, paper=not args.live,
                                allow_live=args.live, allow_short=args.allow_short)
    info = b.prepare()
    print(f"\n  {b.describe()}")
    print(f"  account   {info['broker_account']}   "
          f"{'paper' if info['broker_paper'] else 'LIVE — REAL MONEY'}")
    print(f"  equity    {b.account_equity:,.2f}   cash {b.cash:,.2f}   "
          f"buying power {b.buying_power:,.2f}")
    print(f"  position  {b.units:+,.4f} {args.symbol}")
    print(f"  asset     {info['broker_asset_class']}   "
          f"fractionable {info['broker_fractionable']}   "
          f"shortable {info['broker_shortable']}")
    print(f"  market    {'open' if b.market_open() else 'closed'}")
    for warn in info["broker_warnings"]:
        print(f"  ! {warn}")
    print()
    return 0


def cmd_journal(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    df = run.journal_frame()
    if df.empty:
        print("  empty journal")
        return 0
    cols = [c for c in ("bar", "close", "signal", "target", "equity", "held", "fill")
            if c in df.columns]
    print(df.tail(args.tail)[cols].to_string(index=False))
    return 0


def cmd_fills(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    df = run.fills_frame()
    if df.empty:
        print("  no fills yet")
        return 0
    cols = ["bar", "side", "units", "fill_price", "notional", "commission",
            "slippage_cost", "equity"]
    print(f"\n  {len(df)} fills — {run.journal_path}\n")
    print(df.tail(args.tail)[cols].to_string(index=False))
    print()
    return 0


def cmd_trades(args) -> int:
    run = paper.PaperRun.load(args.dir, args.id)
    df = run.round_trips()
    if df.empty:
        print("  no trades yet")
        return 0
    closed = df[df["exit"] != "OPEN"]
    print(f"\n  {len(closed)} closed round trips"
          + (", 1 still open" if len(df) > len(closed) else "") + "\n")
    print(df.tail(args.tail).to_string(index=False))
    if not closed.empty:
        wins = closed[closed["pnl"] > 0]
        print(f"\n  win rate {len(wins) / len(closed):.1%}   "
              f"total {closed['pnl'].sum():+,.2f}   "
              f"avg {closed['pnl'].mean():+,.2f}   "
              f"best {closed['pnl'].max():+,.2f}   worst {closed['pnl'].min():+,.2f}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="quantlab paper trading (forward test)")
    ap.add_argument("--dir", default="paper_runs", help="where runs are stored")
    sub = ap.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="create a run and seed its history")
    start.add_argument("--strategy", default="tjr", choices=list(REGISTRY))
    start.add_argument("--id", help="run id (default strategy-symbol-date)")
    start.add_argument("--param", action="append",
                       help="override a strategy param, e.g. --param rr=2.0")
    start.add_argument("--tv", action="store_true",
                       help="read the live TradingView chart as-is, without changing it")
    start.add_argument("--tv-symbol", help="live TradingView chart, e.g. NQ1!")
    start.add_argument("--tv-timeframe", help="chart resolution, e.g. 5")
    start.add_argument("--bars", type=int, default=400, help="bars per fetch")
    start.add_argument("--allow-replay", action="store_true",
                       help="ingest TradingView replay-mode bars. Rehearsal only — "
                            "those bars are historical and are not forward results")
    start.add_argument("--alpaca", metavar="SYMBOL",
                       help="bars from Alpaca, e.g. SOL/USD or QQQ. Pair this with "
                            "--broker-symbol SOL/USD so the feed and the orders are "
                            "the same instrument on the same venue")
    start.add_argument("--alpaca-timeframe", default="1D",
                       help="5Min, 15Min, 1H, 1D (default 1D)")
    start.add_argument("--yahoo", help="ticker via yfinance")
    start.add_argument("--interval-yf", default="1d")
    start.add_argument("--period", default="6mo")
    start.add_argument("--csv", help="a file something else appends closed bars to")
    start.add_argument("--replay", help="rehearse on a stored csv (not a forward test)")
    start.add_argument("--replay-start", type=int, default=300)
    start.add_argument("--replay-step", type=int, default=1)
    start.add_argument("--cost-bps", type=float, default=3.0)
    start.add_argument("--vol-target", type=float, default=0.15)
    start.add_argument("--equity", type=float, default=100_000.0)
    start.add_argument("--min-history", type=int, default=200,
                       help="bars required before the first decision (raise it for "
                            "long-lookback strategies like tsmom)")
    start.add_argument("--note", help="why you are running this")

    route = start.add_argument_group(
        "order routing",
        "By default fills are simulated in-process. These send them to a broker "
        "instead. The bar feed is unaffected — the chart can be NQ1! while the "
        "orders are for QQQ.")
    route.add_argument("--broker", choices=["none", "alpaca"], default="none")
    route.add_argument("--broker-symbol",
                       help="the instrument the orders are for, as the broker names "
                            "it (e.g. QQQ, BTC/USD). Required with --broker.")
    route.add_argument("--allow-short", action="store_true",
                       help="let the run actually go short. Without it, short "
                            "targets are clamped to flat and the routed account "
                            "stops matching the strategy")
    route.add_argument("--whole-units", action="store_true",
                       help="never send fractional quantities")
    route.add_argument("--queue-when-closed", action="store_true",
                       help="submit market orders while the market is shut. They "
                            "fill at the next open, hours after the decision")
    route.add_argument("--fill-timeout", type=float, default=90.0,
                       help="seconds to wait for a fill before cancelling (default 90)")
    route.add_argument("--live", action="store_true",
                       help="route to the LIVE account instead of paper. Real money.")
    start.set_defaults(func=cmd_start)

    poll = sub.add_parser("poll", help="one iteration, then exit")
    poll.add_argument("--id", required=True)
    poll.set_defaults(func=cmd_poll)

    loop = sub.add_parser("run", help="poll on an interval until interrupted")
    loop.add_argument("--id", required=True)
    loop.add_argument("--interval", type=int, help="seconds between polls")
    loop.add_argument("--max-polls", type=int)
    loop.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="where the run stands")
    rep.add_argument("--id", required=True)
    rep.set_defaults(func=cmd_report)

    lst = sub.add_parser("list", help="all runs")
    lst.set_defaults(func=cmd_list)

    pf = sub.add_parser("portfolio",
                        help="what every run adds up to — combined exposure and vol")
    pf.add_argument("--bars", type=int, default=400,
                    help="bars of history for the covariance estimate")
    pf.set_defaults(func=cmd_portfolio)

    fil = sub.add_parser("fills", help="the fill ledger — every paper trade")
    fil.add_argument("--id", required=True)
    fil.add_argument("--tail", type=int, default=30)
    fil.set_defaults(func=cmd_fills)

    trd = sub.add_parser("trades", help="fills collapsed into flat-to-flat trades")
    trd.add_argument("--id", required=True)
    trd.add_argument("--tail", type=int, default=30)
    trd.set_defaults(func=cmd_trades)

    jrn = sub.add_parser("journal", help="tail the decision log (every bar, fill or not)")
    jrn.add_argument("--id", required=True)
    jrn.add_argument("--tail", type=int, default=20)
    jrn.set_defaults(func=cmd_journal)

    unf = sub.add_parser("unfilled", help="orders the broker refused or could not fill")
    unf.add_argument("--id", required=True)
    unf.add_argument("--tail", type=int, default=30)
    unf.set_defaults(func=cmd_unfilled)

    brk = sub.add_parser("broker", help="what the broker account looks like right now")
    brk.add_argument("--symbol", required=True, help="e.g. QQQ, SPY, BTC/USD")
    brk.add_argument("--allow-short", action="store_true")
    brk.add_argument("--live", action="store_true", help="check the LIVE account")
    brk.set_defaults(func=cmd_broker)

    args = ap.parse_args()
    try:
        return args.func(args)
    except paper.ParamDrift as exc:
        print(f"\n  REFUSING TO CONTINUE\n\n  {exc}\n")
        return 2
    except feeds.FeedError as exc:
        print(f"\n  feed error: {exc}\n")
        return 3
    except broker_mod.BrokerError as exc:
        print(f"\n  broker error: {exc}\n")
        return 4


if __name__ == "__main__":
    sys.exit(main() or 0)

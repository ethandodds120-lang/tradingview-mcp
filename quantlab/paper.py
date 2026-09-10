"""Paper trading — which is to say forward testing.

It watches bars arrive, runs the same strategy code the backtester runs, and writes
down what it would have done at the moment it would have done it. The output is a
record of decisions made on data that did not exist when the strategy was written.
That record is the only thing in this repo that is genuinely out-of-sample.

By default nothing leaves this process: the Account below simulates the fill. A run
can instead be created with a broker (see broker.py), in which case the same
decisions are sent to an Alpaca paper account and the fills are read back from it.
That changes exactly one thing — where the fill price comes from. It does not make
the run live trading, and it does not make the record worth more; a broker's paper
account is still a simulator, just one with a better quote model than this file's.
Everything below about history, fingerprints and the forming bar applies unchanged,
because those are what make the record mean anything, not the fill.

What makes it worth more than a backtest
----------------------------------------
Only one thing: the decisions are timestamped and append-only. Three properties
enforce it, and they are the whole point of the module —

1. Bars that already existed when the run was created are marked as history. They
   seed the strategy's lookback and are never journaled as paper trades. You cannot
   start a run today and claim last month's bars as forward results.

2. The parameters are fingerprinted at creation and checked on every poll. Edit the
   strategy params of a live run and it refuses to continue. Retuning after seeing
   forward results is how a forward test turns back into a backtest, and it happens
   by accident unless something stops it.

3. The forming bar is dropped (see feeds.py). Decisions are made on closed bars
   only, and the resulting order fills at the OPEN of the following bar.

Fills
-----
The vectorized engine assumes close-to-close fills. This does not: a decision at
the close of bar t fills at the open of bar t+1, plus slippage, plus commission.
That is still optimistic — no partial fills, no gaps through your stop, unlimited
size at the touch — but it is closer, and the gap between the two is worth looking
at. `report()` prints the engine's number next to the paper account over exactly
the same bars so you can see what the fill assumption is worth.

Not modeled: margin interest, overnight financing, borrow on shorts, exchange fees
beyond the flat bps, tax, and the fact that a real fill moves the market.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import broker as broker_mod
from . import data as data_mod
from . import engine, feeds, metrics
from .strategies import REGISTRY

BAR_COLS = ["open", "high", "low", "close", "volume"]

# fields that change what the strategy would do — frozen for the life of a run
_FINGERPRINTED = ("strategy", "params", "cost_bps", "vol_target", "vol_lookback",
                  "max_leverage", "rebalance_band", "feed")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fingerprint(config: dict) -> str:
    subset = {k: config.get(k) for k in _FINGERPRINTED}
    # Only folded in when a broker is actually attached, so runs created before
    # routing existed — and simulated runs, which store None — keep their hashes.
    # Swapping the broker of a live run is a param change like any other.
    if config.get("broker"):
        subset["broker"] = config["broker"]
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ParamDrift(RuntimeError):
    """Raised when a live run's decision parameters were edited after creation."""


# ────────────────────────────── the account ──────────────────────────────

@dataclass
class Account:
    """A cash-and-units ledger. Position is carried as units so the arithmetic is
    auditable rather than a compounding fraction.

    The simulated fill: a flat bps haircut on the reference price, always available
    in unlimited size. broker.AlpacaBroker presents this same surface and gets the
    number from a broker instead."""

    cash: float
    units: float = 0.0
    commission_bps: float = 1.5
    slippage_bps: float = 1.5

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def fraction(self, price: float) -> float:
        eq = self.equity(price)
        return (self.units * price / eq) if eq else 0.0

    def rebalance(self, price: float, target_fraction: float) -> dict | None:
        """Move to `target_fraction` of equity at `price`. Returns the fill, if any."""
        eq = self.equity(price)
        if eq <= 0:
            return None
        target_units = target_fraction * eq / price
        delta = target_units - self.units
        if abs(delta * price) < 1e-9:
            return None

        side = 1.0 if delta > 0 else -1.0
        fill_px = price * (1 + side * self.slippage_bps / 10_000.0)
        notional = abs(delta) * fill_px
        commission = notional * self.commission_bps / 10_000.0

        self.cash -= delta * fill_px + commission
        self.units = target_units
        return {
            "side": "buy" if delta > 0 else "sell",
            "units": round(delta, 8),
            "ref_price": round(price, 6),
            "fill_price": round(fill_px, 6),
            "notional": round(notional, 2),
            "commission": round(commission, 4),
            "slippage_cost": round(abs(delta) * price * self.slippage_bps / 10_000.0, 4),
        }


# ────────────────────────────── the run ──────────────────────────────

@dataclass
class PaperRun:
    root: Path
    config: dict
    state: dict
    _feed: feeds.Feed | None = field(default=None, repr=False)
    _bars: pd.DataFrame | None = field(default=None, repr=False)
    _broker: object | None = field(default=None, repr=False)

    # ---- files ----
    @property
    def bars_path(self) -> Path:
        return self.root / "bars.csv"

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    # ---- lifecycle ----
    @classmethod
    def create(cls, base: str | Path, run_id: str, strategy: str, feed_spec: dict,
               params: dict | None = None, cost_bps: float = 3.0,
               vol_target: float | None = 0.15, vol_lookback: int = 60,
               max_leverage: float = 2.0, rebalance_band: float = 0.10,
               start_equity: float = 100_000.0, min_history: int = 200,
               note: str = "", broker_spec: dict | None = None) -> "PaperRun":
        if strategy not in REGISTRY:
            raise KeyError(f"unknown strategy {strategy!r}; have {list(REGISTRY)}")
        root = Path(base) / run_id
        if root.exists():
            raise FileExistsError(f"{root} already exists — pick another run id")
        root.mkdir(parents=True)

        config = {
            "run_id": run_id,
            "created": _now(),
            "strategy": strategy,
            "params": {**REGISTRY[strategy].params, **(params or {})},
            "feed": feed_spec,
            "broker": broker_spec or None,
            "cost_bps": cost_bps,
            "vol_target": vol_target,
            "vol_lookback": vol_lookback,
            "max_leverage": max_leverage,
            "rebalance_band": rebalance_band,
            "start_equity": start_equity,
            "min_history": max(min_history, vol_lookback + 10),
            "note": note,
        }
        broker = None
        try:
            feed = feeds.build(feed_spec)
            config.update(feed.prepare())
            # Fail here, before any bars are written, rather than on the first fill
            # at some unattended hour. prepare() checks the account, the symbol and
            # whether it can do what the strategy will ask of it.
            broker = broker_mod.build(broker_spec)
            if broker is not None:
                config.update(broker.prepare())
                # the account is the authority on its own money; a --equity flag
                # would just be a number this run wishes were true
                start_equity = broker.account_equity
                config["start_equity"] = start_equity
        except Exception:
            root.rmdir()        # do not leave a half-built run in the way of a retry
            raise
        config["fingerprint"] = fingerprint(config)

        state = {
            "last_bar": None,
            "forward_start": None,
            "cash": broker.cash if broker is not None else start_equity,
            "units": broker.units if broker is not None else 0.0,
            "pending": None,
            "equity": start_equity,
            "peak_equity": start_equity,
            "bars_seen": 0,
            "bars_forward": 0,
            "fills": 0,
            "updated": _now(),
        }
        run = cls(root=root, config=config, state=state, _feed=feed, _broker=broker)
        run.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        run._save_state()
        run._journal({"type": "created", "at": _now(), "config": config})
        return run

    @classmethod
    def load(cls, base: str | Path, run_id: str) -> "PaperRun":
        root = Path(base) / run_id
        if not root.exists():
            raise FileNotFoundError(f"no such run: {root}")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        expected = config.get("fingerprint")
        actual = fingerprint(config)
        if expected != actual:
            raise ParamDrift(
                f"run {run_id!r} was created with fingerprint {expected} but its config "
                f"now hashes to {actual}.\n"
                "Decision parameters were edited after the run started. Changing them "
                "mid-flight turns the forward test back into a backtest — the results "
                "so far were what told you to change them.\n"
                "Start a new run instead; the old one keeps its record.")
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        return cls(root=root, config=config, state=state)

    @staticmethod
    def list_runs(base: str | Path) -> list[str]:
        base = Path(base)
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if (p / "config.json").exists())

    # ---- persistence ----
    def _save_state(self) -> None:
        self.state["updated"] = _now()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _journal(self, record: dict) -> None:
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def bars(self) -> pd.DataFrame:
        if self._bars is None:
            self._bars = (data_mod.load_csv(str(self.bars_path))
                          if self.bars_path.exists()
                          else pd.DataFrame(columns=BAR_COLS))
        return self._bars

    def _store_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append genuinely new bars. The store is append-only like everything else
        here: a bar that has already been decided on is never rewritten, even if the
        feed later hands back a revised copy of it."""
        store = self.bars()
        fresh = df if store.empty else df[df.index > store.index[-1]]
        if fresh.empty:
            return store

        out = fresh.reset_index()
        out.columns = ["time"] + BAR_COLS
        header = not self.bars_path.exists()
        out.to_csv(self.bars_path, mode="a", header=header, index=False)
        self._bars = fresh if store.empty else pd.concat([store, fresh])
        return self._bars

    def journal_frame(self) -> pd.DataFrame:
        if not self.journal_path.exists():
            return pd.DataFrame()
        rows = []
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def fills_frame(self) -> pd.DataFrame:
        """Every fill, one row each. This is the paper trade ledger."""
        jf = self.journal_frame()
        if jf.empty or "fill" not in jf:
            return pd.DataFrame()
        rows = []
        for rec in jf.to_dict("records"):
            fill = rec.get("fill")
            if isinstance(fill, dict):
                rows.append({"bar": rec["bar"], **fill,
                             "equity": rec.get("equity"), "held": rec.get("held")})
        return pd.DataFrame(rows)

    def unfilled_frame(self) -> pd.DataFrame:
        """Orders the broker did not fill, and why.

        Empty for a simulated run — the simulation always fills. On a routed run
        this is the part a backtest cannot show you, so it is kept separate from
        the fill ledger rather than averaged into it.
        """
        jf = self.journal_frame()
        if jf.empty or "unfilled" not in jf:
            return pd.DataFrame()
        rows = [{"bar": rec["bar"], **rec["unfilled"]}
                for rec in jf.to_dict("records")
                if isinstance(rec.get("unfilled"), dict)]
        return pd.DataFrame(rows)

    def round_trips(self) -> pd.DataFrame:
        """Fills collapsed into flat-to-flat trades.

        A vol-targeted position gets resized while it is open, so a "trade" is not
        one buy and one sell. P&L is accumulated from the cash side of the ledger —
        over a stretch that starts and ends flat, the cash deltas sum to exactly the
        realized result, commissions and slippage included. When a fill flips the
        position through zero it is split pro-rata between the trade it closes and
        the one it opens.
        """
        fills = self.fills_frame()
        if fills.empty:
            return pd.DataFrame()

        trades, units, cur = [], 0.0, None

        def open_trade(row, cash, size):
            return {"entry": row["bar"], "side": "long" if size > 0 else "short",
                    "entry_price": row["fill_price"], "equity_at_entry": row["equity"],
                    "cash": cash, "fills": 1, "peak_units": abs(size)}

        def close_trade(cur, row):
            trades.append({
                "entry": cur["entry"], "exit": row["bar"], "side": cur["side"],
                "entry_price": round(cur["entry_price"], 2),
                "exit_price": round(row["fill_price"], 2),
                "units": round(cur["peak_units"], 4),
                "pnl": round(cur["cash"], 2),
                "return": round(cur["cash"] / cur["equity_at_entry"], 5)
                if cur["equity_at_entry"] else float("nan"),
                "fills": cur["fills"],
            })

        for row in fills.to_dict("records"):
            delta, price = float(row["units"]), float(row["fill_price"])
            cash = -(delta * price) - float(row["commission"])

            if cur is None:
                cur, units = open_trade(row, cash, delta), delta
                continue

            if np.sign(delta) == np.sign(units):            # adding to the position
                cur["cash"] += cash
                cur["fills"] += 1
                units += delta
                cur["peak_units"] = max(cur["peak_units"], abs(units))
                continue

            # reducing, closing or flipping
            closing = min(abs(delta), abs(units))
            frac = closing / abs(delta)
            cur["cash"] += cash * frac
            cur["fills"] += 1
            new_units = units + delta

            # journal fills are rounded, so the running total misses exact zero by a
            # hair. An absolute epsilon here invents phantom flip-trades of size 1e-8.
            flat_eps = 1e-6 * max(abs(units), abs(delta), 1.0)
            if abs(new_units) < flat_eps:
                close_trade(cur, row)
                cur, units = None, 0.0
            elif np.sign(new_units) != np.sign(units):      # flipped through flat
                close_trade(cur, row)
                cur, units = open_trade(row, cash * (1 - frac), new_units), new_units
            else:                                           # partial trim
                units = new_units

        if cur is not None:                                 # still open right now
            last = fills.iloc[-1]
            trades.append({
                "entry": cur["entry"], "exit": "OPEN", "side": cur["side"],
                "entry_price": round(cur["entry_price"], 2),
                "exit_price": float("nan"),
                "units": round(cur["peak_units"], 4),
                "pnl": round(cur["cash"] + units * float(last["ref_price"]), 2),
                "return": float("nan"), "fills": cur["fills"],
            })
        return pd.DataFrame(trades)

    @property
    def feed(self) -> feeds.Feed:
        if self._feed is None:
            self._feed = feeds.build(self.config["feed"])
            if self.state.get("last_bar"):
                self._feed.seek(pd.Timestamp(self.state["last_bar"]))
        return self._feed

    @property
    def broker(self):
        """The routed broker, or None when fills are simulated. Cached — it holds a
        session, and rebuilding it per bar would re-authenticate per bar."""
        if self._broker is None and self.config.get("broker"):
            self._broker = broker_mod.build(self.config["broker"])
        return self._broker

    @property
    def routes_orders(self) -> bool:
        return bool(self.config.get("broker"))

    # ---- the loop ----
    def poll(self) -> dict:
        """One iteration: fetch, fill the pending order, decide, journal."""
        if self.routes_orders:
            # Once per poll, before anything reads cash or units. A freshly loaded
            # broker knows nothing about the account until it is asked, and a run
            # resumed in a new process would otherwise mark itself at zero.
            self.broker.refresh()

        fetched = self.feed.bars()
        if fetched.empty:
            return {"new_bars": 0, "note": "feed returned nothing"}

        last_bar = pd.Timestamp(self.state["last_bar"]) if self.state["last_bar"] else None
        store = self._store_bars(fetched)

        if last_bar is None:
            return self._bootstrap(store)

        new = store[store.index > last_bar]
        if new.empty:
            if self.routes_orders:
                # pick up anything that moved the account while we were not looking:
                # a manual trade, a late fill, a margin action
                self._mirror_broker(float(store["close"].iloc[-1]))
                self._save_state()
            return {"new_bars": 0, "waiting_on": str(store.index[-1]),
                    "pending": self.state["pending"]}

        # Only the newest bar is actionable. If several arrived at once the loop is
        # catching up, and the opens the older decisions would have filled at are
        # gone — routing them now would send a market order on a price that is
        # already history. The simulated account has no such problem and fills them
        # all, exactly as it did before.
        newest = new.index[-1]
        events = [self._process_bar(store, ts, stale=(self.routes_orders and ts != newest))
                  for ts in new.index]
        self._save_state()
        return {"new_bars": len(events), "events": events}

    def _mirror_broker(self, price: float) -> None:
        """Copy the broker's view of cash and position into state. poll() has
        already refreshed it."""
        acct = self.broker
        equity = acct.equity(price)
        self.state.update({
            "cash": acct.cash, "units": acct.units, "equity": equity,
            "peak_equity": max(self.state.get("peak_equity", equity), equity),
            "broker_equity": acct.account_equity,
        })

    def _bootstrap(self, store: pd.DataFrame) -> dict:
        """First poll. Everything already on the chart is history, not results."""
        ts = store.index[-1]
        self.state.update({"last_bar": str(ts), "forward_start": str(ts),
                           "bars_seen": len(store)})
        decision = self._decide(store, ts)
        self._journal({
            "type": "backfill", "at": _now(), "bar": str(ts),
            "history_bars": len(store),
            "span": [str(store.index[0]), str(ts)],
            "note": "bars up to and including this one are history — they seed the "
                    "lookback and are not counted as paper trades",
            **decision,
        })
        self._save_state()
        return {"new_bars": 0, "bootstrapped": len(store), "from": str(store.index[0]),
                "to": str(ts), **decision}

    def _process_bar(self, store: pd.DataFrame, ts: pd.Timestamp,
                     stale: bool = False) -> dict:
        bar = store.loc[ts]
        acct = self._account()
        pending = self.state.get("pending")

        # 1. yesterday's decision fills at this bar's open
        fill, unfilled = None, None
        if pending is not None:
            if stale:
                unfilled = {"routed": False, "decided_at": pending["bar"],
                            "target": pending["target"], "ref_price": float(bar["open"]),
                            "why_not": "catch-up bar — this open has already passed, so "
                                       "the order was not sent"}
            else:
                result = acct.rebalance(float(bar["open"]), float(pending["target"]))
                if result and result.get("routed") is False:
                    # the broker took the order and would not or could not fill it.
                    # It is not a fill and must never reach the fill ledger, but it
                    # is the most interesting thing that happened this bar.
                    unfilled = {**result, "decided_at": pending["bar"]}
                    self.state["unfilled"] = self.state.get("unfilled", 0) + 1
                elif result:
                    fill = {**result, "decided_at": pending["bar"]}
                    self.state["fills"] = self.state.get("fills", 0) + 1
            self.state["pending"] = None

        # 2. mark to the close
        close = float(bar["close"])
        equity = acct.equity(close)
        self.state.update({
            "cash": acct.cash, "units": acct.units, "equity": equity,
            "peak_equity": max(self.state.get("peak_equity", equity), equity),
            "last_bar": str(ts),
            "bars_seen": self.state.get("bars_seen", 0) + 1,
            "bars_forward": self.state.get("bars_forward", 0) + 1,
        })
        if self.routes_orders:
            self.state["broker_equity"] = acct.account_equity

        # 3. decide for the next bar
        history = store.loc[:ts]
        decision = self._decide(history, ts, held=acct.fraction(close))

        record = {
            "type": "bar", "at": _now(), "bar": str(ts),
            "open": float(bar["open"]), "close": close,
            "fill": fill,
            "equity": round(equity, 2),
            "held": round(acct.fraction(close), 4),
            "drawdown": round(equity / self.state["peak_equity"] - 1.0, 5),
            **decision,
        }
        if unfilled:
            record["unfilled"] = unfilled
        if self.routes_orders:
            record["broker_equity"] = round(acct.account_equity, 2)
        self._journal(record)
        return record

    def _decide(self, history: pd.DataFrame, ts: pd.Timestamp, held: float = 0.0) -> dict:
        """Signal at the last CLOSED bar, sized, banded. Sets the pending order."""
        cfg = self.config
        if len(history) < cfg["min_history"]:
            self.state["pending"] = None
            return {"signal": 0.0, "target": 0.0, "order": None,
                    "why": f"warmup: {len(history)}/{cfg['min_history']} bars"}

        strat = REGISTRY[cfg["strategy"]]
        sig = float(strat.signal(history, **cfg["params"]).iloc[-1])
        scale = float(engine.vol_scale(history, cfg["vol_target"], cfg["vol_lookback"],
                                       cfg["max_leverage"]).iloc[-1])
        target = sig * scale

        band = cfg["rebalance_band"]
        move = abs(target - held) > band or np.sign(target) != np.sign(held)
        if move:
            self.state["pending"] = {"bar": str(ts), "target": target, "signal": sig}
            order = {"target": round(target, 4), "fills_at": "next bar open"}
        else:
            self.state["pending"] = None
            order = None
        return {"signal": sig, "target": round(target, 4), "vol_scale": round(scale, 4),
                "order": order}

    def _account(self):
        """Whatever turns a target into a fill. Same surface either way."""
        if self.routes_orders:
            return self.broker
        bps = self.config["cost_bps"] / 2
        return Account(cash=self.state["cash"], units=self.state["units"],
                       commission_bps=bps, slippage_bps=bps)

    # ---- reporting ----
    def forward_bars(self) -> pd.DataFrame:
        store = self.bars()
        start = self.state.get("forward_start")
        return store[store.index > pd.Timestamp(start)] if start else store

    def report(self) -> str:
        cfg, st = self.config, self.state
        store = self.bars()
        jf = self.journal_frame()
        bars_df = jf[jf.get("type") == "bar"] if not jf.empty and "type" in jf else pd.DataFrame()

        brk = cfg.get("broker") or {}
        if brk:
            where = ("ALPACA PAPER" if cfg.get("broker_paper", True) else "*** LIVE ACCOUNT ***")
            banner = f"ORDERS ROUTED → {where} {brk.get('symbol')}"
        else:
            banner = "NO ORDERS ARE ROUTED ANYWHERE"

        lines = [
            "",
            "=" * 78,
            f"  PAPER RUN  {cfg['run_id']}",
            f"  {banner}",
            f"  strategy   {cfg['strategy']}  {cfg['params']}",
            f"  feed       {cfg['feed'].get('kind')}  "
            f"{cfg.get('chart_symbol') or cfg['feed'].get('symbol') or cfg['feed'].get('ticker') or cfg['feed'].get('path','')}"
            f"  tf={cfg.get('chart_resolution') or cfg['feed'].get('timeframe') or cfg['feed'].get('interval','')}",
            f"  costs      {cfg['cost_bps']:.1f} bps round trip   vol target "
            f"{(cfg['vol_target'] or 0):.0%}   band {cfg['rebalance_band']:.2f}",
            f"  created    {cfg['created']}",
        ]
        if brk:
            lines += [
                f"  broker     account {cfg.get('broker_account')}  "
                f"{'fractional' if brk.get('fractional', True) else 'whole units'}  "
                f"{'long/short' if brk.get('allow_short') else 'long only'}",
            ]
            for warn in cfg.get("broker_warnings") or []:
                lines.append(f"             ! {warn}")
        lines.append("=" * 78)

        if store.empty or not st.get("forward_start"):
            lines += ["", "  no bars yet — run a poll", ""]
            return "\n".join(lines)

        fwd = self.forward_bars()
        age_days = (pd.Timestamp(st["last_bar"]) - pd.Timestamp(st["forward_start"])).total_seconds() / 86400
        px = float(store["close"].iloc[-1])
        equity = st["equity"]
        ret = equity / cfg["start_equity"] - 1.0

        lines += [
            "",
            f"  history seeded   {len(store) - len(fwd)} bars up to {st['forward_start']}",
            f"  forward bars     {len(fwd)}  ({age_days:.1f} days)  last {st['last_bar']}",
            f"  fills            {st.get('fills', 0)}",
            "",
            f"  equity           {equity:,.2f}   ({ret:+.2%})",
            f"  position         {st['units']:+.4f} units  =  "
            f"{(st['units'] * px / equity if equity else 0):+.2%} of equity",
            f"  cash             {st['cash']:,.2f}",
            f"  pending          {st.get('pending') or 'none'}",
        ]

        if brk:
            fills = self.fills_frame()
            unfilled = self.unfilled_frame()
            lines += ["", f"  routed to        {cfg.get('broker_account')} "
                          f"({'paper' if cfg.get('broker_paper', True) else 'LIVE'})"]
            if st.get("broker_equity") is not None:
                gap = st["broker_equity"] - equity
                lines.append(f"  account equity   {st['broker_equity']:,.2f}   "
                             f"(differs by {gap:+,.2f} — this run marks at the bar "
                             f"close and counts only {brk.get('symbol')})")
            if not fills.empty and "slippage_cost" in fills:
                slip = fills["slippage_cost"].astype(float)
                traded = fills["notional"].astype(float).sum()
                lines += [
                    f"  realized slip    {slip.sum():+,.2f} over {traded:,.0f} traded   "
                    f"({(slip.sum() / traded * 10_000 if traded else 0):+.2f} bps)",
                    f"                   the run assumes {cfg['cost_bps'] / 2:.1f} bps "
                    f"per side — that is the number this line is here to check",
                ]
            if not unfilled.empty:
                lines += ["", f"  UNFILLED         {len(unfilled)} order(s) the broker "
                              "did not fill — these are missing trades, not free passes:"]
                reasons = unfilled["why_not"].value_counts()
                for why, n in reasons.items():
                    lines.append(f"                   {n:>3} x {str(why)[:60]}")

        if not bars_df.empty and "equity" in bars_df and len(bars_df) > 2:
            eq = pd.Series(bars_df["equity"].astype(float).values,
                           index=pd.to_datetime(bars_df["bar"]))
            r = eq.pct_change().dropna()
            ppy = data_mod.periods_per_year(eq.index)
            lines += [
                "",
                f"  paper Sharpe     {metrics.sharpe(r, ppy):>6.2f}   "
                f"maxDD {metrics.max_drawdown(eq):>7.2%}   t {metrics.t_stat(r):>5.2f}",
            ]

            # the same bars through the vectorized engine, for the fill-model gap
            if len(fwd) > cfg["min_history"] // 2:
                strat = REGISTRY[cfg["strategy"]]
                costs = engine.CostModel(commission_bps=cfg["cost_bps"] / 2,
                                         slippage_bps=cfg["cost_bps"] / 2)
                shadow_src = store  # full history so the lookback is populated
                res = engine.run(shadow_src, strat.signal(shadow_src, **cfg["params"]),
                                 costs, vol_target=cfg["vol_target"],
                                 vol_lookback=cfg["vol_lookback"],
                                 max_leverage=cfg["max_leverage"],
                                 rebalance_band=cfg["rebalance_band"])
                sr = res.returns.loc[res.returns.index > pd.Timestamp(st["forward_start"])]
                if len(sr) > 2:
                    seq = (1 + sr).cumprod()
                    lines += [
                        f"  engine Sharpe    {metrics.sharpe(sr, res.ppy):>6.2f}   "
                        f"maxDD {metrics.max_drawdown(seq):>7.2%}   "
                        f"return {float(seq.iloc[-1] - 1):+.2%}",
                        "",
                        "  The engine line is the same strategy over the same bars with "
                        "close-to-close",
                        "  fills. The difference between the two rows is what the fill "
                        "assumption is worth.",
                    ]

        rt = self.round_trips()
        if not rt.empty:
            closed = rt[rt["exit"] != "OPEN"]
            lines += ["", f"  round trips      {len(closed)} closed"
                          + (f", 1 open" if len(rt) > len(closed) else "")]
            if not closed.empty:
                wins = closed[closed["pnl"] > 0]
                lines += [
                    f"  win rate         {len(wins) / len(closed):.1%}   "
                    f"avg {closed['pnl'].mean():+,.2f}   "
                    f"best {closed['pnl'].max():+,.2f}   "
                    f"worst {closed['pnl'].min():+,.2f}",
                    "",
                    "  last trades (paper.py trades --id <run> for all):",
                ]
                tail = closed.tail(5)[["entry", "exit", "side", "units", "pnl", "return"]]
                lines += ["    " + ln for ln in
                          tail.to_string(index=False).splitlines()]

        lines += ["", "-" * 78]
        if age_days < 90:
            lines += [
                f"  {age_days:.0f} days of forward data. This is not yet evidence of anything.",
                "  Months, and a trade count in the dozens at minimum, before the number",
                "  above deserves a second look. Do not change the parameters in the",
                "  meantime — that resets the clock and this run will refuse to continue.",
            ]
        else:
            lines += [
                f"  {age_days:.0f} days of forward data. Compare the trade count to what the",
                "  backtest predicted over the same span before reading anything into P&L.",
            ]
        lines += ["-" * 78, ""]
        return "\n".join(lines)


# ────────────────────────────── driver ──────────────────────────────

def poll_forever(run: PaperRun, interval: int, max_polls: int | None = None,
                 on_event=print) -> None:
    """Poll on a fixed interval until interrupted. One process, no threads.

    Restart-safe: everything needed to continue is on disk after each poll, so
    killing this and starting it again loses nothing but the sleep.
    """
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        try:
            result = run.poll()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # a bot meant to run for months cannot die on one bad fetch
            on_event(f"[{_now()}] {type(exc).__name__}: {exc}")
            run._journal({"type": "error", "at": _now(),
                          "error": f"{type(exc).__name__}: {exc}"})
            result = None
        if result:
            if result.get("new_bars"):
                for ev in result.get("events", []):
                    fill, miss = ev.get("fill"), ev.get("unfilled")
                    tail = ""
                    if fill:
                        tail = (f"  FILL {fill['side']} {fill['units']:+.4f} @ "
                                f"{fill['fill_price']:.2f}")
                        if fill.get("routed"):
                            tail += (f" (vs {fill['ref_price']:.2f} ref, "
                                     f"{fill['slippage_cost']:+.2f} slip)")
                    elif miss:
                        tail = f"  NO FILL — {miss.get('why_not')}"
                    on_event(f"[{ev['bar']}] close {ev['close']:.2f}  "
                             f"sig {ev['signal']:+.0f}  target {ev['target']:+.2f}  "
                             f"equity {ev['equity']:,.2f}" + tail)
            elif result.get("bootstrapped"):
                on_event(f"[{_now()}] seeded {result['bootstrapped']} history bars "
                         f"({result['from']} → {result['to']})")
            else:
                on_event(f"[{_now()}] no new closed bar "
                         f"(last {result.get('waiting_on')})")
        exhausted = getattr(run.feed, "exhausted", False)
        if exhausted:
            on_event(f"[{_now()}] feed exhausted — stopping")
            return
        if max_polls is None or polls < max_polls:
            time.sleep(interval)

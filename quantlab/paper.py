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

Execution (DESIGN-execution.md)
-------------------------------
A run created with an `execution` block in its config can send the order the
moment the decision exists instead of at the next close, apply the book governor's
multiplier to its target, and measure every routed fill against the open the
backtest assumed. A run without the block behaves exactly as it always has; every
new path below is gated on that key.

The one order that cannot go out the moment it exists is an equity's
market-on-open: Alpaca refuses one between 09:28 and 19:00 New York, and the
decision poll runs at 16:05. It is planned at decision time and sent by a later
poll inside the window the broker names (`opg_window`); a window missed because
no poll ran falls back to a market order once the session is open, marked late.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


# ────────────────────────────── execution fields ──────────────────────────────

# the §4 columns, in the order the report reads them
EXECUTION_KEYS = ("bar_price", "arrival_mid", "arrival_bid", "arrival_ask", "arrival_at",
                  "arrival_source", "arrival_spread_bps", "poll_price", "sent_at", "lag_s",
                  "fill_price", "delay_bps", "arrival_drift_bps", "slippage_bps",
                  "total_bps", "assumed_bps", "excess_bps")

# what a leg looks like in the journal — the broker's record carries more
LEG_KEYS = ("client_order_id", "order_id", "type", "limit_price", "status", "filled_qty",
            "avg_price", "latency_s", "cancelled")

STOPPED_FILE = "STOPPED"


def _bps(side_sign: float, earlier, later) -> float | None:
    """Signed so positive always means the run paid more (buy) or got less (sell)."""
    if earlier is None or later is None or not float(earlier):
        return None
    return round(side_sign * (float(later) - float(earlier)) / float(earlier) * 1e4, 1)


def _seconds_between(earlier, later) -> float | None:
    if not earlier or not later:
        return None
    try:
        a = datetime.fromisoformat(str(earlier).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(later).replace("Z", "+00:00"))
    except ValueError:
        return None
    a = a if a.tzinfo else a.replace(tzinfo=timezone.utc)
    b = b if b.tzinfo else b.replace(tzinfo=timezone.utc)
    return round((b - a).total_seconds(), 3)


def _num(value) -> float | None:
    return None if value is None else round(float(value), 6)


def _iso_utc(ts) -> str | None:
    """An instant as the journal spells it: UTC with the offset written out. The
    feed's bar index and broker.opg_window() are naive UTC; both come through here."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _parse_utc(value) -> datetime | None:
    if not value:
        return None
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


@contextlib.contextmanager
def _sigterm_exits():
    """SystemExit on SIGTERM while the body runs, so a poll stopped by systemd
    takes the same save path as any other exit. The broker installs its own
    handler around an order wait; it cancels the order and then calls this one.
    Off the main thread, or where the signal does not exist, nothing is installed."""
    def handler(signum, frame):
        raise SystemExit(128 + int(signum))

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError, AttributeError):
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def execution_fields(side_sign: float, bar_price, arrival: dict | None, poll_price,
                     fill_price, sent_at, cost_bps) -> dict:
    """What a routed fill cost, and where the cost came from (§4).

    `bar_price` is the open the backtest assumed, `arrival` the book when the
    decision was made, `poll_price` the mark the order was sent against and
    `fill_price` what it got. Every bps field is signed with `side_sign` (+1 buy,
    -1 sell) so that positive means worse for the run. Pure: no I/O, and a None
    anywhere stays a None rather than becoming a zero that looks like a number.
    """
    arrival = arrival or {}
    mid, bid, ask = arrival.get("mid"), arrival.get("bid"), arrival.get("ask")
    spread = (round((float(ask) - float(bid)) / float(mid) * 1e4, 1)
              if mid and bid is not None and ask is not None else None)
    total = _bps(side_sign, bar_price, fill_price)
    assumed = None if cost_bps is None else round(float(cost_bps) / 2, 4)
    return {
        "bar_price": _num(bar_price),
        "arrival_mid": _num(mid),
        "arrival_bid": _num(bid),
        "arrival_ask": _num(ask),
        "arrival_at": arrival.get("at"),
        "arrival_source": arrival.get("source"),
        "arrival_spread_bps": spread,
        "poll_price": _num(poll_price),
        "sent_at": sent_at,
        "lag_s": _seconds_between(arrival.get("at"), sent_at),
        "fill_price": _num(fill_price),
        "delay_bps": _bps(side_sign, bar_price, poll_price),
        "arrival_drift_bps": _bps(side_sign, mid, poll_price),
        "slippage_bps": _bps(side_sign, poll_price, fill_price),
        "total_bps": total,
        "assumed_bps": assumed,
        "excess_bps": (round(total - assumed, 1)
                       if total is not None and assumed is not None else None),
    }


def _sanitise_run_id(run_id: str) -> str:
    """A client_order_id is 48 chars at Alpaca: `{run}:{bar}:{leg}` has to fit."""
    clean = re.sub(r"[^A-Za-z0-9-]", "-", str(run_id))
    return clean[:48 - 3 - len(":20260101T0000")]


def _leg_summary(legs: list[dict] | None) -> list[dict]:
    return [{k: leg.get(k) for k in LEG_KEYS} for leg in (legs or [])]


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
    # set for the length of poll(dry_run=True): every write becomes a no-op
    _dry_run: bool = field(default=False, repr=False)

    # ---- files ----
    @property
    def bars_path(self) -> Path:
        return self.root / "bars.csv"

    @property
    def stopped_path(self) -> Path:
        return self.root / STOPPED_FILE

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
               note: str = "", broker_spec: dict | None = None,
               execution: dict | None = None) -> "PaperRun":
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
        if execution is not None:
            # how the order goes out, not what the strategy decides — so it is not
            # fingerprinted, and a run without it keeps the old path unchanged
            config["execution"] = execution
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
        if self._dry_run:
            return
        self.state["updated"] = _now()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _journal(self, record: dict) -> None:
        if self._dry_run:
            return
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # ---- stop marker (§8) ----
    @property
    def is_stopped(self) -> bool:
        return self.stopped_path.exists()

    def _stopped_marker(self) -> dict:
        try:
            return json.loads(self.stopped_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def stop(self, reason: str) -> dict:
        """Leave a STOPPED marker so the tick, the book and poll() all skip this run.
        Nothing else is touched: the journal, bars and state stay as the record."""
        marker = {"at": _now(), "reason": reason}
        tmp = self.stopped_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(marker, indent=2), encoding="utf-8")
        tmp.replace(self.stopped_path)
        self._journal({"type": "stopped", **marker})
        return marker

    def bars(self) -> pd.DataFrame:
        if self._bars is None:
            self._bars = (data_mod.load_csv(str(self.bars_path))
                          if self.bars_path.exists()
                          else pd.DataFrame(columns=BAR_COLS))
        return self._bars

    def _store_bars(self, df: pd.DataFrame, write: bool = True) -> pd.DataFrame:
        """Append genuinely new bars. The store is append-only like everything else
        here: a bar that has already been decided on is never rewritten, even if the
        feed later hands back a revised copy of it."""
        store = self.bars()
        fresh = df if store.empty else df[df.index > store.index[-1]]
        if fresh.empty:
            return store
        if not write:
            # a dry run sees the new bars without the store learning about them
            return fresh if store.empty else pd.concat([store, fresh])

        out = fresh.reset_index()
        out.columns = ["time"] + BAR_COLS
        header = not self.bars_path.exists()
        out.to_csv(self.bars_path, mode="a", header=header, index=False)
        self._bars = fresh if store.empty else pd.concat([store, fresh])
        return self._bars

    def _records(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        return [json.loads(line)
                for line in self.journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def journal_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._records())

    def execution_frame(self) -> pd.DataFrame:
        """Every routed fill that carries the §4 fields, one row each, plus the
        `execution_backfill` records reconstructed for fills from before the fields
        existed. A routed fill without them is left out rather than shown as a row
        of blanks — the backfill is how it gets in."""
        rows = []
        for rec in self._records():
            fill = rec.get("fill")
            if (rec.get("type") == "bar" and isinstance(fill, dict)
                    and fill.get("routed") and "delay_bps" in fill):
                rows.append({"bar": rec["bar"], **fill})
            elif rec.get("type") == "execution_backfill":
                rows.append({k: v for k, v in rec.items() if k not in ("type", "at", "note")})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        lead = [c for c in ("bar", "decided_at", "side", "units", "order_style",
                            *EXECUTION_KEYS, "legs") if c in df.columns]
        return df[lead + [c for c in df.columns if c not in lead]]

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

        `bar` means two things, told apart by `fill_bar`. A row written at the fill
        bar — step 1 of `_process_bar`, or a legacy `rebalance()` refusal — has
        `bar` = the bar the order would have filled at. A row written by
        `_reconcile_inflight`, for a decision-time order that died (cancelled,
        expired, rejected) before its fill bar arrived, has `bar` = the decided
        bar, because the fill bar is not known yet. `decided_at` is the decided
        bar in both; `fill_bar` is set where it is known and None where it is not.
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

    @property
    def _exec(self) -> dict:
        """The §2 block. Empty for a legacy run, and everything new keys off that."""
        return self.config.get("execution") or {}

    @property
    def routes_at_decision(self) -> bool:
        return self.routes_orders and self._exec.get("route_at") == "decision"

    @property
    def uses_book(self) -> bool:
        return bool(self._exec.get("book"))

    # ---- the loop ----
    def poll(self, dry_run: bool = False) -> dict:
        """One iteration: fetch, fill the pending order, decide, journal.

        `dry_run` does the same work in memory and writes nothing — no bars, no
        journal, no state, no orders — and returns what it would have done. State
        is saved at the end of every real poll, new bar or not, so `updated` is a
        heartbeat; a SIGTERM anywhere in the poll is turned into SystemExit so it
        still gets one and the next tick can resume."""
        if self.is_stopped:
            return {"stopped": True, "run_id": self.config["run_id"], **self._stopped_marker()}
        self._dry_run = dry_run
        try:
            with _sigterm_exits():
                result = self._poll()
            self._save_state()
        except SystemExit:
            self._save_state()
            raise
        finally:
            self._dry_run = False
        return result

    def _poll(self) -> dict:
        if self.routes_orders:
            # Once per poll, before anything reads cash or units. A freshly loaded
            # broker knows nothing about the account until it is asked, and a run
            # resumed in a new process would otherwise mark itself at zero.
            self.broker.refresh()
        if self.routes_at_decision and self.state.get("inflight") and not self._dry_run:
            self._reconcile_inflight()

        fetched = self.feed.bars()
        if fetched.empty:
            return {"new_bars": 0, "note": "feed returned nothing"}

        last_bar = pd.Timestamp(self.state["last_bar"]) if self.state["last_bar"] else None
        store = self._store_bars(fetched, write=not self._dry_run)

        if last_bar is None:
            return self._bootstrap(store)

        new = store[store.index > last_bar]
        if new.empty:
            sent = None
            if self.routes_at_decision and not self._dry_run:
                # the fill bar is still ahead: a planned on-open order can go out
                # now, and a send that failed last poll can be tried again
                sent = self._send_deferred() or self._retry_route(store)
            if self.routes_orders:
                # pick up anything that moved the account while we were not looking:
                # a manual trade, a late fill, a margin action
                self._mirror_broker(float(store["close"].iloc[-1]))
            out = {"new_bars": 0, "waiting_on": str(store.index[-1]),
                   "pending": self.state["pending"]}
            if sent:
                out["order_sent"] = sent
            return self._dry_note(out)

        # Only the newest bar is actionable. If several arrived at once the loop is
        # catching up, and the opens the older decisions would have filled at are
        # gone — routing them now would send a market order on a price that is
        # already history. The simulated account has no such problem and fills them
        # all, exactly as it did before.
        newest = new.index[-1]
        events = [self._process_bar(store, ts, stale=(self.routes_orders and ts != newest))
                  for ts in new.index]
        return self._dry_note({"new_bars": len(events), "events": events})

    def _dry_note(self, out: dict) -> dict:
        if self._dry_run:
            out["dry_run"] = True
            if self.state.get("inflight"):
                out["inflight"] = self.state["inflight"]
        return out

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
        out = {"new_bars": 0, "bootstrapped": len(store), "from": str(store.index[0]),
               "to": str(ts), **decision}
        if self.routes_at_decision and self.state.get("pending"):
            self._save_state()                  # the seed is on disk before any routing I/O
            out["order_sent"] = self._route_guarded(ts, decision, float(store["close"].iloc[-1]))
        return out

    def _process_bar(self, store: pd.DataFrame, ts: pd.Timestamp,
                     stale: bool = False) -> dict:
        bar = store.loc[ts]
        acct = self._account()
        pending = self.state.get("pending")

        # 1. yesterday's decision fills at this bar's open
        fill, unfilled, would_send = None, None, None
        if pending is not None:
            inflight = self.state.get("inflight")
            if self.routes_at_decision:
                # the order went out when the decision was made (§3). This open is
                # the price the backtest assumed for it, not where it fills.
                if not inflight:
                    inflight = self._adopt_orphans(pending)
                if inflight and inflight.get("bar") == pending["bar"]:
                    fill, unfilled = self._settle_inflight(inflight, pending, float(bar["open"]))
                else:
                    err = pending.get("route_error")
                    unfilled = {"routed": False, "decided_at": pending["bar"],
                                "target": pending["target"], "ref_price": float(bar["open"]),
                                "why_not": (f"the send failed ({err}) and this open has since "
                                            "passed, so no order went out" if err else
                                            "catch-up bar — the decision was made after this "
                                            "open had passed, so no order was sent")}
                if unfilled:
                    unfilled["fill_bar"] = str(ts)
                self.state["inflight"] = None
            elif stale:
                unfilled = {"routed": False, "decided_at": pending["bar"],
                            "target": pending["target"], "ref_price": float(bar["open"]),
                            "why_not": "catch-up bar — this open has already passed, so "
                                       "the order was not sent"}
            elif self._dry_run and self.routes_orders:
                # a legacy routed run would send a market order here; show it instead
                would_send = self._would_rebalance(float(bar["open"]), float(pending["target"]))
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
        if would_send:
            record["would_send"] = would_send
        # Routed at decision time: the bar record and the pending order are on disk
        # first, then the order goes out and gets its own `order_sent` line. A crash
        # mid-wait, or a quote that fails on the way, can lose neither — and cannot
        # get this bar decided twice. A catch-up bar's decision is not sent — its
        # open has passed.
        if self.routes_at_decision and self.state.get("pending") and not stale:
            self._save_state()
            record["order_sent"] = self._route_guarded(ts, decision, close)
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

        book = {}
        if self.uses_book:
            # the book governor (§6): one multiplier for every run, applied before
            # the band so a scaled-down target is what the band is measured against
            target_raw = target
            k, meta = self._book_multiplier()
            target = target_raw * k
            self.state["target_raw"] = target_raw
            self.state["book_k"] = k
            book = {"target_raw": round(target_raw, 4), **meta}

        band = cfg["rebalance_band"]
        move = abs(target - held) > band or np.sign(target) != np.sign(held)
        if move:
            self.state["pending"] = {"bar": str(ts), "target": target, "signal": sig}
            order = {"target": round(target, 4), "fills_at": "next bar open"}
        else:
            self.state["pending"] = None
            order = None
        return {"signal": sig, "target": round(target, 4), "vol_scale": round(scale, 4),
                "order": order, **book}

    def _book_multiplier(self) -> tuple[float, dict]:
        """k from paper_runs/book.json, or 1.0 with a reason that is never quiet."""
        from . import book as book_mod       # book imports this module

        base = self.root.parent
        tick_s = int(self._exec.get("tick_s") or 300)
        k, meta = book_mod.book_multiplier(base, self.config["run_id"], tick_s)
        if meta["book_stale"]:
            msg = (f"{self.config['run_id']}: book multiplier not applied (k=1.0) — "
                   f"{meta['book_reason']}")
            if self._dry_run:
                print(f"ALERT {_now()} {msg} [dry run — not logged]", file=sys.stderr)
            else:
                book_mod.alert(base, msg)
        return k, meta

    # ---- routing at decision time (§3, §5) ----
    def _prefix(self, bar) -> str:
        """`{run}:{bar}` — every leg of the decision made at `bar` hangs off it."""
        return f"{_sanitise_run_id(self.config['run_id'])}:{pd.Timestamp(bar):%Y%m%dT%H%M}"

    def _route_guarded(self, ts: pd.Timestamp, decision: dict, close: float,
                       retry: bool = False) -> dict:
        """_route_now with its failure written down instead of thrown.

        The bar record and the pending order are already on disk when this runs,
        so a quote or a clock that fails here must not unwind the poll — the next
        one would decide and journal the same bar again. The error gets its own
        line, the pending order keeps the reason, and the next poll looks for the
        order by client id before it considers sending (`_retry_route`).
        """
        try:
            return self._route_now(ts, decision, close, retry=retry)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            inflight = self.state.get("inflight")
            if inflight and inflight.get("bar") == str(ts):
                inflight["error"] = err            # the intent is on disk; §5 finds its legs
            elif self.state.get("pending"):
                self.state["pending"]["route_error"] = err
            self._journal({"type": "error", "at": _now(), "bar": str(ts), "stage": "route",
                           "retry": retry, "error": err})
            self._save_state()
            return {"sent": False, "error": err}

    def _route_now(self, ts: pd.Timestamp, decision: dict, close: float,
                   retry: bool = False) -> dict:
        """Send the pending order now, the way the run's `execution` block says.

        The journal gets an `order_sent` line and state an `inflight` payload
        before anything waits — a poll killed mid-wait can find its own order again
        by client_order_id (§5). What comes back is only a summary for the caller;
        the fill is written by `_settle_inflight` at the next bar, when the open it
        is measured against is known.

        A market-on-open order is the exception. Alpaca rejects one that arrives
        between 09:28 and 19:00 New York, and the decision poll for an equity runs
        at 16:05. So it is planned here — legs, client ids, the window the broker
        will take it in — journaled as `order_planned`, and `_send_deferred` sends
        it on the first poll inside that window. `order_sent` is written only when
        something actually went.
        """
        exe, broker = self._exec, self.broker
        pending = self.state["pending"]
        target = float(pending["target"])
        style = exe.get("style") or ("marketable_limit" if broker.is_crypto else "market_on_open")
        opg = style == "market_on_open"
        limit_bps = float(exe.get("limit_bps", 5.0))
        prefix = self._prefix(ts)

        arrival = broker.quote()
        plan = broker._plan(target, close, opg=opg)
        planned, note, window = [], {}, {}
        if plan is None:
            note = {"routed": None,
                    "why_not": "move is below the broker's minimum notional — nothing to send"}
        elif plan.get("routed") is False:
            note = {"routed": False, "why_not": plan.get("why_not"),
                    "wanted_units": plan.get("wanted_units")}
        else:
            for n, units in enumerate(plan["legs"]):
                planned.append({
                    "client_order_id": broker._client_id(prefix, n),
                    "side": "buy" if units > 0 else "sell",
                    "qty": broker._leg_qty(units, opg=opg),
                    "type": "limit" if style == "marketable_limit" else ("opg" if opg else "market"),
                    "limit_price": (broker._limit_price(arrival["mid"] or plan["price"], units, limit_bps)
                                    if style == "marketable_limit" else None),
                    **({"status": "deferred"} if opg else {}),
                })
            note = {"routed": True, "mark_price": round(plan["price"], 6),
                    "wanted_units": round(plan["delta"], 8),
                    "units_before": round(plan["units_before"], 8)}
            if opg:
                # the quantities above are sized off today's mark; place() sizes
                # again off the mark at send time, against the same target
                not_before, not_after = broker.opg_window()
                window = {"deferred": True, "send_not_before": _iso_utc(not_before),
                          "send_not_after": _iso_utc(not_after)}
        arrival_fields = {f"arrival_{k}": arrival.get(k) for k in ("mid", "bid", "ask", "at", "source")}
        arrival_fields["arrival_spread_bps"] = execution_fields(1, None, arrival, None, None,
                                                               None, None)["arrival_spread_bps"]
        summary = {"route_at": "decision", "order_style": style, "client_order_prefix": prefix,
                   "client_order_ids": [l["client_order_id"] for l in planned],
                   "limit_price": planned[0]["limit_price"] if planned else None,
                   "legs": planned, **note, **window}
        if retry:
            summary.update(retry=True, retry_after=pending.get("route_error"))
        if self._dry_run:
            return {"would_send": {**summary, "quote": arrival}}

        self._journal({"type": "order_planned" if opg else "order_sent", "at": _now(),
                       "bar": str(ts), "target": round(target, 4), "signal": decision.get("signal"),
                       "decision_close": close, **arrival_fields,
                       **{k: v for k, v in summary.items() if k != "legs"},
                       "planned_legs": planned})
        inflight = {"bar": str(ts), "target": target, "client_order_prefix": prefix,
                    "order_style": style, "arrival": arrival, "decision_close": close,
                    "planned_legs": planned, "legs": planned if opg else [], "sent_at": None,
                    "poll_price": note.get("mark_price"), **note, **window}
        pending.pop("route_error", None)
        self.state["inflight"] = inflight
        self._save_state()                          # before anything waits (§3.4)
        if not note.get("routed"):
            return {"sent": False, "why_not": note.get("why_not"), "order_style": style}
        if opg:
            # nothing goes out now. The same poll sends it if the window is already
            # open (a decision made late in the evening); otherwise a later one does.
            out = {"sent": False, "order_style": style,
                   "client_order_ids": summary["client_order_ids"], **window}
            return {**out, **(self._send_deferred() or {})}

        try:
            result = broker.place(target, close, style=style, client_order_prefix=prefix,
                                  limit_bps=limit_bps,
                                  limit_wait_s=float(exe.get("limit_wait_s", 60.0)),
                                  fallback=exe.get("fallback", "market"))
        except broker_mod.BrokerError as exc:
            # a leg may or may not have reached the broker; legs stays empty so the
            # next poll looks for them by client id rather than trusting either
            inflight["error"] = str(exc)
            self._save_state()
            return {"sent": False, "why_not": str(exc), "order_style": style}
        return self._placed(inflight, result)

    def _placed(self, inflight: dict, result: dict | None) -> dict:
        """Fold what place() returned into the inflight payload, save, summarise."""
        if result is None:
            inflight.update(routed=None, why_not="move is below the broker's minimum notional")
        else:
            inflight.update({k: result[k] for k in
                             ("legs", "order_style", "poll_price", "sent_at", "units_before",
                              "wanted_units", "open", "routed", "why_not", "leg_error", "clamped")
                             if k in result})
        self._save_state()
        legs = inflight.get("legs") or []
        out = {"sent": any(l.get("order_id") for l in legs),
               "order_style": inflight.get("order_style"),
               "client_order_ids": [l.get("client_order_id") for l in legs],
               "statuses": [l.get("status") for l in legs],
               "poll_price": inflight.get("poll_price"), "sent_at": inflight.get("sent_at"),
               "open": bool(inflight.get("open"))}
        if inflight.get("routed") is False:
            out["why_not"] = inflight.get("why_not")
        if inflight.get("late"):
            out["late"] = True
        return out

    def _send_deferred(self) -> dict | None:
        """A planned market-on-open order goes out here, on the first poll inside
        its window (§3.3). Past the window with nothing sent — no poll ran, the box
        was down — it goes as a plain market order once the session is open, and
        the fill is marked late. Only called while the fill bar is still ahead;
        once that bar has closed, `_settle_inflight` writes the plan off as never
        sent instead."""
        inflight = self.state.get("inflight")
        if not inflight or not inflight.get("deferred"):
            return None
        window = {k: inflight.get(k) for k in ("send_not_before", "send_not_after")}
        now = datetime.now(timezone.utc)
        not_before, not_after = _parse_utc(window["send_not_before"]), _parse_utc(window["send_not_after"])
        if now < not_before:
            return {"sent": False, "deferred": True, "order_style": "market_on_open",
                    "opens_in_s": round((not_before - now).total_seconds()), **window}
        late = now >= not_after
        if late and not self.broker.market_open():
            return {"sent": False, "deferred": True, "late": True, **window,
                    "why_not": "the on-open window was missed and the market is closed — "
                               "a market order goes out once the session opens"}
        exe = self._exec
        style = "market" if late else "market_on_open"
        # the plan is spent whatever place() says next; a poll that dies inside it
        # finds the legs by client id (§5) rather than sending them twice
        inflight.update(deferred=False, send_attempted_at=_now(), legs=[])
        if late:
            inflight["late"] = True
        self._save_state()
        try:
            result = self.broker.place(float(inflight["target"]), inflight.get("decision_close"),
                                       style=style,
                                       client_order_prefix=inflight["client_order_prefix"],
                                       limit_bps=float(exe.get("limit_bps", 5.0)),
                                       limit_wait_s=float(exe.get("limit_wait_s", 60.0)),
                                       fallback=exe.get("fallback", "market"))
        except broker_mod.BrokerError as exc:
            inflight["error"] = str(exc)
            self._save_state()
            return {"sent": False, "why_not": str(exc), "order_style": style, "late": late}
        out = self._placed(inflight, result)
        if out["sent"]:
            self._journal({"type": "order_sent", "at": _now(), "bar": inflight["bar"],
                           "target": round(float(inflight["target"]), 4),
                           "client_order_prefix": inflight["client_order_prefix"],
                           "client_order_ids": out["client_order_ids"],
                           "order_style": out["order_style"], "poll_price": out["poll_price"],
                           "sent_at": out["sent_at"], "late": late, **window,
                           "planned_legs": inflight.get("planned_legs"),
                           "legs": _leg_summary(inflight.get("legs"))})
        return out

    def _retry_route(self, store: pd.DataFrame) -> dict | None:
        """A pending decision with nothing in flight — its send failed, or died
        before the intent was saved — is routed again while its fill bar is still
        ahead. The broker is asked for the legs first: if they exist under this
        decision's client ids they are adopted, and nothing is sent."""
        pending = self.state.get("pending")
        if (not pending or self.state.get("inflight")
                or pending.get("bar") != self.state.get("last_bar")):
            return None
        ts = pd.Timestamp(pending["bar"])
        if self._adopt_orphans(pending):
            return {"sent": True, "recovered": True,
                    "client_order_ids": [l.get("client_order_id")
                                         for l in self.state["inflight"]["legs"]]}
        return self._route_guarded(ts, {"signal": pending.get("signal")},
                                   float(store.loc[ts, "close"]), retry=True)

    _STYLE_OF_LEG = {"opg": "market_on_open", "limit": "marketable_limit", "market": "market"}

    def _adopt_orphans(self, pending: dict) -> dict | None:
        """Legs the broker holds under this decision's prefix that state never
        recorded — a poll that died after sending and before saving. Adopted into
        `inflight` and journaled as sent, never re-sent. None when there are none."""
        prefix = self._prefix(pending["bar"])
        legs = self._find_legs(prefix)
        if not legs:
            return None
        inflight = {"bar": pending["bar"], "target": float(pending["target"]),
                    "client_order_prefix": prefix,
                    "order_style": self._STYLE_OF_LEG.get(legs[0]["type"], legs[0]["type"]),
                    "arrival": None, "decision_close": None, "planned_legs": [], "legs": legs,
                    "sent_at": None, "poll_price": None, "routed": True, "recovered": True,
                    "open": any(l.get("open") for l in legs)}
        self._journal({"type": "order_sent", "at": _now(), "bar": pending["bar"],
                       "target": round(float(pending["target"]), 4), "recovered": True,
                       "client_order_prefix": prefix, "order_style": inflight["order_style"],
                       "client_order_ids": [l["client_order_id"] for l in legs],
                       "note": "found at the broker under this run's client ids after a poll "
                               "died between sending and saving — adopted, not re-sent"})
        pending.pop("route_error", None)
        self.state["inflight"] = inflight
        self._save_state()
        return inflight

    def _reconcile_inflight(self) -> None:
        """§5, at the top of every poll: an order sent at decision time and not yet
        accounted for is re-queried, never re-sent. Filled or still open — leave it
        for step 1 of the fill bar. Dead with nothing filled — say so now. A
        market-on-open still waiting for its window is not touched here; that is
        `_send_deferred`, once the poll knows the fill bar is still ahead."""
        inflight = self.state["inflight"]
        if inflight.get("deferred"):
            return
        legs = inflight.get("legs") or []
        if not legs and inflight.get("routed") is True and inflight.get("client_order_prefix"):
            # the poll died between persisting the intent and place() returning.
            # Whatever reached the broker carries our client ids: look for them.
            legs = self._find_legs(inflight["client_order_prefix"])
            inflight["legs"] = legs
            if not legs:
                # nothing did. The intent is dropped and the pending order keeps the
                # reason, so the poll can route it again — after this lookup, never
                # instead of it — while its fill bar is still ahead.
                pending = self.state.get("pending")
                if pending and pending.get("bar") == inflight["bar"]:
                    pending["route_error"] = (
                        inflight.get("error") or "no order carrying this client_order_prefix "
                                                 "at the broker — nothing was sent")
                self.state["inflight"] = None
                self._save_state()
                return
            self._save_state()
        if not legs:
            return
        res = self.broker.resolve(legs, poll_price=inflight.get("poll_price"),
                                  units_before=inflight.get("units_before"))
        inflight.update(legs=res["legs"], open=res["open"], last_resolved=_now())
        if res["open"] or res["filled_units"]:
            return
        pending = self.state.get("pending") or {}
        self._journal({
            "type": "unfilled", "at": _now(), "bar": inflight["bar"],
            "unfilled": {"routed": False, "decided_at": inflight["bar"], "fill_bar": None,
                         "target": pending.get("target", inflight.get("target")),
                         "ref_price": inflight.get("poll_price"),
                         "why_not": f"order {', '.join(res['statuses'])} — nothing filled",
                         "order_style": inflight.get("order_style"),
                         "legs": _leg_summary(res["legs"])},
        })
        self.state["unfilled"] = self.state.get("unfilled", 0) + 1
        self.state["inflight"] = None
        if pending.get("bar") == inflight["bar"]:
            self.state["pending"] = None
        self._save_state()                          # the journal line and this agree, whatever the feed does next

    def _find_legs(self, prefix: str) -> list[dict]:
        """The legs the broker has under `prefix`, rebuilt from its own records."""
        legs = []
        for n in range(4):                          # a flip with fallbacks is four
            cid = f"{prefix}:{n}"
            order = self.broker._existing(cid)
            if order is None:
                break
            tif = str(getattr(order, "time_in_force", "") or "").lower()
            kind = str(getattr(order, "order_type", getattr(order, "type", "")) or "").lower()
            status = broker_mod._status(order)
            legs.append({
                "client_order_id": cid, "order_id": str(order.id),
                "type": "opg" if tif.endswith("opg") else ("limit" if "limit" in kind else "market"),
                "side": "sell" if str(getattr(order, "side", "")).lower().endswith("sell") else "buy",
                "qty": broker_mod._f(getattr(order, "qty", 0)),
                "limit_price": broker_mod._r6(getattr(order, "limit_price", None)),
                "status": status,
                "filled_qty": broker_mod._f(getattr(order, "filled_qty", 0)),
                "avg_price": broker_mod._f(getattr(order, "filled_avg_price", 0) or 0) or None,
                "latency_s": None, "cancelled": False,
                "open": status not in broker_mod._TERMINAL, "resumed": True,
            })
        return legs

    def _settle_inflight(self, inflight: dict, pending: dict,
                         bar_open: float) -> tuple[dict | None, dict | None]:
        """Step 1 of the fill bar for a decision-time order: (fill, unfilled)."""
        base = {"routed": False, "decided_at": pending["bar"], "target": pending["target"],
                "ref_price": bar_open}
        legs = inflight.get("legs") or []
        if inflight.get("deferred"):
            # planned for the open and never sent: no poll ran inside the window
            self.state["unfilled"] = self.state.get("unfilled", 0) + 1
            return None, {**base, "order_style": inflight.get("order_style"),
                          "why_not": "market-on-open planned but never sent — no poll ran "
                                     f"between {inflight.get('send_not_before')} and "
                                     f"{inflight.get('send_not_after')}",
                          "legs": _leg_summary(legs)}
        if not legs:
            if inflight.get("routed") is None:      # below min notional: nothing to do
                return None, None
            self.state["unfilled"] = self.state.get("unfilled", 0) + 1
            return None, {**base, "why_not": inflight.get("why_not") or inflight.get("error")
                          or "nothing was sent"}

        res = self.broker.resolve(legs, poll_price=inflight.get("poll_price"),
                                  units_before=inflight.get("units_before"))
        legs = _leg_summary(res["legs"])
        if not res["filled_units"]:
            why = f"order {', '.join(res['statuses'])} — nothing filled"
            if res["open"]:
                why += (" (still open at the next bar; a later fill reaches the account, "
                        "not the ledger)")
            self.state["unfilled"] = self.state.get("unfilled", 0) + 1
            return None, {**base, "why_not": why, "order_style": inflight.get("order_style"),
                          "legs": legs}

        units = float(res["units"])
        side_sign = 1.0 if units > 0 else -1.0
        # the mark the order was sized and sent against; the legacy key stays too
        poll_price = inflight.get("poll_price")
        if poll_price is None:
            poll_price = inflight.get("mark_price") or res["fill_price"]
        fill = {
            "routed": True, "side": res["side"], "units": round(units, 8),
            "ref_price": round(float(poll_price), 6), "bar_price": round(bar_open, 6),
            "fill_price": res["fill_price"], "notional": res["notional"],
            "commission": 0.0, "slippage_cost": res["slippage_cost"],
            "order_ids": res["order_ids"],
            "latency_s": round(sum(l.get("latency_s") or 0 for l in res["legs"]), 2),
            "broker_equity": res["broker_equity"], "broker_units": res["broker_units"],
        }
        for key in ("order_units", "in_kind_fee_units"):
            if key in res:
                fill[key] = res[key]
        wanted = inflight.get("wanted_units")
        if wanted is not None and abs(float(wanted) - units) > 1e-9:
            fill["short_by"] = round(float(wanted) - units, 8)
        if inflight.get("clamped"):
            fill["clamped"] = inflight["clamped"]
        if res["open"]:
            fill["partial"] = True
        if inflight.get("late"):
            fill["late"] = True                     # the on-open window was missed
        fill.update(execution_fields(side_sign, bar_open, inflight.get("arrival"), poll_price,
                                     res["fill_price"], inflight.get("sent_at"),
                                     self.config["cost_bps"]))
        fill.update({"order_style": inflight.get("order_style"), "legs": legs,
                     "decided_at": pending["bar"]})
        self.state["fills"] = self.state.get("fills", 0) + 1
        return fill, None

    def _would_rebalance(self, ref_price: float, target: float) -> dict:
        """What a legacy routed run's market order at this open would be. Reads only."""
        plan = self.broker._plan(target, ref_price)
        out = {"route_at": "next_close", "order_style": "market"}
        if plan is None:
            return {**out, "why_not": "move is below the broker's minimum notional"}
        if plan.get("routed") is False:
            return {**out, **plan}
        return {**out, "mark_price": round(plan["price"], 6),
                "wanted_units": round(plan["delta"], 8),
                "legs": [{"side": "buy" if u > 0 else "sell", "qty": self.broker._leg_qty(u)}
                         for u in plan["legs"]]}

    # ---- backfill (§10) ----
    def backfill_execution(self) -> dict:
        """The §4 fields for the one routed fill made before they were recorded,
        rebuilt from its journal line and bars.csv. Returned, not written — the CLI
        appends it only with --append, and only once."""
        recs = self._records()
        if any(r.get("type") == "execution_backfill" for r in recs):
            raise ValueError("this journal already has an execution_backfill record")
        fills = [r for r in recs if r.get("type") == "bar" and isinstance(r.get("fill"), dict)
                 and r["fill"].get("routed") and "delay_bps" not in r["fill"]]
        if len(fills) != 1:
            raise ValueError(f"expected exactly one routed fill without execution fields, "
                             f"found {len(fills)}")
        rec = fills[0]
        fill = rec["fill"]
        decided_at = fill["decided_at"]
        bars = self.bars()
        # the run had no quote when it decided: the decided bar's close stands in
        # for arrival, stamped at the instant that bar closed — not when the poll
        # got round to it, because the gap between those two is the lag measured
        ts = pd.Timestamp(decided_at)
        arrival_close = float(bars.loc[ts, "close"])
        closed_at = getattr(self.feed, "closed_at", None)
        if closed_at is not None:
            bar_close = closed_at(ts)
        else:
            later = bars.index[bars.index > ts]     # the next bar opens as this one closes
            bar_close = later[0] if len(later) else None
        decided_rec = next((r for r in recs if r.get("type") in ("bar", "backfill")
                            and r.get("bar") == decided_at), None)
        arrival = {"mid": arrival_close, "bid": None, "ask": None,
                   "at": _iso_utc(bar_close), "source": "backfill:decision_close"}
        latency = float(fill.get("latency_s") or 0.0)
        sent_at = (datetime.fromisoformat(rec["at"]) - timedelta(seconds=latency)).isoformat()
        side_sign = 1.0 if fill["side"] == "buy" else -1.0
        order_ids = fill.get("order_ids") or []
        return {
            "type": "execution_backfill", "at": _now(), "bar": rec["bar"],
            "decided_at": decided_at, "side": fill["side"], "units": fill["units"],
            "processed_at": decided_rec["at"] if decided_rec else None,
            "order_style": "market", "poll_source": "backfill:last_trade",
            **execution_fields(side_sign, fill["bar_price"], arrival, fill["ref_price"],
                               fill["fill_price"], sent_at, self.config["cost_bps"]),
            "legs": [{"client_order_id": None, "order_id": order_ids[0] if order_ids else None,
                      "type": "market", "limit_price": None, "status": "filled",
                      "filled_qty": fill.get("order_units", fill["units"]),
                      "avg_price": fill["fill_price"], "latency_s": fill.get("latency_s"),
                      "cancelled": False}],
            "note": "reconstructed from the fill record and bars.csv — arrival is the "
                    "decided bar's close at the instant it closed, processed_at when the "
                    "poll saw it, poll_price the last trade the order was sized on",
        }

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
            if result.get("stopped"):
                on_event(f"[{_now()}] {run.config['run_id']} is STOPPED "
                         f"({result.get('reason')}) — nothing to do")
                return
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

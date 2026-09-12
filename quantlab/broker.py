"""Order routing.

paper.py's Account is a simulation: it decides what a fill would have cost and
writes that down. This module is the alternative — it sends the order to a broker
and records what actually came back. The strategy code, the sizing, the bar feed
and the append-only journal are unchanged; only the thing that turns a target
position into a fill is swapped.

What that buys you, and what it does not
----------------------------------------
It buys you the one number the simulation has to guess: the real fill price. Every
routed fill records `ref_price` (the bar open the simulation would have used) next
to `fill_price` (what the broker actually gave you), so the cost of the fill
assumption stops being an estimate. It also buys you the failures a simulation
cannot have — rejected orders, insufficient buying power, an order that sat
unfilled because the market was closed. Those are journaled, not smoothed over.

It does not make the result a live-trading result. An Alpaca paper account is a
simulator on their side: your order does not reach an exchange, it is filled
against the quote without moving it, and there is no queue position, no partial
fill from thin liquidity, and no borrow to locate on a short. It is a better fill
model, not a real one.

The traded instrument is not the charted one
--------------------------------------------
The bar feed and the broker symbol are deliberately separate. The whole reason to
route at all is usually that the strategy was developed on a chart you cannot
trade — NQ1! bars, QQQ orders. That means the bar's price is the wrong number to
size with: a target of "40% of equity" turned into units at NQ's ~20,000 would buy
a thirtieth of the QQQ position it asked for. So the broker marks its own
instrument. It reads the traded symbol's own last price, sizes with that, and
records the feed's bar price beside it as `ref_price` for the audit trail only.

Two ways in
-----------
`rebalance(ref_price, target)` is the legacy path: a market order at the next
bar's close, waited out. `place(target, ref_price, style=...)` is the path for
runs that route at decision time (DESIGN-execution.md §3): the same sizing and
the same legs, but the order goes out the moment the decision is made, as a
marketable limit (crypto), a market-on-open (equities) or a plain market order,
and every leg carries a client_order_id the run can find again after a crash
with `resolve(legs)`. Both return the same fill dict; `place` adds the execution
fields the report needs.

Safety
------
- Refuses a live (non-paper) account unless allow_live is passed explicitly.
- Refuses a symbol the account cannot actually trade, at creation, not at 3am.
- Never routes a catch-up bar. If the loop resumes after a gap, the opens those
  decisions would have filled at are gone; paper.py marks them and moves on.
- Market orders are not submitted while the market is closed. A market order sent
  on Saturday fills at Monday's open on a decision made from Friday's close, which
  is not the trade the strategy asked for. Set require_open=False to allow it.
  Market-on-open orders are exempt: being sent while closed is the point —
  but not at any hour; opg_window() says when Alpaca will take one.
- Nothing is sent on top of a live order. When a wait runs out the order is
  cancelled and the cancel is confirmed by re-query before any fallback goes
  out; a SIGTERM mid-wait cancels the open leg before the process dies.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# order states that will never fill any more than they already have
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day",
             "stopped", "suspended", "replaced"}

_STYLES = ("market", "marketable_limit", "market_on_open")
_CLIENT_ID_MAX = 48     # Alpaca's limit on client_order_id


class BrokerError(RuntimeError):
    pass


def _load_env() -> None:
    """Pull APCA_* out of a .env if it is sitting next to the package."""
    if os.getenv("APCA_API_KEY_ID"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (Path.cwd() / ".env",
                      Path(__file__).resolve().parents[1] / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _r6(value) -> float | None:
    return None if value is None else round(value, 6)


def _status(order) -> str:
    return str(getattr(order, "status", "")).split(".")[-1].lower()


def _not_found(exc) -> bool:
    """True when an APIError means the thing does not exist, not that the call failed.

    alpaca-py raises APIError(response.text, HTTPError): `status_code` reads the
    HTTP status off the wrapped HTTPError (None when there is none), and the
    body for a missing order is {"code":40410000,"message":"order not found"}.
    Both are checked, so an APIError built from the body alone still classifies.
    """
    try:
        if exc.status_code == 404:
            return True
    except Exception:
        pass
    return "order not found" in str(exc).lower()


def _iso(ts) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()
    return str(ts)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vwap(done: list[dict]) -> tuple[float, float, float]:
    """(signed units, gross notional, average price) over the legs that filled."""
    ordered = sum(f["filled_units"] for f in done)
    gross = sum(abs(f["filled_units"]) * f["avg_price"] for f in done)
    return ordered, gross, gross / sum(abs(f["filled_units"]) for f in done)


@dataclass
class AlpacaBroker:
    """Routes the paper run's target position to an Alpaca account.

    Presents the same surface as paper.Account — cash, units, equity(price),
    fraction(price), rebalance(price, target) — so the run loop does not know or
    care which one it is holding. The difference is that cash and units are not
    ours to compute: they are read back from the broker on every refresh, and the
    broker is the authority. If you trade the same account by hand, this will see
    it and the run's equity will move accordingly.
    """

    symbol: str
    paper: bool = True
    allow_live: bool = False
    fractional: bool = True
    allow_short: bool = False
    require_open: bool = True
    fill_timeout: float = 90.0
    poll_every: float = 1.0
    min_notional: float = 1.0
    cancel_unfilled: bool = True
    # how long to keep re-querying a cancelled order before giving up on the
    # broker saying so. Past this the order is treated as still live.
    cancel_confirm_s: float = 30.0

    kind: str = field(default="alpaca", init=False)
    cash: float = field(default=0.0, init=False)
    units: float = field(default=0.0, init=False)
    mark: float = field(default=0.0, init=False)
    account_equity: float = field(default=0.0, init=False)
    buying_power: float = field(default=0.0, init=False)
    last_refresh: float = field(default=0.0, init=False)
    _client: object | None = field(default=None, init=False, repr=False)
    _data: object | None = field(default=None, init=False, repr=False)
    _asset: object | None = field(default=None, init=False, repr=False)

    # ────────────────────────── connection ──────────────────────────

    @property
    def client(self):
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:      # pragma: no cover - depends on install
                raise BrokerError("alpaca-py is not installed: pip install alpaca-py") from exc
            _load_env()
            key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
            if not key or not secret:
                raise BrokerError(
                    "APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set. Put them in "
                    ".env next to this repo, or export them.")
            base = os.getenv("APCA_API_BASE_URL", "")
            if self.paper and base and "paper-api" not in base:
                raise BrokerError(
                    f"APCA_API_BASE_URL points at {base!r}, which is not the paper "
                    "endpoint, but this broker was created with paper=True. Fix the "
                    "URL, or create the run with paper=False and --allow-live, "
                    "deliberately.")
            if not self.paper and not self.allow_live:
                raise BrokerError(
                    "paper=False routes orders to a live brokerage account with real "
                    "money. Pass allow_live=True if that is genuinely what you want.")
            self._client = TradingClient(api_key=key, secret_key=secret, paper=self.paper)
        return self._client

    @property
    def asset(self):
        if self._asset is None:
            from alpaca.common.exceptions import APIError
            try:
                self._asset = self.client.get_asset(self.symbol)
            except APIError as exc:
                raise BrokerError(
                    f"Alpaca does not know the symbol {self.symbol!r}. Continuous "
                    f"contract tickers like NQ1! are not tradable there — pass the "
                    f"instrument you actually want to trade (e.g. QQQ). ({exc})") from exc
        return self._asset

    @property
    def is_crypto(self) -> bool:
        return str(getattr(self.asset, "asset_class", "")).lower().endswith("crypto")

    def _data_client(self):
        """Market-data client for the traded instrument. Crypto data needs no key."""
        if self._data is None:
            if self.is_crypto:
                from alpaca.data.historical import CryptoHistoricalDataClient
                self._data = CryptoHistoricalDataClient()
            else:
                from alpaca.data.historical import StockHistoricalDataClient
                _load_env()
                self._data = StockHistoricalDataClient(
                    os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
        return self._data

    # ────────────────────────── introspection ──────────────────────────

    def prepare(self) -> dict:
        """Validate everything that can be validated up front. Returns run config."""
        from alpaca.common.exceptions import APIError

        acct = self.client.get_account()
        if str(acct.status).split(".")[-1] != "ACTIVE":
            raise BrokerError(f"Alpaca account is {acct.status}, not ACTIVE.")

        asset = self.asset
        if not asset.tradable:
            raise BrokerError(f"{self.symbol} is not tradable on this account.")

        warnings = []
        if self.fractional and not asset.fractionable:
            warnings.append(f"{self.symbol} is not fractionable — orders round to "
                            "whole units, so small target changes will not trade")
        if self.allow_short and not getattr(asset, "shortable", False):
            raise BrokerError(
                f"{self.symbol} is not shortable on this account, but the run was "
                "created with allow_short. A long/short strategy routed here would "
                "silently become long-only; pick a shortable symbol or drop the flag.")
        if not self.allow_short:
            warnings.append("short targets are clamped to flat (allow_short is off) — "
                            "the routed account will not match the simulated one for "
                            "any strategy that goes short")

        others = [p.symbol for p in self.client.get_all_positions()
                  if p.symbol.replace("/", "") != self.symbol.replace("/", "")]
        if others:
            warnings.append(f"account already holds {', '.join(others)} — this run's "
                            "equity is computed from cash + this symbol only, so those "
                            "positions will make it disagree with the account")
        try:
            resting = [o for o in self.client.get_orders()
                       if o.symbol.replace("/", "") == self.symbol.replace("/", "")]
            if resting:
                warnings.append(f"{len(resting)} open order(s) already resting on "
                                f"{self.symbol} — they are not this run's and will "
                                "change its position when they fill")
        except APIError:
            pass

        self.refresh()
        return {
            "broker_account": acct.account_number,
            "broker_paper": bool(self.paper),
            "broker_asset_class": str(getattr(asset, "asset_class", "")),
            "broker_fractionable": bool(asset.fractionable),
            "broker_shortable": bool(getattr(asset, "shortable", False)),
            "broker_equity_at_start": self.account_equity,
            "broker_warnings": warnings,
        }

    def refresh(self) -> dict:
        """Read cash, equity, the position and the mark back from the broker."""
        from alpaca.common.exceptions import APIError

        acct = self.client.get_account()
        self.cash = _f(acct.cash)
        self.account_equity = _f(acct.equity)
        self.buying_power = _f(acct.buying_power)
        held = None
        try:
            held = self.client.get_open_position(self.symbol.replace("/", ""))
            self.units = _f(held.qty)
        except APIError as exc:
            text = str(exc).lower()
            if "position does not exist" not in text and "404" not in text:
                raise BrokerError(f"could not read the {self.symbol} position: {exc}") from exc
            self.units = 0.0
        self.mark = self._mark_price(held)
        self.last_refresh = time.time()
        return {"cash": self.cash, "units": self.units, "mark": self.mark,
                "equity": self.account_equity}

    def _latest_trade(self):
        """The traded instrument's last print, or None if it cannot be read."""
        try:
            data = self._data_client()
            if self.is_crypto:
                from alpaca.data.requests import CryptoLatestTradeRequest
                got = data.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=self.symbol))
            else:
                from alpaca.data.requests import StockLatestTradeRequest
                got = data.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=self.symbol))
            return next(iter(got.values()))
        except Exception:
            return None

    def _mark_price(self, held=None) -> float:
        """Last price of the instrument being traded — not of the charted one.

        Falls back to the open position's own mark, then to whatever we had, so a
        data hiccup degrades the sizing rather than killing a run that has been
        going for months.
        """
        trade = self._latest_trade()
        if trade is not None:
            price = _f(getattr(trade, "price", 0))
            if price > 0:
                return price
        if held is not None:
            price = _f(getattr(held, "current_price", 0))
            if price > 0:
                return price
        return self.mark

    def quote(self) -> dict:
        """The traded instrument's NBBO right now. Read-only.

        Returns {bid, ask, mid, at, source, age_s}. `source` is "quote" when both
        sides came back, "trade" when the book was missing or one-sided and the
        last print stood in (bid == ask == mid), "unavailable" when neither could
        be read — mid is None then and the caller sizes off the mark instead.
        `age_s` is how old the print is: a stock quote on a Saturday is Friday's
        after-hours book, real but stale, and the report should know that.
        """
        out = {"bid": None, "ask": None, "mid": None, "at": None,
               "source": "unavailable", "age_s": None}
        try:
            data = self._data_client()
            if self.is_crypto:
                from alpaca.data.requests import CryptoLatestQuoteRequest
                got = data.get_crypto_latest_quote(
                    CryptoLatestQuoteRequest(symbol_or_symbols=self.symbol))
            else:
                from alpaca.data.requests import StockLatestQuoteRequest
                got = data.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=self.symbol))
            q = next(iter(got.values()))
            bid, ask = _f(q.bid_price), _f(q.ask_price)
            if bid > 0 and ask > 0:
                out.update(bid=bid, ask=ask, mid=round((bid + ask) / 2, 8),
                           at=_iso(q.timestamp), source="quote")
        except Exception:
            pass
        if out["mid"] is None:
            trade = self._latest_trade()
            price = _f(getattr(trade, "price", 0)) if trade is not None else 0.0
            if price > 0:
                out.update(bid=price, ask=price, mid=price,
                           at=_iso(getattr(trade, "timestamp", None)), source="trade")
        if out["at"]:
            try:
                then = datetime.fromisoformat(out["at"])
                out["age_s"] = round((datetime.now(timezone.utc) - then).total_seconds(), 1)
            except ValueError:
                pass
        return out

    def market_open(self) -> bool:
        if self.is_crypto:
            return True
        return bool(self.client.get_clock().is_open)

    def opg_window(self, now=None) -> tuple:
        """When a market-on-open order can be sent for the next opening auction.

        Alpaca's rule for TimeInForce.OPG: an order that arrives between 09:28 and
        19:00 America/New_York is rejected; one that arrives after 19:00 is queued
        for the next session's opening auction. So a decision made from Friday's
        close cannot go out at the 20:05 UTC poll — that is 16:05 ET, inside the
        rejection window — it waits for 19:00 ET, and the run decides when to
        call place(). This is the window it decides against.

        Returns (send_not_before, send_not_after) as tz-naive UTC pd.Timestamps:
        19:00 New York on the calendar day before the next open, and the next
        open less two minutes. next_open comes from the trading clock; `now` (the
        decision time, default the wall clock) is only a sanity check against a
        clock whose next_open has already gone by. Read-only. Equities only —
        crypto has no auction.
        """
        import pandas as pd
        from zoneinfo import ZoneInfo

        ny = ZoneInfo("America/New_York")
        nxt = pd.Timestamp(self.client.get_clock().next_open)
        nxt = (nxt.tz_localize("UTC") if nxt.tzinfo is None else nxt).tz_convert("UTC")
        now = pd.Timestamp.now("UTC") if now is None else pd.Timestamp(now)
        now = (now.tz_localize("UTC") if now.tzinfo is None else now).tz_convert("UTC")
        if nxt <= now:
            raise BrokerError(f"trading clock says next_open is {nxt.isoformat()}, "
                              f"which is not after {now.isoformat()}")
        # the New York date first, then 19:00 on the day before it as a local wall
        # time. zoneinfo picks the offset for that wall time, so the window is
        # right across the DST changes; adding hours to a UTC instant would not be.
        open_ny = nxt.to_pydatetime().astimezone(ny)
        eve = open_ny.date() - timedelta(days=1)
        not_before = datetime(eve.year, eve.month, eve.day, tzinfo=ny).replace(hour=19)
        send_not_before = pd.Timestamp(not_before).tz_convert("UTC").tz_localize(None)
        send_not_after = (nxt - pd.Timedelta(minutes=2)).tz_localize(None)
        return send_not_before, send_not_after

    def describe(self) -> str:
        return (f"alpaca {'paper' if self.paper else 'LIVE'} {self.symbol} "
                f"({'fractional' if self.fractional else 'whole units'}, "
                f"{'long/short' if self.allow_short else 'long only'})")

    # ────────────────────────── the account surface ──────────────────────────

    def equity(self, price: float | None = None) -> float:
        """Cash plus the position at the traded instrument's own mark.

        `price` is the caller's bar close and is deliberately ignored: it belongs to
        the charted instrument, which is usually not the one being held. Holding
        QQQ and marking it at NQ's close would produce an equity curve made of two
        different instruments. The account's own equity figure is recorded beside
        this one as broker_equity; a persistent gap between the two means the
        account holds something this run did not put there.
        """
        return self.cash + self.units * (self.mark or price or 0.0)

    def fraction(self, price: float | None = None) -> float:
        mark = self.mark or price or 0.0
        eq = self.equity(price)
        return (self.units * mark / eq) if eq else 0.0

    def rebalance(self, ref_price: float, target_fraction: float) -> dict | None:
        """Move the real position to `target_fraction` of equity. Returns the fill.

        `ref_price` is the bar open the simulated account would have filled at. It
        is recorded, not used: sizing and slippage both go off the traded symbol's
        own mark, because that is the instrument the order is for.
        """
        plan = self._plan(target_fraction, ref_price)
        if plan is None or plan.get("routed") is False:
            return plan

        fills, errors = [], []
        for leg in plan["legs"]:
            try:
                fills.append(self._submit(leg, plan["price"]))
            except BrokerError as exc:
                errors.append(str(exc))
                break

        self.refresh()
        done = [f for f in fills if f and f.get("filled_units")]
        if not done:
            why = errors[0] if errors else (fills[0].get("why_not") if fills else "no fill")
            return self._refused(plan, ref_price, why)
        return self._settle(plan, done, ref_price, errors)

    def place(self, target_fraction: float, ref_price: float | None, *,
              style: str = "market", client_order_prefix: str,
              limit_bps: float = 5.0, limit_wait_s: float = 60.0,
              fallback: str = "market") -> dict | None:
        """Move the position to `target_fraction` now, the way `style` says.

        Same sizing and legs as rebalance(); the difference is when it is called
        (at decision time, not the next close) and how each leg goes out:

        - market            what rebalance() sends, waited out.
        - marketable_limit  a limit through the touch by `limit_bps`, GTC, waited
                            `limit_wait_s`; then cancelled and the cancel confirmed;
                            then a market order for whatever is left, if `fallback`
                            is "market". Fill is the VWAP across the legs.
        - market_on_open    TimeInForce.OPG, whole units, sent the moment this is
                            called — the run picks the moment (opg_window()) and
                            require_open does not apply. Returned open: nothing
                            is waited on, resolve(legs) picks it up at the next poll.

        Every leg is `f"{client_order_prefix}:{n}"` — the run can find its own
        orders again after a crash, and a leg whose id already exists is adopted,
        never re-sent. `ref_price` is the feed's price for the audit trail (the
        fill bar's open is not known yet when this runs; the caller fills it in).
        Returns rebalance()'s dict plus poll_price, sent_at, order_style, legs.
        """
        if style not in _STYLES:
            raise BrokerError(f"unknown order style {style!r}; one of {', '.join(_STYLES)}")
        if fallback not in ("market", "none", None):
            raise BrokerError(f"unknown fallback {fallback!r}; 'market' or 'none'")
        prefix = str(client_order_prefix or "")
        if not prefix:
            raise BrokerError("client_order_prefix is required — it is how a poll that "
                              "crashed mid-wait finds its own order again")
        # leave room for ":NN" — a flip with limit fallbacks is four legs
        if len(prefix) + 3 > _CLIENT_ID_MAX:
            raise BrokerError(f"client_order_prefix {prefix!r} is too long "
                              f"({len(prefix)} chars; at most {_CLIENT_ID_MAX - 3})")

        opg = style == "market_on_open"
        plan = self._plan(target_fraction, ref_price, opg=opg)
        if plan is None or plan.get("routed") is False:
            return plan
        price = plan["price"]
        sent_at = _now_iso()
        legs: list[dict] = []
        errors: list[str] = []
        fell_back = False
        withheld = None

        for wanted in plan["legs"]:
            cid = self._client_id(prefix, len(legs))
            try:
                if style == "market":
                    legs.append(self._send_leg(wanted, cid, kind="market"))
                    continue
                if opg:
                    legs.append(self._send_leg(wanted, cid, kind="opg"))
                    continue
                # marketable limit: priced off the book at the moment of sending,
                # off the mark if the book cannot be read
                q = self.quote()
                mid = q["mid"] or price
                leg = self._send_leg(wanted, cid, kind="limit",
                                     limit_price=self._limit_price(mid, wanted, limit_bps),
                                     wait_s=limit_wait_s)
                leg["mid"] = mid
                leg["quote_source"] = q["source"]
                legs.append(leg)
                if leg.get("cancel_pending"):
                    # the broker has not agreed the limit is dead. A market order on
                    # top of it could fill both. Leave it; resolve() gets it later.
                    withheld = (f"cancel of {cid} not confirmed after "
                                f"{self.cancel_confirm_s:.0f}s — fallback withheld")
                    break
                sign = 1.0 if wanted > 0 else -1.0
                left = leg["qty"] - leg["filled_qty"]
                if left > 0 and fallback == "market" and abs(left * price) >= self.min_notional:
                    fell_back = True
                    legs.append(self._send_leg(sign * left, self._client_id(prefix, len(legs)),
                                               kind="market"))
            except BrokerError as exc:
                errors.append(str(exc))
                break

        self.refresh()
        order_style = style
        if style == "marketable_limit" and fell_back:
            order_style = "limit+market"
        extra = {
            "poll_price": round(price, 6),
            "sent_at": sent_at,
            "order_style": order_style,
            "legs": legs,
            "wanted_units": round(plan["delta"], 8),
            "units_before": round(plan["units_before"], 8),
        }
        if withheld:
            extra["leg_error"] = withheld
            extra["open"] = True

        if opg:
            # the fill is hours away. Everything the accounting needs later is here;
            # the fill fields are filled in by resolve() once the open has happened.
            out = {
                "routed": True, "open": True,
                "side": "buy" if plan["delta"] > 0 else "sell",
                "units": None,
                "ref_price": round(price, 6),
                "bar_price": _r6(ref_price),
                "fill_price": None, "notional": None,
                "commission": 0.0, "slippage_cost": None,
                "order_ids": [l["order_id"] for l in legs if l.get("order_id")],
                "latency_s": round(sum(l["latency_s"] or 0 for l in legs), 2),
                "broker_equity": self.account_equity,
                "broker_units": round(self.units, 8),
                **extra,
            }
            if plan["clamped"]:
                out["clamped"] = plan["clamped"]
            if errors:
                out["leg_error"] = errors[0]
            if not out["order_ids"]:
                out.update(routed=False, open=False,
                           why_not=errors[0] if errors else legs[0].get("why_not", "not sent"))
            return out

        done = [self._as_fill(l) for l in legs if l["filled_qty"] > 0]
        if not done:
            if errors:
                why = errors[0]
            elif withheld:
                why = withheld
            elif legs and legs[0].get("why_not"):
                why = legs[0]["why_not"]
            elif legs:
                why = (f"order {legs[-1]['status'] or 'unknown'} after "
                       f"{legs[-1]['latency_s'] or 0:.0f}s"
                       + (" (cancelled)" if legs[-1]["cancelled"] else ""))
            else:
                why = "no fill"
            return {**self._refused(plan, ref_price, why), **extra}
        out = self._settle(plan, done, ref_price, errors)
        out.update(extra)
        return out

    def resolve(self, legs: list[dict], *, poll_price: float | None = None,
                units_before: float | None = None) -> dict:
        """Re-query every leg by client_order_id and re-aggregate the fill.

        For the polls after an order went out: an on-open leg fills hours after it
        was sent, and a limit leg that was still pending_cancel may have filled or
        died since. Returns the updated legs plus filled_units, fill_price (VWAP),
        notional, order_ids and `open` (True while any leg can still fill); with
        `poll_price`, slippage_cost against it; with `units_before`, `units` as the
        position actually moved (crypto fees are taken in kind — see _settle).
        """
        from alpaca.common.exceptions import APIError

        updated = []
        for leg in legs:
            leg = dict(leg)
            cid = leg.get("client_order_id")
            if not cid or leg.get("status") == "not_sent":
                updated.append(leg)
                continue
            try:
                order = self.client.get_order_by_client_id(cid)
            except APIError as exc:
                leg["resolve_error"] = str(exc)
                updated.append(leg)
                continue
            leg.pop("resolve_error", None)
            leg.update(order_id=str(order.id), status=_status(order),
                       filled_qty=_f(getattr(order, "filled_qty", 0)),
                       avg_price=_f(getattr(order, "filled_avg_price", 0) or 0) or None)
            leg["open"] = leg["status"] not in _TERMINAL
            updated.append(leg)
        self.refresh()

        done = [self._as_fill(l) for l in updated if l["filled_qty"] > 0]
        out = {"legs": updated,
               "open": any(l.get("open") for l in updated),
               "statuses": [l["status"] for l in updated],
               "order_ids": [l["order_id"] for l in updated if l.get("order_id")],
               "filled_units": 0.0, "fill_price": None, "notional": None,
               "units": 0.0, "slippage_cost": None,
               "broker_equity": self.account_equity,
               "broker_units": round(self.units, 8)}
        if not done:
            return out
        ordered, gross, avg_price = _vwap(done)
        filled = ordered
        if units_before is not None:
            moved = self.units - units_before
            if abs(moved) >= 1e-12:
                filled = moved
                if abs(ordered - moved) > 1e-12:
                    out["order_units"] = round(ordered, 10)
                    out["in_kind_fee_units"] = round(abs(ordered) - abs(moved), 10)
        out.update(filled_units=round(ordered, 8), units=round(filled, 8),
                   fill_price=round(avg_price, 6), notional=round(gross, 2),
                   side="buy" if filled > 0 else "sell")
        if poll_price:
            side_sign = 1.0 if filled > 0 else -1.0
            out["slippage_cost"] = round(abs(filled) * (avg_price - poll_price) * side_sign, 4)
        return out

    # ────────────────────────── internals ──────────────────────────

    def _whole_units_only(self, target: float, opg: bool = False) -> bool:
        """Fractional quantities are rejected on shorts and on non-fractionable
        assets — and on anything but a DAY order, which rules out on-open."""
        return (opg
                or not self.fractional
                or not self.asset.fractionable
                or target < 0
                or self.units < 0)

    def _target_units(self, equity: float, price: float, fraction: float,
                      opg: bool = False) -> tuple[float, str]:
        """Turn a target fraction of equity into a quantity the broker will accept."""
        raw = fraction * equity / price
        clamped = ""
        if raw < 0 and (not self.allow_short or self.is_crypto):
            raw, clamped = 0.0, ("crypto cannot be shorted at Alpaca — target flattened"
                                 if self.is_crypto else
                                 "short target flattened (allow_short is off)")
        if self._whole_units_only(raw, opg=opg):
            trimmed = math.trunc(raw)
            if trimmed == 0 and raw != 0 and not clamped:
                clamped = (f"target of {raw:.4f} units rounds to 0 — {self.symbol} "
                           "trades in whole units here")
            raw = float(trimmed)
        return raw, clamped

    def _plan(self, target_fraction: float, ref_price: float | None,
              opg: bool = False) -> dict | None:
        """Refresh, size, and split the move into legs.

        Returns the plan, or a refusal (a dict with routed=False) the caller hands
        straight back, or None when the move is below min_notional.
        """
        self.refresh()
        units_before = self.units
        eq = self.equity()
        price = self.mark
        if eq <= 0:
            return {"routed": False, "why_not": "account equity is not positive"}
        if price <= 0:
            return {"routed": False, "why_not": f"no mark available for {self.symbol}"}

        target_units, clamped = self._target_units(eq, price, target_fraction, opg=opg)
        delta = target_units - self.units
        if abs(delta * price) < self.min_notional:
            return None

        # an on-open order is sent while closed by design
        if self.require_open and not opg and not self.market_open():
            return {"routed": False, "side": "buy" if delta > 0 else "sell",
                    "wanted_units": round(delta, 8), "ref_price": _r6(ref_price),
                    "mark_price": round(price, 6),
                    "why_not": "market closed — a market order now would fill at the "
                               "next open on a decision made from this bar's close",
                    "clamped": clamped or None}

        # Alpaca rejects a single order that crosses through flat, so a flip is two:
        # close what is there, then open the other way.
        if self.units and target_units and (self.units > 0) != (target_units > 0):
            legs = [-self.units, target_units]
        else:
            legs = [delta]
        return {"units_before": units_before, "eq": eq, "price": price,
                "target_units": target_units, "clamped": clamped,
                "delta": delta, "legs": legs}

    def _refused(self, plan: dict, ref_price: float | None, why: str) -> dict:
        return {"routed": False, "side": "buy" if plan["delta"] > 0 else "sell",
                "wanted_units": round(plan["delta"], 8), "ref_price": _r6(ref_price),
                "mark_price": round(plan["price"], 6), "why_not": why,
                "broker_equity": self.account_equity}

    def _settle(self, plan: dict, done: list[dict], ref_price: float | None,
                errors: list[str]) -> dict:
        """The fill dict, from the legs that filled and the position as it stands."""
        price, delta = plan["price"], plan["delta"]
        ordered, gross, avg_price = _vwap(done)

        # What the orders say they filled and what the position actually moved by are
        # not always the same number: Alpaca takes crypto fees in kind, so a buy of
        # 0.00031232 BTC leaves 0.00031154 in the account. The ledger has to follow
        # the position, or its reconstructed P&L drifts from the account's by exactly
        # the fees. The order's number is kept alongside when they disagree.
        filled = self.units - plan["units_before"]
        if abs(filled) < 1e-12:
            filled = ordered
        side_sign = 1.0 if filled > 0 else -1.0
        out = {
            "routed": True,
            "side": "buy" if filled > 0 else "sell",
            "units": round(filled, 8),
            # the mark the order was sized and sent against — the honest baseline for
            # what the market order cost. ref_price is the charted instrument's bar
            # open, kept for the audit trail; the two are not comparable unless the
            # feed and the broker are pointed at the same thing.
            "ref_price": round(price, 6),
            "bar_price": _r6(ref_price),
            "fill_price": round(avg_price, 6),
            "notional": round(gross, 2),
            # equities are commission-free at Alpaca; crypto fees land in the account
            # balance rather than on the order, so this stays 0 and the cash read
            # picks them up on the next refresh.
            "commission": 0.0,
            # what the fill actually cost against the price the simulation assumed
            "slippage_cost": round(abs(filled) * (avg_price - price) * side_sign, 4),
            "order_ids": [f["order_id"] for f in done],
            "latency_s": round(sum(f["latency_s"] for f in done), 2),
            "broker_equity": self.account_equity,
            "broker_units": round(self.units, 8),
        }
        if abs(ordered - filled) > 1e-12:
            # the gap is a fee taken in the asset itself
            out["order_units"] = round(ordered, 10)
            out["in_kind_fee_units"] = round(abs(ordered) - abs(filled), 10)
        if abs(filled - delta) > 1e-9:
            out["short_by"] = round(delta - filled, 8)
        if plan["clamped"]:
            out["clamped"] = plan["clamped"]
        if errors:
            out["leg_error"] = errors[0]
        return out

    @staticmethod
    def _as_fill(leg: dict) -> dict:
        """A leg record in the shape _settle() aggregates."""
        sign = 1 if leg["side"] == "buy" else -1
        return {"filled_units": leg["filled_qty"] * sign,
                "avg_price": leg["avg_price"] or leg.get("limit_price") or leg.get("mid") or 0.0,
                "order_id": leg["order_id"], "latency_s": leg["latency_s"] or 0.0}

    @staticmethod
    def _client_id(prefix: str, n: int) -> str:
        cid = f"{prefix}:{n}"
        if len(cid) > _CLIENT_ID_MAX:
            raise BrokerError(f"client_order_id {cid!r} is over {_CLIENT_ID_MAX} chars")
        return cid

    def _leg_qty(self, delta_units: float, opg: bool = False) -> float:
        qty = abs(delta_units)
        if self._whole_units_only(self.units + delta_units, opg=opg):
            return float(math.trunc(qty))
        return round(qty, 9)

    def _limit_price(self, mid: float, delta_units: float, bps: float) -> float:
        """Marketable: through the touch by `bps`, rounded away from the market so
        the rounding cannot make it un-marketable. Crypto rounds to the asset's
        price increment, equities to a cent (sub-penny limits are rejected)."""
        buy = delta_units > 0
        raw = mid * (1 + bps / 1e4) if buy else mid * (1 - bps / 1e4)
        inc = _f(getattr(self.asset, "price_increment", 0) or 0) if self.is_crypto else 0.0
        if inc <= 0:
            inc = 0.01
        steps = raw / inc
        steps = math.ceil(steps - 1e-9) if buy else math.floor(steps + 1e-9)
        decimals = max(0, int(round(-math.log10(inc))))
        return round(steps * inc, decimals)

    def _existing(self, cid: str):
        """The order already carrying this client_order_id, or None.

        Only a 404 — no such order — means absent. Anything else (a 5xx, an auth
        failure) is unknown, and unknown must not turn into a second submit_order:
        it is raised, this poll sends nothing, and the next one looks the id up
        again.
        """
        from alpaca.common.exceptions import APIError
        try:
            return self.client.get_order_by_client_id(cid)
        except APIError as exc:
            if _not_found(exc):
                return None
            raise BrokerError(f"could not look up order {cid!r} before sending "
                              f"(nothing was sent): {exc}") from exc

    def _send_leg(self, delta_units: float, cid: str, *, kind: str,
                  limit_price: float | None = None, wait_s: float | None = None) -> dict:
        """One order for `delta_units` — kind is market, limit or opg — carried to
        a terminal state, or for opg handed back open. Returns the leg record."""
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = "buy" if delta_units > 0 else "sell"
        qty = self._leg_qty(delta_units, opg=(kind == "opg"))
        leg = {"client_order_id": cid, "order_id": None, "type": kind, "side": side,
               "qty": qty, "limit_price": limit_price, "status": None,
               "filled_qty": 0.0, "avg_price": None, "latency_s": None, "cancelled": False}
        if qty <= 0:
            leg.update(status="not_sent", why_not="quantity rounds to zero")
            return leg

        common = dict(symbol=self.symbol, qty=qty, client_order_id=cid,
                      side=OrderSide.BUY if delta_units > 0 else OrderSide.SELL)
        if kind == "limit":
            # fractional equity orders are DAY-only at Alpaca; crypto takes GTC
            req = LimitOrderRequest(**common, limit_price=limit_price,
                                    time_in_force=TimeInForce.GTC if self.is_crypto
                                    else TimeInForce.DAY)
        elif kind == "opg":
            req = MarketOrderRequest(**common, time_in_force=TimeInForce.OPG)
        else:
            req = MarketOrderRequest(**common,
                                     time_in_force=TimeInForce.GTC if self.is_crypto
                                     else TimeInForce.DAY)

        started = time.time()
        open_ids: list = []
        with self._cancel_on_sigterm(open_ids):
            # a poll that died between sending and journaling must not send twice
            order = self._existing(cid)
            if order is None:
                try:
                    order = self.client.submit_order(req)
                except APIError as exc:
                    raise BrokerError(f"order rejected: {exc}") from exc
            else:
                leg["resumed"] = True
            open_ids.append(order.id)
            leg["order_id"] = str(order.id)
            if kind == "opg":
                leg.update(status=_status(order), open=True,
                           latency_s=round(time.time() - started, 3))
                return leg
            order = self._await_fill(order, timeout=wait_s)
            if _status(order) not in _TERMINAL:
                # the wait ran out on a live order. Whatever has not filled must be
                # off the book before anything else goes out.
                order, confirmed = self._cancel_confirm(order)
                leg["cancelled"] = confirmed
                if not confirmed:
                    leg["cancel_pending"] = True

        leg.update(status=_status(order),
                   filled_qty=_f(getattr(order, "filled_qty", 0)),
                   avg_price=_f(getattr(order, "filled_avg_price", 0) or 0) or None,
                   latency_s=round(time.time() - started, 3))
        leg["open"] = leg["status"] not in _TERMINAL
        return leg

    def _submit(self, delta_units: float, ref_price: float) -> dict:
        """One market order for `delta_units`, waited out to a terminal state."""
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        qty = self._leg_qty(delta_units)
        if qty <= 0:
            return {"filled_units": 0.0, "why_not": "quantity rounds to zero"}

        req = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=OrderSide.BUY if delta_units > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if self.is_crypto else TimeInForce.DAY,
        )
        started = time.time()
        open_ids: list = []
        cancelled = False
        with self._cancel_on_sigterm(open_ids):
            try:
                order = self.client.submit_order(req)
            except APIError as exc:
                raise BrokerError(f"order rejected: {exc}") from exc
            open_ids.append(order.id)
            order = self._await_fill(order)
            if self.cancel_unfilled and _status(order) not in _TERMINAL:
                # The wait ran out. A partial fill is the dangerous case: the rest is
                # still working and would fill later, unrecorded. Cancel, and take
                # the filled quantity from the confirmed order, not the stale one.
                order, cancelled = self._cancel_confirm(order)
        filled_qty = _f(getattr(order, "filled_qty", 0))
        avg = _f(getattr(order, "filled_avg_price", 0) or 0)
        status = _status(order)

        if filled_qty <= 0:
            # the suffix follows the setting, not the confirmed cancel: that is
            # what the legacy journals say and they have to keep saying it
            return {"filled_units": 0.0,
                    "why_not": f"order {status or 'unknown'} after "
                               f"{time.time() - started:.0f}s"
                               + (" (cancelled)" if self.cancel_unfilled else "")}

        return {
            "filled_units": filled_qty * (1 if delta_units > 0 else -1),
            "avg_price": avg or ref_price,
            "order_id": str(order.id),
            "status": status,
            "latency_s": time.time() - started,
            "cancelled": cancelled,
        }

    def _await_fill(self, order, timeout: float | None = None):
        """Poll the order until it stops being able to fill, or we run out of patience."""
        from alpaca.common.exceptions import APIError

        deadline = time.time() + (self.fill_timeout if timeout is None else timeout)
        while time.time() < deadline:
            if _status(order) in _TERMINAL:
                return order
            time.sleep(self.poll_every)
            try:
                order = self.client.get_order_by_id(order.id)
            except APIError:
                return order
        return order

    def _cancel_confirm(self, order):
        """Cancel `order` and re-query until the broker says it is over.

        Returns (order, confirmed). Not confirmed means it was still pending_cancel
        after cancel_confirm_s: the order is live and nothing may be sent on top of
        it. The broker's answer, not the cancel request, is what counts — a cancel
        that returns 200 can still race a fill.
        """
        from alpaca.common.exceptions import APIError

        try:
            self.client.cancel_order_by_id(order.id)
        except APIError:
            pass    # already terminal, or a race — the re-query below is the truth
        deadline = time.time() + self.cancel_confirm_s
        while True:
            try:
                order = self.client.get_order_by_id(order.id)
            except APIError:
                pass
            if _status(order) in _TERMINAL:
                return order, True
            if time.time() >= deadline:
                return order, False
            time.sleep(self.poll_every)

    @contextlib.contextmanager
    def _cancel_on_sigterm(self, order_ids: list):
        """While waiting on an order, a SIGTERM (systemd stop) cancels it first.

        Without this a stop mid-wait leaves the order working at the broker with
        nobody watching it. The cancel is fire-and-forget — confirming it is left
        to resolve() on the next tick — and the process then exits 143, so the
        unit shows as failed rather than clean. Any handler that was installed
        before (paper.py's, if it has one) runs after the cancels.
        """
        def handler(signum, frame):
            for oid in list(order_ids):
                try:
                    self.client.cancel_order_by_id(oid)
                except Exception:
                    pass
            signal.signal(signal.SIGTERM, previous)
            if callable(previous):
                previous(signum, frame)
            raise SystemExit(128 + int(signum))

        try:
            previous = signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError, AttributeError):
            # not the main thread, or no SIGTERM on this platform — nothing to install
            yield
            return
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)


def build(spec: dict | None):
    """Rebuild a broker from the dict stored in a run's config.json.

    None, or kind 'ledger', means the simulated account in paper.py — which is the
    default and stays the default.
    """
    if not spec:
        return None
    kind = spec.get("kind")
    if kind in (None, "ledger", "sim", "none"):
        return None
    if kind == "alpaca":
        return AlpacaBroker(**{k: v for k, v in spec.items() if k != "kind"})
    raise BrokerError(f"unknown broker kind: {kind!r}")

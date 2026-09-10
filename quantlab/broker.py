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

Safety
------
- Refuses a live (non-paper) account unless allow_live is passed explicitly.
- Refuses a symbol the account cannot actually trade, at creation, not at 3am.
- Never routes a catch-up bar. If the loop resumes after a gap, the opens those
  decisions would have filled at are gone; paper.py marks them and moves on.
- Market orders are not submitted while the market is closed. A market order sent
  on Saturday fills at Monday's open on a decision made from Friday's close, which
  is not the trade the strategy asked for. Set require_open=False to allow it.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# order states that will never fill any more than they already have
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day",
             "stopped", "suspended", "replaced"}


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

    def _mark_price(self, held=None) -> float:
        """Last price of the instrument being traded — not of the charted one.

        Falls back to the open position's own mark, then to whatever we had, so a
        data hiccup degrades the sizing rather than killing a run that has been
        going for months.
        """
        try:
            if self.is_crypto:
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoLatestTradeRequest
                if self._data is None:
                    self._data = CryptoHistoricalDataClient()
                got = self._data.get_crypto_latest_trade(
                    CryptoLatestTradeRequest(symbol_or_symbols=self.symbol))
            else:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockLatestTradeRequest
                if self._data is None:
                    self._data = StockHistoricalDataClient(
                        os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
                got = self._data.get_stock_latest_trade(
                    StockLatestTradeRequest(symbol_or_symbols=self.symbol))
            price = _f(next(iter(got.values())).price)
            if price > 0:
                return price
        except Exception:
            pass
        if held is not None:
            price = _f(getattr(held, "current_price", 0))
            if price > 0:
                return price
        return self.mark

    def market_open(self) -> bool:
        if self.is_crypto:
            return True
        return bool(self.client.get_clock().is_open)

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
        self.refresh()
        units_before = self.units
        eq = self.equity()
        price = self.mark
        if eq <= 0:
            return {"routed": False, "why_not": "account equity is not positive"}
        if price <= 0:
            return {"routed": False, "why_not": f"no mark available for {self.symbol}"}

        target_units, clamped = self._target_units(eq, price, target_fraction)
        delta = target_units - self.units
        if abs(delta * price) < self.min_notional:
            return None

        if self.require_open and not self.market_open():
            return {"routed": False, "side": "buy" if delta > 0 else "sell",
                    "wanted_units": round(delta, 8), "ref_price": round(ref_price, 6),
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

        fills, errors = [], []
        for leg in legs:
            try:
                fills.append(self._submit(leg, price))
            except BrokerError as exc:
                errors.append(str(exc))
                break

        self.refresh()
        done = [f for f in fills if f and f.get("filled_units")]
        if not done:
            why = errors[0] if errors else (fills[0].get("why_not") if fills else "no fill")
            return {"routed": False, "side": "buy" if delta > 0 else "sell",
                    "wanted_units": round(delta, 8), "ref_price": round(ref_price, 6),
                    "mark_price": round(price, 6), "why_not": why,
                    "broker_equity": self.account_equity}

        ordered = sum(f["filled_units"] for f in done)
        gross = sum(abs(f["filled_units"]) * f["avg_price"] for f in done)
        avg_price = gross / sum(abs(f["filled_units"]) for f in done)

        # What the orders say they filled and what the position actually moved by are
        # not always the same number: Alpaca takes crypto fees in kind, so a buy of
        # 0.00031232 BTC leaves 0.00031154 in the account. The ledger has to follow
        # the position, or its reconstructed P&L drifts from the account's by exactly
        # the fees. The order's number is kept alongside when they disagree.
        filled = self.units - units_before
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
            "bar_price": round(ref_price, 6),
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
        if clamped:
            out["clamped"] = clamped
        if errors:
            out["leg_error"] = errors[0]
        return out

    # ────────────────────────── internals ──────────────────────────

    def _whole_units_only(self, target: float) -> bool:
        """Fractional quantities are rejected on shorts and on non-fractionable assets."""
        return (not self.fractional
                or not self.asset.fractionable
                or target < 0
                or self.units < 0)

    def _target_units(self, equity: float, price: float, fraction: float) -> tuple[float, str]:
        """Turn a target fraction of equity into a quantity the broker will accept."""
        raw = fraction * equity / price
        clamped = ""
        if raw < 0 and (not self.allow_short or self.is_crypto):
            raw, clamped = 0.0, ("crypto cannot be shorted at Alpaca — target flattened"
                                 if self.is_crypto else
                                 "short target flattened (allow_short is off)")
        if self._whole_units_only(raw):
            trimmed = math.trunc(raw)
            if trimmed == 0 and raw != 0 and not clamped:
                clamped = (f"target of {raw:.4f} units rounds to 0 — {self.symbol} "
                           "trades in whole units here")
            raw = float(trimmed)
        return raw, clamped

    def _submit(self, delta_units: float, ref_price: float) -> dict:
        """One market order for `delta_units`, waited out to a terminal state."""
        from alpaca.common.exceptions import APIError
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        qty = abs(delta_units)
        if self._whole_units_only(self.units + delta_units):
            qty = float(math.trunc(qty))
        else:
            qty = round(qty, 9)
        if qty <= 0:
            return {"filled_units": 0.0, "why_not": "quantity rounds to zero"}

        req = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=OrderSide.BUY if delta_units > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if self.is_crypto else TimeInForce.DAY,
        )
        started = time.time()
        try:
            order = self.client.submit_order(req)
        except APIError as exc:
            raise BrokerError(f"order rejected: {exc}") from exc

        order = self._await_fill(order)
        filled_qty = _f(getattr(order, "filled_qty", 0))
        avg = _f(getattr(order, "filled_avg_price", 0) or 0)
        status = str(getattr(order, "status", "")).split(".")[-1].lower()

        if filled_qty <= 0:
            if self.cancel_unfilled and status not in _TERMINAL:
                try:
                    self.client.cancel_order_by_id(order.id)
                except APIError:
                    pass
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
        }

    def _await_fill(self, order):
        """Poll the order until it stops being able to fill, or we run out of patience."""
        from alpaca.common.exceptions import APIError

        deadline = time.time() + self.fill_timeout
        while time.time() < deadline:
            status = str(getattr(order, "status", "")).split(".")[-1].lower()
            if status in _TERMINAL:
                return order
            time.sleep(self.poll_every)
            try:
                order = self.client.get_order_by_id(order.id)
            except APIError:
                return order
        return order


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

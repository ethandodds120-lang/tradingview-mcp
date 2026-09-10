"""Bar feeds for the paper-trading loop.

A feed's only job is to hand back recent OHLCV bars. It does not decide anything
and it never places an order — nothing in this package can.

The one rule every live feed obeys: DROP THE LAST BAR. On a live chart the most
recent bar is still forming. Its high, low and close will all change before it
closes, so a signal computed on it is a signal you could not have acted on, and it
will flicker between polls. `drop_last` is True for anything reading a live source
and False for files of already-closed bars.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod

_COLS = ["open", "high", "low", "close", "volume"]


class FeedError(RuntimeError):
    pass


class Feed:
    """Base class. fetch() returns a datetime-indexed OHLCV frame, newest last."""

    name = "feed"
    drop_last = True

    def prepare(self) -> dict:
        """Optional one-time setup. Returns whatever it wants recorded in the run config."""
        return {}

    def seek(self, last_bar) -> None:
        """Told the newest bar already on disk, when a run resumes in a new process.

        Live feeds ignore it — they always return the most recent bars anyway. Only
        cursor-based feeds like ReplayFeed need it, and without it a resumed
        rehearsal quietly replays bars it has already decided on and makes no
        progress."""
        return None

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError

    def bars(self) -> pd.DataFrame:
        """fetch() with the forming bar removed and the columns normalized."""
        df = self.fetch()
        if df is None or df.empty:
            return pd.DataFrame(columns=_COLS)
        df = df[[c for c in _COLS if c in df.columns]].copy()
        for c in _COLS:
            if c not in df:
                df[c] = np.nan
        df = df[_COLS].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if self.drop_last and len(df):
            df = df.iloc[:-1]
        return df


@dataclass
class CsvFeed(Feed):
    """Re-reads a CSV that something else appends to. Bars are assumed closed."""

    path: str
    name: str = field(default="csv", init=False)
    drop_last: bool = field(default=False, init=False)

    def fetch(self) -> pd.DataFrame:
        return data_mod.load_csv(self.path)


@dataclass
class YahooFeed(Feed):
    """yfinance. Intraday intervals only go back a few days; daily is the usable one."""

    ticker: str
    interval: str = "1d"
    period: str = "6mo"
    name: str = field(default="yahoo", init=False)

    def fetch(self) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(self.ticker, period=self.period, interval=self.interval,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df


@dataclass
class TradingViewFeed(Feed):
    """Bars off the live TradingView Desktop chart, via this repo's `tv` CLI.

    Reads only. The CDP bridge is a chart automation tool and the terms exclude
    automated trading through it; even when this repo grows a real broker layer,
    orders will not go out this way.
    """

    symbol: str | None = None
    timeframe: str | None = None
    count: int = 400
    node: str = "node"
    cli: str | None = None
    timeout: int = 90
    allow_replay: bool = False
    name: str = field(default="tradingview", init=False)

    def __post_init__(self):
        self.cli = self.cli or str(Path(__file__).resolve().parents[1] / "src" / "cli" / "index.js")

    def _call(self, *args: str) -> dict:
        proc = subprocess.run([self.node, self.cli, *args], capture_output=True,
                              text=True, timeout=self.timeout)
        # the CLI reports its own errors as JSON on stderr with a non-zero exit, so
        # try to parse before falling back to complaining about the exit code
        out = proc.stdout.strip() or proc.stderr.strip()
        payload, start = None, out.find("{")
        if start >= 0:
            try:
                payload = json.loads(out[start:])
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            why = (f"exited {proc.returncode}" if proc.returncode else "no JSON returned")
            raise FeedError(f"tv {' '.join(args)}: {why}. {out[:300]}")
        if not payload.get("success", True):
            raise FeedError(f"tv {' '.join(args)}: {payload.get('error')}")
        return payload

    def prepare(self) -> dict:
        try:
            if self.symbol:
                self._call("symbol", self.symbol)
            if self.timeframe:
                self._call("timeframe", str(self.timeframe))
            state = self._call("state")
        except FileNotFoundError as exc:
            raise FeedError("node not found on PATH — the TradingView feed shells out "
                            "to this repo's CLI") from exc
        except subprocess.TimeoutExpired as exc:
            raise FeedError("TradingView did not answer. Is the desktop app running "
                            "with CDP on port 9222?") from exc
        return {"chart_symbol": state.get("symbol"),
                "chart_resolution": state.get("resolution")}

    def fetch(self) -> pd.DataFrame:
        # Replay mode reveals historical bars one at a time and they look exactly
        # like live ones to `ohlcv`. Ingesting them would write simulated bars into
        # a record whose entire value is that its data arrived after the strategy
        # did, so refuse unless the run was explicitly created to do that.
        if not self.allow_replay:
            st = self._call("replay", "status")
            if st.get("is_replay_started"):
                raise FeedError(
                    "the chart is in REPLAY mode. Those bars are historical and would "
                    "be recorded as forward results. Run `tv replay stop`, or create "
                    "the run with allow_replay if you are deliberately rehearsing.")
        payload = self._call("ohlcv", "-n", str(self.count))
        bars = payload.get("bars") or []
        if not bars:
            raise FeedError("chart returned no bars")
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
        return df.set_index("time")


@dataclass
class AlpacaFeed(Feed):
    """Bars from Alpaca. The point of it is that it is the same venue broker.py
    trades on.

    Every other feed here reads a different venue than the one the order goes to,
    which is survivable for a signal but not for sizing: a target expressed as a
    fraction of equity has to be turned into units at the price of the thing you
    are actually holding. Point this at the symbol you route to and the two agree
    by construction.

    Crypto needs no credentials for market data and trades 24/7, which is also what
    makes it the one asset class where a polling loop does not need a market clock.
    """

    symbol: str
    timeframe: str = "1D"
    count: int = 400
    feed: str | None = None          # equities only: iex (free) or sip (paid)
    name: str = field(default="alpaca", init=False)

    _UNITS = {"m": "Minute", "min": "Minute", "h": "Hour", "hour": "Hour",
              "d": "Day", "day": "Day", "w": "Week", "week": "Week"}

    @property
    def is_crypto(self) -> bool:
        return "/" in self.symbol

    def _timeframe(self):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        raw = str(self.timeframe).strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        letters = "".join(ch for ch in raw if ch.isalpha()).lower()
        amount = int(digits) if digits else 1
        # a bare number means minutes, matching the TradingView feed's convention
        unit = self._UNITS.get(letters, "Minute" if not letters else None)
        if unit is None:
            raise FeedError(f"cannot read timeframe {self.timeframe!r} — "
                            "try 5Min, 1H, 1D")
        return TimeFrame(amount, getattr(TimeFrameUnit, unit))

    def _lookback(self) -> pd.Timedelta:
        """Enough history to cover `count` bars, with slack for gaps and halts."""
        tf = self._timeframe()
        per = {"Min": "1min", "Hour": "1h", "Day": "1D", "Week": "1W"}
        step = pd.Timedelta(per.get(str(tf.unit_value), "1D")) * tf.amount_value
        # equities only print during the session, so ask for well more than we need
        return step * self.count * (1 if self.is_crypto else 4)

    def prepare(self) -> dict:
        df = self.fetch()
        if df.empty:
            raise FeedError(f"Alpaca returned no bars for {self.symbol}")
        return {"alpaca_symbol": self.symbol, "alpaca_timeframe": str(self.timeframe),
                "alpaca_first_bar": str(df.index[0]), "alpaca_bars": len(df)}

    def fetch(self) -> pd.DataFrame:
        start = pd.Timestamp.now(tz="UTC") - self._lookback()
        try:
            if self.is_crypto:
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoBarsRequest

                client = CryptoHistoricalDataClient()
                req = CryptoBarsRequest(symbol_or_symbols=self.symbol,
                                        timeframe=self._timeframe(), start=start)
                bars = client.get_crypto_bars(req).df
            else:
                import os

                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest

                from . import broker as broker_mod
                broker_mod._load_env()
                client = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"),
                                                   os.getenv("APCA_API_SECRET_KEY"))
                kw = {"feed": self.feed} if self.feed else {}
                req = StockBarsRequest(symbol_or_symbols=self.symbol,
                                       timeframe=self._timeframe(), start=start, **kw)
                bars = client.get_stock_bars(req).df
        except ImportError as exc:
            raise FeedError("alpaca-py is not installed: pip install alpaca-py") from exc
        except Exception as exc:
            raise FeedError(f"Alpaca bars for {self.symbol}: {exc}") from exc

        if bars is None or bars.empty:
            raise FeedError(f"Alpaca returned no bars for {self.symbol}")
        bars = bars.reset_index()
        ts = "timestamp" if "timestamp" in bars.columns else bars.columns[1]
        bars["time"] = pd.to_datetime(bars[ts], utc=True).dt.tz_localize(None)
        return bars.set_index("time").tail(self.count)


@dataclass
class ReplayFeed(Feed):
    """Rehearsal feed: reveals a stored CSV one bar at a time.

    For testing that the bot itself works — that it fills, sizes, journals and
    recovers from a restart — without waiting days for live bars. It is NOT a
    forward test. The data already existed when the strategy was written, so the
    equity curve it produces is worth exactly as much as a backtest, which is to
    say it is not evidence of anything.
    """

    path: str
    start: int = 300
    step: int = 1
    name: str = field(default="replay", init=False)
    drop_last: bool = field(default=False, init=False)

    def __post_init__(self):
        self._all = data_mod.load_csv(self.path)
        self._cursor = min(self.start, len(self._all))

    def prepare(self) -> dict:
        return {"replay_source": self.path, "replay_total_bars": len(self._all)}

    def seek(self, last_bar) -> None:
        pos = int(self._all.index.searchsorted(pd.Timestamp(last_bar), side="right"))
        self._cursor = max(self._cursor, pos + self.step)

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._all)

    def fetch(self) -> pd.DataFrame:
        window = self._all.iloc[:self._cursor]
        self._cursor = min(self._cursor + self.step, len(self._all))
        return window


def build(spec: dict) -> Feed:
    """Rebuild a feed from the dict stored in a run's config.json."""
    kind = spec.get("kind")
    args = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "csv":
        return CsvFeed(**args)
    if kind == "yahoo":
        return YahooFeed(**args)
    if kind == "tradingview":
        return TradingViewFeed(**args)
    if kind == "alpaca":
        return AlpacaFeed(**args)
    if kind == "replay":
        return ReplayFeed(**args)
    raise FeedError(f"unknown feed kind: {kind!r}")

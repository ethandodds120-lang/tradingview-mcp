"""Data loading. Accepts TradingView CSV exports or any generic OHLCV file."""

from __future__ import annotations

import numpy as np
import pandas as pd

_COL_ALIASES = {
    "time": "date", "date": "date", "datetime": "date", "timestamp": "date",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close", "adj close": "close", "price": "close",
    "volume": "volume", "vol": "volume", "v": "volume",
}


def load_csv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV. Handles TradingView exports (unix or ISO timestamps).

    Returns a DataFrame indexed by datetime with columns open/high/low/close/volume.
    """
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    mapped = {}
    for col in raw.columns:
        key = _COL_ALIASES.get(col)
        if key and key not in mapped:
            mapped[key] = col

    missing = {"date", "close"} - set(mapped)
    if missing:
        raise ValueError(f"{path}: could not find columns for {sorted(missing)}. Found: {list(raw.columns)}")

    df = pd.DataFrame(index=range(len(raw)))
    ts = raw[mapped["date"]]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "s" if ts.max() < 1e11 else "ms"
        df["date"] = pd.to_datetime(ts, unit=unit, utc=True).dt.tz_localize(None)
    else:
        df["date"] = pd.to_datetime(ts, errors="coerce", format="mixed")

    for field in ("open", "high", "low", "close", "volume"):
        if field in mapped:
            df[field] = pd.to_numeric(raw[mapped[field]], errors="coerce")
        elif field == "volume":
            df[field] = np.nan
        else:
            df[field] = df.get("close")

    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]]


def load_yahoo(ticker: str, start: str = "2000-01-01", interval: str = "1d") -> pd.DataFrame:
    """Optional convenience loader. Requires `pip install yfinance` and network access."""
    import yfinance as yf

    df = yf.download(ticker, start=start, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def synthetic(n: int = 4000, seed: int = 0, mu: float = 0.0002, sigma: float = 0.012,
              start: str = "2010-01-01") -> pd.DataFrame:
    """Geometric random walk with NO predictable structure.

    This is the null hypothesis made concrete. Any strategy that shows a strong
    edge here is measuring your own overfitting, not a market effect.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    close = 100 * np.exp(np.cumsum(rets))
    noise = np.abs(rng.normal(0, sigma / 2, n)) * close
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, sigma / 4, n)),
        "high": close + noise,
        "low": close - noise,
        "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualization factor from the bar spacing."""
    if len(index) < 3:
        return 252.0
    median_gap = pd.Series(index).diff().dt.total_seconds().median()
    if not median_gap or np.isnan(median_gap):
        return 252.0
    if median_gap >= 86400 * 25:
        return 12.0
    if median_gap >= 86400 * 5:
        return 52.0
    if median_gap >= 86400:
        # A daily index is not always an equity calendar. Crypto prints every
        # day of the week, and annualising that with 252 understates its vol by
        # sqrt(365/252) — 17% — which quietly turned a 15% vol target into ~18%
        # on the first routed SOL run. A calendar with no weekend gap anywhere
        # in its recent history is a 7-day calendar; anything with a gap of two
        # days or more is trading sessions and stays on 252.
        gaps = pd.Series(index).diff().dt.total_seconds().dropna().tail(90)
        if len(gaps) >= 28 and (gaps < 86400 * 1.5).all():
            return 365.25
        return 252.0
    return 252.0 * (6.5 * 3600 / median_gap)


# ────────────────────────── panel data ──────────────────────────

def load_panel_csv(path: str) -> pd.DataFrame:
    """A dates x tickers close-price frame, from wide or long CSV.

    Wide:  date,SPY,QQQ,...        one column per ticker
    Long:  date,ticker,close       one row per ticker per date

    Long format is what fund and vendor exports look like, wide is what you get
    from a pivot; accept both rather than make the caller reshape. Rows with any
    missing name are dropped: a cross-sectional rank over a universe that changes
    width mid-sample is comparing different things on different days.
    """
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "time", "timestamp")),
                    df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)

    lower = {c.lower(): c for c in df.columns}
    if "ticker" in lower or "symbol" in lower:
        tkr = lower.get("ticker") or lower["symbol"]
        price = lower.get("close") or lower.get("nav") or lower.get("price")
        if price is None:
            raise ValueError("long-format panel needs a close/price/nav column")
        wide = df.pivot(index=date_col, columns=tkr, values=price)
    else:
        wide = df.set_index(date_col)

    wide.index = pd.DatetimeIndex(wide.index).normalize()
    wide = wide[~wide.index.duplicated(keep="last")].sort_index()
    return wide.apply(pd.to_numeric, errors="coerce").dropna(how="any")


def synthetic_panel(n_tickers: int = 20, n: int = 1500, seed: int = 0,
                    mu: float = 0.0002, sigma: float = 0.012,
                    market_beta: float = 0.6) -> pd.DataFrame:
    """Correlated random walks. The null hypothesis for a cross-sectional test.

    A panel of *independent* walks is the wrong null: real universes share a
    market factor, and a dollar-neutral book hedges most of it away. Testing
    against uncorrelated names would flatter any long/short strategy by handing
    it diversification it will not have. `market_beta` is how much of each name's
    move is the common factor.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(mu, sigma, n)
    cols = {}
    for i in range(n_tickers):
        idio = rng.normal(mu, sigma, n)
        r = market_beta * market + np.sqrt(max(0.0, 1 - market_beta ** 2)) * idio
        cols[f"SYN{i:02d}"] = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(cols, index=idx)

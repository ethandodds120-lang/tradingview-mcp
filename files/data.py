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
        return 252.0
    return 252.0 * (6.5 * 3600 / median_gap)

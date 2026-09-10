"""Backtest engine.

Deliberately pessimistic. Costs are charged on every unit of position change,
signals are lagged one full bar, and volatility targeting is computed from
trailing data only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import periods_per_year


@dataclass
class CostModel:
    commission_bps: float = 1.0   # per side, on notional traded
    slippage_bps: float = 2.0     # per side — this is where most edges die
    spread_bps: float = 0.0

    @property
    def per_unit(self) -> float:
        """Total cost in decimal per 1.0 of position change."""
        return (self.commission_bps + self.slippage_bps + self.spread_bps) / 10_000.0


@dataclass
class BacktestResult:
    returns: pd.Series          # net per-bar strategy returns
    gross_returns: pd.Series
    equity: pd.Series
    position: pd.Series         # actual held position, post-lag, post-sizing
    costs: pd.Series
    ppy: float

    @property
    def turnover(self) -> float:
        return float(self.position.diff().abs().sum())


def vol_scale(df: pd.DataFrame,
              vol_target: float | None = 0.15,
              vol_lookback: int = 60,
              max_leverage: float = 2.0,
              ppy: float | None = None) -> pd.Series:
    """Position multiplier that targets a constant annualized volatility.

    Trailing realized vol only — the value at bar t uses returns up to bar t.
    Lives here rather than inside run() because the paper-trading loop sizes with
    exactly this function; two copies of it would drift apart within a month.
    """
    if vol_target is None:
        return pd.Series(1.0, index=df.index)
    ppy = ppy or periods_per_year(df.index)
    asset_ret = df["close"].pct_change().fillna(0.0)
    realized = asset_ret.rolling(vol_lookback).std() * np.sqrt(ppy)
    scale = (vol_target / realized.replace(0, np.nan)).clip(upper=max_leverage)
    return scale.fillna(0.0)


def run(df: pd.DataFrame,
        signal: pd.Series,
        costs: CostModel | None = None,
        vol_target: float | None = 0.15,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        rebalance_band: float = 0.10,
        warmup: int = 0) -> BacktestResult:
    """Backtest a signal series against price data.

    vol_target: annualized volatility to size to. None disables sizing (raw ±1).
    """
    costs = costs or CostModel()
    ppy = periods_per_year(df.index)

    asset_ret = df["close"].pct_change().fillna(0.0)

    # --- position sizing from TRAILING volatility only ---
    scale = vol_scale(df, vol_target, vol_lookback, max_leverage, ppy)
    desired = (signal * scale).fillna(0.0)

    # --- no-trade band: don't churn on tiny vol-targeting adjustments ---
    if rebalance_band > 0:
        vals = desired.values
        out = np.empty_like(vals)
        current = 0.0
        for i, want in enumerate(vals):
            if abs(want - current) > rebalance_band or np.sign(want) != np.sign(current):
                current = want
            out[i] = current
        desired = pd.Series(out, index=desired.index)

    # --- the one-bar lag: you cannot trade on a bar you have not finished ---
    held = desired.shift(1).fillna(0.0)

    if warmup:
        held.iloc[:warmup] = 0.0

    traded = held.diff().abs().fillna(held.abs())
    cost_series = traded * costs.per_unit

    gross = held * asset_ret
    net = gross - cost_series

    equity = (1.0 + net).cumprod()

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity=equity,
        position=held,
        costs=cost_series,
        ppy=ppy,
    )

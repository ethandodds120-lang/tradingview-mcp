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
    # 1 for a single instrument. For a panel this is the width of the universe
    # and `position` holds gross exposure rather than a signed position, which
    # changes what a "trade" means — see metrics.summary.
    n_assets: int = 1
    traded: pd.Series | None = None      # per-bar turnover, panels only

    @property
    def is_panel(self) -> bool:
        return self.n_assets > 1

    @property
    def turnover(self) -> float:
        if self.traded is not None:
            return float(self.traded.sum())
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


def run_panel(prices: pd.DataFrame,
              weights: pd.DataFrame,
              costs: CostModel | None = None,
              vol_target: float | None = 0.15,
              vol_lookback: int = 60,
              max_leverage: float = 2.0,
              rebalance_band: float = 0.10,
              warmup: int = 0) -> BacktestResult:
    """run(), generalised from one instrument to a whole universe.

    `prices` is dates x tickers. `weights` is the same shape: the fraction of
    equity to hold in each name going into the next bar. Rows need not sum to
    anything in particular — dollar-neutral books sum to zero, long-only to one.

    Every rule the single-asset engine enforces is kept, not relaxed:

      one-bar lag      held = desired.shift(1), applied to the whole row. You
                       cannot trade a bar you have not finished, and that is no
                       less true of twenty instruments than of one.
      costs            charged on the L1 turnover of the row, so opening a leg
                       and closing another both cost, as they do in the market.
      vol targeting    sized off the BOOK's trailing vol, never each leg's own.
                       Scaling legs independently is what makes correlated names
                       stack — two trend models on gold and silver at 0.79
                       correlation are one position held twice, and per-leg
                       sizing cannot see that. This is the same defect
                       `paper.py portfolio` exists to report, fixed here by
                       construction.

    The scale is computed from the *unscaled* book's returns to avoid the
    circularity of sizing off a series that already contains the sizing. Those
    returns at bar t use weights from t-1, so the scale at t is knowable at t.
    """
    costs = costs or CostModel()
    ppy = periods_per_year(prices.index)

    asset_ret = prices.pct_change().fillna(0.0)
    w = weights.reindex(prices.index).fillna(0.0).reindex(columns=prices.columns,
                                                          fill_value=0.0)

    # --- size the book from its own trailing volatility, not each leg's ---
    if vol_target is None:
        scale = pd.Series(1.0, index=prices.index)
    else:
        raw_book = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)
        realised = raw_book.rolling(vol_lookback).std() * np.sqrt(ppy)
        scale = (vol_target / realised.replace(0, np.nan)).clip(upper=max_leverage)
        scale = scale.fillna(0.0)

    desired = w.mul(scale, axis=0)

    # --- no-trade band, on the row's L1 distance ---
    # The single-asset band compares one number to a threshold. The portfolio
    # analogue is total turnover: hold the whole row unless the book has drifted
    # far enough to be worth paying for, which also stops a panel rebalancing
    # twenty legs to chase a rounding error in the vol scale.
    if rebalance_band > 0:
        vals = desired.values
        out = np.empty_like(vals)
        current = np.zeros(vals.shape[1])
        for i in range(vals.shape[0]):
            if np.abs(vals[i] - current).sum() > rebalance_band:
                current = vals[i]
            out[i] = current
        desired = pd.DataFrame(out, index=desired.index, columns=desired.columns)

    # --- the one-bar lag ---
    held = desired.shift(1).fillna(0.0)
    if warmup:
        held.iloc[:warmup] = 0.0

    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = held.iloc[0].abs().sum()
    cost_series = traded * costs.per_unit

    gross = (held * asset_ret).sum(axis=1)
    net = gross - cost_series

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity=(1.0 + net).cumprod(),
        position=held.abs().sum(axis=1),     # gross exposure
        costs=cost_series,
        ppy=ppy,
        n_assets=int(prices.shape[1]),
        traded=traded,
    )

"""Performance metrics, including the ones people skip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def sharpe(returns: pd.Series, ppy: float, rf: float = 0.0) -> float:
    ex = returns - rf / ppy
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(ex.mean() / sd * np.sqrt(ppy))


def sortino(returns: pd.Series, ppy: float) -> float:
    downside = returns[returns < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(returns.mean() / downside * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def longest_drawdown_bars(equity: pd.Series) -> int:
    peak = equity.cummax()
    underwater = equity < peak
    best = cur = 0
    for flag in underwater.values:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return int(best)


def cagr(equity: pd.Series, ppy: float) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[-1] <= 0:
        return float("nan")
    years = n / ppy
    return float(equity.iloc[-1] ** (1 / years) - 1.0)


def t_stat(returns: pd.Series) -> float:
    """t-statistic of mean return. Below ~2 you have not shown anything,
    and with parameter search even 3 is weak."""
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(len(returns)))


def summary(res: BacktestResult) -> dict:
    eq, r, ppy = res.equity, res.returns, res.ppy
    if res.is_panel:
        # A panel's `position` is gross exposure and never changes sign, so
        # counting sign flips reports ~1 trade for a book that rebalances every
        # month. Count the bars on which the book actually moved instead.
        trades = int((res.traded > 1e-9).sum())
    else:
        side = np.sign(res.position)
        trades = int((side.diff().abs() > 0).sum())
    gross_sh = sharpe(res.gross_returns, ppy)
    net_sh = sharpe(r, ppy)
    return {
        "cagr": cagr(eq, ppy),
        "sharpe": net_sh,
        "sharpe_gross": gross_sh,
        "cost_drag_sharpe": gross_sh - net_sh,
        "sortino": sortino(r, ppy),
        "max_dd": max_drawdown(eq),
        "calmar": (cagr(eq, ppy) / abs(max_drawdown(eq))) if max_drawdown(eq) < 0 else float("nan"),
        "longest_dd_bars": longest_drawdown_bars(eq),
        "t_stat": t_stat(r),
        "total_return": float(eq.iloc[-1] - 1.0),
        "total_costs": float(res.costs.sum()),
        "turnover": res.turnover,
        "trades": trades,
        "exposure": float((res.position.abs() > 1e-9).mean()),
        "bars": len(r),
    }


def format_summary(name: str, s: dict) -> str:
    return (
        f"{name:<14} "
        f"CAGR {s['cagr']:>7.2%}  "
        f"Sharpe {s['sharpe']:>5.2f} (gross {s['sharpe_gross']:>5.2f})  "
        f"MaxDD {s['max_dd']:>7.2%}  "
        f"t {s['t_stat']:>5.2f}  "
        f"trades {s['trades']:>5d}"
    )

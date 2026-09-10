"""The part that matters.

A causality gate plus four tests, each designed to kill a strategy that only looks
good. The gate is not scored — a strategy that fails it is not a weak strategy, it
is a bug, and the four tests below it are measuring a signal that could not have
been traded. A strategy that passes all of them still might not work. A strategy
that fails any of them almost certainly does not.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine, metrics
from .strategies import Strategy, random_entry


# ────────────────────── 0. causality ──────────────────────

def causality_check(df: pd.DataFrame, strat: Strategy, probes: int = 6) -> dict:
    """Rule 1, enforced instead of assumed.

    A causal strategy computes bar t from bars <= t, so its signal on df[:k] must
    equal its signal on the whole frame across every overlapping bar. Recompute on
    truncated history and compare. Any mismatch means information from after the
    cut leaked backwards — a centred rolling window, a .shift(-n), a pivot marked
    at the bar it printed rather than at the bar it was confirmed.

    Every overlapping bar is compared, including the last one of the truncation —
    that is the only bar a .shift(-1) corrupts, so excluding it would let the most
    common leak through. Path-dependent models must therefore not flatten on the
    final bar of their input; see the eod handling in tjr.py.
    """
    n = len(df)
    full = strat.signal(df)
    worst, first_bad, checked = 0.0, None, 0
    for frac in np.linspace(0.4, 0.95, probes):
        k = int(n * frac)
        if k < 60:
            continue
        part = strat.signal(df.iloc[:k])
        diff = (full.iloc[:k] - part).abs()
        checked += 1
        worst = max(worst, float(diff.max()))
        if first_bad is None and float(diff.max()) > 1e-9:
            first_bad = diff.index[diff.values.argmax()]
    return {"probes": checked, "max_diff": worst, "clean": worst <= 1e-9,
            "first_mismatch": first_bad}


# ────────────────────── 1. walk-forward ──────────────────────

@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame

    def summary(self, ppy: float) -> dict:
        eq = (1 + self.oos_returns).cumprod()
        res = engine.BacktestResult(
            returns=self.oos_returns,
            gross_returns=self.oos_returns,
            equity=eq,
            position=pd.Series(0.0, index=self.oos_returns.index),
            costs=pd.Series(0.0, index=self.oos_returns.index),
            ppy=ppy,
        )
        s = metrics.summary(res)
        s["folds_positive"] = float((self.folds["oos_sharpe"] > 0).mean())
        return s


def walk_forward(df: pd.DataFrame, strat: Strategy, costs: engine.CostModel,
                 n_folds: int = 5, train_frac: float = 0.6, **bt_kwargs) -> WalkForwardResult:
    """Anchored-window walk-forward.

    For each fold: pick the best parameters on the in-sample window, then apply
    them — unchanged — to the never-before-seen out-of-sample window. Only the
    out-of-sample returns are collected.

    This is the single biggest difference between a backtest and evidence.
    """
    if not strat.grid:
        combos = [dict(strat.params)]
    else:
        keys = list(strat.grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*[strat.grid[k] for k in keys])]

    n = len(df)
    fold_size = n // (n_folds + 1)
    rows, oos_chunks = [], []

    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        test_end = min(fold_size * (k + 2), n)
        train = df.iloc[:train_end]
        test = df.iloc[max(0, train_end - 600):test_end]  # carry lookback context

        best, best_sh = None, -np.inf
        for combo in combos:
            try:
                sig = strat.signal(train, **combo)
                r = engine.run(train, sig, costs, **bt_kwargs)
                sh = metrics.sharpe(r.returns, r.ppy)
            except Exception:
                continue
            if sh > best_sh:
                best_sh, best = sh, combo

        if best is None:
            continue

        sig = strat.signal(test, **best)
        r = engine.run(test, sig, costs, **bt_kwargs)
        oos = r.returns.iloc[-(test_end - train_end):]
        oos_chunks.append(oos)

        rows.append({
            "fold": k + 1,
            "train_end": df.index[train_end - 1],
            "is_sharpe": best_sh,
            "oos_sharpe": metrics.sharpe(oos, r.ppy),
            **{f"p_{key}": val for key, val in best.items()},
        })

    oos_returns = pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float)
    return WalkForwardResult(oos_returns, pd.DataFrame(rows))


# ────────────────────── 2. parameter sensitivity ──────────────────────

def parameter_sensitivity(df: pd.DataFrame, strat: Strategy,
                          costs: engine.CostModel, **bt_kwargs) -> pd.DataFrame:
    """Sharpe across the whole parameter grid.

    A real effect is a plateau: neighbouring parameters give similar results.
    A single tall spike surrounded by garbage is a coincidence you found by looking.
    """
    if not strat.grid:
        return pd.DataFrame()
    keys = list(strat.grid)
    rows = []
    for values in itertools.product(*[strat.grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        try:
            sig = strat.signal(df, **combo)
            r = engine.run(df, sig, costs, **bt_kwargs)
            rows.append({**combo, "sharpe": metrics.sharpe(r.returns, r.ppy),
                         "max_dd": metrics.max_drawdown(r.equity)})
        except Exception:
            continue
    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return out


# ────────────────────── 3. random-entry benchmark ──────────────────────

def random_benchmark(df: pd.DataFrame, strat_sharpe: float, trade_rate: float,
                     hold: int, costs: engine.CostModel, n_trials: int = 500,
                     **bt_kwargs) -> dict:
    """Compare against coin-flip strategies with the same trade frequency.

    Returns the percentile your strategy occupies. Below ~95 means you have not
    distinguished your rules from randomness on this data.
    """
    sharpes = []
    for seed in range(n_trials):
        sig = random_entry(df, trade_rate=trade_rate, seed=seed, hold=hold)
        r = engine.run(df, sig, costs, **bt_kwargs)
        sharpes.append(metrics.sharpe(r.returns, r.ppy))
    arr = np.array(sharpes)
    return {
        "percentile": float((arr < strat_sharpe).mean() * 100),
        "random_mean": float(arr.mean()),
        "random_std": float(arr.std()),
        "random_p95": float(np.percentile(arr, 95)),
        "random_max": float(arr.max()),
        "strategy": strat_sharpe,
        "trials": n_trials,
    }


# ────────────────────── 4. cost sweep ──────────────────────

def cost_sweep(df: pd.DataFrame, strat: Strategy,
               bps_levels=(0, 1, 2, 5, 10, 20, 50), **bt_kwargs) -> pd.DataFrame:
    """At what round-trip cost does the edge die?

    If the answer is "just above where I actually trade", you do not have an edge,
    you have a rounding error.
    """
    rows = []
    for bps in bps_levels:
        c = engine.CostModel(commission_bps=bps / 2, slippage_bps=bps / 2)
        sig = strat.signal(df)
        r = engine.run(df, sig, c, **bt_kwargs)
        rows.append({"round_trip_bps": bps, "sharpe": metrics.sharpe(r.returns, r.ppy),
                     "cagr": metrics.cagr(r.equity, r.ppy)})
    return pd.DataFrame(rows)


# ────────────────────── verdict ──────────────────────

def verdict(wf_sharpe: float, folds_positive: float, sens: pd.DataFrame,
            rand_pct: float | None, breakeven_bps: float, live_bps: float) -> list[str]:
    notes = []
    notes.append(("PASS" if wf_sharpe > 0.3 else "FAIL") +
                 f"  walk-forward OOS Sharpe {wf_sharpe:.2f} (want > 0.3)")
    notes.append(("PASS" if folds_positive >= 0.6 else "FAIL") +
                 f"  {folds_positive:.0%} of folds positive (want >= 60%)")
    if not sens.empty:
        frac = (sens["sharpe"] > 0).mean()
        notes.append(("PASS" if frac >= 0.6 else "FAIL") +
                     f"  {frac:.0%} of parameter settings positive (want >= 60%)")
    if rand_pct is None:
        notes.append("SKIP  random-entry benchmark not run (--quick)")
    else:
        notes.append(("PASS" if rand_pct >= 95 else "FAIL") +
                     f"  beats {rand_pct:.0f}% of random entries (want >= 95%)")
    notes.append(("PASS" if breakeven_bps > live_bps * 3 else "FAIL") +
                 f"  survives to {breakeven_bps:.0f}bps cost vs {live_bps:.0f}bps live (want 3x headroom)")
    return notes

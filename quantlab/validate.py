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
from statistics import NormalDist

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

def trial_matrix(df: pd.DataFrame, strat: Strategy, costs: engine.CostModel,
                 **bt_kwargs) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Run the whole parameter grid once, keeping every trial's return series.

    Returns (rows, matrix, ppy) where `rows` is one summary record per surviving
    combination in grid order, and `matrix` is the T x N array of per-bar returns
    with columns in that same order. The tests below need the full series, not
    just the summary: deflating a Sharpe needs the dispersion of Sharpes across
    trials, and the overfitting probability needs to re-rank every trial on
    subsets of the timeline.

    One pass, because on a path-dependent model like tjr the grid is the
    expensive part of the gauntlet and running it twice would double the wait.
    """
    if not strat.grid:
        return pd.DataFrame(), np.empty((len(df), 0)), 0.0
    keys = list(strat.grid)
    rows, cols, ppy = [], [], 0.0
    for values in itertools.product(*[strat.grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        try:
            sig = strat.signal(df, **combo)
            r = engine.run(df, sig, costs, **bt_kwargs)
            rows.append({**combo, "sharpe": metrics.sharpe(r.returns, r.ppy),
                         "max_dd": metrics.max_drawdown(r.equity)})
            cols.append(r.returns.values)
            ppy = r.ppy
        except Exception:
            continue
    matrix = np.column_stack(cols) if cols else np.empty((len(df), 0))
    return pd.DataFrame(rows), matrix, ppy


def parameter_sensitivity(df: pd.DataFrame, strat: Strategy,
                          costs: engine.CostModel, **bt_kwargs) -> pd.DataFrame:
    """Sharpe across the whole parameter grid.

    A real effect is a plateau: neighbouring parameters give similar results.
    A single tall spike surrounded by garbage is a coincidence you found by looking.
    """
    rows, _, _ = trial_matrix(df, strat, costs, **bt_kwargs)
    if rows.empty:
        return pd.DataFrame()
    return rows.sort_values("sharpe", ascending=False)


# ────────────────── 2b. what the search itself cost you ──────────────────
#
# Everything above reports the best of N parameter combinations. That number is
# biased upward by construction: search enough settings on noise and one of them
# looks good. These two tests price that bias in. Neither is optional once a grid
# exists — sensitivity already tells you the spread across trials, and these turn
# that spread into "how much of the winner was luck".

_EULER = 0.5772156649015329          # Euler-Mascheroni


def _phi(x: float) -> float:
    return NormalDist().cdf(x)


def _phi_inv(p: float) -> float:
    return NormalDist().inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def expected_max_sharpe(trial_sharpes: np.ndarray, n_trials: int) -> float:
    """The Sharpe the *best* of N independent noise strategies would show anyway.

    Bailey & Lopez de Prado's approximation to the expected maximum of N draws
    from a normal, scaled by the dispersion of Sharpes actually observed across
    the grid. This is the benchmark a real strategy has to clear — not zero.
    """
    if n_trials < 2 or len(trial_sharpes) < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * ((1 - _EULER) * _phi_inv(1 - 1 / n_trials)
                 + _EULER * _phi_inv(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(returns: pd.Series, ppy: float, trial_sharpes: np.ndarray,
                    n_trials: int | None = None) -> dict:
    """Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014).

    The probability that the observed Sharpe exceeds what the best of `n_trials`
    tries would produce on data with no edge at all, corrected for the skew and
    fat tails of the actual return distribution.

    Read it as a probability, not a ratio: below ~0.95 means the result is not
    distinguishable from the best of however many things you tried. Note the
    Sharpes here are per-bar, not annualized — annualizing before deflating is
    the standard way to get a wrong answer, because the sqrt(ppy) factor inflates
    the observed Sharpe and the benchmark by different amounts.
    """
    n = len(returns)
    if n < 3 or len(trial_sharpes) < 2:
        return {"dsr": float("nan"), "psr": float("nan"), "sr0": float("nan"),
                "trials": n_trials or len(trial_sharpes)}

    n_trials = n_trials or len(trial_sharpes)
    scale = np.sqrt(ppy)
    sr = metrics.sharpe(returns, ppy) / scale            # back to per-bar
    sr0 = expected_max_sharpe(np.asarray(trial_sharpes) / scale, n_trials)

    r = returns.values
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return {"dsr": float("nan"), "psr": float("nan"), "sr0": sr0 * scale,
                "trials": n_trials}
    z = (r - r.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())                        # non-excess; normal = 3

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return {"dsr": float("nan"), "psr": float("nan"), "sr0": sr0 * scale,
                "trials": n_trials}

    dsr = _phi((sr - sr0) * np.sqrt(n - 1) / np.sqrt(denom))
    psr = _phi(sr * np.sqrt(n - 1) / np.sqrt(denom))     # against a zero benchmark
    return {"dsr": float(dsr), "psr": float(psr), "sr0": float(sr0 * scale),
            "trials": n_trials, "skew": skew, "kurtosis": kurt}


def probability_of_backtest_overfitting(matrix: np.ndarray, n_splits: int = 8) -> dict:
    """PBO via combinatorially symmetric cross-validation — Bailey et al. (2017).

    Chop the timeline into `n_splits` blocks, and for every way of splitting them
    into equal in-sample and out-of-sample halves, pick the trial that won
    in-sample and see where it ranks out-of-sample. PBO is how often the winner
    lands in the bottom half — i.e. how often your selection procedure would have
    chosen a below-median strategy.

    Above 0.5 means the search is worse than picking at random, which is the
    signature of a grid fitted to noise rather than to an effect.

    READ A SINGLE ESTIMATE LOOSELY. Measured on pure noise across 25 seeds this
    returns a mean of 0.54 — correctly centred — but with a standard deviation of
    0.19 and a range of 0.21 to 0.90. The splits share blocks, so they are not
    independent and the effective sample is far smaller than the split count
    suggests. A value of 0.6 is barely half a standard deviation from neutral and
    means very little on its own; 0.85 means something. That is why the gate sits
    at 0.6 rather than 0.5 — a gate on the nominal boundary would fail an
    honest strategy roughly half the time.
    """
    T, N = matrix.shape
    if N < 2 or T < n_splits * 2:
        return {"pbo": float("nan"), "splits": 0, "trials": N}

    n_splits -= n_splits % 2                              # needs an even count
    edges = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [matrix[edges[i]:edges[i + 1]] for i in range(n_splits)]

    def sharpes(rows: np.ndarray) -> np.ndarray:
        sd = rows.std(axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = np.nan
        return rows.mean(axis=0) / sd

    below, total = 0, 0
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        rest = [i for i in range(n_splits) if i not in combo]
        is_s = sharpes(np.vstack([blocks[i] for i in combo]))
        oos_s = sharpes(np.vstack([blocks[i] for i in rest]))
        if not np.isfinite(is_s).any() or not np.isfinite(oos_s).any():
            continue
        best = int(np.nanargmax(is_s))
        # rank of the in-sample winner among out-of-sample results
        rank = float((oos_s < oos_s[best]).sum()) / max(1, np.isfinite(oos_s).sum() - 1)
        total += 1
        if rank < 0.5:
            below += 1
    return {"pbo": below / total if total else float("nan"),
            "splits": total, "trials": N}


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


# ────────── 5. can the sizing survive the strategy's own bad luck? ──────────

def _stationary_bootstrap(n: int, n_paths: int, mean_block: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (n_paths, n).

    Blocks, not independent draws. Resampling returns one at a time destroys
    volatility clustering, and volatility clustering is *where drawdowns come
    from* — a bad month is consecutive bad days, not bad days sprinkled evenly.
    An IID bootstrap would therefore report a comfortable drawdown distribution
    for a strategy that dies in a real streak, which is the opposite of useful.

    Block lengths are geometric with mean `mean_block`, and the walk wraps at the
    end so every starting point is equally likely.
    """
    p = 1.0 / max(1, mean_block)
    idx = np.empty((n_paths, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_paths)
    for j in range(1, n):
        jump = rng.random(n_paths) < p
        idx[:, j] = np.where(jump, rng.integers(0, n, size=n_paths),
                             (idx[:, j - 1] + 1) % n)
    return idx


def _longest_underwater(equity: np.ndarray) -> np.ndarray:
    """Longest consecutive run below the running peak, per row."""
    uw = equity < np.maximum.accumulate(equity, axis=1)
    run = np.zeros(uw.shape[0], dtype=np.int64)
    best = np.zeros(uw.shape[0], dtype=np.int64)
    for j in range(uw.shape[1]):
        run = (run + 1) * uw[:, j]
        np.maximum(best, run, out=best)
    return best


def drawdown_distribution(returns: pd.Series, ppy: float, n_paths: int = 2000,
                          mean_block: int = 20, ruin_threshold: float = 0.50,
                          seed: int = 0, batch: int = 250) -> dict:
    """What drawdown should you actually expect — not the one you happened to get.

    The backtest's max drawdown is a single draw from a distribution. Sizing to it
    is the standard way to be surprised in month three: the realized figure is
    usually nearer the median than the tail, and the tail is what ends accounts.
    Resample the strategy's own returns in blocks, rebuild the equity curve, and
    read the distribution of worst drawdowns and longest time under water.

    `ruin_threshold` is the loss that would end the experiment — a prop firm's
    max-drawdown rule, or the point past which you would stop. p_ruin is how often
    a path built from this strategy's own returns breaches it.
    """
    r = returns.dropna().values.astype(float)
    n = len(r)
    if n < 60:
        return {"paths": 0, "p_ruin": float("nan")}

    rng = np.random.default_rng(seed)
    dds, tuws = [], []
    for start in range(0, n_paths, batch):
        k = min(batch, n_paths - start)
        idx = _stationary_bootstrap(n, k, mean_block, rng)
        eq = np.cumprod(1.0 + r[idx], axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dds.append((eq / peak - 1.0).min(axis=1))
        tuws.append(_longest_underwater(eq))
    dd = np.concatenate(dds)
    tuw = np.concatenate(tuws)

    realized_eq = np.cumprod(1.0 + r)
    realized_dd = float((realized_eq / np.maximum.accumulate(realized_eq) - 1.0).min())

    return {
        "paths": int(n_paths),
        "realized_dd": realized_dd,
        # percentiles of a negative quantity: 5th is the *worst* tail
        "dd_median": float(np.percentile(dd, 50)),
        "dd_p95": float(np.percentile(dd, 5)),
        "dd_p99": float(np.percentile(dd, 1)),
        "realized_pct": float((dd < realized_dd).mean() * 100),
        "tuw_median_bars": float(np.percentile(tuw, 50)),
        "tuw_p95_bars": float(np.percentile(tuw, 95)),
        "tuw_p95_years": float(np.percentile(tuw, 95) / ppy) if ppy else float("nan"),
        "p_ruin": float((dd <= -abs(ruin_threshold)).mean()),
        "ruin_threshold": abs(ruin_threshold),
    }


# ────────────────────── verdict ──────────────────────

def verdict(wf_sharpe: float, folds_positive: float, sens: pd.DataFrame,
            rand_pct: float | None, breakeven_bps: float, live_bps: float,
            dsr: float | None = None, pbo: float | None = None,
            t_stat: float | None = None, ruin: dict | None = None) -> list[str]:
    """The gates. The last three default to None so older callers keep working.

    dsr / pbo / t_stat price in the cost of the search itself. A grid of 36
    settings gets 36 chances to look good on noise, and nothing above this line
    knows that. The t-stat bar is 3.0, not 2.0, following Harvey, Liu & Zhu —
    with the number of strategies the literature has already tried, 2.0 stopped
    meaning anything.
    """
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

    # ── gates that price in the search itself ──
    if dsr is not None and np.isfinite(dsr):
        notes.append(("PASS" if dsr >= 0.95 else "FAIL") +
                     f"  deflated Sharpe {dsr:.2f} (want >= 0.95 — the probability "
                     f"the edge survives the search)")
    if pbo is not None and np.isfinite(pbo):
        # 0.6, not the nominal 0.5: a single PBO estimate has a standard deviation
        # near 0.19, so gating on the boundary itself fails honest strategies about
        # half the time. See probability_of_backtest_overfitting.
        notes.append(("PASS" if pbo <= 0.6 else "FAIL") +
                     f"  overfitting probability {pbo:.0%} (want <= 60%; noisy, "
                     f"sd ~19%)")
    if t_stat is not None and np.isfinite(t_stat):
        notes.append(("PASS" if abs(t_stat) >= 3.0 else "FAIL") +
                     f"  t-stat {t_stat:.2f} (want >= 3.0, Harvey-Liu-Zhu bar)")
    if ruin and np.isfinite(ruin.get("p_ruin", float("nan"))):
        # A signal can be real and still end the account. This gate is about the
        # sizing, not the edge, which is why it is scored separately from all of
        # the above rather than folded into the drawdown number.
        notes.append(("PASS" if ruin["p_ruin"] <= 0.05 else "FAIL") +
                     f"  {ruin['p_ruin']:.0%} of resampled paths breach "
                     f"-{ruin['ruin_threshold']:.0%} (want <= 5%)")
    return notes

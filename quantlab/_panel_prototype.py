"""Prototype: what a panel-aware engine would take, and what of the gauntlet survives.

The existing Strategy contract is (df, **params) -> position Series for ONE
instrument, and engine.run backtests one position series against one price
series. Cross-sectional strategies — flow reversal, XS momentum, pairs — rank a
universe and hold many names at once. Nothing in quantlab can express that.

This builds the smallest thing that could work and then tries to run the real
validators against its output, to find out which of them care.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\Redux\tradingview-mcp")

from quantlab import engine, metrics, validate  # noqa: E402
from quantlab.data import periods_per_year      # noqa: E402

CACHE = Path(__file__).parent / "etf_panel.csv"
UNIVERSE = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
            "IWM", "IJH", "IWD", "IWF",
            "AGG", "LQD", "HYG", "TLT", "IEF",
            "GLD", "SLV"]


def load_panel() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    from quantlab import feeds
    cols = {}
    for t in UNIVERSE:
        try:
            s = feeds.AlpacaFeed(symbol=t, timeframe="1D", count=1000).fetch()["close"]
            s.index = s.index.normalize()
            cols[t] = s[~s.index.duplicated(keep="last")]
        except Exception as exc:
            print(f"  ! {t}: {str(exc)[:60]}")
    px = pd.DataFrame(cols).dropna()
    px.to_csv(CACHE)
    return px


# ─────────────────────────── the generalisation ───────────────────────────

def run_panel(prices: pd.DataFrame, weights: pd.DataFrame,
              costs: engine.CostModel | None = None,
              vol_target: float | None = 0.15, vol_lookback: int = 60,
              max_leverage: float = 2.0) -> engine.BacktestResult:
    """engine.run, but the position is a row of weights instead of a scalar.

    Every rule the single-asset engine enforces is preserved and generalised:
      - the one-bar lag becomes weights.shift(1), applied to the whole row
      - cost is charged on the L1 turnover of the row, not on |delta| of a scalar
      - vol targeting scales the whole book off its own trailing realised vol,
        which is the only honest way to do it: scaling each leg separately is
        what makes correlated legs stack, the exact bug `paper.py portfolio`
        exists to report.

    Returns the same BacktestResult the rest of the package already consumes, so
    metrics and the validators do not need to learn a new type.
    """
    costs = costs or engine.CostModel()
    ppy = periods_per_year(prices.index)
    asset_ret = prices.pct_change().fillna(0.0)

    w = weights.reindex(prices.index).fillna(0.0)
    held = w.shift(1).fillna(0.0)                    # the one-bar lag, unchanged

    gross = (held * asset_ret).sum(axis=1)

    if vol_target is not None:
        realised = gross.rolling(vol_lookback).std() * np.sqrt(ppy)
        scale = (vol_target / realised.replace(0, np.nan)).clip(upper=max_leverage)
        # trailing only, and shifted so today's size uses vol known yesterday
        scale = scale.shift(1).fillna(0.0)
    else:
        scale = pd.Series(1.0, index=prices.index)

    held = held.mul(scale, axis=0)
    gross = (held * asset_ret).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    cost_series = turnover * costs.per_unit
    net = gross - cost_series

    return engine.BacktestResult(
        returns=net, gross_returns=gross, equity=(1.0 + net).cumprod(),
        position=held.abs().sum(axis=1),          # gross exposure, for reporting
        costs=cost_series, ppy=ppy,
    )


def xs_momentum(prices: pd.DataFrame, lookback: int = 126, cut: float = 0.2,
                hold: int = 21) -> pd.DataFrame:
    """Cross-sectional momentum: long the best `cut` of the universe, short the
    worst, rebalanced every `hold` bars. Dollar-neutral, equal-weight.

    Uses only trailing returns, and the weights row for date t is built from data
    through t — the engine's shift(1) is what stops it being traded on t.
    """
    signal = prices.pct_change(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    k = max(1, int(len(prices.columns) * cut))
    for i in range(lookback, len(prices), hold):
        row = signal.iloc[i].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values()
        block = pd.Series(0.0, index=prices.columns)
        block[ranked.tail(k).index] = 0.5 / k
        block[ranked.head(k).index] = -0.5 / k
        w.iloc[i:i + hold] = block.values
    return w


if __name__ == "__main__":
    px = load_panel()
    print(f"panel: {px.shape[1]} ETFs x {len(px)} bars  "
          f"{px.index[0].date()} -> {px.index[-1].date()}\n")

    costs = engine.CostModel(commission_bps=0.0, slippage_bps=2.0)
    res = run_panel(px, xs_momentum(px), costs)
    s = metrics.summary(res)
    print(f"  xs_momentum   Sharpe {s['sharpe']:.2f}  CAGR {s['cagr']:.2%}  "
          f"MaxDD {s['max_dd']:.2%}  t {s['t_stat']:.2f}")

    # Do the new gates work on a cross-sectional return series, unchanged?
    print("\n  running the gauntlet's return-series gates against panel output:")
    sh = []
    for lb in (63, 126, 252):
        for cut in (0.1, 0.2):
            r = run_panel(px, xs_momentum(px, lookback=lb, cut=cut), costs)
            sh.append(metrics.sharpe(r.returns, r.ppy))
    best = int(np.argmax(sh))
    lb, cut = [(l, c) for l in (63, 126, 252) for c in (0.1, 0.2)][best]
    best_res = run_panel(px, xs_momentum(px, lookback=lb, cut=cut), costs)

    ds = validate.deflated_sharpe(best_res.returns, best_res.ppy, np.array(sh))
    dd = validate.drawdown_distribution(best_res.returns, best_res.ppy, n_paths=1000)
    print(f"    deflated_sharpe        -> DSR {ds['dsr']:.3f} over {ds['trials']} trials  OK")
    print(f"    drawdown_distribution  -> realised {dd['realized_dd']:.1%}, "
          f"95th pct {dd['dd_p95']:.1%}  OK")

    mat = np.column_stack([run_panel(px, xs_momentum(px, lookback=l, cut=c),
                                     costs).returns.values
                           for l in (63, 126, 252) for c in (0.1, 0.2)])
    pb = validate.probability_of_backtest_overfitting(mat)
    print(f"    pbo                    -> {pb['pbo']:.0%} over {pb['splits']} splits  OK")

    # And the ones that take a Strategy + single-asset df?
    print("\n  validators that take (df, Strategy):")
    for fn in ("causality_check", "walk_forward", "parameter_sensitivity",
               "random_benchmark", "cost_sweep"):
        print(f"    {fn:<22} -> needs a single-asset df and a Strategy object")

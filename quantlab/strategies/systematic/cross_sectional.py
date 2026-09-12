"""Cross-sectional strategies — rank a universe, hold the ends.

These take a dates x tickers price frame and return a dates x tickers weight
frame. That is a different contract from the rest of the package, and the reason
is not cosmetic: a cross-sectional claim is about *relative* performance, so it
cannot be expressed one instrument at a time. "Long the strongest fifth" has no
meaning without the other four fifths to rank against.

The claim these make is the same shape as the rest of the systematic family — a
statistical property measured across the whole sample — but measured across
names at a point in time rather than across time for one name. The evidence is
also the strongest in the literature: Jegadeesh & Titman (1993) and the
"momentum everywhere" replications are cross-sectional results.

Two things to remember when writing another one:

  * Weights for date t may use data through t. The engine's shift(1) is what
    stops them being traded on t, exactly as with a single-asset signal.
  * Rows do not have to sum to anything. Dollar-neutral books sum to zero and
    long-only books sum to one; run_panel sizes whatever it is given off the
    book's own realised volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def xs_momentum(prices: pd.DataFrame, lookback: int = 126, cut: float = 0.2,
                hold: int = 21, long_only: bool = False) -> pd.DataFrame:
    """Long the best `cut` of the universe, short the worst, rebalanced every
    `hold` bars.

    The oldest documented anomaly in the equity literature and still the most
    replicated. It is also the one with the ugliest tail: Daniel & Moskowitz
    document momentum crashes of up to -90% in a month, concentrated after market
    declines, driven by the short leg behaving like a written option. The vol
    targeting in run_panel damps that but does not remove it, and `long_only`
    sidesteps the short leg entirely at the cost of carrying full market beta.

    Equal weight within each leg. Weighting by signal strength is a second
    parameter, and every parameter added here raises the bar the strategy has to
    clear once the search is deflated.
    """
    ranked_on = prices.pct_change(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    k = max(1, int(prices.shape[1] * cut))

    for i in range(lookback, len(prices), hold):
        row = ranked_on.iloc[i].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values()
        block = pd.Series(0.0, index=prices.columns)
        if long_only:
            block[ranked.tail(k).index] = 1.0 / k
        else:
            block[ranked.tail(k).index] = 0.5 / k
            block[ranked.head(k).index] = -0.5 / k
        w.iloc[i:i + hold] = block.values
    return w


def xs_reversal(prices: pd.DataFrame, lookback: int = 5, cut: float = 0.2,
                hold: int = 5) -> pd.DataFrame:
    """The mirror: long the worst performers over a short window, short the best.

    Short-horizon reversal is real in the literature but Heston, Korajczyk &
    Sadka attribute most of it to temporary liquidity imbalance and bid-ask
    bounce — effects a retail taker pays rather than earns. Included because a
    family needs its failure case, and this is the cross-sectional one: expect
    the cost sweep to kill it.
    """
    ranked_on = prices.pct_change(lookback)
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    k = max(1, int(prices.shape[1] * cut))

    for i in range(lookback, len(prices), hold):
        row = ranked_on.iloc[i].dropna()
        if len(row) < 2 * k:
            continue
        ranked = row.sort_values()
        block = pd.Series(0.0, index=prices.columns)
        block[ranked.head(k).index] = 0.5 / k          # losers -> long
        block[ranked.tail(k).index] = -0.5 / k         # winners -> short
        w.iloc[i:i + hold] = block.values
    return w

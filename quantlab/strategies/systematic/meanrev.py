"""Mean reversion — strategies that harvest short-horizon negative autocorrelation.

The mirror of momentum.py: over short windows, an unusual move tends to be
partially given back. Same kind of claim as trend — a statistical property
measured across the whole sample, not a reading of any individual bar — and the
same obligation to survive costs, which at short horizons is where it usually
dies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_meanrev(df: pd.DataFrame, length: int = 2, lower: float = 10, upper: float = 90) -> pd.Series:
    """Short-term mean reversion. Included because it backtests beautifully and
    usually dies on costs — a useful demonstration."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    pos = np.where(rsi < lower, 1.0, np.where(rsi > upper, -1.0, np.nan))
    return pd.Series(pos, index=df.index).ffill().fillna(0.0)

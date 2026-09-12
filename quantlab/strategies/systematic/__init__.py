"""Systematic strategies — a statistical property persists.

These take no view on any individual bar. They measure one property across the
whole dataset — return autocorrelation, volatility clustering, drift — and hold a
position sized by it. There is no story about why price *should* move here, and no
setup being read; on any given bar the rule is as likely to be wrong as right, and
the claim was never about that bar.

They still predict. A momentum rule says next period's return is related to the
last period's, which is a prediction and a falsifiable one. What distinguishes it
from the predictive family is what the claim rests on: a property that has
persisted across decades, markets and independent research groups, rather than a
formation implying intent.

That pedigree is a reason to take the prior seriously. It is not a reason to skip
the gauntlet — an effect published on equity indices from 1965 is not thereby true
of SOL/USD on daily bars in 2026, and the harness exists to say which.
"""

from __future__ import annotations

from .cross_sectional import xs_momentum, xs_reversal
from .meanrev import rsi_meanrev
from .momentum import donchian, ma_cross, trend_filter, tsmom

__all__ = ["tsmom", "ma_cross", "donchian", "trend_filter", "rsi_meanrev",
           "xs_momentum", "xs_reversal"]

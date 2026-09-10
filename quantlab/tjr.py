"""Compatibility shim. The TJR model moved to strategies/predictive/tjr.py.

It moved because it is one member of a family — pattern strategies that claim a
setup tells you what price does next — and its swing, sweep, BOS and gap logic is
now shared vocabulary in strategies/predictive/primitives.py rather than inlined
here.

Nothing about the model changed. This module re-exports the public surface so that
existing imports keep working:

    from quantlab import tjr
    tjr.simulate(df, **params)
    tjr.signal(df, **params)
    tjr.trade_stats(sim.trades)
    tjr.format_trade_stats(stats)

New code should import from the new location:

    from quantlab.strategies.predictive import tjr
"""

from __future__ import annotations

from .strategies.predictive.tjr import (  # noqa: F401
    SimResult,
    _TRADE_COLS,
    format_trade_stats,
    signal,
    simulate,
    trade_stats,
)

__all__ = ["SimResult", "simulate", "signal", "trade_stats", "format_trade_stats"]

"""TJR / ICT model: liquidity sweep -> break of structure -> fair value gap entry.

This one is path-dependent. Entry price, stop and target all depend on the order
events actually happened in, so it cannot be written as a vectorized signal the
way the systematic family is. It gets its own module and exposes:

    simulate(df, **params) -> SimResult    trades + position series
    signal(df, **params)   -> pd.Series    adapter for the engine

The model, in words
-------------------
1. SWEEP    price trades through a confirmed swing low (taking the resting stops
            under it) and closes back above it.
2. BOS      within `bos_window` bars, price closes above the most recent confirmed
            swing high that formed after that low. Structure is broken.
3. FVG      the displacement leg that broke structure left a three-bar imbalance
            (low[k] > high[k-2]). A limit order waits at the edge of that gap.
4. RISK     stop beyond the sweep extreme plus an ATR buffer, target at `rr` x risk.

Short setups are the mirror image.

Built from primitives
---------------------
The pattern vocabulary — confirmed_swings, detect_sweep, structure_level,
detect_bos, find_fvg — now lives in primitives.py and is shared with every other
predictive strategy. What stays here is only the state machine: which primitive is
consulted in which order, and what happens to the position when one fires. Adding
order blocks or turtle soup means writing a new state machine over the same
vocabulary, not another copy of the pivot loop.

Lookahead - read this before changing anything
----------------------------------------------
A swing pivot at index p is NOT knowable at bar p. It needs `swing_right` bars to
its right to confirm, so it only becomes admissible at bar p + swing_right. That
guard now lives in primitives.confirmed_swings, which tags every pivot with the
bar it was confirmed on; the loop below admits a pivot only once bar i has reached
that tag, and never reaches past bar i. Skipping this is the single most common
way an ICT backtest produces fake results: detect pivots with a centered window
and every sweep gets identified using bars that had not printed yet.

Two more conservative choices, both deliberate:
  - No fill on the BOS bar itself. Within one bar you cannot tell whether the
    retracement into the gap happened before or after the breaking close.
  - If a bar touches both the entry gap and the stop, the trade is treated as
    stopped out on that bar.

Position series vs trade list
-----------------------------
simulate() reports two different things and they will not agree exactly:
  - `trades` uses the actual limit/stop/target fill prices.
  - `position` is what the vectorized engine consumes; it is close-to-close, held
    from the close of the entry bar to the close of the bar before the exit. A
    trade entered and stopped inside the same bar therefore shows up in `trades`
    as a loss and in `position` as no exposure at all.
Neither is the truth. Report both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .primitives import (
    Bars,
    confirmed_swings,
    detect_bos,
    detect_sweep,
    find_fvg,
    structure_level,
)


@dataclass
class SimResult:
    trades: pd.DataFrame
    position: pd.Series
    params: dict = field(default_factory=dict)


_TRADE_COLS = ["side", "entry_time", "exit_time", "entry", "exit", "stop",
               "target", "r", "bars", "reason"]


def simulate(df: pd.DataFrame,
             swing_left: int = 3,
             swing_right: int = 3,
             bos_window: int = 20,
             fvg_window: int = 12,
             min_fvg_atr: float = 0.10,
             stop_buffer_atr: float = 0.25,
             rr: float = 2.0,
             max_hold: int = 60,
             atr_len: int = 14,
             allow_short: bool = True) -> SimResult:
    """Run the state machine bar by bar. One setup and one position at a time."""
    swing_right = max(1, int(swing_right))
    swing_left = max(1, int(swing_left))

    bars = Bars.from_frame(df, atr_len)
    n = bars.n
    o, h, l, c = bars.open, bars.high, bars.low, bars.close
    atr = bars.atr
    idx = bars.index

    # Every pivot in the frame, each tagged with the bar it became knowable on.
    # Computing them up front is not lookahead: the cursors below refuse to admit
    # one until bar i has actually reached its confirmed_at tag, which is the same
    # schedule the old inlined loop confirmed them on.
    all_highs, all_lows = confirmed_swings(bars, swing_left, swing_right)
    hi_cur = lo_cur = 0

    pos = np.zeros(n)
    swing_highs: list = []      # admitted pivots, oldest -> newest
    swing_lows: list = []
    trades: list = []

    setup = None    # {side, sweep_idx, bos_level, stage, bos_idx, zone}
    trade = None    # {side, entry_idx, entry, stop, target, risk}

    def close_trade(i, price, reason):
        nonlocal trade
        trades.append({
            "side": trade["side"],
            "entry_time": idx[trade["entry_idx"]],
            "exit_time": idx[i],
            "entry": trade["entry"],
            "exit": price,
            "stop": trade["stop"],
            "target": trade["target"],
            "r": trade["side"] * (price - trade["entry"]) / trade["risk"],
            "bars": i - trade["entry_idx"],
            "reason": reason,
        })
        trade = None

    for i in range(n):
        # -- 1. admit the pivots whose right-hand window just closed -----------
        while hi_cur < len(all_highs) and all_highs[hi_cur].confirmed_at <= i:
            swing_highs.append(all_highs[hi_cur])
            hi_cur += 1
        while lo_cur < len(all_lows) and all_lows[lo_cur].confirmed_at <= i:
            swing_lows.append(all_lows[lo_cur])
            lo_cur += 1

        # -- 2. manage the open position --------------------------------------
        if trade is not None:
            side = trade["side"]
            price, reason = None, None
            if side > 0:
                if l[i] <= trade["stop"]:
                    price, reason = trade["stop"], "stop"
                elif h[i] >= trade["target"]:
                    price, reason = trade["target"], "target"
            else:
                if h[i] >= trade["stop"]:
                    price, reason = trade["stop"], "stop"
                elif l[i] <= trade["target"]:
                    price, reason = trade["target"], "target"
            if price is None and i - trade["entry_idx"] >= max_hold:
                price, reason = c[i], "timeout"
            if price is None and i == n - 1:
                price, reason = c[i], "eod"

            if price is not None:
                close_trade(i, price, reason)
                # On the last bar the position value is never used (the engine
                # lags by one and there is no next bar), so hold it rather than
                # flatten it. That keeps the series identical to the one a run on
                # a longer history would produce, which is what makes
                # validate.causality_check able to compare every overlapping bar.
                if reason == "eod":
                    pos[i] = side
            else:
                pos[i] = side
            continue        # no new setup work on a bar we were already in

        # -- 3. advance an existing setup, take the entry if it fills ----------
        if setup is not None:
            side = setup["side"]

            if setup["stage"] == "bos":
                if i - setup["sweep_idx"] > bos_window:
                    setup = None
                elif detect_bos(bars, i, setup["bos_level"], side):
                    setup["stage"], setup["bos_idx"] = "entry", i
                    # the displacement leg may already have left a gap behind it
                    for k in range(i, max(1, setup["sweep_idx"]) - 1, -1):
                        z = find_fvg(bars, k, side, min_fvg_atr)
                        if z and ((side > 0 and z.high <= c[i]) or (side < 0 and z.low >= c[i])):
                            setup["zone"] = z
                            break

            if setup is not None and setup["stage"] == "entry":
                z = find_fvg(bars, i, side, min_fvg_atr)
                if z:
                    setup["zone"] = z           # freshest gap wins
                if i - setup["bos_idx"] > fvg_window:
                    setup = None
                elif setup["zone"] and i > max(setup["bos_idx"], setup["zone"].index):
                    gap_lo, gap_hi = setup["zone"].low, setup["zone"].high
                    entry = None
                    if side > 0 and l[i] <= gap_hi:
                        entry = min(gap_hi, o[i])
                    elif side < 0 and h[i] >= gap_lo:
                        entry = max(gap_lo, o[i])

                    if entry is not None:
                        buf = stop_buffer_atr * atr[i]
                        if side > 0:
                            stop = min(l[setup["sweep_idx"]:i + 1].min(), entry) - buf
                        else:
                            stop = max(h[setup["sweep_idx"]:i + 1].max(), entry) + buf
                        risk = side * (entry - stop)
                        if risk > 0:
                            trade = {"side": side, "entry_idx": i, "entry": entry,
                                     "stop": stop, "target": entry + side * rr * risk,
                                     "risk": risk}
                            pos[i] = side
                            # the same bar can take out the stop - assume it did
                            if (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop):
                                pos[i] = 0.0
                                close_trade(i, stop, "stop")
                        setup = None

            if trade is not None or setup is not None:
                continue

        # -- 4. look for a fresh sweep of resting liquidity ---------------------
        # A pierced level is retired whether or not price reclaimed it: the stops
        # under it are gone either way, so it is no longer liquidity worth taking.
        swept = detect_sweep(bars, i, swing_lows, 1)
        if swept is not None:
            swing_lows.pop()
            if swept.reclaimed:                  # ... and price closed back above it
                level = structure_level(swing_highs, swept.level.index, 1)
                if level is not None:
                    setup = {"side": 1, "sweep_idx": i, "bos_level": level,
                             "stage": "bos", "bos_idx": None, "zone": None}

        if setup is None and allow_short:
            swept = detect_sweep(bars, i, swing_highs, -1)
            if swept is not None:
                swing_highs.pop()
                if swept.reclaimed:
                    level = structure_level(swing_lows, swept.level.index, -1)
                    if level is not None:
                        setup = {"side": -1, "sweep_idx": i, "bos_level": level,
                                 "stage": "bos", "bos_idx": None, "zone": None}

    return SimResult(
        trades=pd.DataFrame(trades, columns=_TRADE_COLS),
        position=pd.Series(pos, index=idx),
        params={"swing_left": swing_left, "swing_right": swing_right,
                "bos_window": bos_window, "fvg_window": fvg_window,
                "min_fvg_atr": min_fvg_atr, "stop_buffer_atr": stop_buffer_atr,
                "rr": rr, "max_hold": max_hold, "atr_len": atr_len,
                "allow_short": allow_short},
    )


def signal(df: pd.DataFrame, **params) -> pd.Series:
    """Adapter so the engine can treat this like any other strategy."""
    return simulate(df, **params).position


def trade_stats(trades: pd.DataFrame) -> dict:
    """Trade-level view. These use real fill prices, not close-to-close."""
    if trades is None or trades.empty:
        return {"trades": 0}
    r = trades["r"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "trades": int(len(r)),
        "long": int((trades["side"] > 0).sum()),
        "short": int((trades["side"] < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "total_r": float(r.sum()),
        "best_r": float(r.max()),
        "worst_r": float(r.min()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_bars": float(trades["bars"].mean()),
        "exits": trades["reason"].value_counts().to_dict(),
    }


def format_trade_stats(s: dict) -> str:
    if not s.get("trades"):
        return "  no trades"
    exits = "  ".join(f"{k} {v}" for k, v in sorted(s["exits"].items()))
    return (
        f"  {s['trades']} trades ({s['long']}L / {s['short']}S)   "
        f"win rate {s['win_rate']:.1%}   avg {s['avg_r']:+.2f}R   "
        f"total {s['total_r']:+.1f}R\n"
        f"  profit factor {s['profit_factor']:.2f}   "
        f"best {s['best_r']:+.1f}R / worst {s['worst_r']:+.1f}R   "
        f"avg hold {s['avg_bars']:.0f} bars\n"
        f"  exits: {exits}"
    )

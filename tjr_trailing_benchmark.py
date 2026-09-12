#!/usr/bin/env python3
"""Part 1 of DESIGN-tjr-human.md: mechanical trailing stops on the existing TJR trades.

Round 1's geometry said the direction was right on 5 of 8 trades and the
median excursion in the trade's favour, after the fixed stop had fired, was
3.4 R. This asks whether any simple "manage the stop" rule harvests that. Same
entries, same initial stop, same target, same 15:55 flat; only the stop's
subsequent behaviour changes. Five rules, pre-specified in §1 of the contract,
counted as five trials.

Read-only. Prints the per-trade table, the totals, and the one structural
number that decides the question: how far each trade went in its favour
BEFORE the fixed stop fired. A trailing stop can only ever tighten, so if that
number is under 1 R the breakeven rule never arms and the ATR rules never
ratchet above the initial stop — the excursion that came after the stop is
unreachable by any exit management that starts from that stop.

    python tjr_trailing_benchmark.py            > results/tjr_intraday/part1_trailing.txt
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantlab import data
from quantlab.strategies import REGISTRY
from quantlab.strategies.predictive import tjr_intraday as M
from quantlab.strategies.predictive.primitives import confirmed_swings

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RULES = ("fixed", "be_1r", "atr1.0", "atr1.5", "swing", "hybrid")
FLAT_MINUTES = 15 * 60 + 55

# every entry the two rounds produced under the round-1 initial stop
SOURCES = [
    # label, timeframe, traded csv, pair csv, registry name
    ("NQ 5m", 5, "data/NQ_5min_tv.csv", "data/ES_5min_tv.csv", "tjr_intraday"),
    ("ES 5m", 5, "data/ES_5min_tv.csv", "data/NQ_5min_tv.csv", "tjr_intraday"),
    ("NQ 1m", 1, "data/NQ_1min_tv.csv", "data/ES_1min_tv.csv", "tjr_intraday_1m"),
    ("ES 1m", 1, "data/ES_1min_tv.csv", "data/NQ_1min_tv.csv", "tjr_intraday_1m"),
]


def replay(df, ctx, clock, swings_low, swings_high, trade, rule) -> dict:
    """One trade under one rule. Everything the stop uses at bar j comes from
    bars <= j-1; the entry bar itself is tested against the initial stop only,
    as the model does."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    side = int(trade.side)
    entry, stop0, target = float(trade.entry), float(trade.stop), float(trade.target)
    i0 = df.index.get_loc(pd.Timestamp(trade.entry_time))
    day = clock.day[i0]
    risk = abs(entry - stop0)
    fav = (lambda p: (p - entry) * side / risk)           # excursion in R, signed

    # entry bar: the same-bar stop rule, nothing else
    if (side > 0 and l[i0] <= stop0) or (side < 0 and h[i0] >= stop0):
        return {"exit_idx": i0, "exit": stop0, "reason": "stop", "r": -1.0,
                "stop_path": [stop0], "armed": False}

    stop = stop0
    armed = False                     # be_1r / hybrid: has +1 R been seen?
    path = [stop0]
    best = h[i0] if side > 0 else l[i0]
    for j in range(i0 + 1, len(df)):
        if clock.day[j] != day:                          # ran off the session
            return {"exit_idx": j - 1, "exit": c[j - 1], "reason": "eod", "r": fav(c[j - 1]),
                    "stop_path": path, "armed": armed}
        # ---- stop for bar j, from bars <= j-1 ----
        best = max(best, h[j - 1]) if side > 0 else min(best, l[j - 1])
        mfe = fav(best)
        if mfe >= 1.0:
            armed = True
        k = ctx.last[j - 1]                              # newest completed context bar
        atr = float(ctx.bars.atr[k]) if k >= 0 else np.nan
        if rule == "fixed":
            want = stop0
        elif rule == "be_1r":
            want = entry if armed else stop0
        elif rule in ("atr1.0", "atr1.5"):
            m = 1.0 if rule == "atr1.0" else 1.5
            want = (best - m * atr) if side > 0 else (best + m * atr)
        elif rule == "swing":
            pool = [s for s in (swings_low if side > 0 else swings_high) if s.confirmed_at <= k]
            want = pool[-1].price if pool else stop0
        elif rule == "hybrid":
            if armed:
                trail = (best - 1.0 * atr) if side > 0 else (best + 1.0 * atr)
                want = max(entry, trail) if side > 0 else min(entry, trail)
            else:
                want = stop0
        else:
            raise ValueError(rule)
        if np.isnan(want):
            want = stop
        stop = max(stop, want) if side > 0 else min(stop, want)   # only ever tightens
        path.append(stop)
        # ---- bar j against that stop, then the target, then the flat ----
        if side > 0 and l[j] <= stop:
            px = min(stop, o[j])                          # a gap through the stop fills at the open
            return {"exit_idx": j, "exit": px, "reason": "stop" if stop == stop0 else "trail",
                    "r": fav(px), "stop_path": path, "armed": armed}
        if side < 0 and h[j] >= stop:
            px = max(stop, o[j])
            return {"exit_idx": j, "exit": px, "reason": "stop" if stop == stop0 else "trail",
                    "r": fav(px), "stop_path": path, "armed": armed}
        if (side > 0 and h[j] >= target) or (side < 0 and l[j] <= target):
            return {"exit_idx": j, "exit": target, "reason": "target", "r": fav(target),
                    "stop_path": path, "armed": armed}
        if clock.minutes[j] >= FLAT_MINUTES:
            return {"exit_idx": j, "exit": c[j], "reason": "flat", "r": fav(c[j]),
                    "stop_path": path, "armed": armed}
    j = len(df) - 1
    return {"exit_idx": j, "exit": c[j], "reason": "eod", "r": fav(c[j]),
            "stop_path": path, "armed": armed}


def excursions(df, clock, trade, exit_idx) -> dict:
    """MAE inside the trade, and what happened after the fixed stop: the best
    favourable excursion over the rest of the session and whether T1 printed."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    side = int(trade.side); entry = float(trade.entry); risk = abs(entry - float(trade.stop))
    i0 = df.index.get_loc(pd.Timestamp(trade.entry_time))
    day = clock.day[i0]
    fav = lambda p: (p - entry) * side / risk
    inside = slice(i0, exit_idx + 1)
    mae = fav(l[inside].min()) if side > 0 else fav(h[inside].max())
    # The excursion a rule could have USED is over completed bars strictly before
    # the stop bar. The stop bar's own high is not usable — a trade stopped on
    # its entry bar went nowhere first, whatever that bar's high says.
    usable = slice(i0, exit_idx)
    mfe_before = (fav(h[usable].max()) if side > 0 else fav(l[usable].min())) if exit_idx > i0 else 0.0
    # "After the stop" means the rest of the same trading session, up to the
    # 15:55 flat. Masking on the ET calendar date instead would sweep in the
    # 18:00 evening bars of the NEXT session, which is how the round-1 write-up
    # came to say 5 of 8 and 3.4 R; the session-bounded numbers are 4 of 8 and
    # about 3.1 R.
    rest = np.arange(exit_idx + 1, len(df))
    rest = rest[(clock.day[rest] == day) & (clock.minutes[rest] <= FLAT_MINUTES)]
    if len(rest) == 0:
        return {"mae": mae, "mfe_before": mfe_before,
                "mfe_after": np.nan, "t1_after": False, "t1_when": None}
    best_after = h[rest].max() if side > 0 else l[rest].min()
    hit = (h[rest] >= trade.target) if side > 0 else (l[rest] <= trade.target)
    when = df.index[rest[np.argmax(hit)]] if hit.any() else None
    return {"mae": mae, "mfe_before": mfe_before,
            "mfe_after": fav(best_after), "t1_after": bool(hit.any()), "t1_when": when}


def et(ts) -> str:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(M.ET).strftime("%H:%M")


def main() -> int:
    rows, excluded = [], []
    for label, tf, csv, pair, name in SOURCES:
        df = data.load_futures_pair(csv, pair)
        params = dict(REGISTRY[name].params)              # stop_mode='wick', the round-1 stop
        sim = M.simulate(df, **params)
        wide = M.simulate(df, **{**params, "stop_mode": "atr1.0"})
        for d in set(wide.trades.session_day) - set(sim.trades.session_day):
            excluded.append(f"{label} {str(d)[:10]}: no valid wick stop (risk <= 0); round 2 admitted it under atr1.0 only")
        if sim.trades.empty:
            continue
        ctx = M.build_context(df, tf, params["atr_len"])
        clock = M.session_clock(df.index, params["flat_at"])
        sh, sl = confirmed_swings(ctx.bars, params["swing_left"], params["swing_right"])
        for _, t in sim.trades.iterrows():
            rec = {"src": label, "day": str(t.session_day)[:10], "side": "L" if t.side > 0 else "S",
                   "level": t.sweep_level, "entry_et": et(t.entry_time), "entry": float(t.entry),
                   "stop": float(t.stop), "target": float(t.target),
                   "risk_bps": abs(t.entry - t.stop) / t.entry * 1e4, "model_r": float(t.r)}
            fixed = None
            for rule in RULES:
                out = replay(df, ctx, clock, sl, sh, t, rule)
                rec[rule] = out["r"]; rec[rule + "_why"] = out["reason"]
                if rule == "fixed":
                    fixed = out
            ex = excursions(df, clock, t, fixed["exit_idx"])
            rec.update({"stop_hit_et": et(df.index[fixed["exit_idx"]]), "mae": ex["mae"],
                        "mfe_before_stop": ex["mfe_before"], "mfe_after_stop": ex["mfe_after"],
                        "t1_after": ex["t1_after"], "t1_when": et(ex["t1_when"]) if ex["t1_when"] is not None else "-"})
            rows.append(rec)
    tbl = pd.DataFrame(rows)

    print("PART 1 — trailing-stop benchmark on the existing TJR entries (initial stop = round 1's wick rule)\n")
    print("reproduction check: 'fixed' replay vs the model's own r —",
          "identical" if np.allclose(tbl["fixed"], tbl["model_r"], atol=1e-9) else "DIFFERS", "\n")

    print(f"{'src':<6}{'day':<12}{'S':>2}{'level':>8}{'entry':>7}{'risk':>7}{'stopped':>9}"
          + "".join(f"{r:>8}" for r in RULES) + f"{'mfe<stop':>10}{'mfe>stop':>10}{'T1 later':>10}")
    for _, r in tbl.iterrows():
        print(f"{r.src:<6}{r.day:<12}{r.side:>2}{r.level:>8}{r.entry_et:>7}{r.risk_bps:>6.1f}b{r.stop_hit_et:>9}"
              + "".join(f"{r[k]:>+8.2f}" for k in RULES)
              + f"{r.mfe_before_stop:>+10.2f}{r.mfe_after_stop:>+10.2f}{('yes ' + r.t1_when) if r.t1_after else 'no':>10}")
    print()
    print("totals (R):   " + "  ".join(f"{k} {tbl[k].sum():+.2f}" for k in RULES))
    print("wins:         " + "  ".join(f"{k} {int((tbl[k] > 0).sum())}/{len(tbl)}" for k in RULES))
    print("exit reasons: " + "  ".join(f"{k} {dict(tbl[k + '_why'].value_counts())}" for k in RULES))
    best = max(RULES[1:], key=lambda k: tbl[k].sum())
    print(f"\nbest trailing rule: {best} at {tbl[best].sum():+.2f} R vs fixed {tbl['fixed'].sum():+.2f} R "
          f"(the human's benchmark, §5)")

    n_armed = int((tbl["mfe_before_stop"] >= 1.0).sum())
    print(f"\nthe structural number: trades whose favourable excursion reached +1 R BEFORE the fixed stop fired: "
          f"{n_armed} of {len(tbl)}")
    print(f"  median favourable excursion before the stop: {tbl['mfe_before_stop'].median():+.2f} R;"
          f"  after it: {tbl['mfe_after_stop'].median():+.2f} R;  T1 printed later on {int(tbl['t1_after'].sum())} of {len(tbl)}")
    print("  a trailing stop only tightens, so nothing that starts from the wick stop can reach an excursion that\n"
          "  arrives after that stop has fired — the rules above can only differ from 'fixed' on the trades where\n"
          "  price first went the trade's way")

    if excluded:
        print("\nexcluded from the baseline (no wick stop to trail from):")
        for e in excluded:
            print("  " + e)

    # ── the eight round-1 trades, the ones T1 printed for beside the ones it did not ──
    r1 = tbl[tbl["src"].str.endswith("5m")].copy()
    print(f"\n\nROUND-1 TRADES ({len(r1)}), split by whether T1 printed later in the session\n")
    hdr = (f"{'day':<12}{'inst':<5}{'S':>2}{'entry':>10}{'stop':>10}{'risk':>7}{'stop hit':>9}"
           f"{'MAE':>7}{'mfe>stop':>10}{'T1 later':>10}{'atr1.0':>8}")
    for flag, title in ((True, "T1 printed after the stop"), (False, "T1 never printed")):
        part = r1[r1["t1_after"] == flag]
        print(f"  {title} — {len(part)} trades")
        print("  " + hdr)
        for _, r in part.iterrows():
            print(f"  {r.day:<12}{r.src[:2]:<5}{r.side:>2}{r.entry:>10.2f}{r.stop:>10.2f}{r.risk_bps:>6.1f}b"
                  f"{r.stop_hit_et:>9}{r.mae:>+7.2f}{r.mfe_after_stop:>+10.2f}"
                  f"{('yes ' + r.t1_when) if r.t1_after else 'no':>10}{r['atr1.0']:>+8.2f}")
        print()
    print(f"  median favourable excursion after the stop, round-1 trades: {r1['mfe_after_stop'].median():+.2f} R "
          f"(the round-1 write-up said 3.4 R over 5 of 8; that measured the ET calendar date and so included the\n"
          f"  next session's evening bars — see excursions())")
    return 0


if __name__ == "__main__":
    sys.exit(main())

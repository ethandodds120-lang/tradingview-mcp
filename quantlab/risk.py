"""Kill rules, the HALTED marker, the epoch, and the heartbeat (DESIGN-risk.md).

Plain code evaluated on every poll of every run that is neither STOPPED nor
HALTED, after the poll has processed its bars. No model anywhere in this loop.
Three rules, each read since the run's epoch (the forward start, or the most
recent `resume`):

    R1  profit factor over the last 60 closed round trips        trips  < 1.0
    R2  ln(E / E_0) against a zero-drift band of the backtest's
        per-bar sigma:  -2 * sigma_bar * sqrt(n)                 trips  below it
    R3  annualised sd of the epoch's per-bar equity returns       trips  > 2 x vol_target

A rule is *armed* only once it has enough data to mean anything (60 trades, 20
bars); an unarmed rule cannot trip. `evaluate` returns every number each rule
read, so the HALTED marker, the journal line and the alert can all carry the
same figures. Nothing here decides, routes, or reads a feed or a broker: it
works from config.json, state.json, journal.jsonl and bars.csv only, which is
what lets `paper.py risk --all` run on a copy of the VPS's directories.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from . import engine
from .strategies import REGISTRY

HALTED_FILE = "HALTED"
UNHALT = "human — paper.py resume"

PF_WINDOW = 60            # R1: closed round trips
PF_MIN = 1.0              # R1 trips below this
ARM_BARS = 20             # R2 and R3 arm at this many epoch bars
BAND_SIGMAS = 2.0         # R2: the band is -BAND_SIGMAS * sigma_bar * sqrt(n)
VOL_MULT = 2.0            # R3 trips above VOL_MULT * vol_target

RULES = ("r1_profit_factor", "r2_equity_band", "r3_realised_vol")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _f(value) -> float | None:
    """A JSON-clean float: None stays None, everything else rounded so the
    marker, the journal line and the alert print the same digits. A non-finite
    value passes through unchanged, so callers must not store one — R1 stores
    a lossless window as None + `no_losses` rather than an `Infinity` token."""
    if value is None:
        return None
    value = float(value)
    return value if not math.isfinite(value) else round(value, 8)


# ────────────────────────────── epoch (§1.1) ──────────────────────────────

def default_epoch(run) -> dict:
    """The forward start: every rule reads the run from here unless a resume
    has moved it. Not stored — derived, so state.json gains nothing for it."""
    return {"bar": run.state.get("forward_start"),
            "equity": float(run.config.get("start_equity") or 0.0),
            "closed_trades_before": 0}


def epoch(run) -> dict:
    return dict(run.state.get("risk_epoch") or default_epoch(run))


def new_epoch(run) -> dict:
    """An epoch starting now: the run's current bar and equity, and every round
    trip closed so far left behind. What `resume` writes."""
    rt = run.round_trips()
    closed = 0 if rt.empty else int((rt["exit"] != "OPEN").sum())
    return {"bar": run.state.get("last_bar"),
            "equity": float(run.state.get("equity") or 0.0),
            "closed_trades_before": closed}


def epoch_bar_records(run, ep: dict) -> list[dict]:
    """Journal bar records after the epoch bar, in journal order."""
    start = pd.Timestamp(ep["bar"]) if ep.get("bar") else None
    out = []
    for rec in run._records():
        if rec.get("type") != "bar" or "equity" not in rec:
            continue
        if start is not None and pd.Timestamp(rec["bar"]) <= start:
            continue
        out.append(rec)
    return out


# ────────────────────────────── HALTED marker (§1.3) ──────────────────────────────

def halted_path(root: str | Path) -> Path:
    return Path(root) / HALTED_FILE


def read_halted(root: str | Path) -> dict:
    try:
        return json.loads(halted_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def make_marker(rule: str, value, threshold, inputs: dict, ep: dict,
                at: str | None = None) -> dict:
    """The §1.3 marker. The same dict goes to the HALTED file, the `halted`
    journal line and the alert, so the three cannot disagree."""
    return {"at": at or _now(), "rule": rule, "value": _f(value),
            "threshold": _f(threshold), "inputs": inputs, "epoch": ep,
            "unhalt": UNHALT}


def write_halted(root: str | Path, marker: dict) -> Path:
    path = halted_path(root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


# ────────────────────────────── the seed distribution (R2) ──────────────────────────────

MIN_SEED_BARS = ARM_BARS      # R2 needs at least max(vol_lookback, this) seed returns


def _empty_profile(note: str, **extra) -> dict:
    return {"returns": pd.Series(dtype=float), "seed_bars": 0, "vol_lookback": None,
            "first_position_bar": None, "warmup_bars": 0, "min_bars": None,
            "note": note, **extra}


def seed_profile(run, bars: pd.DataFrame | None = None) -> dict:
    """The engine's per-bar net returns for this run's strategy and config over
    its seed history — bars with index <= forward_start — once the strategy
    exists: after the vol lookback is populated AND after the strategy's own
    warmup, i.e. from the engine's first non-zero position. A strategy with a
    252-bar lookback holds nothing for its first 252 bars, and those exact
    zeros are not its noise; counting them shrank sigma_bar by a third on the
    live tsmom run and made the band trip on ordinary drawdowns. Bars the
    strategy is genuinely flat AFTER it has started are kept: the band counts
    every epoch bar, flat or not, so the marginal per-bar sd is the one that
    matches it (recorded as a deviation, see DESIGN-risk.md §8).

    Returns the series plus what was dropped, so R2's inputs can say so. The
    `returns` are empty, with `note` saying why, when there is nothing to
    compute on or fewer than max(vol_lookback, MIN_SEED_BARS) bars survive."""
    cfg = run.config
    start = run.state.get("forward_start")
    strat = REGISTRY.get(cfg.get("strategy"))
    if not start:
        return _empty_profile("no forward start yet")
    if strat is None:
        return _empty_profile(f"unknown strategy {cfg.get('strategy')!r}")
    if strat.is_panel:
        return _empty_profile("panel strategy — no single-instrument seed")
    bars = run.bars() if bars is None else bars
    seed = bars[bars.index <= pd.Timestamp(start)]
    lookback = int(cfg.get("vol_lookback") or 0)
    min_bars = max(lookback, MIN_SEED_BARS)
    if len(seed) <= lookback + 2:
        return _empty_profile("seed history too short for the engine's sigma — rule disabled",
                              seed_bars=int(len(seed)), vol_lookback=lookback, min_bars=min_bars)
    bps = float(cfg.get("cost_bps") or 0.0) / 2
    res = engine.run(seed, strat.signal(seed, **cfg.get("params", {})),
                     engine.CostModel(commission_bps=bps, slippage_bps=bps),
                     vol_target=cfg.get("vol_target"), vol_lookback=lookback,
                     max_leverage=cfg.get("max_leverage", 2.0),
                     rebalance_band=cfg.get("rebalance_band", 0.10))
    pos = res.position.fillna(0.0).to_numpy()
    invested = np.flatnonzero(pos != 0)
    first = int(invested[0]) if len(invested) else None
    if first is None:
        return _empty_profile("the strategy never holds a position over the seed — "
                              "no sigma to build a band from — rule disabled",
                              seed_bars=int(len(seed)), vol_lookback=lookback, min_bars=min_bars)
    start_i = max(lookback, first)
    returns = res.returns.iloc[start_i:].astype(float)
    out = {"returns": returns, "seed_bars": int(len(seed)), "vol_lookback": lookback,
           "first_position_bar": str(seed.index[first]),
           "warmup_bars": int(max(0, first - lookback)), "min_bars": min_bars, "note": None}
    if len(returns) < min_bars:
        out["returns"] = pd.Series(dtype=float)
        out["note"] = (f"only {len(returns)} seed bars after the strategy's warmup "
                       f"(first position at bar {first} of {len(seed)}), need {min_bars} — "
                       "rule disabled")
    return out


def seed_returns(run, bars: pd.DataFrame | None = None) -> pd.Series:
    """`seed_profile(run)["returns"]` — the series R2's sigma_bar is the sd of."""
    return seed_profile(run, bars)["returns"]


def false_trip_rate(seed_returns: pd.Series | np.ndarray, block: int = 10,
                    horizon: int = 250, paths: int = 1000, seed: int = 0,
                    sigmas: float = BAND_SIGMAS) -> float | None:
    """How often R2's band is crossed by paths that are only the seed's own
    noise: the fraction of `paths` stationary-block-bootstrap resamples of the
    seed returns (mean block length `block`, `horizon` bars) whose cumulative
    log return dips below -sigmas * sigma * sqrt(n) at any n <= horizon.

    A 2-sigma band checked at every n is crossed far more often than the
    one-shot 2.3 %; this is the number that says how much more, for this run.
    Deterministic: a numpy Generator seeded with `seed`."""
    r = np.asarray(seed_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return None
    sigma = float(r.std(ddof=1))
    if not sigma > 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(r)
    log_r = np.log1p(r)
    band = -sigmas * sigma * np.sqrt(np.arange(1, horizon + 1))
    tripped = 0
    for _ in range(paths):
        idx = np.empty(horizon, dtype=int)
        filled = 0
        while filled < horizon:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(1.0 / block))
            take = min(length, horizon - filled)
            idx[filled:filled + take] = (start + np.arange(take)) % n
            filled += take
        cum = np.cumsum(log_r[idx])
        if bool((cum < band).any()):
            tripped += 1
    return tripped / paths


# ────────────────────────────── the rules (§1.2) ──────────────────────────────

def _r1_profit_factor(run, ep: dict) -> dict:
    rt = run.round_trips()
    closed = rt[rt["exit"] != "OPEN"] if not rt.empty else rt
    before = int(ep.get("closed_trades_before") or 0)
    in_epoch = closed.iloc[before:] if len(closed) > before else closed.iloc[0:0]
    window = in_epoch.tail(PF_WINDOW)
    pnl = window["pnl"].astype(float) if len(window) else pd.Series(dtype=float)
    gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
    if len(window):
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        pf = None
    armed = len(window) >= PF_WINDOW
    no_losses = bool(len(window)) and not gross_loss > 0
    # The trip is decided on the raw float (inf < 1.0 is False). What is
    # stored is JSON-clean: a window with no losses is `value: None` plus
    # `inputs.no_losses: true`, not an `Infinity` token the stdlib happens to
    # write and stricter readers reject.
    return {
        "value": None if no_losses else _f(pf), "threshold": PF_MIN, "armed": armed,
        "trip": bool(armed and pf is not None and pf < PF_MIN),
        "inputs": {
            "window": PF_WINDOW, "closed_in_epoch": int(len(in_epoch)),
            "closed_before_epoch": before, "trades_in_window": int(len(window)),
            "no_losses": no_losses,
            "gross_profit": _f(gross_profit), "gross_loss": _f(gross_loss),
            "wins": int((pnl > 0).sum()) if len(pnl) else 0,
            "losses": int((pnl < 0).sum()) if len(pnl) else 0,
            "first_entry": str(window["entry"].iloc[0]) if len(window) else None,
            "last_exit": str(window["exit"].iloc[-1]) if len(window) else None,
        },
    }


def _r2_equity_band(run, ep: dict, bars: list[dict], profile: dict) -> dict:
    seed = profile["returns"]
    n = len(bars)
    e0 = float(ep.get("equity") or 0.0)
    e = float(run.state.get("equity") or 0.0)
    sigma = float(seed.std(ddof=1)) if len(seed) > 1 else None
    value = math.log(e / e0) if e > 0 and e0 > 0 else None
    note = profile.get("note")
    if sigma is None:
        note = note or "seed history too short for the engine's sigma — rule disabled"
    elif not sigma > 0:
        note = "engine's per-bar sigma is zero over the seed — rule disabled"
    threshold = (-BAND_SIGMAS * sigma * math.sqrt(n)) if sigma and n else None
    armed = bool(n >= ARM_BARS and sigma and note is None)
    out = {
        "value": _f(value), "threshold": _f(threshold), "armed": armed,
        "trip": bool(armed and value is not None and threshold is not None
                     and value < threshold),
        "inputs": {
            "sigma_bar": _f(sigma), "sigmas": BAND_SIGMAS, "n": n,
            "arm_bars": ARM_BARS, "E_0": _f(e0), "E": _f(e),
            "seed_bars": int(len(seed)), "seed_history_bars": profile.get("seed_bars"),
            "warmup_bars": profile.get("warmup_bars"),
            "first_position_bar": profile.get("first_position_bar"),
            "min_seed_bars": profile.get("min_bars"), "drift": 0.0,
            "strategy": run.config.get("strategy"),
            "cost_bps": run.config.get("cost_bps"),
            "vol_target": run.config.get("vol_target"),
            "vol_lookback": run.config.get("vol_lookback"),
            "max_leverage": run.config.get("max_leverage"),
            "rebalance_band": run.config.get("rebalance_band"),
        },
    }
    if note:
        out["note"] = note
    return out


def _r3_realised_vol(run, ep: dict, bars: list[dict], store: pd.DataFrame) -> dict:
    vol_target = run.config.get("vol_target")
    lookback = int(run.config.get("vol_lookback") or ARM_BARS)
    n = len(bars)
    # the first epoch bar's return is measured from the epoch's own equity, so n
    # bars give n returns and the arming threshold means what it says
    eq = pd.Series([float(ep.get("equity") or 0.0)] + [float(b["equity"]) for b in bars])
    rets = eq.pct_change().dropna().tail(lookback)
    used = bars[-len(rets):] if len(rets) else []
    # Annualised with the periods-per-year of the run's whole bar store — the
    # same figure engine.vol_scale sizes the run to vol_target with — so the
    # realised vol and the target are on one scale. Inferred from the <= 60
    # epoch bars alone, an intraday window shorter than a session reads "bars
    # in the window" as "bars per day" and understates the vol 2-4x.
    ppy = float(data_mod.periods_per_year(store.index)) if len(store) >= 3 else None
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else None
    value = sd * math.sqrt(ppy) if sd is not None and ppy else None
    disabled = vol_target is None
    threshold = None if disabled else VOL_MULT * float(vol_target)
    armed = bool(not disabled and n >= ARM_BARS and value is not None)
    out = {
        "value": _f(value), "threshold": _f(threshold), "armed": armed,
        "trip": bool(armed and value is not None and value > threshold),
        "inputs": {"n": n, "arm_bars": ARM_BARS, "lookback": lookback,
                   "bars_used": int(len(rets)), "ppy": _f(ppy),
                   "ppy_from_bars": int(len(store)),
                   "sd_per_bar": _f(sd), "vol_target": vol_target, "mult": VOL_MULT,
                   "first_bar": used[0]["bar"] if used else None,
                   "last_bar": used[-1]["bar"] if used else None},
    }
    if disabled:
        out["note"] = "vol_target is None — rule disabled"
    return out


def evaluate(run, now: str | None = None, seed: pd.Series | dict | None = None) -> dict:
    """The three rules on the run as it stands. Reads files only; never the
    feed or the broker. `seed` lets a caller that already has `seed_profile`
    (or a bare `seed_returns` series) pass it in rather than run the engine
    twice."""
    ep = epoch(run)
    bars = epoch_bar_records(run, ep)
    store = run.bars()
    if seed is None:
        profile = seed_profile(run, store)
    elif isinstance(seed, dict):
        profile = seed
    else:
        profile = {"returns": seed, "seed_bars": None, "warmup_bars": None,
                   "first_position_bar": None, "min_bars": None, "note": None}
    rules = {
        "r1_profit_factor": _r1_profit_factor(run, ep),
        "r2_equity_band": _r2_equity_band(run, ep, bars, profile),
        "r3_realised_vol": _r3_realised_vol(run, ep, bars, store),
    }
    tripped = next((name for name in RULES if rules[name]["trip"]), None)
    return {"at": now or _now(), "epoch": ep, "epoch_bars": len(bars),
            "rules": rules, "tripped": tripped,
            "armed": [name for name in RULES if rules[name]["armed"]]}


def flags(ev: dict | None) -> dict:
    """What the journal cares about: per rule, armed and trip. A missing
    evaluation is every flag off, so the first poll only journals if
    something is already armed."""
    rules = (ev or {}).get("rules") or {}
    return {name: (bool(rules.get(name, {}).get("armed")),
                   bool(rules.get(name, {}).get("trip"))) for name in RULES}


def changed(prev: dict | None, cur: dict) -> bool:
    return flags(prev) != flags(cur)


def describe(ev: dict) -> str:
    """One paragraph per rule, for `paper.py risk` and the dry run."""
    out = []
    ep = ev["epoch"]
    out.append(f"  epoch      bar {ep.get('bar')}  equity {float(ep.get('equity') or 0):,.2f}  "
               f"closed trades before {ep.get('closed_trades_before')}  "
               f"({ev.get('epoch_bars')} bars since)")
    for name in RULES:
        r = ev["rules"][name]
        value = r["value"]
        if value is None and r["inputs"].get("no_losses"):
            vs = "inf (no losses)"
        else:
            vs = "-" if value is None else (f"{value:.6g}" if math.isfinite(value) else "inf")
        ts = "-" if r["threshold"] is None else f"{r['threshold']:.6g}"
        status = ("TRIP" if r["trip"] else "armed" if r["armed"] else "unarmed")
        out.append(f"  {name:<18} value {vs:>12}  threshold {ts:>12}  {status}")
        if r.get("note"):
            out.append(f"  {'':<18} {r['note']}")
        items = ", ".join(f"{k}={v}" for k, v in r["inputs"].items())
        out.append(f"  {'':<18} inputs: {items}")
    out.append(f"  would halt: {ev['tripped'] or 'no'}")
    return "\n".join(out)


# ────────────────────────────── heartbeat (§3) ──────────────────────────────

HEARTBEAT_FILE = "heartbeat.json"
STALE_REPEAT_S = 6 * 3600
RETRY_BACKOFF_S = 15 * 60     # after a failed push: no retry before this has passed
# The heartbeat timer fires with RandomizedDelaySec=5 and the stamps are
# truncated to the second, so the pass "fifteen minutes later" can land a few
# seconds BEFORE the stamped time and would wait a whole extra pass. Both
# clocks are therefore read with this much slack: a push is never more frequent
# than specified by more than the slack, and never a pass late.
TIMER_SLACK_S = 30
DEFAULT_TICK_S = 300


def _parse(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def heartbeat_items(base_dir: str | Path, now: datetime,
                    tick_default: int = DEFAULT_TICK_S) -> list[dict]:
    """What is fresh and what is stale, right now. Reads state.json, config.json
    and book.json as plain files — no PaperRun, no feed, no broker."""
    from . import paper as paper_mod        # paper imports this module

    base = Path(base_dir)
    items = []
    for run_id in paper_mod.PaperRun.list_runs(base):
        root = base / run_id
        if (root / paper_mod.STOPPED_FILE).exists():
            items.append({"item": run_id, "kind": "run", "skipped": "stopped"})
            continue
        cfg = _read_json(root / "config.json") or {}
        st = _read_json(root / "state.json") or {}
        tick_s = int((cfg.get("execution") or {}).get("tick_s") or tick_default)
        limit = 2 * tick_s
        updated = _parse(st.get("updated"))
        age = (now - updated).total_seconds() if updated else None
        items.append({"item": run_id, "kind": "run", "field": "state.updated",
                      "at": st.get("updated"), "age_s": None if age is None else round(age),
                      "limit_s": limit, "stale": age is None or age > limit,
                      "halted": halted_path(root).exists()})
    doc = _read_json(base / "book.json")
    if doc is None:
        items.append({"item": "book.json", "kind": "book", "field": "as_of", "at": None,
                      "age_s": None, "limit_s": 2 * tick_default, "stale": True,
                      "why": "missing or unreadable"})
    else:
        limit = 2 * int(doc.get("tick_s") or tick_default)
        as_of = _parse(doc.get("as_of"))
        age = (now - as_of).total_seconds() if as_of else None
        items.append({"item": "book.json", "kind": "book", "field": "as_of",
                      "at": doc.get("as_of"), "age_s": None if age is None else round(age),
                      "limit_s": limit, "stale": age is None or age > limit})
    return items


def _stale_message(due: list[dict]) -> tuple[str, dict]:
    """One `stale` alert per pass, whatever the count: a pass that pushed one
    message per stale item could hold the unit for (items x 20 s) and be killed
    by systemd before its state was on disk."""
    def one(it: dict) -> str:
        age = "never written" if it["age_s"] is None else f"{it['age_s']} s old"
        return (f"{it['item']} is stale — {it['field']} {it.get('at') or it.get('why')} is "
                f"{age}, limit {it['limit_s']} s")

    if len(due) == 1:
        it = due[0]
        return one(it), {"item": it["item"], "age_s": it["age_s"], "limit_s": it["limit_s"]}
    lines = [f"{len(due)} things are stale:"] + [f"  {one(it)}" for it in due]
    return "\n".join(lines), {"count": len(due), "items": ", ".join(it["item"] for it in due)}


def _write_heartbeat(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


def heartbeat(base_dir: str | Path, now: datetime | None = None, dry_run: bool = False,
              push_recovery: bool = False, tick_default: int = DEFAULT_TICK_S,
              notify=None) -> dict:
    """One heartbeat pass. Alerts on fresh -> stale, again at most every
    STALE_REPEAT_S while stale; recovery is log-only unless `push_recovery`.
    State lives in paper_runs/heartbeat.json as
    {item: {stale_since, last_alert[, last_attempt, last_error, retry_not_before]}}.

    Two things the order of operations guarantees. The state file is written
    BEFORE the push, so a pass that systemd kills mid-push still leaves
    `stale_since` on disk and the next pass carries on rather than starting
    over. And `last_alert` is stamped only when the alert was delivered as far
    as it can be — pushed, or log-only because Telegram is not configured — so
    a push that fails on the transport does not silence the item for six hours:
    it stays due. It is not retried on every five-minute pass, though: a
    failed push stamps `retry_not_before = now + RETRY_BACKOFF_S` on the item
    and a due item is not pushed before that (its action is `backoff`, and the
    report says so). A delivered push starts the six-hour clock from the pass
    that delivered it, not from the first attempt. Every stale item that is due
    goes out in ONE message, so a pass makes at most one push (<= 20 s) plus
    any recoveries.

    `now` is injectable so the transitions can be tested with a clock; `notify`
    defaults to alerts.notify. A dry run computes and returns everything and
    writes nothing — no state, no log line, no push."""
    from . import alerts as alerts_mod

    notify = notify or alerts_mod.notify
    base = Path(base_dir)
    now = now or datetime.now(timezone.utc)
    now_s = now.replace(microsecond=0).isoformat()
    hb_path = base / HEARTBEAT_FILE
    state = _read_json(hb_path) or {}
    tracked = dict(state.get("items") or {})
    items = heartbeat_items(base, now, tick_default)
    alerts_out = []
    due = []

    for it in items:
        if it.get("skipped"):
            continue
        name = it["item"]
        prev = tracked.get(name)
        if it["stale"]:
            if prev is None:
                tracked[name] = {"stale_since": now_s, "last_alert": None}
                action = "alert"
            else:
                last = _parse(prev.get("last_alert"))
                action = ("alert" if last is None
                          or (now - last).total_seconds() >= STALE_REPEAT_S - TIMER_SLACK_S
                          else "suppressed")
                hold = _parse(prev.get("retry_not_before"))
                if (action == "alert" and hold is not None
                        and now + timedelta(seconds=TIMER_SLACK_S) < hold):
                    # the last push failed: due, but not before the backoff has run
                    action = "backoff"
                    it["retry_not_before"] = prev.get("retry_not_before")
                    it["last_error"] = prev.get("last_error")
            it["action"] = action
            if action == "alert":
                due.append(it)
        elif prev is not None:
            it["action"] = "recovered"
            msg = (f"{name} is fresh again — {it['field']} {it['at']} is {it['age_s']} s old; "
                   f"stale since {prev.get('stale_since')}")
            entry = {"kind": "recovered", "item": name, "message": msg}
            alerts_out.append(entry)
            if not dry_run:
                entry["result"] = notify(base, "recovered", msg,
                                         fields={"item": name, "stale_since": prev.get("stale_since")},
                                         push=push_recovery)
            tracked.pop(name, None)
        else:
            it["action"] = "ok"

    push = None
    if due:
        message, fields = _stale_message(due)
        for it in due:
            alerts_out.append({"kind": "stale", "item": it["item"], "message": message})
        push = {"kind": "stale", "items": [it["item"] for it in due], "message": message}

    doc = {"checked_at": now_s, "items": tracked}
    if not dry_run:
        _write_heartbeat(hb_path, doc)             # the transition is on disk before any push
        if due:
            res = notify(base, "stale", message, fields=fields)
            push["result"] = res
            # delivered: pushed, or never attempted a transport (log-only /
            # Telegram not configured). A transport failure leaves the item due.
            delivered = bool(res.get("pushed")) or "telegram" not in res
            push["delivered"] = delivered
            retry_at = (now + timedelta(seconds=RETRY_BACKOFF_S)).replace(microsecond=0).isoformat()
            for it in due:
                entry = tracked[it["item"]]
                entry["last_attempt"] = now_s
                if delivered:
                    entry["last_alert"] = now_s          # the six hours start now
                    entry.pop("last_error", None)
                    entry.pop("retry_not_before", None)
                else:
                    entry["last_error"] = (res.get("telegram") or {}).get("error") or "push failed"
                    entry["retry_not_before"] = retry_at
                    it["retry_not_before"] = retry_at
                it["delivered"] = delivered
            _write_heartbeat(hb_path, doc)
    return {"checked_at": now_s, "dry_run": dry_run, "items": items,
            "alerts": alerts_out, "push": push, "state": doc}


def format_heartbeat(report: dict) -> str:
    out = [f"  heartbeat at {report['checked_at']}"
           + ("  (DRY RUN — nothing written, nothing sent)" if report["dry_run"] else "")]
    for it in report["items"]:
        if it.get("skipped"):
            out.append(f"  {it['item']:<28} skipped — {it['skipped']}")
            continue
        age = "never" if it["age_s"] is None else f"{it['age_s']:>7} s"
        flag = "STALE" if it["stale"] else "fresh"
        tail = {"alert": "  -> alert", "suppressed": "  (alerted within 6 h — not repeated)",
                "recovered": "  -> recovered (logged)"}.get(it.get("action"), "")
        if it.get("action") == "alert" and it.get("delivered") is False:
            tail = (f"  -> alert (push FAILED — still due; not retried before "
                    f"{it.get('retry_not_before')})")
        elif it.get("action") == "backoff":
            tail = (f"  (due, but the last push failed: {it.get('last_error')} — "
                    f"backoff, retry not before {it.get('retry_not_before')})")
        halted = "  HALTED" if it.get("halted") else ""
        out.append(f"  {it['item']:<28} {it['field']:<14} {age}  limit {it['limit_s']:>5} s  "
                   f"{flag}{halted}{tail}")
    push = report.get("push")
    if push:
        out.append("")
        out.append(f"  [stale] one message for {len(push['items'])} item(s): "
                   + ", ".join(push["items"]))
        for line in push["message"].splitlines():
            out.append(f"    {line}")
    for a in report["alerts"]:
        if a["kind"] != "stale":
            out.append(f"  [{a['kind']}] {a['message']}")
    tracked = report["state"]["items"]
    if tracked:
        out.append("")
        for name, v in tracked.items():
            last = v.get("last_alert")
            if last is None:
                last = "none delivered yet" + (f" (last attempt {v['last_attempt']}: "
                                               f"{v.get('last_error')})" if v.get("last_attempt")
                                               else "")
            hold = (f", retry not before {v['retry_not_before']}"
                    if v.get("retry_not_before") else "")
            out.append(f"  tracking {name}: stale since {v['stale_since']}, last alert {last}{hold}")
    return "\n".join(out)

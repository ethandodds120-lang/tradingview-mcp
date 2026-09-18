"""Reports from the journal, by the box (DESIGN-tjr-human.md sections 3.6, 3.7, 5, 9.5).

    weekly(records)        section 3.6: human R per trade against each of the ten exits and the
                           random stop, behaviour, branches, trades to go, trajectory
    maybe_review(journal)  section 3.7: ONE review at the 25th and ONE at the 50th filled trade,
                           journaled with its stamp. Descriptive: no deflated statistic, declares
                           nothing, binds nothing
    final(journal)         section 5: refuses below 100 filled trades and says how many to go; at
                           100, d_i = R_human,i - R_best_of_ten,i with the best of ten chosen on
                           all 100 entries, through validate.deflated_sharpe with n_trials = 14;
                           PASS iff DSR >= 0.95 and mean(d) > 0
    status(records)        where the run is. Nothing about whether it is winning.

Everything is computed from `journal.rebuild`: the human's R is recomputed there
from the authoritative entry, the exit price and the initial stop, net of costs,
and each mechanical exit's R comes from that trade's `benchmarks` record (its own
initial risk, net of the same costs). A trade counts in a table once it is
COMPLETE: exited and benchmarked. The first N filled trades, in fill order, are
the sample of a review or of the final test; later trades never displace them.

These reports read the live journal of the experiment itself. They are never
run on the output of a replay over market data files (section 9.8).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import N_TRIALS         # 14: section 5, second amendment (ten mechanical exits + four human controls)
from . import journal as J
from .exits import EXIT_NAMES, RANDOM_EXIT

SAMPLE = 100
PASS_BAR = 0.95
REVIEW_AT = (25, 50)
EARLY_MOVE_S = 120.0
WEEK_S = 7 * 86400.0


# ────────────────────────────── arithmetic ──────────────────────────────

def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _se(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None


def _f(x, width: int = 7, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-".rjust(width)
    return (f"{x:+.3f}" if signed else f"{x:.3f}").rjust(width)


def exit_r(trade: dict, name: str) -> float | None:
    b = trade.get("benchmarks") or {}
    rec = b.get(RANDOM_EXIT) if name == RANDOM_EXIT else (b.get("exits") or {}).get(name)
    return None if rec is None else rec.get("r")


def complete(trades: list[dict]) -> list[dict]:
    return [t for t in trades if t["complete"] and t["r"] is not None
            and all(exit_r(t, n) is not None for n in EXIT_NAMES)]


def exit_table(trades: list[dict]) -> dict:
    """R per trade: the human, each of the ten, the random stop — on the same entries."""
    rows = {"human": [t["r"] for t in trades]}
    for name in EXIT_NAMES + (RANDOM_EXIT,):
        rows[name] = [exit_r(t, name) for t in trades]
    out = {k: {"n": len([x for x in v if x is not None]), "mean": _mean(v), "se": _se(v),
               "total": float(sum(x for x in v if x is not None))} for k, v in rows.items()}
    return out


def best_of_ten(trades: list[dict]) -> str | None:
    """The mechanical exit with the highest mean net R on these entries (in-sample
    for the rules: the intended direction of the bias). Ties: the first in EXIT_NAMES."""
    if not trades:
        return None
    means = {n: _mean([exit_r(t, n) for t in trades]) for n in EXIT_NAMES}
    return max(EXIT_NAMES, key=lambda n: (means[n], -EXIT_NAMES.index(n)))


def paired(trades: list[dict], name: str) -> dict:
    d = [t["r"] - exit_r(t, name) for t in trades]
    return {"against": name, "n": len(d), "mean": _mean(d), "se": _se(d),
            "beaten": int(sum(1 for x in d if x > 0)), "tied": int(sum(1 for x in d if x == 0)),
            "lost": int(sum(1 for x in d if x < 0))}


# ────────────────────────────── behaviour and branches ──────────────────────────────

def behaviour(trades: list[dict], skipped: list[dict]) -> dict:
    widths: dict[str, list] = {}
    for t in trades:
        widths.setdefault(t["stop_mode"], []).append(t)
    width_rows = {m: {"n": len(v), "r_mean": _mean([t["r"] for t in v]),
                      "atr_multiple": _mean([t["stop_atr_multiple"] for t in v])} for m, v in sorted(widths.items())}
    when = {}
    for t in trades:
        when[t["stop_when"]] = when.get(t["stop_when"], 0) + 1
    secs = [t["stop_seconds_from_fill"] for t in trades if t["stop_seconds_from_fill"] is not None]
    moves = [m for t in trades for m in t["moves"]]
    early = [m for m in moves if m.get("seconds_since_entry") is not None
             and m["seconds_since_entry"] <= EARLY_MOVE_S]
    rejects = [r for t in trades for r in t["rejects"]]
    why: dict[str, int] = {}
    for r in rejects:
        why[r.get("why")] = why.get(r.get("why"), 0) + 1
    widen = [r for r in rejects if r.get("why") == "move_widens"]

    under = []
    for t in trades:
        x = t["exit"]
        if x is None or x.get("reason") != "exit_now" or not (x.get("r_gross") is not None and x["r_gross"] < 0):
            continue
        b = t.get("benchmarks") or {}
        first_rule = min(((b.get("exits") or {}).get(n) or {}).get("exit_time") or "9999" for n in EXIT_NAMES)
        if x.get("exit_bar") is not None and x["exit_bar"] < first_rule:        # a strictly earlier minute
            held = ((b.get("human") or {}).get("held") or {})
            under.append({"setup_id": t["setup_id"], "n": t["n"], "r_at_exit": t["r"],
                          "reason_code": t["human_reason"], "held_r": held.get("r"),
                          "held_reason": held.get("reason"), "held_exit_et": held.get("exit_et")})
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
    sk = []
    for s in skipped:
        b = s.get("benchmarks") or {}
        rs = {n: ((b.get("exits") or {}).get(n) or {}).get("r") for n in EXIT_NAMES} if b else {}
        sk.append({"setup_id": s["setup_id"], "reason_code": s["reason"], "ended": s["ended"],
                   "zone": s["zone"], "level_type": s["level_type"], "conf": s["conf"],
                   "wick_fixed_r": rs.get("wick_fixed"), "ten_mean_r": _mean(rs.values()) if rs else None,
                   "ten_best_r": max((v for v in rs.values() if v is not None), default=None) if rs else None})
    return {"stop_width": width_rows, "stop_when": when,
            "stop_seconds_from_fill_median": float(np.median(secs)) if secs else None,
            "moves": len(moves), "trades_with_a_move": sum(1 for t in trades if t["moves"]),
            "moves_within_2min": len(early), "widen_attempts": len(widen),
            "widen_requested": [r.get("requested") for r in widen], "rejects": why,
            "underwater_exits_before_any_rule": {"count": len(under), "r_at_exit_mean": _mean([u["r_at_exit"] for u in under]),
                                                 "held_r_mean": _mean([u["held_r"] for u in under]), "trades": under},
            "exit_reasons": reasons,
            "skipped": {"count": len(skipped), "would_have_filled": sum(1 for s in skipped if s["would_fill"]),
                        "wick_fixed_r_mean": _mean([s["wick_fixed_r"] for s in sk]),
                        "ten_mean_r_mean": _mean([s["ten_mean_r"] for s in sk]), "setups": sk}}


def branches(trades: list[dict], best: str | None) -> dict:
    def group(key):
        out: dict[str, list] = {}
        for t in trades:
            out.setdefault(str(key(t)), []).append(t)
        return {k: {"n": len(v), "human": _mean([t["r"] for t in v]),
                    "wick_fixed": _mean([exit_r(t, "wick_fixed") for t in v]),
                    "best": _mean([exit_r(t, best) for t in v]) if best else None} for k, v in sorted(out.items())}

    eq = [t for t in trades if t["zone"] == "eq"]
    co: dict[str, list] = {}
    for t in eq:
        for k in (t["cooccur"] or ["none"]):
            co.setdefault(k, []).append(t)
    return {"level": group(lambda t: t["level_type"]), "confirmation": group(lambda t: t["conf"]),
            "zone": group(lambda t: t["zone"]), "instrument": group(lambda t: t["instrument"]),
            "eq_by_cooccurring_zone": {k: {"n": len(v), "human": _mean([t["r"] for t in v])}
                                       for k, v in sorted(co.items())}}


def _describe(trades: list[dict], skipped: list[dict]) -> dict:
    best = best_of_ten(trades)
    return {"n": len(trades), "table": exit_table(trades), "best_so_far": best,
            "paired_vs_best": paired(trades, best) if best else None,
            "behaviour": behaviour(trades, skipped), "branches": branches(trades, best)}


# ────────────────────────────── text ──────────────────────────────

def _render(d: dict, title: str, notes: list[str]) -> str:
    L = [title, "=" * len(title)]
    L += notes + [""]
    L.append(f"R per trade, net of costs, each against its own initial risk — {d['n']} complete trades")
    L.append(f"  {'exit':<16}{'n':>4}{'mean R':>10}{'s.e.':>9}{'total R':>10}")
    for name, row in d["table"].items():
        mark = "  <- best of ten so far" if name == d["best_so_far"] else ""
        L.append(f"  {name:<16}{row['n']:>4}{_f(row['mean'], 10)}{_f(row['se'], 9, False)}{_f(row['total'], 10)}{mark}")
    p = d["paired_vs_best"]
    if p:
        L += ["", f"paired difference, human - {p['against']} (the best of ten SO FAR; it binds nothing):",
              f"  mean {_f(p['mean'])}   s.e. {_f(p['se'], 7, False)}   human better on {p['beaten']} of {p['n']}"
              f" (tied {p['tied']}, worse {p['lost']})"]
    b = d["behaviour"]
    L += ["", "behaviour", f"  stop width chosen ({', '.join(f'{k} {v}' for k, v in sorted(b['stop_when'].items())) or '-'}):"]
    for mode, row in b["stop_width"].items():
        L.append(f"    {mode:<10}{row['n']:>4} trades   R {_f(row['r_mean'])}   width {_f(row['atr_multiple'], 6, False)} ATR")
    med = b["stop_seconds_from_fill_median"]
    L.append(f"  seconds from fill to the stop choice, median: {'-' if med is None else f'{med:.0f}'}")
    L.append(f"  stop moves: {b['moves']} on {b['trades_with_a_move']} trades; within 2 minutes of entry: {b['moves_within_2min']}")
    L.append(f"  widen attempts (rejected): {b['widen_attempts']}; all rejections: "
             + (", ".join(f"{k} {v}" for k, v in sorted(b["rejects"].items(), key=lambda kv: str(kv[0]))) or "none"))
    u = b["underwater_exits_before_any_rule"]
    L.append(f"  EXIT NOW while underwater before any of the ten exits would have closed the trade: {u['count']}"
             + (f"; R at exit {_f(u['r_at_exit_mean'])}; held under the human's own initial stop {_f(u['held_r_mean'])}"
                if u["count"] else ""))
    for x in u["trades"]:
        L.append(f"    #{x['n']} {x['setup_id']}: exit {_f(x['r_at_exit'])} ({x['reason_code']}); held -> "
                 f"{_f(x['held_r'])} by {x['held_reason']} at {x['held_exit_et']} ET")
    L.append("  exits by reason: " + (", ".join(f"{k} {v}" for k, v in sorted(b["exit_reasons"].items())) or "-"))
    s = b["skipped"]
    L.append(f"  signals skipped: {s['count']} ({s['would_have_filled']} would have filled); on those, wick_fixed "
             f"{_f(s['wick_fixed_r_mean'])} R per trade, mean of the ten {_f(s['ten_mean_r_mean'])}")
    for x in s["setups"]:
        L.append(f"    {x['setup_id']} ({x['reason_code']}): {x['ended']}; wick_fixed {_f(x['wick_fixed_r'])}, "
                 f"best of the ten {_f(x['ten_best_r'])}")
    L += ["", "branches (n, human R per trade, wick_fixed, best of ten so far)"]
    for title_, key in (("level", "level"), ("confirmation", "confirmation"), ("zone", "zone"), ("instrument", "instrument")):
        for k, row in d["branches"][key].items():
            L.append(f"  {title_:<15}{k:<16}{row['n']:>4}{_f(row['human'], 9)}{_f(row['wick_fixed'], 9)}{_f(row['best'], 9)}")
    for k, row in d["branches"]["eq_by_cooccurring_zone"].items():
        L.append(f"  {'eq fills with':<15}{k:<16}{row['n']:>4}{_f(row['human'], 9)}")
    return "\n".join(L)


_NOT_A_TEST = ("Descriptive only: no deflated statistic is computed and nothing is declared. It cannot stop, "
               "extend or change the experiment; the scored benchmark is chosen on all 100 entries.")


# ────────────────────────────── weekly (section 3.6) ──────────────────────────────

def weekly(records: list[dict], now: float | None = None) -> dict:
    state = J.rebuild(records)
    done = complete(state["trades"])
    now_ms = int((now if now is not None else (records[-1]["ts_ms"] / 1000.0 if records else 0.0)) * 1000)
    since = now_ms - int(WEEK_S * 1000)
    week = [r for r in records if r["ts_ms"] > since]
    wk = {k: sum(1 for r in week if r["kind"] == k and not r.get("provisional")
                 and not (k == "signal" and r.get("event") == "zone") and not (k == "skip" and r.get("stage") != "command"))
          for k in ("signal", "fill", "skip", "invalidated", "exit", "reject", "gap")}
    d = _describe(done, state["skipped"])
    to_go = max(0, SAMPLE - state["filled"])
    notes = [f"week to {J.iso_ms(now_ms / 1000.0)}: " + ", ".join(f"{k} {v}" for k, v in wk.items()),
             f"filled trades {state['filled']} of {SAMPLE}; to go {to_go}; complete (exited and benchmarked) {len(done)}",
             f"the section 5 bar: deflated Sharpe of d = human - best of ten, n_trials = {N_TRIALS}, >= {PASS_BAR} and "
             f"mean(d) > 0 — computed once, at {SAMPLE}; the trajectory below is the paired difference only"]
    text = _render(d, "tjr_human — weekly report", notes)
    return {"kind": "weekly", "filled": state["filled"], "to_go": to_go, "week": wk, "data": d, "text": text}


# ────────────────────────────── interim reviews (section 3.7) ──────────────────────────────

def review_due(records: list[dict]) -> list[int]:
    """Which of 25 / 50 can be written now: reached, its first N trades all exited
    and benchmarked, and not written before."""
    state = J.rebuild(records)
    due = []
    for n in REVIEW_AT:
        if n in state["reviews"] or state["filled"] < n:
            continue
        first = state["trades"][:n]
        if len(complete(first)) == n:
            due.append(n)
    return due


def build_review(records: list[dict], n: int) -> dict:
    state = J.rebuild(records)
    first = state["trades"][:n]
    if len(first) < n or len(complete(first)) != n:
        raise ValueError(f"tjr_human.report: the first {n} filled trades are not all exited and benchmarked yet")
    cut = first[-1]["fill_seq"]
    skipped = [s for s in state["skipped"] if s["skip_seq"] <= (first[-1]["exit"] or {}).get("seq", cut)]
    d = _describe(first, skipped)
    notes = [f"interim review at the {n}th filled trade (section 3.7). {_NOT_A_TEST}",
             f"trades #{first[0]['n']}..#{first[-1]['n']}: {first[0]['setup_id']} .. {first[-1]['setup_id']}; "
             f"to go {SAMPLE - n}"]
    return {"n": n, "trades": [t["setup_id"] for t in first], "data": d,
            "text": _render(d, f"tjr_human — interim review at {n} filled trades", notes)}


def maybe_review(journal: J.Journal, now: float | None = None, write_text: bool = True) -> list[dict]:
    """Write the reviews that are due — each ONCE, journaled with its stamp so the
    trades before and after it can be told apart. Returns the records written."""
    out = []
    for n in review_due(journal.records()):
        recs = journal.records()
        if n in J.rebuild(recs)["reviews"]:
            continue
        body = build_review(recs, n)
        rec = journal.write("review", body, now=now)
        out.append(rec)
        if write_text:
            try:
                Path(journal.base, f"review-{n}.txt").write_text(body["text"] + "\n", encoding="utf-8")
            except OSError:
                pass
    return out


# ────────────────────────────── the final test (section 5) ──────────────────────────────

def final(journal_or_records, now: float | None = None, write: bool = True) -> dict:
    """Refuses before 100 filled trades. At 100: the pre-registered test, on the first 100."""
    from ..validate import deflated_sharpe

    journal = journal_or_records if isinstance(journal_or_records, J.Journal) else None
    records = journal.records() if journal is not None else list(journal_or_records)
    state = J.rebuild(records)
    filled = state["filled"]
    if filled < SAMPLE:
        to_go = SAMPLE - filled
        return {"ok": False, "ran": False, "filled": filled, "to_go": to_go, "n_trials": N_TRIALS,
                "text": f"tjr_human final: REFUSED — {filled} filled trades, {to_go} to go. The sample is {SAMPLE} "
                        f"(section 5); the test (n_trials = {N_TRIALS}) runs once, at {SAMPLE}, not before."}
    first = state["trades"][:SAMPLE]
    done = complete(first)
    if len(done) != SAMPLE:
        waiting = [t["setup_id"] for t in first if t not in done]
        return {"ok": False, "ran": False, "filled": filled, "to_go": 0, "n_trials": N_TRIALS, "waiting": waiting,
                "text": f"tjr_human final: REFUSED — {len(waiting)} of the first {SAMPLE} trades are not exited and "
                        f"benchmarked yet: {', '.join(waiting[:5])}"}
    best = best_of_ten(done)
    d = np.array([t["r"] - exit_r(t, best) for t in done], dtype=float)
    # the ten ways the same statistic could have been framed: human minus each mechanical exit,
    # each as a per-trade Sharpe. ppy = 1: one period is one trade, nothing is annualised.
    trial_sharpes = []
    for name in EXIT_NAMES:
        dk = np.array([t["r"] - exit_r(t, name) for t in done], dtype=float)
        sd = dk.std(ddof=1)
        trial_sharpes.append(float(dk.mean() / sd) if sd > 0 else 0.0)
    res = deflated_sharpe(pd.Series(d), 1.0, np.array(trial_sharpes), n_trials=N_TRIALS)
    dsr = res.get("dsr")
    mean_d = float(d.mean())
    passed = bool(dsr is not None and math.isfinite(dsr) and dsr >= PASS_BAR and mean_d > 0)
    table = exit_table(done)
    out = {"ok": True, "ran": True, "filled": filled, "sample": SAMPLE, "n_trials": N_TRIALS, "best_of_ten": best,
           "mean_d": mean_d, "se_d": _se(d), "beaten": int((d > 0).sum()), "dsr": dsr, "psr": res.get("psr"),
           "sr0": res.get("sr0"), "sharpe_d_per_trade": float(mean_d / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
           "trial_sharpes": dict(zip(EXIT_NAMES, trial_sharpes)), "pass_bar": PASS_BAR, "passed": passed,
           "table": table, "trades": [t["setup_id"] for t in done]}
    L = ["tjr_human — the final test (DESIGN-tjr-human.md section 5)", "=" * 58,
         f"sample: the first {SAMPLE} filled trades ({done[0]['setup_id']} .. {done[-1]['setup_id']})",
         f"benchmark: best of the ten mechanical exits on these {SAMPLE} entries = {best}",
         f"  {'exit':<16}{'mean R':>10}"]
    for name, row in table.items():
        L.append(f"  {name:<16}{_f(row['mean'], 10)}")
    L += [f"d_i = R_human,i - R_{best},i:  mean {_f(mean_d)}  s.e. {_f(out['se_d'], 7, False)}  "
          f"human better on {out['beaten']} of {SAMPLE}",
          f"validate.deflated_sharpe(d, ppy=1, trial_sharpes=<ten per-trade Sharpes of human - exit_k>, "
          f"n_trials = {N_TRIALS})",
          f"  deflated Sharpe {_f(dsr, 7, False)}   (bar {PASS_BAR}; expected best-of-{N_TRIALS} noise Sharpe "
          f"{_f(res.get('sr0'), 7, False)} per trade; PSR {_f(res.get('psr'), 7, False)})",
          f"RESULT: {'PASS' if passed else 'FAIL'} — DSR {'>=' if (dsr is not None and math.isfinite(dsr) and dsr >= PASS_BAR) else '<'} "
          f"{PASS_BAR} and mean(d) {'>' if mean_d > 0 else '<='} 0",
          "A pass justifies a second pre-registered sample, nothing more (section 6). Every number stays in the record."]
    out["text"] = "\n".join(L)
    if write and journal is not None and state["final"] is None:
        journal.write("final", {k: v for k, v in out.items() if k != "table"}, now=now)
    return out


# ────────────────────────────── status (section 9.5) ──────────────────────────────

def _integrity_text(chk: dict) -> str:
    """ok | n torn line(s) skipped | BROKEN: why. A torn line is a write cut short
    (skipped, never repaired); BROKEN is for the sequence, the stamps, the files."""
    torn = int(chk.get("bad_lines") or 0)
    tail = f"{torn} torn line(s) skipped (a write was cut short; the records around them are whole)" if torn else ""
    if chk.get("problems"):
        return "BROKEN: " + "; ".join(chk["problems"][:3]) + (f"; {tail}" if tail else "")
    return tail or "ok"


def status(records: list[dict], base_dir=None) -> dict:
    """Where the run is — and nothing about whether it is winning: no R anywhere."""
    state = J.rebuild(records)
    armed = state["armed"]
    open_trades = [t["setup_id"] for t in state["trades"] if not t["closed"]]
    waiting = [t["setup_id"] for t in state["trades"] if t["closed"] and t["benchmarks"] is None]
    setups = state["setups"].values()
    counts = {"signals": sum(1 for s in setups if s["signal"] is not None),
              "filled": state["filled"], "skipped": len(state["skipped"]),
              "invalidated": sum(1 for s in setups if s["status"] == "invalidated"),
              "expired": sum(1 for s in setups if s["status"] == "expired"),
              "observe_records": state["observed"], "commands": state["counts"].get("command", 0),
              "rejects": state["counts"].get("reject", 0), "gaps": len(state["gaps"])}
    out = {"mode": "armed" if armed else "observe",
           "armed_at": armed.get("ts") if armed else None,
           "armed": {k: armed.get(k) for k in ("commit", "prereg_sha256", "target_rule", "reason")} if armed else None,
           "filled": state["filled"], "sample": SAMPLE, "to_go": max(0, SAMPLE - state["filled"]),
           "open_trades": open_trades, "awaiting_benchmarks": waiting, "counts": counts,
           "reviews": {n: r["ts"] for n, r in sorted(state["reviews"].items())},
           "reviews_due": review_due(records), "final_written": state["final"] is not None,
           "last_record": records[-1]["ts"] if records else None, "records": len(records), "n_trials": N_TRIALS}
    if base_dir is not None:
        out["integrity"] = J.integrity(base_dir)
    L = ["tjr_human — status", "=" * 18,
         f"mode: {out['mode']}" + (f" since {out['armed_at']}  (target rule {armed.get('target_rule')}, "
                                   f"commit {str(armed.get('commit'))[:10]})" if armed else " (no entry is taken until `arm`)"),
         f"filled trades: {out['filled']} of {SAMPLE}; to go {out['to_go']}",
         f"open: {', '.join(open_trades) or 'none'}; awaiting benchmarks: {', '.join(waiting) or 'none'}",
         "counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
         "interim reviews written: " + (", ".join(f"{n} at {ts}" for n, ts in out["reviews"].items()) or "none")
         + (f"; due now: {out['reviews_due']}" if out["reviews_due"] else ""),
         f"final test: {'written' if out['final_written'] else f'not before {SAMPLE} filled trades'} (n_trials = {N_TRIALS})",
         f"journal: {out['records']} records, last {out['last_record']}"
         + (f"; integrity {_integrity_text(out['integrity'])}" if "integrity" in out else "")]
    out["text"] = "\n".join(L)
    return out

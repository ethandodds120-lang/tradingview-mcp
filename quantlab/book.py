"""The book governor — one multiplier for every run, from the whole book's vol.

Every run sizes itself to its own vol target as though it were the only thing you
own. It is not, and correlated runs stack: two trend models on gold and silver are
close to the same position twice. `paper.py portfolio` shows that; this module
acts on it, in the smallest way that is still honest — one scalar k, applied to
every run's target, chosen so the book as a whole runs at the vol target. It only
ever scales down (k <= cap, cap defaults to 1.0): a governor, not an allocator.
It does not move capital between runs and it does not know which run is right.

The maths, per run i (DESIGN-execution.md §6):

    E_i    equity            raw_i   what the run wants to hold, before the book
    w_i  = E_i / sum(E) * raw_i      the run's exposure as a fraction of the book
    D_i  = std(last `window` native returns) * sqrt(ppy_i)
    R    = corr of inner-joined single-day returns, last `window` joined rows
    Sigma = D R D      book_vol = sqrt(w' Sigma w)
    k_uncapped = vol_target / book_vol      k = min(cap, k_uncapped)
    after_vol = k * book_vol               (valid because k applies to every run)

Vol is on each run's own calendar because crypto trades seven days and equities
five, and a daily std over a mixed calendar is a number that means nothing. The
correlation joins on the calendar date so the two can meet at all — crypto bars are
stamped 00:00 UTC, equity bars 04:00 or 05:00 — and drops the days one side does
not have. `ppy` comes from the run's own bar index — 365.25 when the index trades
weekends, otherwise what the engine would use (see `native_ppy`).

Files. `book.json` is the current document, written atomically; `book.jsonl` is
every tick's document, one per line, so the history of k is on disk. A run reads
`book.json` and applies k only if the document is fresh (see `read_book`); a stale
or missing book means k = 1.0, and the run says so rather than quietly sizing off
a number from last week.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from . import paper as paper_mod

BOOK_FILE = "book.json"
HISTORY_FILE = "book.jsonl"
ALERTS_FILE = "alerts.log"
STOPPED_FILE = "STOPPED"
VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_stopped(base: str | Path, run_id: str) -> bool:
    """`paper.py stop` leaves a STOPPED marker; everything here skips such a run."""
    return (Path(base) / run_id / STOPPED_FILE).exists()


def active_runs(base: str | Path) -> tuple[list[str], list[dict]]:
    """Run ids the tick should touch, and the ones it should not, with why."""
    keep, skipped = [], []
    for run_id in paper_mod.PaperRun.list_runs(base):
        if is_stopped(base, run_id):
            skipped.append({"run": run_id, "reason": "stopped"})
        else:
            keep.append(run_id)
    return keep, skipped


# ────────────────────────────── the maths ──────────────────────────────

def native_ppy(index: pd.DatetimeIndex) -> float:
    """Periods per year on the run's own calendar.

    This used to carry its own weekend-gap test because `periods_per_year`
    answered 252 for every daily index, crypto included. The engine now makes
    that distinction itself (a daily index with no two-day gap in its recent
    history is a seven-day calendar and gets 365.25), and the book must size off
    the same number the runs size themselves with — two copies of the rule would
    drift apart the first time one was touched. Kept as a name so the call sites
    say what they mean."""
    return data_mod.periods_per_year(index)


def _daily_close(bars: pd.DataFrame) -> pd.Series:
    """One close per calendar date. Collapses the 00:00 / 04:00 / 05:00 UTC stamps
    so different venues line up, and intraday bars down to their last close."""
    close = bars["close"].copy()
    close.index = pd.DatetimeIndex(close.index).normalize()
    return close[~close.index.duplicated(keep="last")]


def _run_row(base: Path, run_id: str) -> tuple[dict | None, dict | None]:
    """Everything the book needs from one run, or the reason it is left out."""
    if is_stopped(base, run_id):
        return None, {"run": run_id, "reason": "stopped"}
    try:
        run = paper_mod.PaperRun.load(base, run_id)
    except paper_mod.ParamDrift:
        return None, {"run": run_id, "reason": "param drift"}
    cfg, st = run.config, run.state
    broker = cfg.get("broker") or {}
    feed = cfg.get("feed") or {}
    # yahoo runs store the instrument as `ticker`, every other feed as `symbol`
    symbol = broker.get("symbol") or feed.get("symbol") or feed.get("ticker")
    if not symbol:
        return None, {"run": run_id, "reason": "no symbol"}
    if not run.bars_path.exists():
        return None, {"run": run_id, "reason": "no bars.csv"}

    equity = float(st.get("equity") or 0.0)
    cash = float(st.get("cash") or 0.0)
    # what the run holds now, as a fraction of its own equity (same as `portfolio`)
    held = (equity - cash) / equity if equity else 0.0
    # ... and what it wants. A run on the new execution path records the target
    # before the book touched it; a legacy run only leaves its pending order, and
    # a run with nothing pending wants what it has. A run with no `execution`
    # block will never write target_raw, so the table says `legacy:` up front
    # rather than leaving it to look like a new run that has not decided yet.
    pending = st.get("pending") or {}
    legacy = "legacy:" if "execution" not in cfg else ""
    if st.get("target_raw") is not None:
        raw, source = float(st["target_raw"]), "target_raw"
    elif pending.get("target") is not None:
        raw, source = float(pending["target"]), legacy + "pending"
    else:
        raw, source = held, legacy + "held"

    bars = data_mod.load_csv(str(run.bars_path))
    return {
        "run": run_id, "symbol": symbol,
        "broker": broker.get("kind") or "simulated",
        "equity": equity, "held": held, "raw": raw, "raw_source": source,
        "_bars": bars,
    }, None


def compute_book(base_dir: str | Path, window: int = 60, vol_target: float = 0.15,
                 cap: float = 1.0, tick_s: int = 300) -> dict:
    """The book document for right now. Reads only; `write_book` persists it."""
    base = Path(base_dir)
    rows, skipped = [], []
    for run_id in paper_mod.PaperRun.list_runs(base):
        row, skip = _run_row(base, run_id)
        (rows if row else skipped).append(row or skip)

    n = len(rows)
    total = sum(r["equity"] for r in rows)
    weights = np.zeros(n)
    vols = np.zeros(n)
    ppys = np.zeros(n)
    daily = {}
    for i, r in enumerate(rows):
        bars = r.pop("_bars")
        weights[i] = (r["equity"] / total) * r["raw"] if total else 0.0
        # vol on the run's own calendar — 365.25 for a seven-day crypto index, the
        # engine's 252 for everything else (see native_ppy)
        ppys[i] = native_ppy(bars.index)
        ret = bars["close"].pct_change().dropna().tail(window)
        vols[i] = float(ret.std() * np.sqrt(ppys[i])) if len(ret) > 1 else 0.0
        daily[r["run"]] = _daily_close(bars).pct_change().dropna()
        r.update({"weight": float(weights[i]), "native_vol": float(vols[i]),
                  "ppy": float(ppys[i]), "native_bars": int(len(ret))})

    # correlation from the days every run has; identity if there is nothing to join
    joined = pd.DataFrame(daily).dropna().tail(window) if daily else pd.DataFrame()
    if n and len(joined) >= 3:
        R = joined[[r["run"] for r in rows]].corr().fillna(0.0).to_numpy(copy=True)
        np.fill_diagonal(R, 1.0)
        basis = f"inner-joined daily returns, {len(joined)} rows"
    else:
        R = np.eye(n)
        basis = ("identity — fewer than 3 overlapping days" if n
                 else "identity — no runs")

    D = np.diag(vols)
    cov = D @ R @ D
    book_vol = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    k_uncapped = vol_target / book_vol if book_vol > 0 else float("inf")
    k = min(cap, k_uncapped)
    for r in rows:
        r["k"] = k
        r["target_k"] = r["raw"] * k

    return {
        "version": VERSION,
        "as_of": _now(),
        "tick_s": tick_s,
        "vol_target": vol_target,
        "cap": cap,
        "k": k,
        # json has no inf; None here means "book vol is zero, cap applies"
        "k_uncapped": k_uncapped if np.isfinite(k_uncapped) else None,
        "book_vol": book_vol,
        "after_vol": k * book_vol,
        "applies_to": [r["run"] for r in rows],
        "rows": rows,
        "skipped": skipped,
        "cov": {
            "basis": basis,
            "window": window,
            "D": [float(v) for v in vols],
            "ppy": [float(p) for p in ppys],
            "R": [[float(x) for x in line] for line in R],
            "symbols": [r["symbol"] for r in rows],
            "runs": [r["run"] for r in rows],
        },
    }


# ────────────────────────────── the files ──────────────────────────────

def write_book(base_dir: str | Path, doc: dict) -> Path:
    """book.json atomically, and the same document appended to book.jsonl."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / BOOK_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    with (base / HISTORY_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc) + "\n")
    return path


def read_book(base_dir: str | Path, tick_s: int = 300) -> tuple[dict | None, str]:
    """(doc, "") if book.json is there and fresh, else (None, why).

    Fresh means `as_of` is no older than two ticks. A book older than that was
    written by a tick that has since been missed, and the runs must not size off
    it — they fall back to k = 1.0 and say so. The reason string carries as_of, so
    a caller that wants to record what it ignored still can.
    """
    path = Path(base_dir) / BOOK_FILE
    if not path.exists():
        return None, f"{BOOK_FILE} missing"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        as_of = datetime.fromisoformat(doc["as_of"])
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
    except Exception as exc:                      # malformed, half-written, no as_of
        return None, f"{BOOK_FILE} unreadable: {str(exc)[:80]}"
    age = (datetime.now(timezone.utc) - as_of).total_seconds()
    limit = 2 * int(doc.get("tick_s") or tick_s)
    if age > limit:
        return None, (f"{BOOK_FILE} stale: as_of {doc['as_of']} is {age:.0f}s old, "
                      f"limit {limit}s")
    return doc, ""


def book_multiplier(base_dir: str | Path, run_id: str, tick_s: int = 300) -> tuple[float, dict]:
    """What a run should multiply its raw target by, and the fields to record.

    Never raises and never silent: a missing, stale or non-applicable book gives
    k = 1.0 with `book_stale` set and `book_reason` saying why. The caller decides
    whether to shout (see `alert`).
    """
    doc, reason = read_book(base_dir, tick_s)
    if doc is None:
        return 1.0, {"book_k": 1.0, "book_k_uncapped": None, "book_as_of": None,
                     "book_stale": True, "book_reason": reason}
    if run_id not in (doc.get("applies_to") or []):
        return 1.0, {"book_k": 1.0, "book_k_uncapped": doc.get("k_uncapped"),
                     "book_as_of": doc.get("as_of"), "book_stale": True,
                     "book_reason": f"{run_id} not in book applies_to"}
    k = float(doc.get("k", 1.0))
    return k, {"book_k": k, "book_k_uncapped": doc.get("k_uncapped"),
               "book_as_of": doc.get("as_of"), "book_stale": False, "book_reason": ""}


def alert(base_dir: str | Path, message: str) -> None:
    """One line to stderr and to alerts.log, kind `book`. Log-only — the book-stale
    alert fires on every decision while the book is stale, and the heartbeat's
    `stale` on book.json already carries the cause (DESIGN-risk.md §4)."""
    from . import alerts as alerts_mod

    alerts_mod.notify(base_dir, "book", message, push=False)


# ────────────────────────────── printing ──────────────────────────────

def format_book(doc: dict) -> str:
    """The table `paper.py book` prints."""
    rows = doc["rows"]
    out = []
    if not rows:
        out.append("  no runs in the book")
    else:
        total = sum(r["equity"] for r in rows)
        out.append(f"  {len(rows)} runs, {total:,.2f} total equity, "
                   f"vol target {doc['vol_target']:.0%}, cap {doc['cap']:.2f}\n")
        out.append(f"  {'run':<24}{'symbol':<9}{'broker':<10}{'raw':<15}{'held':>7}"
                   f"{'raw tgt':>9}{'of book':>9}{'nat vol':>9}{'ppy':>8}{'k_unc':>8}"
                   f"{'k':>7}{'tgt*k':>9}")
        for r in rows:
            k_unc = doc["k_uncapped"]
            out.append(f"  {r['run']:<24}{r['symbol']:<9}{r['broker']:<10}"
                       f"{r['raw_source']:<15}{r['held']:>7.1%}{r['raw']:>9.1%}"
                       f"{r['weight']:>9.1%}{r['native_vol']:>9.1%}{r['ppy']:>8.2f}"
                       f"{(f'{k_unc:.3f}' if k_unc is not None else 'inf'):>8}"
                       f"{r['k']:>7.3f}{r['target_k']:>9.1%}")
    for s in doc.get("skipped") or []:
        out.append(f"  skipped {s['run']} — {s['reason']}")

    if len(rows) > 1:
        labels = [r["run"] for r in rows]
        R = pd.DataFrame(doc["cov"]["R"], index=labels, columns=labels)
        out.append(f"\n  correlation — {doc['cov']['basis']}, window {doc['cov']['window']}\n")
        out.append("   " + R.round(2).to_string().replace("\n", "\n   "))
    elif rows:
        out.append(f"\n  one run — {doc['cov']['basis']}")
    if rows:
        # the diagonal of Sigma, so the basis of book_vol is on the page too
        out.append("\n  D (native vol, annualised on each run's own ppy)")
        for r in rows:
            out.append(f"    {r['run']:<24}{r['native_vol']:>8.2%}   ppy {r['ppy']:.2f}")

    k_unc = doc["k_uncapped"]
    out.append(f"\n  book_vol   {doc['book_vol']:.2%}")
    out.append(f"  after_vol  {doc['after_vol']:.2%}")
    out.append(f"  k          {doc['k']:.4f}   (uncapped "
               f"{f'{k_unc:.4f}' if k_unc is not None else 'inf'}, cap {doc['cap']:.2f})")
    out.append(f"  as_of      {doc['as_of']}")
    return "\n".join(out)

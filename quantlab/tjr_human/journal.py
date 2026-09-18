"""The journal of DESIGN-tjr-human.md section 9.4 — the bot writes everything, the human writes nothing.

One JSONL file per instrument-month under a base directory (default
`tjr_human_runs/`), append-only. Every record carries

    seq      a sequence number, strictly increasing over the WHOLE journal (all files)
    ts       the UTC stamp to the millisecond, ISO with a Z
    ts_ms    the same as epoch milliseconds (never decreases)
    kind     one of KINDS
    instrument, and setup_id when the record belongs to a setup

and is flushed and fsynced before `write` returns. The month in the file name
is the month of the record's own stamp; records that belong to no instrument
(ARMED, a gap, a review, a message nobody can place) go to the `RUN` file.

The kinds of section 9.4: signal, command, fill, stop_set, stop_move, reject,
exit, skip, invalidated, benchmarks, review, observe. Four more, because the
run needs them and the contract names the things they record: `event` (the
detector's other events — levels, sweep, no_sweep, confirmation, touch,
entry_trigger, expired — kept so a restart can tell what it has already seen),
`armed` (section 9.7), `gap` (section 9.7: a restart is journaled as a gap) and
`final` (the section 5 test, once). Two more belong to the runner: `run` (a start
or a stop: mode, feed, warm history, what the code and section 5 hash to) and
`feed` (section 9.6: a feed failure is journaled and retried; a recovery; a hole
in the store). Neither belongs to a setup and no reader counts them as anything.

Readers rebuild the trades, the skipped setups and the filled-trade count from
the records alone. A torn last line (the process died inside a write) is
skipped and counted, never repaired: the file is append-only.

Nothing here knows a token. `write` refuses any record that carries the value
of TELEGRAM_BOT_TOKEN anywhere in it.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .detector import level_class, level_type
from .exits import cost_points, net_r

DEFAULT_DIR = "tjr_human_runs"
RUN = "RUN"
SCHEMA = 1
CONTRACT_KINDS = ("signal", "command", "fill", "stop_set", "stop_move", "reject", "exit", "skip",
                  "invalidated", "benchmarks", "review", "observe")
EXTRA_KINDS = ("event", "armed", "gap", "final", "run", "feed")
KINDS = CONTRACT_KINDS + EXTRA_KINDS
GAP_MIN_S = 60.0
TOKEN_VAR = "TELEGRAM_BOT_TOKEN"
WRITE_ATTEMPTS = 5            # of one record, on OSError (a sharing violation on Windows)
WRITE_RETRY_S = 0.1


def token_forms(token: str | None) -> list[str]:
    """Every spelling of the token a record or an error message could carry: as
    it is in the environment, stripped, and ESCAPED — a token with a trailing CR
    or LF (a CRLF launcher) appears in http.client's InvalidURL message as
    repr() text, which a plain replace of the raw value never matches."""
    from ..alerts import secret_forms               # one list for the log, the journal and the error texts:
    return secret_forms(token)                      # raw, stripped, percent-encoded, repr- and JSON-escaped


_NAME =re.compile(r"^([A-Z0-9]+)-(\d{4})-(\d{2})\.jsonl$")


def iso_ms(epoch_s: float) -> str:
    """UTC, to the millisecond: 2026-09-18T13:50:00.123Z"""
    ms = int(round(float(epoch_s) * 1000.0))
    dt = datetime.fromtimestamp(ms // 1000, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}.{ms % 1000:03d}Z"


def bar_epoch(stamp: str) -> float:
    """A detector stamp ("YYYY-MM-DDTHH:MM:SS", tz-naive UTC) as epoch seconds."""
    return datetime.fromisoformat(str(stamp)).replace(tzinfo=timezone.utc).timestamp()


def epoch_bar(epoch_s: float) -> str:
    """The stamp of the 1-minute bar that contains `epoch_s`."""
    dt = datetime.fromtimestamp(int(epoch_s // 60) * 60, tz=timezone.utc)
    return f"{dt:%Y-%m-%dT%H:%M:%S}"


def clean(obj):
    """Plain JSON-safe data: numpy scalars to Python, NaN / inf to None, tuples to lists."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def file_key(instrument: str | None) -> str:
    """NQ, ES, CME_MINI:NQ1! -> NQ / ES; nothing -> RUN."""
    if not instrument:
        return RUN
    root = re.sub(r"[^A-Z0-9]", "", str(instrument).upper().split(":")[-1])
    for known in ("NQ", "ES"):
        if root.startswith(known):
            return known
    return root or RUN


# ────────────────────────────── reading ──────────────────────────────

def _read_file(path: Path) -> tuple[list[dict], int]:
    """Records of one file and the count of lines that are not records. The file
    is read as BYTES and each line decoded on its own: a line torn inside a
    multi-byte character (the process died mid-write) is one bad line, never a
    UnicodeDecodeError out of every reader."""
    out, bad = [], 0
    try:
        with path.open("rb") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except ValueError:                      # UnicodeDecodeError is a ValueError
                    bad += 1
                    continue
                if isinstance(rec, dict) and "seq" in rec:
                    out.append(rec)
                else:
                    bad += 1
    except OSError:
        pass
    return out, bad


def journal_files(base_dir: str | Path = DEFAULT_DIR) -> list[Path]:
    base = Path(base_dir)
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if _NAME.match(p.name))


def read_journal(base_dir: str | Path = DEFAULT_DIR) -> list[dict]:
    """Every record of every file, in sequence order."""
    recs: list[dict] = []
    for p in journal_files(base_dir):
        recs.extend(_read_file(p)[0])
    recs.sort(key=lambda r: r["seq"])
    return recs


def integrity(base_dir: str | Path = DEFAULT_DIR) -> dict:
    """Machinery facts: is the journal whole? Sequence numbers unique and
    increasing within each file, stamps never decreasing along the sequence,
    every kind known, every record in the file its instrument and month name."""
    problems: list[str] = []
    bad_lines = 0
    allrecs: list[dict] = []
    for p in journal_files(base_dir):
        recs, bad = _read_file(p)
        bad_lines += bad
        key, yy, mm = _NAME.match(p.name).groups()
        last = -1
        for r in recs:
            if r["seq"] <= last:
                problems.append(f"{p.name}: seq {r['seq']} after {last}")
            last = r["seq"]
            if r.get("kind") not in KINDS:
                problems.append(f"{p.name}: seq {r['seq']} unknown kind {r.get('kind')!r}")
            if file_key(r.get("instrument")) != key or str(r.get("ts", ""))[:7] != f"{yy}-{mm}":
                problems.append(f"{p.name}: seq {r['seq']} is in the wrong file")
        allrecs.extend(recs)
    allrecs.sort(key=lambda r: r["seq"])
    seqs = [r["seq"] for r in allrecs]
    if len(set(seqs)) != len(seqs):
        problems.append("duplicate sequence numbers")
    missing = (seqs[-1] - seqs[0] + 1 - len(set(seqs))) if seqs else 0
    if missing:
        problems.append(f"{missing} sequence numbers missing")
    for a, b in zip(allrecs, allrecs[1:]):
        if b["ts_ms"] < a["ts_ms"]:
            problems.append(f"seq {b['seq']}: stamp before seq {a['seq']}'s")
    return {"ok": not problems and not bad_lines, "records": len(allrecs), "files": len(journal_files(base_dir)),
            "bad_lines": bad_lines, "first_seq": seqs[0] if seqs else None, "last_seq": seqs[-1] if seqs else None,
            "problems": problems}


# ────────────────────────────── writing ──────────────────────────────

class Journal:
    """Append-only writer. `clock` returns epoch seconds (UTC); a replay passes its
    simulated clock, or passes `now` to every `write`."""

    def __init__(self, base_dir: str | Path = DEFAULT_DIR, clock=time.time, fsync: bool = True):
        self.base = Path(base_dir)
        self.clock = clock
        self.fsync = bool(fsync)
        self._sizes: dict[str, int] = {}
        self._seq = 0
        self._last_ms = 0
        self._lock = threading.RLock()        # the inbound thread journals too
        self._rescan()

    # another process may have appended (the `arm` command while the runner is up):
    # a changed file size is noticed before the next write and the sequence is re-read
    def _stat(self) -> dict[str, int]:
        out = {}
        for p in journal_files(self.base):
            try:
                out[p.name] = p.stat().st_size
            except OSError:
                pass
        return out

    def _rescan(self) -> None:
        recs = read_journal(self.base)
        if recs:
            self._seq = max(self._seq, recs[-1]["seq"])
            self._last_ms = max(self._last_ms, max(r.get("ts_ms", 0) for r in recs))
        self._sizes = self._stat()

    @property
    def last_seq(self) -> int:
        return self._seq

    def records(self) -> list[dict]:
        return read_journal(self.base)

    def write(self, kind: str, body: dict | None = None, instrument: str | None = None,
              now: float | None = None) -> dict:
        """One record, on disk before this returns. Returns the record as written."""
        if kind not in KINDS:
            raise ValueError(f"tjr_human.journal: unknown kind {kind!r}; known: {KINDS}")
        with self._lock:
            return self._write(kind, body, instrument, now)

    def _write(self, kind: str, body: dict | None, instrument: str | None, now: float | None) -> dict:
        if self._stat() != self._sizes:
            self._rescan()
        t = float(self.clock() if now is None else now)
        ms = max(int(round(t * 1000.0)), self._last_ms)
        body = clean(body or {})
        inst = instrument or body.get("instrument")
        rec = {"seq": self._seq + 1, "ts": iso_ms(ms / 1000.0), "ts_ms": ms, "v": SCHEMA, "kind": kind,
               "instrument": file_key(inst) if inst else RUN}
        for k, v in body.items():
            if k not in ("seq", "ts", "ts_ms", "v", "kind", "instrument"):
                rec[k] = v
        line = json.dumps(rec, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        if any(form in line for form in token_forms(os.environ.get(TOKEN_VAR))):
            raise ValueError("tjr_human.journal: refusing to write a record that carries the bot token")
        self.base.mkdir(parents=True, exist_ok=True)
        path = self.base / f"{rec['instrument']}-{rec['ts'][:7]}.jsonl"
        # A Windows sharing violation (antivirus, a backup, the file open in another
        # program) is usually over in a moment: a short bounded retry, then the error.
        for attempt in range(WRITE_ATTEMPTS):
            try:
                torn = False
                if path.exists() and path.stat().st_size:
                    with path.open("rb") as fh:
                        fh.seek(-1, os.SEEK_END)
                        torn = fh.read(1) != b"\n"
                with path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(("\n" if torn else "") + line + "\n")
                    fh.flush()
                    if self.fsync:
                        os.fsync(fh.fileno())
                break
            except OSError:
                if attempt + 1 >= WRITE_ATTEMPTS:
                    raise
                time.sleep(WRITE_RETRY_S)
        self._seq, self._last_ms = rec["seq"], ms
        try:
            self._sizes[path.name] = path.stat().st_size
        except OSError:
            self._sizes = self._stat()
        return rec

    def mark_start(self, now: float | None = None, min_gap_s: float = GAP_MIN_S, process: str = "runner") -> dict | None:
        """Called once when the runner starts. If the journal is not empty and its
        last record is older than `min_gap_s`, or a trade was open when the writing
        stopped, a `gap` record says so: how long, since which record, which trades
        were open. Nothing is written into an empty journal."""
        recs = self.records()
        if not recs:
            return None
        t = float(self.clock() if now is None else now)
        last = recs[-1]
        seconds = max(0.0, t - last["ts_ms"] / 1000.0)
        state = rebuild(recs)
        open_ids = [tr["setup_id"] for tr in state["trades"] if not tr["closed"]]
        pending = [s["setup_id"] for s in state["setups"].values()
                   if s["status"] == "signal" and s["armed"]]
        if seconds < min_gap_s and not open_ids:
            return None
        return self.write("gap", {"process": process, "last_seq": last["seq"], "last_ts": last["ts"],
                                  "seconds": round(seconds, 3), "open_trades": open_ids,
                                  "setups_waiting": pending}, now=t)


# ────────────────────────────── rebuilding the run from the records ──────────────────────────────

def armed_record(records: list[dict]) -> dict | None:
    """The ARMED record in force, or None: observe mode. The last `armed` record
    decides; one written with `"armed": false` is a disarm (section 9.7 allows it
    only while no trade has filled — `filled_count` is what the CLI asks first)."""
    got = [r for r in records if r["kind"] == "armed"]
    if not got or got[-1].get("armed") is False:
        return None
    return got[-1]


def _new_setup(setup_id: str, rec: dict) -> dict:
    return {"setup_id": setup_id, "instrument": rec.get("instrument"), "day": rec.get("day"), "armed": False,
            "status": "signal", "signal": None, "signals": [], "skip": None, "would_fill": None, "fill": None,
            "preview": None, "stop_set": None, "stop_sets": [], "moves": [], "rejects": [], "commands": [],
            "exit": None, "benchmarks": None, "invalidated": None, "expired": None}


def rebuild(records: list[dict]) -> dict:
    """The run, from the journal alone.

    Returns {"setups": {setup_id: raw}, "trades": [trade...] in fill order,
    "skipped": [...], "filled": int, "armed": record|None, "reviews": {n: record},
    "final": record|None, "gaps": [...], "observed": int, "counts": {kind: n}}.

    A filled trade is an armed setup with an authoritative `fill` record (not a
    provisional one) that was not skipped. Its R is RECOMPUTED here from the
    fill's entry, the exit price and the initial stop on the exit record, net of
    the section 9.2 cost — so the number cannot drift from the entry the ten
    mechanical exits were replayed from.
    """
    setups: dict[str, dict] = {}
    counts: dict[str, int] = {}
    reviews: dict[int, dict] = {}
    gaps, final, observed = [], None, 0
    reasons: dict[int, str] = {}
    for r in records:
        k = r["kind"]
        counts[k] = counts.get(k, 0) + 1
        if k == "review":
            reviews.setdefault(int(r.get("n", 0)), r)
            continue
        if k == "final":
            final = final or r
            continue
        if k == "gap":
            gaps.append(r)
            continue
        if k == "observe":
            observed += 1
            continue
        sid = r.get("setup_id")
        if k == "command" and r.get("stage") == "reason":
            reasons[r.get("attaches_to")] = r.get("reason")      # keyed by the command's own sequence number
            continue
        if not sid:
            continue
        s = setups.get(sid)
        if s is None:
            s = setups[sid] = _new_setup(sid, r)
        if k == "signal":
            s["signals"].append(r)
            if s["signal"] is None:
                s["signal"] = r
                s["armed"] = bool(r.get("armed"))
        elif k == "skip":
            if r.get("stage") == "would_fill":
                s["would_fill"] = r
            else:
                s["skip"] = r
                s["status"] = "skipped"
        elif k == "fill":
            if r.get("provisional"):
                s["preview"] = r
            else:
                s["fill"] = r
            if s["status"] == "signal":
                s["status"] = "live"
        elif k == "stop_set":
            s["stop_sets"].append(r)
            s["stop_set"] = r
        elif k == "stop_move":
            s["moves"].append(r)
        elif k == "reject":
            s["rejects"].append(r)
        elif k == "command":
            s["commands"].append(r)
        elif k == "exit":
            s["exit"] = r
            s["status"] = "closed"
        elif k == "benchmarks":
            s["benchmarks"] = r
        elif k == "invalidated":
            s["invalidated"] = r
            if r.get("void"):
                s["fill"] = s["preview"] = None
            if s["status"] in ("signal", "live"):
                s["status"] = "invalidated"
        elif k == "event" and r.get("event") == "expired":
            s["expired"] = r
            if s["status"] == "signal":
                s["status"] = "expired"

    def why(rec):
        """The reason code of an action: one attached later wins over `unspecified`."""
        return reasons.get(rec.get("command_seq")) or rec.get("human_reason")

    for s in setups.values():
        for k in ("stop_sets", "moves", "rejects"):
            for rec in s[k]:
                rec["reason_code"] = why(rec)
        for k in ("skip", "exit"):
            if s[k] is not None:
                s[k]["reason_code"] = why(s[k])

    trades, skipped = [], []
    for s in setups.values():
        if s["skip"] is not None:
            skipped.append(_skipped_view(s))
        elif s["fill"] is not None and s["armed"]:
            trades.append(_trade_view(s))
    trades.sort(key=lambda t: t["fill_seq"])
    for n, t in enumerate(trades, 1):
        t["n"] = n
    skipped.sort(key=lambda t: t["skip_seq"])
    return {"setups": setups, "trades": trades, "skipped": skipped, "filled": len(trades),
            "armed": armed_record(records), "reviews": reviews, "final": final, "gaps": gaps,
            "observed": observed, "counts": counts}


def filled_count(records: list[dict]) -> int:
    """Filled trades: the count section 5's 100 is a count of. Skips are not in it."""
    return rebuild(records)["filled"]


def entered_count(records: list[dict]) -> int:
    """Armed setups that have been ENTERED: an authoritative fill or a provisional
    one (the fill is known at the open of the entry minute, a minute before the
    closed bar confirms it), not skipped, not voided. This — not `filled_count` —
    is what `arm` / `disarm` ask: nothing changes after trade one has been entered
    (sections 5 and 9.7), and the entry minute is part of trade one."""
    return sum(1 for s in rebuild(records)["setups"].values()
               if s["armed"] and s["skip"] is None and (s["fill"] is not None or s["preview"] is not None))


def origin(records: list[dict]) -> dict:
    """Who wrote this journal: {"replay": n, "live": n, "first_live": record|None}.
    A replay writes `armed` with replay:true and `run` records with mode "replay";
    everything between a replay's run start and its stop is the replay's. Any other
    `armed` or `run` record, and any `gap` or `feed` record, is the experiment's."""
    replay = live = 0
    first_live = None
    inside = False
    for r in records:
        k = r.get("kind")
        if k == "run" and r.get("mode") == "replay":
            replay += 1
            inside = r.get("what") == "start"
            continue
        if k == "armed" and r.get("replay"):
            replay += 1
            continue
        if inside:
            replay += 1
            continue
        if k in ("armed", "run", "gap", "feed") or r.get("setup_id"):
            live += 1
            first_live = first_live or r
    return {"replay": replay, "live": live, "first_live": first_live}


def _branches(ev: dict) -> dict:
    level = ev.get("level") or {}
    name = level.get("name") or ""
    conf = list(ev.get("conf_types") or [])
    zone = (ev.get("zone") or {}).get("kind")
    return {"level": level, "level_type": level_type(name) if name else None,
            "level_class": level.get("class") or (level_class(name) if name else None),
            "conf_types": conf, "conf": "+".join(conf) if conf else None, "zone": zone,
            "cooccur": list(ev.get("cooccur") or [])}


def _trade_view(s: dict) -> dict:
    f = s["fill"]
    ev = f.get("ev") or {}
    side, entry = int(ev["side"]), float(ev["entry"])
    x = s["exit"]
    inst = s["instrument"]
    stop0 = float(x["stop0"]) if x else (float(s["stop_set"]["price"]) if s["stop_set"] else float(ev["stops"]["wick"]))
    r = None
    if x is not None and side * (entry - stop0) > 0:
        r = net_r(side, entry, float(x["exit"]), stop0, cost_points(inst, entry))
    chosen = None
    for ss in s["stop_sets"]:
        if not ss.get("void"):
            chosen = ss
    if x is not None and x.get("chosen_stop_void"):
        chosen = None
    out = {"setup_id": s["setup_id"], "instrument": inst, "day": s["day"], "side": side,
           "direction": "long" if side > 0 else "short", "entry": entry, "entry_time": ev.get("bar_time"),
           "fill_seq": f["seq"], "fill_ts_ms": f["ts_ms"], "stops": dict(ev.get("stops") or {}), "atr": ev.get("atr"),
           "t1": ev.get("t1"), "t2": ev.get("t2"), "target_rule": ev.get("target_rule"),
           "stop0": stop0,
           "stop_mode": x["stop_mode"] if x and x.get("stop_mode") else ((chosen or {}).get("mode") or "wick"),
           "stop_when": (chosen or {}).get("when", "default") if chosen else "default",
           "stop_seconds_from_fill": (chosen or {}).get("seconds_from_fill") if chosen else None,
           "stop_atr_multiple": (side * (entry - stop0) / float(ev["atr"])) if ev.get("atr") else None,
           "moves": s["moves"], "rejects": s["rejects"], "exit": x, "closed": x is not None,
           "r": r, "r_journaled": x.get("r") if x else None,
           "mae_r": x.get("mae_r") if x else None, "mfe_r": x.get("mfe_r") if x else None,
           "exit_reason": x.get("reason") if x else None,
           "human_reason": x.get("reason_code") if x else None,
           "benchmarks": s["benchmarks"], "complete": x is not None and s["benchmarks"] is not None}
    out.update(_branches(ev))
    return out


def _skipped_view(s: dict) -> dict:
    sk = s["skip"]
    wf = s["would_fill"]
    ev = (wf or {}).get("ev") or (s["signal"] or {}).get("ev") or {}
    end = "would_fill" if wf else ("invalidated" if s["invalidated"] else ("expired" if s["expired"] else "open"))
    out = {"setup_id": s["setup_id"], "instrument": s["instrument"], "day": s["day"], "skip_seq": sk["seq"],
           "skip_ts_ms": sk["ts_ms"], "reason": sk.get("reason_code"),
           "side": ev.get("side"), "ended": end, "would_fill": wf is not None,
           "entry": (wf or {}).get("ev", {}).get("entry"), "benchmarks": s["benchmarks"]}
    out.update(_branches(ev))
    return out

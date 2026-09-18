"""The loop of DESIGN-tjr-human.md sections 9.1 / 9.6: feed -> detector -> alert -> commands -> trade -> journal.

One `Runner`, two feeds:

  LiveTvFeed   the interim path of section 9.6: TradingView Desktop on this PC, a
               two-pane layout (NQ1! and ES1!, 1 minute), read through the repo's
               `tv` CLI. Closed bars only — a bar is closed once a newer one exists,
               so the newest bar of every read is the FORMING one and only its open
               (the entry, for `Detector.preview_fill`) and its last price are used.
               Closed bars are merged append-only into a local 1-minute store
               (`tjr_human_runs/store/{NQ,ES}_1min.csv`, seeded from
               `data/{NQ,ES}_1min_tv.csv`). The chart in replay mode is refused, as
               `TradingViewFeed` refuses it. Every CLI call is time-bounded; a feed
               failure is journaled and retried, never fatal.
  ReplayFeed   stored 1-minute csv files under a simulated clock, with a scripted
               human (a json list of timed commands) or none (section 9.8).

HOW A SPECIFIC PANE IS READ. `tv ohlcv` reads the ACTIVE chart only and has no pane
argument. Two ways are implemented:
  method "eval"  (default) one `tv ui eval <js>` per poll that walks
                 `_chartWidgetCollection.getAll()` — the same objects `tv pane list`
                 reads — and returns symbol, resolution and the last n bars of EVERY
                 pane. One process per poll, and the human's active pane is left alone.
  method "focus" the documented commands only: `tv pane list`, then per instrument
                 `tv pane focus <i>`, `tv state` (verify symbol and resolution — a
                 focus click that did not take would otherwise file ES bars as NQ)
                 and `tv ohlcv -n <n>`. Seven processes per poll, and the active pane
                 flips under the human's hands.
Both check `tv replay status` first. Whichever is used, the store refuses bars that
do not continue the series it holds (the wrong pane's prices are 4x away).

WHAT OBSERVED PRICES DO. Closed bars decide stops and targets, by exactly the
conventions the ten mechanical exits are replayed with, so a human who does nothing
is `wick_fixed` to the cent. The forming bar never reaches the store or the detector.
It is used for: its open is the entry (`preview_fill`); its last price is the
`market_price` the action records carry and the MOVE STOP market test; its running
high and low test a MOVED stop (on the minute of the move only a new extreme after
the move counts) and come BEFORE a pending EXIT NOW is filled at the last price — a
stop that traded earlier in the minute is the exit, the later price cannot undo it
(`TradeManager.on_forming`). The default and the chosen stop are never tested on it.

STARTING. The runner refuses to start when the store does not hold the 20 completed
sessions and the 100 four-hour bins the levels need (`detector.check_warm`), when
ARMED names another TARGET_RULE than the code, when the chart is in replay mode, and
— unless `accept_holes` — when the two newest sessions of the store have a hole of
ten or more trading minutes (the PC was off: PDH / PDL / ASIA / LON would be wrong),
when the journal was written by a replay, and — on the first poll — when the PC
clock and the chart's newest bar are minutes apart while the market is open.
It starts in OBSERVE mode unless the journal holds an ARMED record (section 9.7).
One `detect` per directory: an OS lock on `<dir>/runner.lock` is held for the life
of the loop. Once a trade has filled, a store that starts somewhere else than ARMED
recorded is refused, and so is a backfill of older history: H1 / H4 swing levels are
read over all the history held, so the store's start is part of their definition.

FAILURES ARE TOLD. A feed failure, a failing getUpdates (the commands do not
arrive), an exception inside a cycle: each is one state per scope, journaled and
logged on the transition and every ten minutes, and pushed to the phone once when
it falls in 09:25-10:15 ET or while a setup or a trade is on, and once on recovery.
A chart that still answers but no longer delivers bars is a failure too (STALLED:
no new bar for three trading minutes — a session taken over by another device, the
"reconnecting" banner, a PC back from sleep), and so is a PC clock that disagrees
with the bar stamps or with Telegram's message dates. A hole that opens while the
runner is up is pushed, and a day whose levels froze on a store with an unaccepted
hole is not routed. The live loop asks Windows not to sleep while it runs.
What a bar produced is delivered to the trade manager at least once: an exception
leaves it queued for the next cycle. A torn last row of a store csv is cut and
journaled; a token with whitespace or a control character in it turns Telegram off.

THE BARS ARE CHECKED BEFORE THEY ARE KEPT. The feed verifies symbol and resolution;
the chart STYLE is a third thing: Heikin Ashi, Renko or Line Break bars under the
right symbol at "1" would be merged into the append-only store as real prices. So
every fetched batch passes `check_batch` before anything is merged: every OHLC
value on the 0.25 tick grid (Heikin Ashi averages fall off it), every stamp a
whole minute, stamps strictly increasing and 60 s apart apart from the known
breaks (the daily halt, the weekend) and a small allowance for a minute in which
nothing traded (Renko and Line Break bricks are nowhere near it), high >= max(open,
close) and low <= min(open, close) — and not EVERY bar without a wick. Where the page says which style a pane shows,
Candles or Bars is required too. A failing batch is never merged: it is a feed
failure (kind "bars") — journaled, pushed, retried.

ALIVE IS SAID, NOT ASSUMED. A dead runner and a day without a setup look the same
from a phone, so the box says it is there: once per trading day at the first poll
at or after 09:25 ET (mode, filled n of 100, per instrument how fresh the store is
and whether it is warm), and at 10:10 ET one line per instrument that did not
fill, with its terminal event. A FAILING state is pushed again once per trading
day while it lasts, and a refusal raised after the start is pushed before the
process ends. All of it is kind "status"; kind "trade" is the fill, the stop set
and the exit, nothing else. None of it carries an R or a P&L.

A TRADE IS NEVER ABANDONED. A filled trade without an exit from a PAST session (a
crash, a PC switched off) is closed at the next start, however long ago it was:
its session is re-fed from the store and the bars the process missed are tested
against the stop in force, the target and the 15:55 flat (`restart_replay`: the
human had no chance to act), then it is benchmarked on exactly that session's
window of the store. If the store does not hold that session through 15:55 ET
the start is refused, naming the session to backfill.

Nothing here constructs a broker or sends an order. The only network is Telegram
(`quantlab.alerts.notify` out, `commands.TelegramInbound` in), and with the two
variables unset neither makes a request. Pushes and chart drawing run on their own
threads: neither can delay the detector.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import detector as D
from . import journal as J
from . import report as R
from .chart import Chart, TvDraw
from .commands import TelegramInbound
from .commands import _mask as mask_token
from .commands import configured as telegram_configured
from .commands import parse as parse_command
from .commands import telegram_problem
from .detector import Detector, check_warm, et_clock
from .exits import FLAT_MINUTES
from .trade import LATE_FILL_S, STOP_LABEL, TradeManager, send_notices

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "DESIGN-tjr-human.md"
INSTRUMENTS = ("NQ", "ES")
SYMBOLS = {"NQ": "NQ1!", "ES": "ES1!"}
SEEDS = {k: REPO / "data" / f"{k}_1min_tv.csv" for k in INSTRUMENTS}
STORE_DIR = "store"
COLS = ["open", "high", "low", "close", "volume"]

POLL_FAST_S = 5.0                 # 09:25-10:15 ET, a setup waiting for its entry, a trade on
POLL_SLOW_S = 60.0
FAST_FROM, FAST_TO = 9 * 60 + 25, 10 * 60 + 15
CLI_TIMEOUT_S = 15.0
FETCH_MIN, FETCH_MAX_EVAL, FETCH_MAX_FOCUS = 30, 1500, 500        # `tv ohlcv` caps at 500
LATE_S = 120.0                    # an alert about a bar that closed longer ago than this says so
HOLE_MIN = 10                     # trading minutes missing in a row before it is called a hole
STILL_FAILING_S = 600.0
INBOUND_FAIL_AFTER = 2            # consecutive failed getUpdates before it is called a failure (one timeout is weather)
STALL_MIN = 3                     # trading minutes without a new bar before the feed is called STALLED
CLOCK_SLACK_S = 90.0              # the PC clock may sit this far outside the forming bar's minute, no further
CLOCK_REFUSE_S = 180.0            # at the first poll: PC clock this far BEHIND the newest bar -> refuse to start
CLOCK_REFUSE_MIN = 5              # at the first poll: newest bar this many trading minutes behind the PC clock -> refuse
CMD_CLOCK_SLACK_S = 5.0           # a Telegram message dated this far AFTER its receipt: the PC clock is slow
CMD_LATE_AFTER = 2                # consecutive commands older than the stale limit before the clock is suspected
LOCK_FILE = "runner.lock"
FUNNEL_SESSIONS = 149             # the depth of history section 2.6's fill count was measured with
TAIL_BYTES = 1 << 16
WRONG_SERIES_OVERLAP, WRONG_SERIES_JUMP = 0.005, 0.15
SESSION_TAIL_ROWS = 20000         # the window `close_session` is given ENDS with the fill's session: ~14 sessions to warm the ATR
TICK_SIZE = {"NQ": 0.25, "ES": 0.25}
GRID_TOL = 1e-6                   # price units: how far from the tick grid a real price may sit (float noise)
GAP_ALLOW_MIN, GAP_ALLOW_FRAC = 2, 0.05   # minutes in which nothing traded, per batch: the stored ES file has 6 in 41,000
WICKLESS_MIN = 10                 # this many closed bars and not one wick among them: bricks / lines, not minutes
CHART_TYPES = ("Bars", "Candles", "Line", "Area", "Renko", "Kagi", "PointAndFigure", "LineBreak", "HeikinAshi",
               "HollowCandles")   # src/core/chart.js: what `tv state` reports as `chartType`
CHART_TYPES_OK = (0, 1)           # Bars, Candles: true OHLC of the minute
ALIVE_AT, TERMINAL_AT, TERMINAL_LATEST, DAY_END = 9 * 60 + 25, 10 * 60 + 10, 10 * 60 + 12, 17 * 60
MARKER_SUFFIX = ".synthetic.json"
PANES_MARK = "/*tjr_human:panes*/"


class FeedError(RuntimeError):
    def __init__(self, message: str, kind: str = "failure"):
        super().__init__(message)
        self.kind = kind          # failure | timeout | replay | layout | series


class Refused(RuntimeError):
    """The runner (or `arm`) will not do this, and says why."""


# ────────────────────────────── time ──────────────────────────────

def _stamp(epoch_s: float) -> pd.Timestamp:
    return pd.Timestamp(int(round(float(epoch_s) * 1000)), unit="ms")


def _epoch(stamp) -> float:
    return float(pd.Timestamp(stamp).value) / 1e9


def et_minute_and_day(epoch_s: float) -> tuple[int, str]:
    """(ET minute of the day, CME session date 'YYYY-MM-DD') of an instant."""
    minutes, day = et_clock(pd.DatetimeIndex([_stamp(epoch_s)]))
    return int(minutes[0]), str(day[0])[:10]


def poll_interval(now: float, busy: bool) -> float:
    """Section 9.6: every 5 s from 09:25 to 10:15 ET and while a setup or a trade is on, slower otherwise."""
    m, _ = et_minute_and_day(now)
    return POLL_FAST_S if busy or FAST_FROM <= m < FAST_TO else POLL_SLOW_S


def trading_minutes_between(a, b) -> int:
    """Minutes strictly between two bar stamps (tz-naive UTC) on which CME index
    futures trade: not 17:00-18:00 ET, not Friday 17:00 to Sunday 18:00. Holidays
    are not known here — an early close reads as a hole, and a person decides."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if b - a <= pd.Timedelta(minutes=1):
        return 0
    ix = pd.date_range(a + pd.Timedelta(minutes=1), b - pd.Timedelta(minutes=1), freq="1min")
    if len(ix) > 60 * 24 * 60:
        return len(ix)
    et = ix.tz_localize("UTC").tz_convert(D.M.ET)
    m, wd = et.hour * 60 + et.minute, et.weekday
    shut = ((m >= 17 * 60) & (m < 18 * 60)) | (wd == 5) | ((wd == 4) & (m >= 17 * 60)) | ((wd == 6) & (m < 18 * 60))
    return int((~shut).sum())


def recent_holes(frame: pd.DataFrame, sessions: int = 2) -> list[dict]:
    """Holes of HOLE_MIN or more trading minutes that END inside the newest `sessions` session days of a store."""
    if frame is None or len(frame) < 2:
        return []
    _, day = et_clock(frame.index)
    days = np.unique(day)
    keep = day >= days[max(0, len(days) - sessions)]
    first = int(np.argmax(keep))
    idx = frame.index[max(0, first - 1):]
    gaps = np.diff(idx.values).astype("timedelta64[s]").astype(np.int64)
    out = []
    for k in np.where(gaps > 60)[0]:
        missing = trading_minutes_between(idx[k], idx[k + 1])
        if missing >= HOLE_MIN:
            out.append({"after": idx[k].isoformat(), "before": idx[k + 1].isoformat(), "trading_minutes": missing})
    return out


def stalled_minutes(newest_bar, now: float) -> int:
    """Trading minutes that have gone by since the newest bar the chart shows (its
    forming bar, else the newest closed one) without a newer bar appearing. 0 in
    normal running: the forming bar is the minute `now` is in. The daily halt and
    the weekend do not count; holidays are not known."""
    if newest_bar is None:
        return 0
    return trading_minutes_between(pd.Timestamp(newest_bar), _stamp(now).floor("1min"))


_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on: bool) -> bool:
    """Ask Windows not to sleep while the live loop runs (Windows 11 Home sleeps by
    default, and a sleeping PC is a hole in the store). A request of the calling
    thread, undone with `keep_awake(False)` and by the process ending; no power
    setting is changed. False when it could not be asked (not Windows). Never raises."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))
    except Exception:
        return False


# ────────────────────────────── synthetic files (section 9.8) ──────────────────────────────

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _market_hashes() -> set[str]:
    out = set()
    try:
        for p in (REPO / "data").glob("*.csv"):
            out.add(file_sha256(p))
    except OSError:
        pass
    return out


def mark_synthetic(csv_path, generator: str, note: str = "") -> Path:
    """Flag a csv as synthetic: a sidecar `<file>.synthetic.json` bound to the
    file's bytes by their SHA-256. It is the only thing `--show-outcomes` believes.
    Refuses the repo's `data/` directory and any file whose bytes are a file in it."""
    p = Path(csv_path).resolve()
    if (REPO / "data").resolve() in p.parents:
        raise Refused(f"{p.name} is under data/: market files are never marked synthetic")
    sha = file_sha256(p)
    if sha in _market_hashes():
        raise Refused(f"{p.name} is byte-identical to a file under data/: it is market data")
    side = Path(str(p) + MARKER_SUFFIX)
    side.write_text(json.dumps({"synthetic": True, "generator": str(generator), "note": str(note),
                                "sha256": sha, "file": p.name}, indent=2) + "\n", encoding="utf-8")
    return side


def synthetic_status(csv_path) -> dict:
    """{"synthetic": bool, "why": str}. Synthetic means: the sidecar exists, says
    `"synthetic": true`, names a generator, and its SHA-256 is the SHA-256 of the
    csv as it is now. Never the file name, never a column, never a guess: a file
    without a valid sidecar is market data."""
    p = Path(csv_path)
    side = Path(str(p) + MARKER_SUFFIX)
    if not side.exists():
        return {"synthetic": False, "why": f"no {side.name} beside it"}
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"synthetic": False, "why": f"{side.name} is unreadable ({type(exc).__name__})"}
    if not isinstance(meta, dict) or meta.get("synthetic") is not True or not meta.get("generator"):
        return {"synthetic": False, "why": f"{side.name} does not say synthetic: true with a generator"}
    try:
        sha = file_sha256(p)
    except OSError as exc:
        return {"synthetic": False, "why": f"cannot read the file ({type(exc).__name__})"}
    if meta.get("sha256") != sha:
        return {"synthetic": False, "why": f"{side.name} was written for other bytes (sha256 differs)"}
    if sha in _market_hashes():
        return {"synthetic": False, "why": "byte-identical to a file under data/"}
    return {"synthetic": True, "why": f"marked by {meta['generator']}", "generator": meta["generator"]}


def write_synthetic_csv(path, frame: pd.DataFrame, generator: str, note: str = "") -> Path:
    """A 1-minute frame to a store-format csv, with its synthetic marker."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(p, frame)
    mark_synthetic(p, generator, note)
    return p


def synthetic_walk(days: int = 25, seed: int = 1, start: str = "2026-01-05", price: float = 20000.0,
                   sd: float = 4.0, tick: float = 0.25) -> pd.DataFrame:
    """Seeded random-walk sessions on 1-minute bars, 18:00-16:59 ET, with volume, on the
    tick grid (rounding is monotonic: high >= open, close >= low survives it). No market in it."""
    rng = np.random.default_rng(int(seed))
    parts, last = [], float(price)
    for day in pd.bdate_range(start, periods=int(days)):
        ix = pd.date_range(pd.Timestamp(f"{(day - pd.Timedelta(days=1)).date()} 18:00", tz=D.M.ET),
                           pd.Timestamp(f"{day.date()} 16:59", tz=D.M.ET), freq="1min")
        c = last + np.cumsum(rng.normal(0.0, sd, len(ix)))
        o = np.concatenate([[last], c[:-1]])
        wig = np.abs(rng.normal(0.0, sd / 2.0, (2, len(ix))))
        parts.append(pd.DataFrame({"open": o, "high": np.maximum(o, c) + wig[0], "low": np.minimum(o, c) - wig[1],
                                   "close": c, "volume": rng.integers(50, 500, len(ix)).astype(float)},
                                  index=ix.tz_convert("UTC").tz_localize(None)))
        last = float(c[-1])
    out = pd.concat(parts)
    out[COLS[:4]] = (out[COLS[:4]] / tick).round() * tick
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "time"
    return out


# ────────────────────────────── the 1-minute store ──────────────────────────────

def _norm(frame: pd.DataFrame) -> pd.DataFrame:
    """tz-naive UTC DatetimeIndex, the five columns, sorted, no duplicate stamps, no NaN prices."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="time"), dtype=float)
    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("tjr_human.runner: a bar frame needs a DatetimeIndex")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLS].astype(float)
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time"
    return df


def _lines(frame: pd.DataFrame) -> str:
    return "".join(f"{ts:%Y-%m-%d %H:%M:%S},{r.open!r},{r.high!r},{r.low!r},{r.close!r},{r.volume!r}\n"
                   for ts, r in zip(frame.index, frame.itertuples(index=False)))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write("time,open,high,low,close,volume\n")
        fh.write(_lines(_norm(frame)))
    os.replace(tmp, path)


def load_bars(path) -> pd.DataFrame:
    from ..data import load_csv
    return _norm(load_csv(str(path)))


def _row_ok(line: bytes) -> bool:
    """A whole store row: six fields, a stamp and five finite numbers."""
    try:
        parts = line.decode("utf-8").rstrip("\r").split(",")
        if len(parts) != 6 or len(parts[0]) < 10:
            return False
        pd.Timestamp(parts[0])
        return all(np.isfinite(float(x)) for x in parts[1:])
    except (ValueError, TypeError):
        return False


def repair_tail(path) -> dict | None:
    """A kill or a power cut inside an append leaves a torn last row. A row cut
    after five commas still parses — as a bar with a wrong close — and a row
    without its newline glues itself to the next append. So: a file that does not
    end in a newline loses its last (partial) line, and trailing lines that are
    not whole rows go with it. Nothing else is touched; the feed brings the minute
    back. Returns what was cut, or None."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    with p.open("rb") as fh:
        fh.seek(max(0, size - TAIL_BYTES))
        tail = fh.read()
    start = size - len(tail)
    keep, dropped = len(tail), 0
    if not tail.endswith(b"\n"):
        cut = tail.rfind(b"\n")
        if cut < 0 and start > 0:
            return None                                 # no line end in 64 KB: not a store csv; the loader will say so
        keep, dropped = cut + 1, 1
    while keep > 0:
        prev = tail.rfind(b"\n", 0, keep - 1)
        if prev < 0 and start > 0:
            break
        line = tail[prev + 1:keep - 1]
        if _row_ok(line) or line.startswith(b"time,"):
            break
        keep, dropped = prev + 1, dropped + 1
    if keep == len(tail):
        return None
    with p.open("r+b") as fh:
        fh.truncate(start + keep)
        fh.flush()
        os.fsync(fh.fileno())
    return {"file": p.name, "lines_dropped": dropped, "bytes_dropped": size - (start + keep)}


class BarStore:
    """One instrument's closed 1-minute bars: a csv that only grows at its end.

    `merge(bars)` appends the rows stamped after the newest row it holds and
    nothing else — a row it already has is never rewritten, an older one is never
    inserted: the store is what the detector saw. `backfill` (the union, the
    store's own rows winning) exists for the hour before a run, never during one.
    `path=None` keeps it in memory (replay)."""

    def __init__(self, path=None, seed=None, frame: pd.DataFrame | None = None):
        self.path = Path(path) if path is not None else None
        self.seeded_from = None
        self.repairs: list[dict] = []                  # torn tails cut from the csv; the runner journals them
        if frame is not None:
            self.frame = _norm(frame)
        elif self.path is not None and self.path.exists():
            fixed = repair_tail(self.path)
            if fixed:
                self.repairs.append(fixed)
            self.frame = load_bars(self.path)
        elif seed is not None and Path(seed).exists():
            self.frame = load_bars(seed)
            self.seeded_from = str(seed)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                _write_frame(self.path, self.frame)
        else:
            self.frame = _norm(None)

    @property
    def last(self) -> pd.Timestamp | None:
        return self.frame.index[-1] if len(self.frame) else None

    def tail(self, rows: int = SESSION_TAIL_ROWS) -> pd.DataFrame:
        return self.frame.iloc[-int(rows):]

    def session_rows(self, day) -> np.ndarray:
        """Row numbers of the session dated `day` ('YYYY-MM-DD', the CME session date)."""
        if not len(self.frame):
            return np.array([], dtype=np.int64)
        _, d = et_clock(self.frame.index)
        return np.where(d == np.datetime64(pd.Timestamp(day).normalize(), "ns"))[0]

    def session_window(self, day, rows: int = SESSION_TAIL_ROWS) -> pd.DataFrame | None:
        """Exactly what benchmarking one session needs, however old it is: the bars of
        that session, the `rows` before its end (the context ATR's warm-up) and the first
        bar of the next session when there is one (it says this one is over). At the
        15:55 close of the newest session this IS the old `tail()`; for a session a
        month back it is the same window a tail would have been then."""
        idx = self.session_rows(day)
        if not len(idx):
            return None
        end = int(idx[-1]) + 1
        if end < len(self.frame):
            end += 1
        return self.frame.iloc[max(0, end - int(rows)):end]

    def session_complete(self, day) -> tuple[bool, str | None, bool]:
        """(holds a minute stamped >= 15:55 ET of that session, its last stamp, a later session exists)."""
        idx = self.session_rows(day)
        if not len(idx):
            return False, None, False
        m, _ = et_clock(self.frame.index[idx[-1]:idx[-1] + 1])
        return bool(int(m[0]) >= FLAT_MINUTES), self.frame.index[idx[-1]].isoformat(), bool(idx[-1] + 1 < len(self.frame))

    def session_gaps(self, day, start=None) -> list[dict]:
        """Trading minutes the store does NOT hold between `start` (a bar stamp of that session: the
        entry minute of a trade; None = the session's first row held) and the minute stamped 15:55 ET
        — or the session's last row when it holds no 15:55. `session_complete` reads the LAST stamp
        only; a session can end at 15:55 and still lack the minutes in which a stop or a target was
        hit. ANY missing trading minute counts: one is enough to hide the touch."""
        idx = self.session_rows(day)
        if not len(idx):
            return []
        stamps = self.frame.index[idx]
        m, _ = et_clock(stamps)
        first = stamps[0] if start is None else pd.Timestamp(start)
        keep = np.asarray(stamps >= first)
        flat = np.where(keep & (np.asarray(m) >= FLAT_MINUTES))[0]
        if len(flat):
            keep &= np.asarray(stamps <= stamps[flat[0]])
        sel = stamps[keep]
        out = []
        if not len(sel):
            return [{"after": None, "before": None, "from": first.isoformat(), "trading_minutes": None}]
        if sel[0] > first:                              # the entry minute itself is not held
            out.append({"after": None, "before": sel[0].isoformat(), "from": first.isoformat(),
                        "trading_minutes": trading_minutes_between(first - pd.Timedelta(minutes=1), sel[0])})
        gaps = np.diff(sel.values).astype("timedelta64[s]").astype(np.int64)
        for k in np.where(gaps > 60)[0]:
            missing = trading_minutes_between(sel[k], sel[k + 1])
            if missing >= 1:
                out.append({"after": sel[k].isoformat(), "before": sel[k + 1].isoformat(), "trading_minutes": missing})
        return out

    def after(self, stamp) -> pd.DataFrame:
        return self.frame if stamp is None else self.frame.loc[self.frame.index > pd.Timestamp(stamp)]

    def _check_series(self, bars: pd.DataFrame) -> list:
        """Do these bars continue the series held? Returns the stamps of overlapping rows that differ
        (the chart revised a bar the store already holds; the store is never rewritten)."""
        if not len(self.frame) or not len(bars):
            return []
        both = bars.index.intersection(self.frame.index)
        if len(both):
            old, new = self.frame.loc[both], bars.loc[both]
            rel = float(np.median(np.abs(new["close"].to_numpy() - old["close"].to_numpy())
                                  / np.abs(old["close"].to_numpy())))
            if rel > WRONG_SERIES_OVERLAP:
                raise FeedError(f"the bars read do not match the store on {len(both)} shared minutes "
                                f"(median close difference {rel:.1%}): another instrument's pane?", "series")
            differ = (np.abs(new[COLS[:4]].to_numpy() - old[COLS[:4]].to_numpy()) > 1e-9).any(axis=1)
            return list(both[differ])
        newer = bars.loc[bars.index > self.frame.index[-1]]
        if len(newer):
            a, b = float(self.frame["close"].iloc[-1]), float(newer["open"].iloc[0])
            if abs(b - a) / abs(a) > WRONG_SERIES_JUMP:
                raise FeedError(f"the bars read start {abs(b - a) / abs(a):.0%} away from the store's last close: "
                                "another instrument's pane?", "series")
        return []

    def merge(self, bars: pd.DataFrame, normalised: bool = False) -> tuple[pd.DataFrame, dict]:
        bars = bars if normalised else _norm(bars)
        revised = self._check_series(bars)
        last = self.last
        new = bars if last is None else bars.loc[bars.index > last]
        info = {"new": int(len(new)), "overlap": int(len(bars) - len(new)), "revised_upstream": len(revised),
                "revised_stamps": revised, "hole": None}
        if not len(new):
            return new, info
        if last is not None:
            missing = trading_minutes_between(last, new.index[0])
            if missing >= HOLE_MIN:
                info["hole"] = {"after": last.isoformat(), "before": new.index[0].isoformat(),
                                "trading_minutes": missing}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self.path.exists() or self.path.stat().st_size == 0
            if not fresh:                               # never append to a torn last row
                fixed = repair_tail(self.path)
                if fixed:
                    self.repairs.append(fixed)
            with self.path.open("a", encoding="utf-8", newline="") as fh:
                if fresh:
                    fh.write("time,open,high,low,close,volume\n")
                fh.write(_lines(new))
                fh.flush()
                os.fsync(fh.fileno())
        self.frame = pd.concat([self.frame, new]) if len(self.frame) else new
        return new, info

    def backfill(self, bars: pd.DataFrame, allow_older: bool = True) -> dict:
        bars = _norm(bars)
        self._check_series(bars)
        add = bars.loc[~bars.index.isin(self.frame.index)]
        if not allow_older and len(add) and len(self.frame) and add.index[0] < self.frame.index[0]:
            raise Refused(f"the backfill starts at {add.index[0]}, before the store's first bar {self.frame.index[0]}, "
                          "and a trade has filled: H1 / H4 swing levels are read over ALL the history in the store, "
                          "so older history would change the level definition mid-sample (section 5). "
                          "Backfill only from the store's first bar on")
        if len(add):
            self.frame = _norm(pd.concat([self.frame, add]))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                _write_frame(self.path, self.frame)
        return {"added": int(len(add)), "rows": int(len(self.frame))}


# ────────────────────────────── the tv CLI ──────────────────────────────

class TvCli:
    """One bounded call of the repo's `tv` CLI. `argv` is the command prefix — by
    default `node src/cli/index.js`; a test passes a fake. `lock` serialises the
    focus-then-act pairs of the feed and of the chart layer."""

    def __init__(self, argv: list[str] | None = None, timeout: float = CLI_TIMEOUT_S):
        self.argv = list(argv) if argv else ["node", str(REPO / "src" / "cli" / "index.js")]
        self.timeout = float(timeout)
        self.lock = threading.RLock()
        self.calls = 0

    def call(self, *args: str, timeout: float | None = None) -> dict:
        budget = self.timeout if timeout is None else float(timeout)
        what = " ".join(str(a) for a in args[:2])
        self.calls += 1
        try:
            proc = subprocess.run([*self.argv, *[str(a) for a in args]], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=budget, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise FeedError(f"tv {what}: no answer within {budget:.0f} s", "timeout") from None
        except OSError as exc:
            raise FeedError(f"tv {what}: cannot run {self.argv[0]!r} ({type(exc).__name__})") from None
        out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        payload, start = None, out.find("{")
        if start >= 0:
            try:
                payload = json.loads(out[start:])
            except ValueError:
                payload = None
        if not isinstance(payload, dict):
            why = f"exited {proc.returncode}" if proc.returncode else "no JSON returned"
            raise FeedError(f"tv {what}: {why}. {' '.join(out[:200].split())}")
        if not payload.get("success", True):
            raise FeedError(f"tv {what}: {payload.get('error')}")
        return payload


def check_batch(rows, tick: float = 0.25, what: str = "") -> None:
    """App-independent sanity of one fetched batch, BEFORE it is normalised or merged
    (normalising would sort and de-duplicate what Renko gets wrong). `rows` is what
    the chart returned: lists `[time, o, h, l, c, ...]` or dicts, oldest first, the
    last one still forming. Raises FeedError(kind="bars"); returns None when it is
    plain 1-minute OHLC. The forming bar is held to the stamp rules and its open to
    the grid — its open is the entry `preview_fill` is given."""
    if not rows:
        return
    who = f"{what}: " if what else ""
    try:
        if isinstance(rows[0], dict):
            a = np.asarray([[r.get(k) for k in ("time", "open", "high", "low", "close")] for r in rows], dtype=float)
        else:
            a = np.asarray([list(r[:5]) for r in rows], dtype=float)
    except (TypeError, ValueError):
        raise FeedError(f"{who}the batch is not rows of time, open, high, low, close", "bars") from None
    if a.ndim != 2 or a.shape[1] != 5 or not np.isfinite(a).all():
        raise FeedError(f"{who}the batch has missing or non-finite values", "bars")
    t, px = a[:, 0], a[:, 1:5]
    closed = px[:-1]
    n = len(closed)
    off = np.abs(closed / tick - np.round(closed / tick)) * tick > GRID_TOL
    o_form = px[-1, 0]
    if off.any() or abs(o_form / tick - round(o_form / tick)) * tick > GRID_TOL:
        raise FeedError(f"{who}{int(off.any(axis=1).sum())} of {n} closed bars have a price off the {tick} tick grid — "
                        "averaged candles (HEIKIN ASHI?), not traded prices. Set the pane to plain Candles; "
                        "nothing was merged", "bars")
    o, h, l, c = closed[:, 0], closed[:, 1], closed[:, 2], closed[:, 3]          # noqa: E741
    bad = (h < np.maximum(o, c) - GRID_TOL) | (l > np.minimum(o, c) + GRID_TOL)
    if bad.any():
        raise FeedError(f"{who}{int(bad.sum())} of {n} closed bars have a high under, or a low over, their own open "
                        "or close: not OHLC bars; nothing was merged", "bars")
    if (np.abs(t - np.round(t)) > 1e-6).any() or (np.round(t) % 60 != 0).any():
        raise FeedError(f"{who}bar stamps that are not whole minutes: not a 1-minute time chart "
                        "(Renko / Line Break / a seconds or tick chart?); nothing was merged", "bars")
    d = np.diff(t)
    if (d <= 0).any():
        raise FeedError(f"{who}{int((d <= 0).sum())} bar stamps repeat or run backwards: not a 1-minute time chart "
                        "(RENKO / LINE BREAK bricks share stamps); nothing was merged", "bars")
    odd = 0
    for k in np.where(d != 60)[0]:
        if trading_minutes_between(_stamp(t[k]), _stamp(t[k + 1])) > 0:          # 0: the daily halt, the weekend
            odd += 1
    allowed = max(GAP_ALLOW_MIN, int(GAP_ALLOW_FRAC * len(d)))
    if odd > allowed:
        raise FeedError(f"{who}{odd} of {len(d)} consecutive bars are not 60 s apart inside a session (allowed "
                        f"{allowed}: a minute without a trade, one hole): not a 1-minute time chart "
                        "(RENKO / LINE BREAK?); nothing was merged", "bars")
    if n >= WICKLESS_MIN and (np.abs(h - np.maximum(o, c)) <= GRID_TOL).all() \
            and (np.abs(l - np.minimum(o, c)) <= GRID_TOL).all():
        raise FeedError(f"{who}not one of {n} closed bars has a wick (high = max(open, close), low = min(open, close) on "
                        "every bar): bricks or lines (RENKO / LINE BREAK?), not minutes of trading; nothing was merged",
                        "bars")


def check_chart_type(value, what: str = "") -> None:
    """The style the page reports for a pane (`tv state`'s chartType; the eval's best-effort `style`).
    None = the page did not say: `check_batch` is then the only guard."""
    if value is None:
        return
    try:
        k = int(value)
    except (TypeError, ValueError):
        return
    if k not in CHART_TYPES_OK:
        name = CHART_TYPES[k] if 0 <= k < len(CHART_TYPES) else str(k)
        raise FeedError(f"{what + ': ' if what else ''}the pane shows chart type {name}; the store takes plain "
                        "Candles or Bars only (other styles are not the minute's traded OHLC). Set it to Candles; "
                        "nothing was merged", "bars")


@dataclass
class Poll:
    closed: pd.DataFrame                  # bars that can no longer change
    forming: dict | None                  # {"time", "open", "high", "low", "last"}: the bar still being made.
    #                                       high / low are its RUNNING extremes (None when the feed cannot know
    #                                       them); they are never stored and never reach the detector


def _bars_frame(rows) -> pd.DataFrame:
    if not rows:
        return _norm(None)
    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"][:len(rows[0])])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    return _norm(df.set_index("time"))


def _split(df: pd.DataFrame) -> Poll:
    """The closed-bar rule: a bar is closed once a newer one exists."""
    if not len(df):
        raise FeedError("the chart returned no bars")
    last = df.iloc[-1]
    forming = {"time": df.index[-1].isoformat(), "open": float(last["open"]), "high": float(last["high"]),
               "low": float(last["low"]), "last": float(last["close"])}
    return Poll(df.iloc[:-1], forming)


def _js(n: int) -> str:
    return ("(function(){" + PANES_MARK + "var n=" + str(int(n)) + ";"
            "var all=window.TradingViewApi._chartWidgetCollection.getAll();var out=[];"
            "for(var i=0;i<all.length;i++){try{var ms=all[i].model().mainSeries();var b=ms.bars();"
            "var end=b.lastIndex();var start=Math.max(b.firstIndex(),end-n+1);var rows=[];"
            "for(var k=start;k<=end;k++){var v=b.valueAt(k);if(v)rows.push([v[0],v[1],v[2],v[3],v[4],v[5]||0]);}"
            "var st=null;try{st=ms.style();}catch(e1){}"
            "if(typeof st!=='number'){try{st=ms.properties().childs().style.value();}catch(e2){st=null;}}"
            "out.push({index:i,symbol:ms.symbol(),resolution:ms.interval(),style:(typeof st==='number'?st:null),bars:rows});}"
            "catch(e){out.push({index:i,error:String(e&&e.message||e)});}}return {panes:out};})()")


class LiveTvFeed:
    """Section 9.6. See the module docstring for the two methods. `cycle` never
    returns a partial truth: per instrument it gives a `Poll` or the `FeedError`
    that instrument's read ended in; a replay-mode chart fails the whole cycle."""

    name, live = "tradingview", True

    def __init__(self, cli: TvCli | None = None, symbols: dict | None = None, method: str = "eval",
                 ticks: dict | None = None):
        if method not in ("eval", "focus"):
            raise ValueError("LiveTvFeed: method is 'eval' or 'focus'")
        self.cli = cli or TvCli()
        self.symbols = dict(symbols or SYMBOLS)
        self.ticks = {**TICK_SIZE, **(ticks or {})}     # a test on hand-built prices names its own grid
        self.method = method
        self.panes: dict[str, int] = {}

    def pane_of(self, instrument: str) -> int | None:
        return self.panes.get(instrument)

    @staticmethod
    def _is(symbol, want: str) -> bool:
        return str(symbol or "").split(":")[-1].upper() == want.upper()

    def _map(self, panes: list[dict], instruments) -> None:
        found = {}
        for inst in instruments:
            want = self.symbols[inst]
            hit = [p for p in panes if self._is(p.get("symbol"), want)]
            if not hit:
                raise FeedError(f"no pane shows {want}: the layout is "
                                f"{[p.get('symbol') for p in panes]} (section 9.6 wants NQ1! and ES1!)", "layout")
            if str(hit[0].get("resolution")) != "1":
                raise FeedError(f"the {want} pane is at resolution {hit[0].get('resolution')!r}, not 1 minute", "layout")
            found[inst] = int(hit[0]["index"])
        self.panes = found

    def check_replay(self) -> None:
        st = self.cli.call("replay", "status")
        if st.get("is_replay_started"):
            raise FeedError("the chart is in REPLAY mode: those bars are historical and would be journaled as "
                            "the experiment's. Run `tv replay stop`.", "replay")

    def cycle(self, now: float, want: dict[str, int]) -> dict:
        with self.cli.lock:
            self.check_replay()
            if self.method == "eval":
                return self._cycle_eval(want)
            return self._cycle_focus(want)

    def _cycle_eval(self, want: dict[str, int]) -> dict:
        n = min(FETCH_MAX_EVAL, max([FETCH_MIN, *want.values()]))
        res = self.cli.call("ui", "eval", _js(n)).get("result") or {}
        panes = res.get("panes") if isinstance(res, dict) else None
        if not isinstance(panes, list):
            raise FeedError("tv ui eval: the page returned no pane list")
        self._map(panes, want)
        out = {}
        for inst in want:
            p = next(p for p in panes if int(p.get("index", -1)) == self.panes[inst])
            try:
                if p.get("error"):
                    raise FeedError(f"{inst} pane: {p['error']}")
                check_chart_type(p.get("style"), inst)
                check_batch(p.get("bars") or [], self.ticks.get(inst, 0.25), inst)
                out[inst] = _split(_bars_frame(p.get("bars") or []))
            except FeedError as exc:
                out[inst] = exc
        return out

    def _cycle_focus(self, want: dict[str, int]) -> dict:
        self._map(self.cli.call("pane", "list").get("panes") or [], want)
        out = {}
        for inst, n in want.items():
            try:
                self.cli.call("pane", "focus", str(self.panes[inst]))
                st = self.cli.call("state")
                if not self._is(st.get("symbol"), self.symbols[inst]) or str(st.get("resolution")) != "1":
                    raise FeedError(f"after focusing pane {self.panes[inst]} the active chart is "
                                    f"{st.get('symbol')} @ {st.get('resolution')}, not {self.symbols[inst]} @ 1", "layout")
                check_chart_type(st.get("chartType", st.get("chart_type")), inst)
                bars = self.cli.call("ohlcv", "-n", str(min(FETCH_MAX_FOCUS, max(FETCH_MIN, int(n))))).get("bars")
                check_batch(bars or [], self.ticks.get(inst, 0.25), inst)
                out[inst] = _split(_bars_frame(bars or []))
            except FeedError as exc:
                out[inst] = exc
        return out


class ReplayFeed:
    """Stored 1-minute frames under a simulated clock. At `now` a bar stamped s is
    closed iff s + 60 <= now; of the bar that is forming only the OPEN is exposed
    (as `open` and as `last`): its high, low and close are the future."""

    name, live = "replay", False

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = {k: _norm(v) for k, v in frames.items()}
        self._epochs = {k: (v.index.values.astype("datetime64[s]").astype(np.int64)) for k, v in self.frames.items()}
        self._pos = {k: 0 for k in self.frames}

    def skip_to(self, instrument: str, row: int) -> pd.DataFrame:
        """Hand over rows [pos, row) as history; they will not be polled again."""
        out = self.frames[instrument].iloc[self._pos[instrument]:row]
        self._pos[instrument] = max(self._pos[instrument], int(row))
        return out

    def timeline(self) -> np.ndarray:
        """Every instant something becomes knowable: each remaining bar's open and its close."""
        ts = [self._epochs[k][self._pos[k]:] for k in self.frames]
        allt = np.concatenate([np.concatenate([t, t + 60]) for t in ts]) if ts else np.array([], dtype=np.int64)
        return np.unique(allt)

    def cycle(self, now: float, want: dict[str, int] | None = None) -> dict:
        out = {}
        for inst, ep in self._epochs.items():
            i = self._pos[inst]
            j = int(np.searchsorted(ep, now - 60.0, side="right"))
            closed = self.frames[inst].iloc[i:max(i, j)]
            self._pos[inst] = max(i, j)
            forming = None
            k = self._pos[inst]
            if k < len(ep) and ep[k] <= now < ep[k] + 60:
                o = float(self.frames[inst]["open"].iloc[k])
                forming = {"time": self.frames[inst].index[k].isoformat(), "open": o, "high": None, "low": None,
                           "last": o}
            out[inst] = Poll(closed, forming)
        return out


# ────────────────────────────── pushes that cannot delay the loop ──────────────────────────────

class Pusher:
    """Notices -> `quantlab.alerts.notify`, on a daemon thread when `background`.
    `notify=False` records and sends nothing at all (replay: no network, no
    alerts.log, no stderr line carrying an R)."""

    def __init__(self, base_dir, notify=None, background: bool = True):
        self.base_dir = base_dir
        self.notify = notify
        self.background = bool(background) and notify is not False
        self.by_kind: dict[str, int] = {}
        self.sent: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def push(self, notices: list[dict]) -> None:
        for n in notices or []:
            self.by_kind[n.get("kind", "?")] = self.by_kind.get(n.get("kind", "?"), 0) + 1
        if not notices or self.notify is False:
            return
        if not self.background:
            self.sent += send_notices(self.base_dir, notices, self.notify)
            return
        self._queue.put(list(notices))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="tjr-human-push", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.sent += send_notices(self.base_dir, batch, self.notify)
            except Exception:
                pass

    def close(self, timeout: float = 25.0) -> None:
        end = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < end:
            time.sleep(0.05)
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(max(0.0, end - time.monotonic()))


# ────────────────────────────── one `detect` per directory ──────────────────────────────

class RunLock:
    """An exclusive OS lock on `<dir>/runner.lock`, held for the life of the live
    loop. Two `detect` processes on one directory would journal every signal and
    fill twice, double the alerts and split each other's getUpdates. The OS drops
    the lock when the process dies, so a crash never leaves it stuck. `arm`,
    `status` and the reports do not take it."""

    def __init__(self, base_dir):
        self.path = Path(base_dir) / LOCK_FILE
        self._fh = None

    def acquire(self) -> None:
        if self._fh is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise Refused(f"another `detect` is running on this directory ({self.path.name} is held): two runners "
                          "would journal and alert everything twice and steal each other's commands") from None
        self._fh = fh

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fh.close()
        except OSError:
            pass


# ────────────────────────────── ARMED (section 9.7) ──────────────────────────────

def prereg_sha256(path=DESIGN) -> str:
    """SHA-256 of section 5 of DESIGN-tjr-human.md: from the line that starts
    `## 5.` up to (not including) the next `## ` heading, line endings normalised
    to LF, trailing blank lines dropped, UTF-8."""
    lines = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    try:
        a = next(i for i, ln in enumerate(lines) if ln.startswith("## 5."))
    except StopIteration:
        raise Refused(f"{Path(path).name} has no '## 5.' section to hash") from None
    b = next((i for i in range(a + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return hashlib.sha256(("\n".join(lines[a:b]).rstrip() + "\n").encode("utf-8")).hexdigest()


CODE_PATHS = ("quantlab/tjr_human", "tjr_human.py", "DESIGN-tjr-human.md",
              "quantlab/strategies/predictive/primitives.py", "quantlab/strategies/predictive/tjr_intraday.py",
              "quantlab/alerts.py", "quantlab/validate.py")


def git_state(repo=REPO) -> dict:
    """{"commit": str|None, "dirty": [porcelain lines for the code this experiment runs]}. Read-only git."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20,
                              stdin=subprocess.DEVNULL)
    try:
        head = run("rev-parse", "HEAD")
        commit = head.stdout.strip() if head.returncode == 0 and len(head.stdout.strip()) >= 7 else None
        st = run("status", "--porcelain", "--", *CODE_PATHS)
        dirty = [ln for ln in st.stdout.splitlines() if ln.strip()] if st.returncode == 0 else ["git status failed"]
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": []}
    return {"commit": commit, "dirty": dirty}


def store_first_bars(base_dir, instruments=INSTRUMENTS) -> dict:
    """{instrument: stamp of the first bar of its store csv, or None}. H1 / H4 swing
    levels are the nearest admissible swing over ALL the history the store holds, so
    where the store starts is part of the level definition; ARMED records it."""
    out = {}
    for inst in instruments:
        path = Path(base_dir) / STORE_DIR / f"{inst}_1min.csv"
        first = None
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.readline()
                row = fh.readline().split(",")
            if len(row) == 6:
                first = pd.Timestamp(row[0]).isoformat()
        except (OSError, ValueError):
            first = None
        out[inst] = first
    return out


def arm(base_dir=J.DEFAULT_DIR, reason: str = "", now: float | None = None, git=git_state,
        design=DESIGN, allow_dirty: bool = False) -> dict:
    """Write ARMED: the commit of the code, the SHA-256 of section 5 and the
    TARGET_RULE in force. From the next signal the bot takes entries and counts.
    Refuses: no reason; no commit; uncommitted changes in the code it names
    (unless `allow_dirty`, which is journaled); once a trade has been ENTERED —
    a provisional fill in its entry minute included, armed at this moment or not
    (`journal.entered_count`); a directory that holds a replay's journal."""
    reason = (reason or "").strip()
    if not reason:
        raise Refused("arm needs --reason: trade one starts the pre-registered sample, and the record says why now")
    journal = J.Journal(base_dir)
    records = journal.records()
    current = J.armed_record(records)
    who = J.origin(records)
    if who["replay"]:
        raise Refused(f"{Path(base_dir)} holds a replay's journal ({who['replay']} record(s)): the pre-registered "
                      "sample is never started on top of replayed trades — use another directory")
    entered = J.entered_count(records)
    if entered > 0:                                   # armed or not: a provisional fill in its entry minute counts
        raise Refused(f"{entered} trade(s) have been entered"
                      + (f" and the run is armed (seq {current['seq']})" if current is not None else "")
                      + ": nothing changes after trade one (section 5) — ARMED cannot be rewritten")
    g = git()
    if not g.get("commit"):
        raise Refused("cannot read the git commit hash: ARMED must name the code it arms")
    if g.get("dirty") and not allow_dirty:
        raise Refused("uncommitted changes in the code ARMED would name — commit first, or pass --allow-dirty "
                      "(it is journaled): " + "; ".join(g["dirty"][:8]))
    body = {"armed": True, "reason": reason, "commit": g["commit"], "dirty": list(g.get("dirty") or []),
            "prereg_sha256": prereg_sha256(design), "prereg_file": Path(design).name,
            "target_rule": D.TARGET_RULE, "n_trials": R.N_TRIALS, "sample": R.SAMPLE, "pass_bar": R.PASS_BAR,
            "late_fill_s": LATE_FILL_S, "store_first_bar": store_first_bars(base_dir),
            "supersedes": current["seq"] if current is not None else None}
    return journal.write("armed", body, now=now)


def disarm(base_dir=J.DEFAULT_DIR, reason: str = "", now: float | None = None) -> dict:
    """Back to observe mode — refused once a trade has been entered, the provisional
    fill of its entry minute included (section 9.7)."""
    journal = J.Journal(base_dir)
    records = journal.records()
    entered = J.entered_count(records)
    if entered > 0:                                   # the entry minute of trade one is already trade one
        raise Refused(f"refused: {entered} trade(s) have been entered. Nothing changes after trade one (section 5); "
                      "stopping the process is always possible and is journaled as a gap")
    if J.armed_record(records) is None:
        raise Refused("not armed: nothing to disarm")
    if not (reason or "").strip():
        raise Refused("disarm needs --reason")
    return journal.write("armed", {"armed": False, "reason": reason.strip()}, now=now)


# ────────────────────────────── the scripted human (replay) ──────────────────────────────

def load_script(source) -> list[tuple[float, str, object, str | None]]:
    """A json list of timed items -> [(epoch, "cmd"|"price", value, instrument)], sorted.

        {"at": "2026-07-15 09:45:00", "text": "stop session gut"}
        {"at": "2026-07-15T13:55:10Z", "price": 99.4, "instrument": "NQ"}

    `at` is ET wall-clock time (what the human reads off the chart) unless it ends
    in Z, which is UTC. `text` is a Telegram message exactly as typed."""
    if source is None or source == "none":
        return []
    items = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else list(source)
    out = []
    for k, it in enumerate(items):
        if not isinstance(it, dict) or "at" not in it or ("text" in it) == ("price" in it):
            raise ValueError(f"script item {k}: needs 'at' and exactly one of 'text' / 'price': {it!r}")
        at = str(it["at"]).strip()
        ts = pd.Timestamp(at[:-1], tz="UTC") if at.upper().endswith("Z") else pd.Timestamp(at, tz=D.M.ET)
        if "text" in it:
            out.append((ts.timestamp(), "cmd", str(it["text"]), it.get("instrument")))
        else:
            if not it.get("instrument"):
                raise ValueError(f"script item {k}: a price needs an instrument")
            out.append((ts.timestamp(), "price", float(it["price"]), str(it["instrument"])))
    return sorted(out, key=lambda x: x[0])


# ────────────────────────────── the runner ──────────────────────────────

class Runner:
    def __init__(self, base_dir, feed, instruments=INSTRUMENTS, clock=time.time, sleep=time.sleep,
                 journal: J.Journal | None = None, notify=None, background: bool = True,
                 chart: Chart | None = None, inbound="auto", stores: dict | None = None, seeds: dict | None = None,
                 allow_cold: bool = False, accept_holes: bool = False, preview: bool = True, log=None,
                 keep_awake: bool = True, accept_short_sessions=()):
        self.base = Path(base_dir)
        #: session days a PERSON says ended early (a holiday close): a trade a crash left open in one of
        #: them ends at that session's last close instead of refusing the start for want of a 15:55 bar
        self.accept_short_sessions = {str(d)[:10] for d in (accept_short_sessions or ())}
        self.keep_awake = bool(keep_awake)
        self.feed = feed
        self.live = bool(getattr(feed, "live", False))
        self.instruments = tuple(instruments)
        self.clock, self.sleep = clock, sleep
        self.journal = journal or J.Journal(self.base, clock=clock, fsync=self.live)
        records = self.journal.records()
        self.armed_rec = J.armed_record(records)
        self.mgr = TradeManager(self.journal, armed=self.armed_rec is not None, clock=clock)
        self.pusher = Pusher(self.base, notify=notify, background=background and self.live)
        self.chart = chart if chart is not None else Chart(executor=None)
        self.inbound = (TelegramInbound(self.journal, clock=clock) if self.live else None) if inbound == "auto" else inbound
        self.allow_cold, self.accept_holes, self.preview = bool(allow_cold), bool(accept_holes), bool(preview)
        self.log = log or (lambda line: print(line, file=sys.stderr))
        if stores is None:
            seeds = SEEDS if seeds is None else seeds
            stores = {}
            for i in self.instruments:
                path = self.base / STORE_DIR / f"{i}_1min.csv"
                try:
                    stores[i] = BarStore(path, seed=seeds.get(i))
                except Refused:
                    raise
                except Exception as exc:                # a csv pandas cannot parse is a refusal, not a traceback
                    raise Refused(f"{i}: the 1-minute store {path} cannot be read ({type(exc).__name__}: "
                                  f"{' '.join(str(exc).split())[:200]}). Nothing was changed. Move the file aside "
                                  "(the store is then re-seeded from data/ and needs a backfill) or repair it") from None
        self.stores: dict[str, BarStore] = stores
        self._todo: dict[str, list] = {i: [] for i in self.instruments}     # undelivered work, oldest first
        self._lock = RunLock(self.base)
        self._inbound_up = False
        self._repairs_told: dict[str, int] = {i: 0 for i in self.instruments}
        self.det = {i: Detector(i) for i in self.instruments}
        self.start_day: str | None = None
        self._fed: dict[str, pd.Timestamp | None] = {i: None for i in self.instruments}
        self._route: dict[str, bool] = {}
        self._pending: dict[str, str] = {}              # instrument -> stamp of the entry minute awaited
        self._stop_drawn: dict[str, tuple] = {}
        self._fill_drawn: dict[str, tuple] = {}
        self._holes_checked: set[str] = set()
        self._clock_checked = False
        self._accepted_holes: set[tuple] = set()        # (instrument, after, before) a person accepted at the start
        self._holes_told: set[tuple] = set()
        self._clock_why: dict[str, str | None] = {"bar": None, "cmd": None}
        self._cmd_late = 0
        self._clock_skew: float | None = None
        self._awake = False
        self._failing: dict[str, dict] = {}
        self._run_sizes: dict[str, int] = {}
        self._started = False                           # start_live has returned: a refusal from here on is pushed
        self._started_at: float | None = None
        self._abandoned: dict[str, list[dict]] = {i: [] for i in self.instruments}   # past-session fills without an exit
        self._abandoned_pending: set[str] = set()
        self._live_told: set[tuple] = set()             # ("alive" | "terminal", session day) already said
        self._revised: dict[str, dict[str, set]] = {i: {} for i in self.instruments}  # session day -> revised stamps
        self.facts = {"events": {i: {} for i in self.instruments}, "routed_days": {i: 0 for i in self.instruments},
                      "routed": {i: {} for i in self.instruments},
                      "cold_days": {i: 0 for i in self.instruments}, "bars": {i: 0 for i in self.instruments},
                      "previews": 0, "cycles": 0, "feed_failures": 0, "commands": 0, "prices": 0}

    # ---- starting -------------------------------------------------------------------------------------------

    def _refuse(self, why: str, now: float, quiet: bool = False, **body) -> None:
        try:
            if not quiet:                               # quiet: the journal is not this run's to write into
                self.journal.write("run", {"what": "refused", "why": why, **body}, now=now)
        except Exception:
            pass
        exc = Refused(why)
        if self.live and self._started:                 # after startup nobody is at the terminal: the box tells
            self._push_refusal(why)
            exc.pushed = True
        self._lock.release()
        raise exc

    def _push_refusal(self, why: str) -> None:
        try:
            self.pusher.push([{"kind": "status", "fields": {"refused": True},
                               "text": f"tjr_human REFUSED and is STOPPING — {mask_token(' '.join(str(why).split()))[:900]}"}])
        except Exception:
            pass

    def _tell_repairs(self, now: float) -> None:
        """A torn last row cut from a store csv is journaled and logged, once."""
        for inst in self.instruments:
            fixes = getattr(self.stores[inst], "repairs", None) or []
            for fix in fixes[self._repairs_told.get(inst, 0):]:
                self.log(f"tjr_human store: {inst}: a torn last row was cut from {fix.get('file')} "
                         f"({fix.get('lines_dropped')} line(s)); the feed brings the minute back")
                try:
                    self.journal.write("feed", {"what": "store_repaired", **fix}, instrument=inst, now=now)
                except Exception:
                    pass
            self._repairs_told[inst] = len(fixes)

    def start_live(self) -> dict:
        now = float(self.clock())
        _, today = et_minute_and_day(now)
        if self.live:
            try:
                self._lock.acquire()                    # before anything is journaled: a second `detect` writes nothing
            except Refused as exc:
                self._refuse(str(exc), now)
        bad_lines = 0
        try:
            bad_lines = int(J.integrity(self.base).get("bad_lines") or 0)
        except Exception:
            pass
        who = J.origin(self.journal.records())
        if self.live and who["replay"]:
            self._refuse(f"{self.base} holds a replay's journal ({who['replay']} record(s)): the live run never shares "
                         "a journal with replayed trades (section 9.8) — use another directory", now, quiet=True)
        gap = self.journal.mark_start(now=now)
        records = self.journal.records()
        state = J.rebuild(records)
        if self.armed_rec is not None and self.armed_rec.get("target_rule") != D.TARGET_RULE:
            self._refuse(f"ARMED (seq {self.armed_rec['seq']}) names TARGET_RULE {self.armed_rec.get('target_rule')!r}, "
                         f"the code says {D.TARGET_RULE!r}: it cannot change after ARMED — restore it, or disarm and "
                         "re-arm while no trade has filled", now)
        if self.armed_rec is not None and "late_fill_s" in self.armed_rec \
                and self.armed_rec.get("late_fill_s") != LATE_FILL_S:
            self._refuse(f"ARMED (seq {self.armed_rec['seq']}) recorded LATE_FILL_S = {self.armed_rec.get('late_fill_s')!r}, "
                         f"the code says {LATE_FILL_S!r}: which fills count cannot change after ARMED", now)
        self._tell_repairs(now)
        depth_warn = []
        recorded = (self.armed_rec or {}).get("store_first_bar") or {}
        for inst in self.instruments:                   # where the store starts is part of the H1 / H4 level definition
            frame = self.stores[inst].frame
            first = frame.index[0].isoformat() if len(frame) else None
            was = recorded.get(inst)
            if self.armed_rec is not None and was != first:
                if was is not None and state["filled"] > 0:
                    self._refuse(f"{inst}: the store starts at {first}, ARMED (seq {self.armed_rec['seq']}) recorded "
                                 f"{was}, and {state['filled']} trade(s) have filled. H1 / H4 swing levels are read over "
                                 "all the history in the store, so another start is another level definition "
                                 "mid-sample (section 5). Restore the store the sample was started with", now)
                depth_warn.append(f"{inst}: the store starts at {first}; ARMED recorded {was}")
        warm = {}
        for inst in self.instruments:
            warm[inst] = check_warm(self.stores[inst].frame, for_day=today)
            if not warm[inst]["ok"] and not self.allow_cold:
                self._refuse(f"{inst}: the store does not cover the history the levels need — {warm[inst]['reason']} "
                             f"(store {warm[inst]['first_bar']} .. {warm[inst]['last_bar']})", now, warm=warm[inst])
        info = self.mgr.resume(records)
        # NEVER BY AGE. Every setup that still owes something — an exit, its benchmarks — has its session
        # re-fed from the store, however long ago it was: a trade a crash left open is closed, not dropped.
        owing = [s for s in state["setups"].values()
                 if s.get("armed") and s.get("day") and
                 (s["status"] == "live" or ((s["fill"] or s["would_fill"]) and s["benchmarks"] is None))]
        unfinished = sorted(s["day"] for s in owing)
        self.start_day = today
        self.mgr.restart_day = today                    # an exit of an earlier session found now is `restart_replay`
        self._started_at = now
        for s in owing:
            key = J.file_key(s.get("instrument"))
            if (s["day"] < today and s["skip"] is None and s["exit"] is None and key in self._abandoned
                    and (s["fill"] is not None or (s["preview"] is not None and s["invalidated"] is None))):
                entry = ((s["fill"] or s["preview"] or {}).get("ev") or {}).get("bar_time")
                self._abandoned[key].append({"setup_id": s["setup_id"], "day": s["day"], "entry_bar": entry})
                self._abandoned_pending.add(key)
        for r in records:
            if r.get("kind") == "run" and r.get("what") == "liveness" and r.get("day"):
                self._live_told.add((str(r.get("part")), str(r["day"])))
        first_day = min([today, *unfinished])
        for inst in self.instruments:                   # history: fed, never routed
            frame = self.stores[inst].frame
            if len(frame):
                _, day = et_clock(frame.index)
                hist = frame.loc[day < np.datetime64(pd.Timestamp(first_day), "ns")]
                if len(hist):
                    self.det[inst].feed(hist)
                    self._fed[inst] = hist.index[-1]
        current = {"commit": None, "dirty": []}
        try:
            current = git_state()
        except Exception:
            pass
        sha = None
        try:
            sha = prereg_sha256()
        except Exception:
            pass
        warn = list(depth_warn)
        for inst in self.instruments:
            n = int(warm[inst].get("sessions") or 0)
            if n < FUNNEL_SESSIONS:
                warn.append(f"{inst}: the store holds {n} completed sessions; H1 / H4 swing levels are the nearest "
                            f"admissible swing over ALL the history held, and section 2.6's count was made with "
                            f"{FUNNEL_SESSIONS} — with less history the levels, and so the sweeps, can differ "
                            "(a decision the record still owes: DESIGN-tjr-human.md section 2.4 / 9.6)")
        if bad_lines:
            warn.append(f"the journal has {bad_lines} torn line(s) — a write was cut short; they are skipped, never repaired")
        for inst in self.instruments:
            for a in self._abandoned[inst]:
                warn.append(f"{inst}: the trade of {a['day']} ({a['setup_id']}) has no exit — the process was down. It is "
                            "closed at the first poll by replaying the stored bars of that session (restart_replay)")
            for d in sorted({s["day"] for s in owing if J.file_key(s.get("instrument")) == inst and s["day"] < today}):
                if not len(self.stores[inst].session_rows(d)):
                    warn.append(f"{inst}: the setup of {d} still owes its benchmarks and the store holds no bar of "
                                "that session — backfill it (detect --backfill)")
        problem = telegram_problem()
        if problem is not None and os.environ.get("TELEGRAM_BOT_TOKEN"):
            warn.append(f"Telegram: {problem}")
        if self.armed_rec is not None:
            if sha and self.armed_rec.get("prereg_sha256") != sha:
                warn.append("section 5 of DESIGN-tjr-human.md no longer hashes to what ARMED recorded")
            if current.get("commit") and self.armed_rec.get("commit") != current["commit"]:
                warn.append("the code is at another commit than ARMED recorded")
        inbound_up = bool(self.inbound.start()) if self.inbound is not None else False
        self._inbound_up = inbound_up
        rec = self.journal.write("run", {
            "what": "start", "mode": "armed" if self.mgr.armed else "observe", "feed": self.feed.name,
            "method": getattr(self.feed, "method", None), "today": today, "re_fed_from": first_day,
            "warm": warm, "restored": info, "gap_seq": gap["seq"] if gap else None,
            "telegram_configured": telegram_configured(), "inbound_thread": inbound_up,
            "journal_bad_lines": bad_lines, "late_fill_s": LATE_FILL_S,
            "store_first_bar": {i: (self.stores[i].frame.index[0].isoformat() if len(self.stores[i].frame) else None)
                                for i in self.instruments},
            "chart": self.chart.enabled, "commit": current.get("commit"), "dirty": bool(current.get("dirty")),
            "prereg_sha256": sha, "target_rule": D.TARGET_RULE, "warnings": warn,
            "accept_holes": self.accept_holes, "allow_cold": self.allow_cold, "keep_awake": self._awake,
            "abandoned_trades": {i: list(v) for i, v in self._abandoned.items() if v},
            "accept_short_sessions": sorted(self.accept_short_sessions)}, now=now)
        for w in warn:
            self.log(f"tjr_human WARNING: {w}")
        self._started = True
        return rec

    # ---- a trade a crash left open ------------------------------------------------------------------------------

    def _abandoned_sessions_held(self, inst: str, now: float) -> None:
        """After the first merge (the chart gives back the minutes the process missed): does the store
        hold the session of every abandoned trade through 15:55 ET — EVERY trading minute of it from the
        entry minute on, not only its last stamp? If not, REFUSE — naming the session and the minutes
        missing — rather than close the trade on half a session, on a session with a hole in it (the
        stop or the target may have been hit in the minutes not held, and a wrong exit would enter the
        pre-registered sample), or leave it open for ever. `--accept-short-session` says a session ENDED
        early; it never unlocks minutes missing in front of bars the store does hold."""
        for a in self._abandoned[inst]:
            done, last, later = self.stores[inst].session_complete(a["day"])
            gaps = self.stores[inst].session_gaps(a["day"], a.get("entry_bar")) if last is not None else []
            if gaps:
                said = "; ".join((f"{g['trading_minutes']} trading minute(s) between {g['after']}Z and {g['before']}Z"
                                  if g.get("after") else
                                  f"the minutes from the entry minute {g.get('from')}Z"
                                  + (f" to {g['before']}Z" if g.get("before") else " on")) for g in gaps[:4])
                self._refuse(f"{inst}: the trade filled on {a['day']} ({a['setup_id']}) was left open when the process "
                             f"stopped, and the store does not hold that session continuously from the entry minute "
                             f"through 15:55 ET — missing: {said}"
                             + (f" (and {len(gaps) - 4} more)" if len(gaps) > 4 else "")
                             + ". The stop or the target may have been hit in minutes the store does not hold, so the "
                             "trade is NOT closed on these bars. Backfill it — detect --backfill "
                             f"{inst}=<1-minute csv export covering ALL of {a['day']} from the entry minute to 15:55 ET; "
                             "an export that starts mid-session leaves the hole> — and start again. "
                             "--accept-short-session does not unlock a hole", now,
                             session=a["day"], setup_id=a["setup_id"], store_last_of_session=last,
                             entry_bar=a.get("entry_bar"), missing=gaps[:20], missing_ranges=len(gaps))
            if done or (later and last is not None and a["day"] in self.accept_short_sessions):
                continue                                # through 15:55, or an early close a PERSON named for this day
            self._refuse(f"{inst}: the trade filled on {a['day']} ({a['setup_id']}) was left open when the process "
                         f"stopped, and the store does not hold that session through 15:55 ET (it "
                         + (f"ends at {last}Z" if last else "holds no bar of it")
                         + f"). Backfill it — detect --backfill {inst}=<1-minute csv export covering {a['day']}> — "
                         "and start again: the trade is then closed by replaying that session (the stop in force, "
                         "the target, the 15:55 flat)"
                         + (f". If that session really closed early (a holiday), start with --accept-short-session "
                            f"{a['day']}: the trade then ends at that session's last close" if later and last else ""), now,
                         session=a["day"], setup_id=a["setup_id"], store_last_of_session=last)

    def _abandoned_closed(self, inst: str, now: float) -> None:
        """After the re-fed sessions went through the manager: every abandoned trade has an exit (or was
        voided by its closed entry bar). One that has not is never left silently open."""
        if inst not in self._abandoned_pending or self._todo[inst]:
            return
        setups = J.rebuild(self.journal.records())["setups"]
        for a in self._abandoned[inst]:
            st = setups.get(a["setup_id"]) or {}
            if st.get("exit") is None and st.get("skip") is None and not (st.get("invalidated") or {}).get("void") \
                    and not (st.get("fill") is None and st.get("preview") is None):
                self._refuse(f"{inst}: the trade of {a['day']} ({a['setup_id']}) was left open when the process stopped "
                             "and re-feeding its session from the store did NOT close it: the store no longer "
                             "reproduces the journaled setup (was it replaced or re-seeded?). Restore the store the "
                             "trade was taken on, or backfill that session, and start again", now,
                             session=a["day"], setup_id=a["setup_id"])
        self._abandoned_pending.discard(inst)

    # ---- one closed bar at a time ---------------------------------------------------------------------------

    def _should_route(self, inst: str, ev: dict) -> bool:
        if ev["kind"] == "levels":
            ok = bool(ev.get("warm")) or self.allow_cold
            if ok and self.live and (self.start_day is None or ev["day"] >= self.start_day):
                ok = not self._holed_day(inst, ev)
            self._route[inst] = ok
            self.facts["routed_days" if ok else "cold_days"][inst] += 1
        if not self._route.get(inst, False):
            return False
        if self.start_day is not None and ev["day"] < self.start_day:
            return self.mgr.seen(ev)                  # an earlier day is re-fed to restore, never to enter
        return True

    def _holed_day(self, inst: str, ev: dict) -> bool:
        """The levels of this day froze on a store with a hole (ten or more trading
        minutes missing) in its two newest sessions that nobody accepted: PDH / PDL /
        ASIA / LON / the swings may be wrong, so the day is NOT routed — no signal, no
        entry, nothing counted. Journaled and pushed once. A person accepts a hole
        (an early close) by restarting with --accept-holes, or closes it with
        --backfill; the append-only store never fills it by itself."""
        try:
            frame = self.stores[inst].frame
            holes = [h for h in recent_holes(frame.loc[:pd.Timestamp(ev["bar_time"])])
                     if (inst, h["after"], h["before"]) not in self._accepted_holes]
        except Exception as exc:                        # never let the check itself stop a bar
            self.log(f"tjr_human: {inst}: the hole check failed ({type(exc).__name__}) — the day is routed")
            return False
        if not holes:
            return False
        now = float(self.clock())
        self._journal_quietly("feed", {"what": "day_not_routed", "scope": inst, "day": ev["day"], "holes": holes[:5],
                                       "why": "levels frozen on a store with an unaccepted hole"}, now, instrument=inst)
        self.log(f"tjr_human: {inst} {ev['day']}: NOT ROUTED — the levels froze on a store with a hole {holes[:3]}")
        self.pusher.push([{"kind": "status", "fields": {"scope": inst, "day": ev["day"], "holes": holes[:5]},
                           "text": f"tjr_human {inst} {ev['day']}: this day is NOT traded — the 1-minute store has a hole "
                                   f"({holes[0]['trading_minutes']} trading minutes missing after {holes[0]['after']}), "
                                   "so today's levels may be wrong. Restart detect with --backfill "
                                   f"{inst}=<csv export>, or with --accept-holes if it was an early close."}])
        return True

    def _process(self, inst: str, rows: pd.DataFrame, now: float, out: list | None = None) -> list[dict]:
        """Closed bars -> detector -> trade manager, AT LEAST ONCE. The detector never
        takes a bar twice, so what a bar produced (its routed events, then the bar
        itself) is queued the moment it is fed and an item leaves the queue only
        when the manager has taken it. An exception (the journal file held by
        another program) leaves the rest queued; the next cycle delivers it before
        it feeds anything new. `out` collects notices in place, so what was
        delivered before an exception is still pushed."""
        notices: list[dict] = out if out is not None else []
        self._drain(inst, now, notices)
        det = self.det[inst]
        for k in range(len(rows)):
            one = rows.iloc[k:k + 1]
            stamp = one.index[0]
            r = one.iloc[0]
            events = det.feed(one)
            self._fed[inst] = stamp
            self.facts["bars"][inst] += 1
            todo = self._todo[inst]
            for ev in events:
                kinds = self.facts["events"][inst]
                kinds[ev["kind"]] = kinds.get(ev["kind"], 0) + 1
                route = self._should_route(inst, ev)
                if self._route.get(inst, False):
                    self._count_routed(inst, ev)
                if route:
                    todo.append(("event", stamp, ev))
            todo.append(("bar", stamp, (float(r.open), float(r.high), float(r.low), float(r.close))))
            self._drain(inst, now, notices)
        self._sync_stop(inst, now)
        return notices

    def _count_routed(self, inst: str, ev: dict) -> None:
        """Counts over ROUTED (warm) sessions, by branch. Counts of events only: nothing here is a price or a result."""
        book = self.facts["routed"][inst]

        def bump(name: str, key) -> None:
            d = book.setdefault(name, {})
            d[str(key)] = d.get(str(key), 0) + 1

        k = ev["kind"]
        bump("detector_events", k)
        zone = (ev.get("zone") or {}).get("kind")
        if k == "sweep":
            bump("sweeps_by_class_and_direction", f"{(ev.get('level') or {}).get('class')}/{ev.get('direction')}")
        elif k == "confirmation":
            bump("confirmations_by_types", "+".join(ev.get("types") or []) or "-")
        elif k == "signal":
            bump("signals_by_zone", zone)
        elif k == "entry_trigger":
            bump("entry_triggers_by_zone", zone)
        elif k == "invalidated":
            bump("invalidations_by_why_and_stage", f"{ev.get('reason')}@{ev.get('stage')}")
        elif k == "expired":
            bump("expiries_by_why", ev.get("reason"))

    def _drain(self, inst: str, now: float, notices: list) -> None:
        todo = self._todo[inst]
        while todo:
            what, stamp, payload = todo[0]
            if what == "event":
                got = self._event(inst, payload, now)
            else:
                got = self.mgr.on_bar(inst, stamp.isoformat(), *payload, now)
            todo.pop(0)                                 # taken: only now does it leave the queue
            lag = now - (_epoch(stamp) + 60.0)
            if lag > LATE_S:
                got = [{**n, "text": f"[LATE {int(lag)} s: caught up from the store] {n['text']}"} for n in got]
            notices += got

    def _event(self, inst: str, ev: dict, now: float) -> list[dict]:
        got = self.mgr.on_event(ev, now)
        if ev["kind"] in ("signal", "zone"):
            self.chart.on_event(ev)
        elif ev["kind"] == "entry_trigger":
            self._pending[inst] = ev["entry_at"]
        elif ev["kind"] in ("fill", "invalidated", "expired"):
            self._pending.pop(inst, None)
            if ev["kind"] == "fill":
                tr = self.mgr.trade(inst)
                mark = (ev["setup_id"], float(ev["entry"]))
                drawn = self._fill_drawn.get(inst) == mark      # the preview drew the same fill already
                if tr is not None and tr.fill is not None and tr.phase in ("live", "closed") \
                        and tr.setup_id == ev["setup_id"] and not drawn:
                    self._fill_drawn[inst] = mark
                    self.chart.on_event(ev)
        return got

    def _forming(self, inst: str, forming: dict | None, now: float) -> list[dict]:
        if not forming:
            return []
        out: list[dict] = []
        want = self._pending.get(inst)
        if self.preview and want is not None and forming["time"] == want:
            ev = self.det[inst].preview_fill(float(forming["open"]))
            out += self.mgr.on_preview(ev, now)
            self._pending.pop(inst, None)               # after it was taken: a failed journal write is tried again
            self.facts["previews"] += 1
            tr = self.mgr.trade(inst)
            if ev and ev.get("kind") == "fill" and tr is not None and tr.phase == "live":
                self._fill_drawn[inst] = (ev["setup_id"], float(ev["entry"]))
                self.chart.on_event(ev)
        # the running range tests a MOVED stop and comes before a pending EXIT NOW's fill; it never
        # touches the default or the chosen stop (closed bars decide those) and never reaches the detector
        out += self.mgr.on_forming(inst, forming["time"], forming["open"], forming.get("high"), forming.get("low"),
                                   forming["last"], now)
        self._sync_stop(inst, now)
        return out

    def _sync_stop(self, inst: str, now: float) -> None:
        tr = self.mgr.trade(inst)
        if tr is None or tr.phase != "live" or not tr.hist:
            return
        s = tr.newest_stop()
        key = (tr.setup_id, float(s["price"]))
        if self._stop_drawn.get(inst) != key:
            if self._stop_drawn.get(inst) is not None or s["source"] != "default":
                self.chart.on_stop(inst, tr.day, s["price"], STOP_LABEL.get(s["mode"], "price"), now)
            self._stop_drawn[inst] = key

    def _close_sessions(self, inst: str, now: float, force: bool = False) -> list[dict]:
        mine = [day for i, day in self.mgr.awaiting() if i == J.file_key(inst)]
        frame = self.stores[inst].frame
        if not mine or not len(frame):
            return []
        m, day = et_clock(frame.index[-1:])
        newest_day = str(day[0])[:10]
        due = sorted({d for d in mine if force or newest_day > d or (newest_day == d and int(m[0]) >= FLAT_MINUTES)})
        written = []
        for d in due:                                   # oldest first, each on the window of ITS session — never a
            window = self.stores[inst].session_window(d)    # tail: a fill older than the tail could never be benchmarked
            if window is not None and len(window):
                written += self.mgr.close_session(inst, window, now)
        out = []
        if written:
            for rec in R.maybe_review(self.journal, now=now, write_text=self.live):
                out.append({"kind": "status", "fields": {"review": rec.get("n")},
                            "text": f"tjr_human: interim review at {rec.get('n')} filled trades written "
                                    f"(review-{rec.get('n')}.txt). It decides nothing and changes nothing (section 3.7)."})
        return out

    def _command(self, cmd, now: float) -> list[dict]:
        self.facts["commands"] += 1
        if self.live and not self._is_backlog(cmd, now):  # a startup backlog says nothing about the PC clock
            self._cmd_clock(cmd, now)
        out = self.mgr.on_command(cmd, now)
        for inst in self.instruments:
            self._sync_stop(inst, now)
        return out

    def _commands(self, cmds: list, now: float) -> list[dict]:
        """Every command is applied and journaled one by one, as ever. The REPLIES to a backlog — messages
        sent before this run started and too old to act on, which Telegram hands over in one go at the
        first getUpdates — are folded into one line with their count: the push queue is single, and a
        hundred 'too old' answers would sit in it ahead of a real alert."""
        out: list[dict] = []
        backlog = 0
        for cmd in cmds:
            old = self._is_backlog(cmd, now)
            got = self._command(cmd, now)
            if old:                                     # rejected (stale, or unreadable) and journaled as ever; not answered one by one
                backlog += 1
                got = [n for n in got if n.get("kind") != "reply"]
            out += got
        if backlog:
            out.append({"kind": "reply", "fields": {"why": "stale", "ignored": backlog},
                        "text": f"{backlog} old message(s) IGNORED: sent before this run started and more than "
                                f"{self.mgr.max_command_age_s:.0f} s ago. Each is journaled as rejected; "
                                "nothing was acted on. Send it again if you still mean it."})
        return out

    def _is_backlog(self, cmd, now: float) -> bool:
        """Sent before this run started AND too old to act on: what Telegram kept while nothing was polling."""
        sent = getattr(cmd, "sent_ts", None)
        return (self._started_at is not None and sent is not None and float(sent) < self._started_at
                and now - float(sent) > self.mgr.max_command_age_s)

    # ---- live -----------------------------------------------------------------------------------------------

    _SAYS = {"telegram_inbound": "commands are NOT being received (SKIP / STOP / MOVE STOP / EXIT NOW do not arrive)",
             "cycle": "the loop is failing inside a cycle: bars, events or the journal may be stuck",
             "clock": "the PC CLOCK is suspect: the STOP window, the SKIP boundary, stale-command and LATE tests all "
                      "read it — set the clock (Settings > Time > Sync now)"}
    _RUN_WHAT = {"cycle": ("cycle_error", "cycle_error_still", "cycle_recovered"),
                 "clock": ("clock_suspect", "clock_suspect_still", "clock_recovered")}

    def _matters_now(self, now: float) -> bool:
        m, _ = et_minute_and_day(now)
        return FAST_FROM <= m < FAST_TO or self.mgr.needs_fast_polling()

    def _journal_quietly(self, kind: str, body: dict, now: float, instrument: str | None = None) -> None:
        try:
            self.journal.write(kind, body, instrument=instrument, now=now)
        except Exception:                               # the journal may be the thing that is failing
            pass

    def _feed_failed(self, key: str, exc: Exception, now: float) -> None:
        """One failure state per scope — the whole feed, one instrument's pane, the
        Telegram inbound thread, the cycle itself. Journaled and logged on the
        TRANSITION and every ten minutes, never per poll; pushed to the phone once
        when it falls inside 09:25-10:15 ET or while a setup or a trade is on (the
        box does the telling), and once more when it recovers. Never raises."""
        if key not in ("telegram_inbound", "cycle", "clock"):
            self.facts["feed_failures"] += 1
        st = self._failing.get(key)
        line = mask_token(" ".join(f"{type(exc).__name__}: {exc}".split()) if key == "cycle"
                          else " ".join(str(exc).split()))[:400]
        run = key in self._RUN_WHAT
        extra = {"skew_s": getattr(exc, "skew_s", None)} if key == "clock" else {}
        if st is None:
            st = self._failing[key] = {"since": now, "n": 1, "told": now, "pushed": False}
            if run:
                self._journal_quietly("run", {"what": self._RUN_WHAT[key][0], "error": line, **extra}, now)
            else:
                self._journal_quietly("feed", {"what": "failure", "scope": key,
                                               "error_kind": getattr(exc, "kind", "error"), "error": line}, now)
            self.log(f"tjr_human {key if run else 'feed'}: {key}: {line} — retrying")
        else:
            st["n"] += 1
            if now - st["told"] >= STILL_FAILING_S:
                st["told"] = now
                body = {"scope": key, "failures": st["n"], "seconds": round(now - st["since"], 1), "error": line}
                self._journal_quietly("run" if run else "feed",
                                      {"what": self._RUN_WHAT[key][1] if run else "still_failing", **body, **extra}, now)
        # a suspect clock is pushed whenever it happens: the 09:25-10:15 test itself reads that clock; so are
        # bars that are not plain 1-minute OHLC (a chart style): until it is fixed nothing can be merged
        m, day = et_minute_and_day(now)
        what = self._SAYS.get(key, f"{key}: no bar is being read — no signal can be detected, stops and targets "
                                   "are not being tested")
        if not st["pushed"] and (key == "clock" or getattr(exc, "kind", None) == "bars" or self._matters_now(now)):
            st["pushed"], st["pushed_day"] = True, day
            self.pusher.push([{"kind": "status", "fields": {"scope": key},
                               "text": f"tjr_human FAILING since {J.iso_ms(st['since'])[11:19]} UTC — {what}. {line[:200]}"}])
        elif st["pushed"] and st.get("pushed_day") != day and self._trading_hours(m, day):
            st["pushed_day"] = day                       # once per trading day while it lasts: never a silent week
            self.pusher.push([{"kind": "status", "fields": {"scope": key, "failures": st["n"]},
                               "text": f"tjr_human STILL FAILING since {J.iso_ms(st['since'])[:19]}Z "
                                       f"({st['n']} failures) — {what}. {line[:200]}"}])

    @staticmethod
    def _trading_hours(m: int, day: str) -> bool:
        return pd.Timestamp(day).weekday() < 5 and ALIVE_AT <= m < DAY_END

    def _feed_ok(self, key: str, now: float) -> None:
        st = self._failing.pop(key, None)
        if st is not None:
            body = {"scope": key, "failures": st["n"], "seconds": round(now - st["since"], 1)}
            if key in self._RUN_WHAT:
                self._journal_quietly("run", {"what": self._RUN_WHAT[key][2], **body}, now)
            else:
                self._journal_quietly("feed", {"what": "recovered", **body}, now)
            if st.get("pushed"):
                self.pusher.push([{"kind": "status", "fields": {"scope": key},
                                   "text": f"tjr_human RECOVERED: {key} after {int(now - st['since'])} s "
                                           f"({st['n']} failure(s))"}])

    def _watch_inbound(self, now: float) -> None:
        """A failing getUpdates is never silent: wrong token (401), another poller
        (409), no network, a journal that cannot be written. Outbound may still work,
        so the phone is told that its commands do not arrive."""
        if self.inbound is None or not self._inbound_up:
            return
        err = getattr(self.inbound, "last_error", None)
        n = int(getattr(self.inbound, "error_count", 1 if err else 0) or 0)
        alive = getattr(self.inbound, "alive", None)
        if callable(alive) and not alive():
            err, n = "the inbound thread has stopped", INBOUND_FAIL_AFTER
            try:
                self.inbound.start()
            except Exception:
                pass
        if err and n >= INBOUND_FAIL_AFTER:
            self._feed_failed("telegram_inbound", RuntimeError(err), now)
        elif not err:
            self._feed_ok("telegram_inbound", now)

    def _bar_clock(self, inst: str, forming: dict | None, now: float) -> None:
        """The PC clock against the bar stamps. While the chart is live, `now` lies
        inside the forming bar's minute. A clock BEHIND the forming bar's open is
        unambiguous (the chart cannot show the future): refused at the first poll
        beyond CLOCK_REFUSE_S, a `clock_suspect` state beyond CLOCK_SLACK_S after.
        A clock AHEAD of the newest bar cannot be told from a stalled chart by the
        stamps alone: at the first poll CLOCK_REFUSE_MIN trading minutes of it refuse
        the start (either way nothing can be traded), later it is the STALLED
        failure of `_cycle_live`, whose text names both causes."""
        if not forming:
            return
        ahead = now - _epoch(forming["time"])             # 0..60 s in normal running
        first = not self._clock_checked and self.facts["cycles"] <= 1
        self._clock_checked = True
        if first and self.live:                         # only the very first poll of a run refuses: a loop that has
            #                                             been up for hours (the CLI was failing) is never ended by this
            if ahead < -CLOCK_REFUSE_S:
                self._refuse(f"the PC clock ({J.iso_ms(now)}) is {int(-ahead)} s BEHIND the newest bar the chart shows "
                             f"({inst} {forming['time']}Z): the clock is wrong. The STOP window, the SKIP boundary and "
                             "the session day are read from it — set the clock, then start again", now,
                             skew_s=round(ahead, 1))
            behind = stalled_minutes(forming["time"], now)
            if behind >= CLOCK_REFUSE_MIN:
                self._refuse(f"the newest bar the chart shows ({inst} {forming['time']}Z) is {behind} trading minutes "
                             f"older than the PC clock ({J.iso_ms(now)}): either the chart is not receiving data "
                             "(reconnecting, a session taken over by another device, a holiday) or the PC clock is "
                             "fast. Nothing can be traded on either — start again when the chart is live and the "
                             "clock is right", now, skew_s=round(ahead, 1), trading_minutes=behind)
        self._clock_why["bar"] = (f"the PC clock is {int(-ahead)} s BEHIND the forming bar {inst} {forming['time']}Z"
                                  if ahead < -CLOCK_SLACK_S else None)
        self._clock_skew = round(ahead, 1)

    def _cmd_clock(self, cmd, now: float) -> None:
        """The PC clock against Telegram's own message date (whole seconds, never
        later than the true receipt). Dated AFTER the receipt: the PC clock is slow,
        and the STOP window stretches by that much (the SKIP boundary does not: a fill is voided on
        Telegram's date only, `trade._skip_can_void`). Two commands
        running older than the stale limit: the clock is fast, or delivery is slow."""
        sent = getattr(cmd, "sent_ts", None)
        if sent is None:
            return
        delay = now - float(sent)
        if delay < -CMD_CLOCK_SLACK_S:
            self._clock_why["cmd"] = f"a Telegram message is dated {-delay:.0f} s AFTER the PC clock received it: the clock is slow"
            self._clock_skew = round(delay, 1)
        elif delay > self.mgr.max_command_age_s:
            self._cmd_late += 1
            if self._cmd_late >= CMD_LATE_AFTER:
                self._clock_why["cmd"] = (f"{self._cmd_late} commands in a row arrived more than "
                                          f"{self.mgr.max_command_age_s:.0f} s after Telegram dated them ({delay:.0f} s): "
                                          "the PC clock is fast, or delivery is that slow — they were rejected as stale")
                self._clock_skew = round(delay, 1)
        else:
            self._cmd_late = 0
            self._clock_why["cmd"] = None
        self._watch_clock(now)

    def _watch_clock(self, now: float) -> None:
        why = [w for w in self._clock_why.values() if w]
        if why:
            exc = RuntimeError("; ".join(why))
            exc.skew_s = self._clock_skew
            self._feed_failed("clock", exc, now)
        else:
            self._feed_ok("clock", now)

    def _tell_hole(self, inst: str, hole: dict, now: float) -> None:
        """A hole that opened while the runner was up (the PC slept, the chart could
        not serve the minutes missed): journaled, logged, and pushed once."""
        key = (inst, hole.get("after"), hole.get("before"))
        self.journal.write("feed", {"what": "hole", **hole}, instrument=inst, now=now)
        self.log(f"tjr_human feed: {inst}: HOLE in the store {hole}")
        if key in self._holes_told or inst not in self._holes_checked or not self._matters_now(now):
            return                                       # before the first check the start rule (refuse / accept) speaks
        self._holes_told.add(key)
        self.pusher.push([{"kind": "status", "fields": {"scope": inst, "hole": hole},
                           "text": f"tjr_human {inst}: HOLE in the 1-minute store — {hole.get('trading_minutes')} trading "
                                   f"minutes missing after {hole.get('after')}Z. The store never fills it by itself; a day "
                                   "whose levels depend on it is not traded (restart with --backfill, or --accept-holes)."}])

    def _refresh_armed(self, now: float) -> None:
        sizes = {}
        for p in J.journal_files(self.base):
            if p.name.startswith(J.RUN + "-"):
                try:
                    sizes[p.name] = p.stat().st_size
                except OSError:
                    pass
        if sizes == self._run_sizes:
            return
        self._run_sizes = sizes
        rec = J.armed_record(self.journal.records())
        if (rec is not None) != self.mgr.armed:
            if rec is not None and rec.get("target_rule") != D.TARGET_RULE:
                self.log("tjr_human: ARMED names another TARGET_RULE than the code — staying in observe mode")
                return
            self.armed_rec = rec
            self.mgr.set_armed(rec is not None)
            self.log(f"tjr_human: now {'ARMED — entries are taken from the next signal' if rec else 'in OBSERVE mode'}")

    def _want(self, now: float) -> dict[str, int]:
        out = {}
        for inst in self.instruments:
            last = self.stores[inst].last
            behind = FETCH_MAX_EVAL if last is None else int((now - _epoch(last)) / 60.0) + 10
            out[inst] = max(FETCH_MIN, behind)
        return out

    def cycle_live(self) -> list[dict]:
        """One poll. Never raises except `Refused` (replay mode or a hole, before the first bar is processed)."""
        now = float(self.clock())
        self.facts["cycles"] += 1
        notices: list[dict] = []
        try:
            return self._cycle_live(now, notices)
        finally:
            self.pusher.push(notices)                   # what was delivered before an exception is still told

    def _cycle_live(self, now: float, notices: list) -> list[dict]:
        self._refresh_armed(now)
        self._watch_inbound(now)
        try:
            polls = self.feed.cycle(now, self._want(now))
            self._feed_ok("feed", now)
        except FeedError as exc:
            if exc.kind == "replay" and not self._holes_checked:
                self._refuse(str(exc), now)
            self._feed_failed("feed", exc, now)
            polls = {}
        except Exception as exc:                        # a feed bug is a feed failure, not the end of the run
            self._feed_failed("feed", exc, now)
            polls = {}
        for inst in self.instruments:
            got = polls.get(inst)
            if got is None:
                continue
            if isinstance(got, Exception):
                self._feed_failed(inst, got, now)
                continue
            try:
                new, info = self.stores[inst].merge(got.closed)
            except (FeedError, OSError) as exc:         # OSError: the store csv is held by another program
                self._feed_failed(inst, exc, now)
                continue
            self._note_revised(inst, info)
            self._bar_clock(inst, got.forming, now)
            # a chart that answers but no longer delivers bars: the same state machine as a failing CLI
            newest = self.stores[inst].last
            if got.forming and (newest is None or pd.Timestamp(got.forming["time"]) > newest):
                newest = pd.Timestamp(got.forming["time"])
            idle = stalled_minutes(newest, now)
            stalled = idle >= STALL_MIN
            if stalled:
                self._feed_failed(inst, FeedError(
                    f"STALLED: no new bar for {idle} trading minutes (newest {newest.isoformat()}Z) though the chart "
                    "answers: its data has stopped, or the PC clock is fast (reconnecting / another device logged in "
                    "/ back from sleep / holiday)", "stalled"), now)
            else:
                self._feed_ok(inst, now)
            self._tell_repairs(now)
            if info.get("hole"):
                self._tell_hole(inst, info["hole"], now)
            if inst not in self._holes_checked:
                holes = recent_holes(self.stores[inst].frame)
                if holes and not self.accept_holes:
                    self._refuse(f"{inst}: the store has a hole in its two newest sessions — {holes[:3]}. The levels "
                                 "that depend on those minutes would be wrong. Backfill the store (detect --backfill "
                                 f"{inst}=<csv export>) or, if it is an early close, start with --accept-holes", now,
                                 holes=holes)
                if holes:
                    self.journal.write("feed", {"what": "holes_accepted", "holes": holes}, instrument=inst, now=now)
                    self._accepted_holes |= {(inst, h["after"], h["before"]) for h in holes}
                self._abandoned_sessions_held(inst, now)
                self._holes_checked.add(inst)
            rows = self.stores[inst].after(self._fed[inst])
            self._process(inst, rows, now, notices)
            self._abandoned_closed(inst, now)
            if not stalled:                             # a stale forming bar is not a price: no preview, no EXIT NOW fill
                notices += self._forming(inst, got.forming, now)
            notices += self._close_sessions(inst, now)
        self._watch_clock(now)
        notices += self._commands(self.inbound.drain() if self.inbound is not None else [], now)
        self._maybe_weekly(now)
        self._flush_revised(now)
        self._liveness(now, notices)
        return notices

    # ---- bars the chart revised after the store took them ---------------------------------------------------------

    def _note_revised(self, inst: str, info: dict) -> None:
        stamps = info.get("revised_stamps") or []
        if not len(stamps):
            return
        _, day = et_clock(pd.DatetimeIndex(stamps))
        for ts, d in zip(stamps, day):
            self._revised[inst].setdefault(str(d)[:10], set()).add(pd.Timestamp(ts).isoformat())

    def _flush_revised(self, now: float, final: bool = False) -> None:
        """One `feed` record per instrument per session: how many bars the chart showed differently from
        what the store already held, and the first and the last of them. Written when the session is over
        (and, `partial`, when the run stops inside it). The store is append-only: it is NOT rewritten —
        what the detector saw stays what the record holds."""
        _, today = et_minute_and_day(now)
        for inst in self.instruments:
            for d in sorted(self._revised[inst]):
                if not final and d >= today:
                    continue
                stamps = sorted(self._revised[inst].pop(d))
                self._journal_quietly("feed", {"what": "revised_upstream", "day": d, "count": len(stamps),
                                               "first": stamps[0], "last": stamps[-1], "partial": bool(final and d >= today),
                                               "store_rewritten": False}, now, instrument=inst)

    # ---- alive is said, not assumed ---------------------------------------------------------------------------------

    def _liveness(self, now: float, notices: list) -> None:
        if not self.live:
            return
        m, day = et_minute_and_day(now)
        if not self._trading_hours(m, day):
            return
        if ("alive", day) not in self._live_told:
            self._live_told.add(("alive", day))
            notices.append(self._alive_notice(now, m, day))
        if m >= TERMINAL_AT and ("terminal", day) not in self._live_told:
            lines, settled = self._terminal_lines(day)
            if settled or m >= TERMINAL_LATEST:
                self._live_told.add(("terminal", day))
                self._journal_quietly("run", {"what": "liveness", "part": "terminal", "day": day, "lines": lines}, now)
                if lines:
                    notices.append({"kind": "status", "fields": {"day": day},
                                    "text": f"tjr_human {day} {D.hhmm(m)} ET — no fill: " + " | ".join(lines)})

    def _alive_notice(self, now: float, m: int, day: str) -> dict:
        per, body = [], {}
        for inst in self.instruments:
            last = self.stores[inst].last
            behind = stalled_minutes(last, now) if last is not None else None
            try:
                warm = bool(check_warm(self.stores[inst].frame, for_day=day)["ok"])
            except Exception:
                warm = False
            body[inst] = {"store_last": last.isoformat() if last is not None else None, "trading_minutes_behind": behind,
                          "warm": warm}
            per.append(f"{inst}: store to " + (f"{last:%H:%M}Z, {behind} trading min behind" if last is not None else "EMPTY")
                       + f", warm {'yes' if warm else 'NO'}")
        try:
            filled = int(J.filled_count(self.journal.records()))
        except Exception:
            filled = None
        failing = sorted(self._failing)
        self._journal_quietly("run", {"what": "liveness", "part": "alive", "day": day, "mode": "armed" if self.mgr.armed
                                      else "observe", "filled": filled, "sample": R.SAMPLE, "instruments": body,
                                      "failing": failing}, now)
        return {"kind": "status", "fields": {"day": day, "filled": filled},
                "text": f"tjr_human ALIVE {day} {D.hhmm(m)} ET — mode "
                        f"{'ARMED' if self.mgr.armed else 'OBSERVE (no entry is taken)'}; filled "
                        f"{'?' if filled is None else filled} of {R.SAMPLE}; " + "; ".join(per)
                        + ("; OPEN PROBLEMS: " + ", ".join(failing) if failing else "; no open problem")}

    def _terminal_lines(self, day: str) -> tuple[list[str], bool]:
        """One line per instrument that did not fill today: its terminal event and why. Event names and
        reasons only. (lines, every instrument has reached a terminal state)."""
        lines, settled = [], True
        for inst in self.instruments:
            tr = self.mgr.trade(inst)
            mine = tr if tr is not None and tr.day == day else None
            if mine is not None and mine.fill is not None and mine.phase in ("live", "closed"):
                continue                                # it filled: the trade alerts speak for it
            evs = []
            for e in reversed(self.det[inst].events):
                if e.get("day") != day:
                    break
                evs.append(e)
            if not any(e["kind"] == "levels" for e in evs):
                lines.append(f"{inst}: no levels today — the 09:29 bar never reached the detector (the feed?)")
            elif not self._route.get(inst, False):
                lines.append(f"{inst}: not routed today (a cold store or an unaccepted hole): nothing could be traded")
            elif mine is not None and mine.phase == "skipped":
                lines.append(f"{inst}: skipped")
            else:
                last = next((e for e in evs if e["kind"] in ("no_sweep", "expired", "invalidated", "fill")), None)
                if last is None:
                    settled = False
                    lines.append(f"{inst}: no terminal event yet (detector stage {self.det[inst].snapshot().get('stage')})")
                elif last["kind"] == "no_sweep":
                    lines.append(f"{inst}: no_sweep ({last.get('reason')})")
                elif last["kind"] == "expired":
                    lines.append(f"{inst}: expired ({last.get('reason')}; {last.get('because')})")
                elif last["kind"] == "invalidated":
                    lines.append(f"{inst}: invalidated ({last.get('reason')} at stage {last.get('stage')})")
                else:
                    lines.append(f"{inst}: a fill was seen and NOT taken ("
                                 + ("observe mode" if mine is None or mine.phase == "observe" else mine.phase) + ")")
        return lines, settled

    def _maybe_weekly(self, now: float) -> None:
        """Section 3.6: Sunday, from the journal, by the box — once, as a file beside the journal."""
        ts = _stamp(now).tz_localize("UTC").tz_convert(D.M.ET)
        if ts.weekday() != 6:
            return
        path = self.base / f"weekly-{ts.date()}.txt"
        if path.exists():
            return
        try:
            path.write_text(R.weekly(self.journal.records(), now=now)["text"] + "\n", encoding="utf-8")
            self.pusher.push([{"kind": "status", "text": f"tjr_human: weekly report written ({path.name})", "fields": {}}])
        except Exception as exc:
            self.log(f"tjr_human: weekly report failed: {type(exc).__name__}: {exc}")

    def _wait(self, seconds: float) -> None:
        end = float(self.clock()) + seconds
        while float(self.clock()) < end:
            cmds = self.inbound.drain() if self.inbound is not None else []
            if cmds:
                now = float(self.clock())
                self.pusher.push(self._commands(cmds, now))
                if any(t.exit_request is not None for t in self.mgr.open_trades()):
                    return                                  # EXIT NOW: go and observe a price
            self.sleep(min(0.5, max(0.0, end - float(self.clock()))))

    def run_live(self, max_cycles: int | None = None) -> dict:
        n = 0
        started = False
        if self.live and self.keep_awake:
            self._awake = keep_awake(True)              # this thread's request; undone in the finally below
        try:
            self.start_live()
            started = True
            while max_cycles is None or n < max_cycles:
                try:
                    self.cycle_live()
                    self._feed_ok("cycle", float(self.clock()))
                except Refused:
                    raise
                except Exception as exc:                    # nothing in a cycle may end the run; told once, not per poll
                    self._feed_failed("cycle", exc, float(self.clock()))
                n += 1
                if max_cycles is not None and n >= max_cycles:
                    break
                self._wait(poll_interval(float(self.clock()), self.mgr.needs_fast_polling()))
        except KeyboardInterrupt:
            self.log("tjr_human: stopping (the next start journals the gap)")
        except Refused as exc:                          # raised after startup: pushed before the process ends
            if started and not getattr(exc, "pushed", False):
                self._push_refusal(str(exc))
            raise
        finally:
            if self._awake:
                keep_awake(False)
                self._awake = False
            self.stop(journal_it=started or bool(self.facts["cycles"]))
        return self.machinery_facts()

    def stop(self, journal_it: bool = True) -> None:
        try:
            if journal_it and self.live:
                self._flush_revised(float(self.clock()), final=True)
        except Exception:
            pass
        try:
            if journal_it:
                self.journal.write("run", {"what": "stop", "cycles": self.facts["cycles"],
                                           "open_trades": [t.setup_id for t in self.mgr.open_trades()]},
                                   now=float(self.clock()))
        except Exception:
            pass
        if self.inbound is not None:
            try:
                self.inbound.stop()
            except Exception:
                pass
        self.pusher.close()
        self.chart.close()
        self._lock.release()

    def __del__(self):
        try:
            self._lock.release()
        except Exception:
            pass

    # ---- replay ---------------------------------------------------------------------------------------------

    def run_replay(self, script=None, synthetic: bool = False, sources: dict | None = None) -> dict:
        """Drive the same loop over a `ReplayFeed`. Per instant T of the timeline:
        first the human's items timed before T, then the bars that closed at T
        (their events, then the bar), then the open of the bar forming at T."""
        if self.live:
            raise Refused("run_replay needs a ReplayFeed")
        scripted = isinstance(script, list) and bool(script) and isinstance(script[0], tuple)
        items = list(script) if scripted else load_script(script)
        times = self.feed.timeline()
        if not len(times):
            return self.machinery_facts()
        self.journal.write("run", {"what": "start", "mode": "replay", "armed": self.mgr.armed, "synthetic": bool(synthetic),
                                   "sources": sources or {}, "human": len(items), "allow_cold": self.allow_cold,
                                   "target_rule": D.TARGET_RULE}, now=float(times[0]))
        if not self.allow_cold:                         # the cold sessions in one call: nothing in them is routed
            for inst in self.instruments:
                frame = self.feed.frames[inst]
                _, day = et_clock(frame.index)
                days = np.unique(day)
                if len(days) > D.WARM_SESSIONS:
                    row = int(np.argmax(day >= days[D.WARM_SESSIONS]))
                    hist = self.feed.skip_to(inst, row)
                    cold = self.det[inst].feed(hist)
                    for ev in cold:
                        k = self.facts["events"][inst]
                        k[ev["kind"]] = k.get(ev["kind"], 0) + 1
                    self.facts["cold_days"][inst] += sum(1 for ev in cold if ev["kind"] == "levels")
                    self.facts["bars"][inst] += len(hist)
                    self.stores[inst].merge(hist, normalised=True)
                    self._fed[inst] = hist.index[-1] if len(hist) else None
            times = self.feed.timeline()
        q = 0
        now = float(times[0]) if len(times) else 0.0
        for T in times:
            now = float(T)
            while q < len(items) and items[q][0] < now:
                self._scripted(items[q])
                q += 1
            polls = self.feed.cycle(now)
            notices: list[dict] = []
            for inst in self.instruments:
                got = polls.get(inst)
                if got is None:
                    continue
                if len(got.closed):
                    new, _ = self.stores[inst].merge(got.closed, normalised=True)
                    notices += self._process(inst, new, now)
                    notices += self._close_sessions(inst, now)
                notices += self._forming(inst, got.forming, now)
            self.pusher.push(notices)
        while q < len(items):
            self._scripted(items[q])
            q += 1
        for inst in self.instruments:
            self.pusher.push(self._close_sessions(inst, now + 60.0, force=True))
        self.journal.write("run", {"what": "stop", "mode": "replay"}, now=now + 60.0)
        return self.machinery_facts(scripted=len(items))

    def _scripted(self, item) -> None:
        t, kind, value, inst = item
        if kind == "cmd":
            self.pusher.push(self._command(parse_command(value), float(t)))
        else:
            self.facts["prices"] += 1
            self.pusher.push(self.mgr.on_price(inst, float(value), float(t)))
            self._sync_stop(J.file_key(inst), float(t))

    # ---- what a replay may say on market data (section 9.8) --------------------------------------------------

    def machinery_facts(self, scripted: int | None = None) -> dict:
        """Counts of what the machine did. No R, no price, no exit reason, no
        per-exit figure: nothing from which a result could be read."""
        records = self.journal.records()
        by_kind: dict[str, int] = {}
        rejects: dict[str, int] = {}
        stages: dict[str, int] = {}
        rejected_seqs = set()
        inbound = []
        for r in records:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
            if r["kind"] == "reject":
                rejects[str(r.get("why"))] = rejects.get(str(r.get("why")), 0) + 1
                rejected_seqs.add(r.get("command_seq"))
            if r["kind"] == "command":
                stages[str(r.get("stage"))] = stages.get(str(r.get("stage")), 0) + 1
                if r.get("stage") == "inbound":
                    inbound.append(r["seq"])
        st = R.status(records, self.base)
        integ = st.get("integrity") or {}
        return {
            "mode": "live" if self.live else "replay", "armed": self.mgr.armed,
            "instruments": {i: {"bars": self.facts["bars"][i], "routed_days": self.facts["routed_days"][i],
                                "cold_days_not_routed": self.facts["cold_days"][i],
                                "detector_events_basis": "ALL sessions fed, cold and routed",
                                "detector_events": dict(sorted(self.facts["events"][i].items())),
                                "routed_sessions": {"basis": "ROUTED (warm) sessions only",
                                                    **{k: dict(sorted(v.items())) for k, v in
                                                       sorted(self.facts["routed"][i].items())}},
                                "detector": self.det[i].snapshot()} for i in self.instruments},
            "setups": {"basis": "ROUTED (warm) sessions only, from the journal",
                       **{k: st["counts"][k] for k in ("signals", "filled", "skipped", "invalidated", "expired",
                                                        "observe_records")}},
            "trades": {"basis": "ROUTED (warm) sessions only, from the journal",
                       "filled": st["filled"], "exit_records": by_kind.get("exit", 0),
                       "benchmarks_records": by_kind.get("benchmarks", 0), "open_at_end": len(st["open_trades"]),
                       "awaiting_benchmarks": len(st["awaiting_benchmarks"]), "previews": self.facts["previews"]},
            "commands": {"scripted": scripted, "received": len(inbound),
                         "accepted": sum(1 for s in inbound if s not in rejected_seqs),
                         "rejected": sum(1 for s in inbound if s in rejected_seqs),
                         "rejected_by_rule": dict(sorted(rejects.items())), "stages": dict(sorted(stages.items())),
                         "price_observations": self.facts["prices"]},
            "journal": {"records": len(records), "by_kind": dict(sorted(by_kind.items())),
                        "integrity_ok": bool(integ.get("ok")), "problems": list(integ.get("problems") or [])[:5],
                        "files": integ.get("files")},
            "alerts": {"notices": sum(self.pusher.by_kind.values()), "by_kind": dict(sorted(self.pusher.by_kind.items())),
                       "sent_to_notify": len(self.pusher.sent)},
            "draws": self.chart.counts(),
            "feed": {"name": self.feed.name, "cycles": self.facts["cycles"], "failures": self.facts["feed_failures"]},
        }


# ────────────────────────────── entry points the CLI calls ──────────────────────────────

def replay(csvs: dict[str, str], human=None, base_dir=None, show_outcomes: bool = False, armed: bool = True,
           allow_cold: bool = False, preview: bool = True) -> dict:
    """Section 9.8. `csvs`: {instrument: path}. Returns {"facts", "synthetic",
    "outcomes"?}. The journal lives in a temporary directory that is deleted —
    a replay journal of market data holds R on disk, so it is never kept; only a
    replay of synthetic files may be kept (`base_dir`) or show outcomes."""
    status = {inst: synthetic_status(p) for inst, p in csvs.items()}
    synthetic = all(s["synthetic"] for s in status.values())
    if show_outcomes and not synthetic:
        bad = {i: s["why"] for i, s in status.items() if not s["synthetic"]}
        raise Refused("--show-outcomes REFUSED: not flagged synthetic — " + "; ".join(f"{i}: {w}" for i, w in bad.items())
                      + ". On market data the replay prints machinery facts only (section 9.8): R, win rate or "
                        "per-exit totals of this variant would be a third round of a closed test.")
    if base_dir is not None and not synthetic:
        raise Refused("--journal-dir REFUSED for market data: the replay journal holds R per trade; it is kept only "
                      "for files flagged synthetic (section 9.8)")
    if base_dir is not None:
        who = J.origin(J.read_journal(base_dir))
        if who["live"]:
            r0 = who["first_live"] or {}
            raise Refused(f"--journal-dir REFUSED: {Path(base_dir)} holds records a replay did not write "
                          f"(first: seq {r0.get('seq')} kind {r0.get('kind')!r}). A replay's fills would count toward "
                          "the 100 of the pre-registered sample and lock ARMED — keep a replay in a directory of its own")
    frames = {inst: load_bars(p) for inst, p in csvs.items()}
    tmp = None
    if base_dir is None:
        tmp = tempfile.mkdtemp(prefix="tjr_human_replay_")
        base_dir = tmp
    try:
        t0 = min(float(f.index[0].value) / 1e9 for f in frames.values() if len(f))
        journal = J.Journal(base_dir, clock=lambda: t0, fsync=False)
        if armed and J.armed_record(journal.records()) is None:
            journal.write("armed", {"armed": True, "reason": "replay: the machinery is tested with entries on",
                                    "commit": None, "prereg_sha256": None, "target_rule": D.TARGET_RULE,
                                    "replay": True}, now=t0)
        runner = Runner(base_dir, ReplayFeed(frames), instruments=tuple(frames), clock=lambda: t0, journal=journal,
                        notify=False, background=False, chart=Chart(executor=None), inbound=None,
                        stores={i: BarStore(None) for i in frames}, allow_cold=allow_cold, preview=preview)
        facts = runner.run_replay(human, synthetic=synthetic,
                                  sources={i: {"file": Path(p).name, **status[i]} for i, p in csvs.items()})
        out = {"facts": facts, "synthetic": synthetic, "sources": status}
        if show_outcomes:
            out["outcomes"] = R.weekly(journal.records())["text"]
        return out
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def journal_is_reportable(records: list[dict]) -> tuple[bool, str]:
    """`report`, `review` and `final` read the experiment's own journal. A journal
    a replay wrote is refused unless the replay ran on files flagged synthetic."""
    for r in records:
        if r.get("kind") == "run" and r.get("what") == "start" and r.get("mode") == "replay" and not r.get("synthetic"):
            return False, "this journal was written by a replay over files not flagged synthetic (section 9.8)"
    who = J.origin(records)
    if who["replay"] and who["live"]:
        return False, (f"this journal mixes a replay's records ({who['replay']}) with the experiment's own "
                       f"({who['live']}): replayed trades must never be reported as the sample (section 9.8)")
    return True, ""


def live_runner(base_dir=J.DEFAULT_DIR, method: str = "eval", draw: bool = True, accept_holes: bool = False,
                cli: TvCli | None = None, backfill: dict | None = None, keep_awake: bool = True,
                accept_short_sessions=()) -> Runner:
    """The `detect` command's runner: the TradingView feed, the chart layer on, Telegram if it is configured."""
    cli = cli or TvCli()
    feed = LiveTvFeed(cli, method=method)
    chart = Chart(executor=TvDraw(cli, feed.pane_of) if draw else None, log_dir=base_dir)
    who = J.origin(J.read_journal(base_dir))
    if who["replay"]:                                 # before anything (a backfill record, the store) is written there
        raise Refused(f"{Path(base_dir)} holds a replay's journal ({who['replay']} record(s)): the live run never "
                      "shares a journal with replayed trades (section 9.8) — use another directory")
    runner = Runner(base_dir, feed, chart=chart, accept_holes=accept_holes, keep_awake=keep_awake,
                    accept_short_sessions=accept_short_sessions)
    records = runner.journal.records()
    locked = J.entered_count(records) > 0
    for inst, path in (backfill or {}).items():
        info = runner.stores[inst].backfill(load_bars(path), allow_older=not locked)
        runner.journal.write("feed", {"what": "backfill", "file": Path(path).name, **info}, instrument=inst)
    return runner

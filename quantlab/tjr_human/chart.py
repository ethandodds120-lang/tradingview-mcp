"""The display layer of DESIGN-tjr-human.md section 3.2, through the repo's `tv` CLI (`tv draw shape`).

At signal time it draws, on the pane of the instrument: every active key level
labelled by type, the sweep bar, the confirmation bar and its type, the zone box
and its type, the stop line (indicative wick stop), T1 and T2. A `zone` event
redraws the box; the fill replaces the indicative stop line with the exact stop
in force and marks the entry; a STOP / MOVE STOP moves the line.

It is a display and nothing else (sections 7, 9.3):

  * nothing here can delay or stop the detector: `Chart` only ENQUEUES; a daemon
    thread does the drawing, one shape at a time, and the queue is bounded — when
    it is full the oldest job is dropped and counted;
  * every failure — TradingView down, a CLI timeout, a pane that is not there,
    a shape TradingView refuses — is logged (`chart.log` beside the journal) and
    swallowed. No exception leaves this module;
  * it removes only shapes it drew itself (by entity id), never `draw clear`:
    the human's own drawings are not ours to touch;
  * in replay it is DISABLED: `Chart(executor=None)` shells out to nothing. The
    shapes it would have drawn are still built and counted, because the replay
    reports draw calls as a machinery fact (section 9.8).

`tv draw shape` draws on the ACTIVE pane, so a shape for NQ needs `tv pane focus
<NQ's index>` first. Focus-and-draw runs under the `TvCli` lock the live feed
also takes, one shape per lock hold, so a poll waits for at most one shape.

`shapes_for(ev)` is pure: event in, list of shape dicts out. That is what the
tests check; the executor is a few lines around the CLI.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .detector import level_class, level_sort_key, level_type
from .journal import bar_epoch

CHART_LOG = "chart.log"
QUEUE_MAX = 400
DRAW_TIMEOUT_S = 8.0
ZONE_MINUTES = 45                 # how far right the zone box is drawn from where it formed
BAR_S = 60

#: line colour by level type (section 3.2: "labelled by type")
TYPE_COLOR = {"ASIA": "#7e57c2", "LON": "#1e88e5", "PD": "#fb8c00", "H1": "#00897b", "H4": "#00695c",
              "POC": "#c2185b", "HVN": "#ad1457"}
ZONE_COLOR = {"fvg": "#42a5f5", "eq": "#ffee58", "ob": "#66bb6a", "breaker": "#ef5350"}
STOP_COLOR, T1_COLOR, T2_COLOR, MARK_COLOR = "#e53935", "#43a047", "#2e7d32", "#90a4ae"


def _epoch(stamp) -> int | None:
    try:
        return int(bar_epoch(stamp))
    except Exception:
        return None


def _line(tag: str, t: int, price: float, text: str, color: str, style: int = 0, width: int = 1) -> dict:
    return {"tag": tag, "shape": "horizontal_line", "time": t, "price": float(price), "text": text,
            "overrides": {"linecolor": color, "linestyle": style, "linewidth": width, "showLabel": True,
                          "textcolor": color, "horzLabelsAlign": "right"}}


def _text(tag: str, t: int, price: float, text: str, color: str = MARK_COLOR) -> dict:
    return {"tag": tag, "shape": "text", "time": t, "price": float(price), "text": text,
            "overrides": {"color": color, "fontsize": 11}}


def _zone(z: dict, fallback_t: int) -> dict | None:
    if not z or z.get("low") is None or z.get("high") is None:
        return None
    t0 = _epoch(z.get("formed_time")) or fallback_t
    color = ZONE_COLOR.get(z.get("kind"), MARK_COLOR)
    return {"tag": "zone", "shape": "rectangle", "time": t0, "price": float(z["high"]),
            "time2": t0 + ZONE_MINUTES * BAR_S, "price2": float(z["low"]), "text": f"zone {z.get('kind')}",
            "overrides": {"color": color, "backgroundColor": color, "transparency": 80, "showLabel": True,
                          "textColor": color}}


def _target(tag: str, t: int, tgt: dict | None, color: str, rule) -> dict | None:
    if not tgt or tgt.get("price") is None:
        return None
    return _line(tag, t, tgt["price"], f"{tag.upper()} {tgt.get('names', '')} ({rule})", color, style=2, width=2)


def shapes_for(ev: dict) -> list[dict]:
    """Everything section 3.2 draws for one detector event. Pure. Unknown kinds draw nothing."""
    kind = ev.get("kind")
    t = _epoch(ev.get("bar_time"))
    if t is None:
        return []
    out: list[dict | None] = []
    if kind in ("signal", "zone"):
        if kind == "signal":
            levels = ev.get("levels") or {}
            for name in sorted(levels, key=level_sort_key):
                price = levels[name]
                if price is None:
                    continue
                typ = level_type(name)
                out.append(_line(f"level:{name}", t, price, f"{name} [{typ} / {level_class(name)}]",
                                 TYPE_COLOR.get(typ, MARK_COLOR), style=1))
            lv = ev.get("level") or {}
            ts = _epoch(ev.get("sweep_bar_time")) or _epoch(ev.get("sweep_time")) or t
            if ev.get("extreme") is not None:
                out.append(_text("sweep", ts, ev["extreme"],
                                 f"SWEEP {lv.get('name', '')} ({ev.get('direction', '')})"))
            tc = _epoch(ev.get("conf_time")) or t
            out.append(_text("confirmation", tc, ev.get("ref_price") if ev.get("ref_price") is not None
                             else (ev.get("eq") or 0.0), "CONF " + "+".join(ev.get("conf_types") or [])))
            rule = ev.get("target_rule")
            out.append(_target("t1", t, ev.get("t1"), T1_COLOR, rule))
            out.append(_target("t2", t, ev.get("t2"), T2_COLOR, rule))
        out.append(_zone(ev.get("zone"), t))
        wick = (ev.get("stops_indicative") or {}).get("wick")
        if wick is not None:
            out.append(_line("stop", t, wick, "STOP wick (indicative)", STOP_COLOR, width=2))
    elif kind == "fill":
        if ev.get("entry") is not None:
            out.append(_text("entry", t, ev["entry"], f"ENTRY {ev.get('direction', '')} {float(ev['entry']):.2f}"))
        wick = (ev.get("stops") or {}).get("wick")
        if wick is not None:
            out.append(_line("stop", t, wick, "STOP wick", STOP_COLOR, width=2))
        rule = ev.get("target_rule")
        out.append(_target("t1", t, ev.get("t1"), T1_COLOR, rule))
        out.append(_target("t2", t, ev.get("t2"), T2_COLOR, rule))
    return [s for s in out if s is not None]


def stop_shape(price: float, label: str, epoch_s: float) -> dict:
    return _line("stop", int(epoch_s), price, f"STOP {label}", STOP_COLOR, width=2)


def draw_args(shape: dict) -> list[str]:
    """The `tv draw shape` command line for one shape."""
    args = ["draw", "shape", "--type", shape["shape"], "--price", repr(float(shape["price"])),
            "--time", str(int(shape["time"]))]
    if shape.get("price2") is not None:
        args += ["--price2", repr(float(shape["price2"])), "--time2", str(int(shape["time2"]))]
    if shape.get("text"):
        args += ["--text", str(shape["text"])]
    if shape.get("overrides"):
        args += ["--overrides", json.dumps(shape["overrides"], separators=(",", ":"))]
    return args


class TvDraw:
    """The executor: focus the instrument's pane, draw (or remove) one shape. Raises; `Chart` swallows."""

    def __init__(self, cli, pane_of, timeout: float = DRAW_TIMEOUT_S):
        self.cli = cli
        self.pane_of = pane_of                        # callable: instrument -> pane index, or None
        self.timeout = float(timeout)

    def draw(self, instrument: str, shape: dict) -> str | None:
        with self.cli.lock:
            self._focus(instrument)
            res = self.cli.call(*draw_args(shape), timeout=self.timeout)
        return res.get("entity_id")

    def remove(self, instrument: str, entity_id: str) -> None:
        with self.cli.lock:
            self._focus(instrument)
            self.cli.call("draw", "remove", str(entity_id), timeout=self.timeout)

    def _focus(self, instrument: str) -> None:
        idx = self.pane_of(instrument)
        if idx is None:
            raise RuntimeError(f"no pane known for {instrument}")
        self.cli.call("pane", "focus", str(int(idx)), timeout=self.timeout)


class Chart:
    """`on_event` / `on_stop` enqueue and return at once. `executor=None` disables drawing (replay)."""

    #: tags redrawn in place: the old shape is removed when a new one with the same tag arrives
    REPLACED = ("zone", "stop", "t1", "t2")

    def __init__(self, executor=None, log_dir: str | Path | None = None, background: bool = True):
        self.executor = executor
        self.log_path = Path(log_dir) / CHART_LOG if log_dir is not None else None
        self.background = bool(background) and executor is not None
        self.built: list[dict] = []                   # every shape asked for: {"instrument","day","tag","shape"}
        self.drawn = 0
        self.failed = 0
        self.dropped = 0
        self._ids: dict[tuple, str] = {}              # (instrument, day, tag) -> entity id
        self._day: dict[str, str] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.executor is not None

    # ---- what the runner calls: never raises, never waits ---------------------------------------------------

    def on_event(self, ev: dict) -> int:
        try:
            shapes = shapes_for(ev)
            for s in shapes:
                self._ask(ev.get("instrument"), ev.get("day"), s)
            return len(shapes)
        except Exception as exc:
            self._log(f"on_event {ev.get('kind') if isinstance(ev, dict) else '?'}: {type(exc).__name__}: {exc}")
            return 0

    def on_stop(self, instrument: str, day: str, price: float, label: str, epoch_s: float) -> int:
        try:
            self._ask(instrument, day, stop_shape(float(price), label, epoch_s))
            return 1
        except Exception as exc:
            self._log(f"on_stop: {type(exc).__name__}: {exc}")
            return 0

    def counts(self) -> dict:
        by: dict[str, int] = {}
        for b in self.built:
            by[b["shape"]["shape"]] = by.get(b["shape"]["shape"], 0) + 1
        return {"enabled": self.enabled, "asked": len(self.built), "by_shape": by, "drawn": self.drawn,
                "failed": self.failed, "dropped": self.dropped}

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout)

    # ---- the queue ------------------------------------------------------------------------------------------

    def _ask(self, instrument, day, shape: dict) -> None:
        job = {"instrument": str(instrument), "day": str(day), "tag": shape["tag"], "shape": shape}
        self.built.append(job)
        if self.executor is None:
            return
        if not self.background:
            self._do(job)
            return
        while True:
            try:
                self._queue.put_nowait(job)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="tjr-human-chart", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._do(job)

    def _do(self, job: dict) -> None:
        inst, day, tag = job["instrument"], job["day"], job["tag"]
        try:
            if self._day.get(inst) != day:                          # a new day: yesterday's shapes of ours go
                for key in [k for k in self._ids if k[0] == inst and k[1] != day]:
                    self._remove(inst, self._ids.pop(key))
                self._day[inst] = day
            if tag in self.REPLACED and (inst, day, tag) in self._ids:
                self._remove(inst, self._ids.pop((inst, day, tag)))
            eid = self.executor.draw(inst, job["shape"])
            if eid:
                self._ids[(inst, day, tag)] = eid
            self.drawn += 1
        except Exception as exc:
            self.failed += 1
            self._log(f"draw {inst} {tag}: {type(exc).__name__}: {exc}")

    def _remove(self, inst: str, eid: str) -> None:
        try:
            self.executor.remove(inst, eid)
        except Exception as exc:
            self._log(f"remove {inst} {eid}: {type(exc).__name__}: {exc}")

    def _log(self, line: str) -> None:
        try:
            if self.log_path is None:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {' | '.join(str(line).splitlines())}\n")
        except Exception:
            pass

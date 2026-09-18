"""One setup, the trade it may become, and the rules of the four human controls
(DESIGN-tjr-human.md sections 3.3, 3.4, 9.3, 9.7).

No broker, no order: a simulated ledger. `TradeManager` is driven by four things
and owns no clock, no feed and no network of its own:

    on_event(ev, now)                  a detector event (closed bars only)
    on_preview(ev, now)                Detector.preview_fill(observed open): the entry, the moment it happens
    on_bar(inst, bar_time, o,h,l,c)    each closed 1-minute bar, AFTER the events it produced
    on_price(inst, price, now)         optionally, a last-price observation between bars
    on_command(cmd, now)               a parsed human command

Every method returns a list of notices `{"kind": "signal"|"trade"|"reply",
"text", "fields"}` for the runner to push through `quantlab.alerts.notify`;
nothing here can delay the detector.

THE STOP, precisely. A stop has a time from which the market is tested against it:

  * chosen BEFORE the entry instant (STOP from the signal on): in force at the fill;
  * chosen in the 60 s AFTER the fill: accepted, but the entry minute belongs to
    the stop that was in force at the fill — the wick stop unless a choice came
    earlier. "The 60 s are a grace for the message, not for the market": a trade
    the wick stop takes in its entry minute stays stopped, the later choice is
    void and R is against the wick. From the next minute the chosen stop is the
    initial stop and R is against it;
  * MOVE STOP: only tighter, only after the STOP window. It owns closed bars
    from the first bar that opens at or after the command. Inside the minute it
    was made in, only what the market did AFTER it counts: the runner hands over
    the forming bar's running high and low (`on_forming`), the adverse extreme
    seen at the move is remembered, and a new extreme beyond it that reaches the
    moved stop stops the trade at the moved stop — seen on the forming bar, or
    on the closed bar of that minute. Without forming extremes (a closed-bars-only
    feed, the replay) the move's own minute cannot be split and the moved stop is
    live from the next minute. `on_price` (a scripted tick) tests it at once.
    The default and the chosen stop are decided by CLOSED bars only, so a human
    who does nothing is `wick_fixed` exactly.
  * A NAMED width (wick, 1.0, 1.5, 2.0, session) is held to the same band as a
    typed price: between the wick stop and the 2.0 ATR stop, inclusive, or
    rejected and journaled with the price it resolved to (sections 3.3, 5, 9.3).

Fill conventions are `exits.py`'s: the entry minute is tested against the stop
only; later minutes stop first (a gap through the stop fills at the open), then
T1 (the whole position), then the close of the minute stamped 15:55; a session
that ends earlier exits at its last close. EXIT NOW fills at the next observed
price: the forming bar's last price (`on_forming`) or the next `on_price`, or the
close of the next bar that closes after the command — in every case AFTER the
stop and target tests of that minute: if the forming bar's running range has
already reached the stop in force (or T1), that is the exit, not the later price. Realised R is against the
initial stop, net of the section 9.2 round trip on the entry notional.

Observe mode (not armed): everything is journaled as `observe`, no entry is
taken, commands are rejected. A setup is armed or not at its SIGNAL; arming
later in the day starts with the next signal (section 9.7).
"""

from __future__ import annotations

import math
import time

import pandas as pd

from . import journal as J
from .commands import REASON_WINDOW_S, REASONS, UNSPECIFIED, USAGE, Command
from .commands import malformed as telegram_malformed
from .detector import et_clock, hhmm
from .exits import FLAT_MINUTES, Session, benchmark_record, cost_points, net_r, replay_exit

STOP_WINDOW_S = 60.0
ENTRY_MINUTE_S = 60.0          # the entry bar: it belongs to the stop in force at the fill
MAX_COMMAND_AGE_S = 120.0
BAND = ("wick", "atr2.0")
#: A fill the process learns of more than this many seconds after the entry minute
#: closed (a restart later the same day) was never the human's to manage. None =
#: the contract as written: it is taken and counts (section 9.7 says only that a
#: stop is journaled as a gap). A number = such a fill is journaled `observe`
#: (why: late_fill) and is not a trade. The user decides before `arm`; ARMED records it.
LATE_FILL_S: float | None = None
STOP_LABEL = {"wick": "wick", "atr1.0": "1.0 ATR", "atr1.5": "1.5 ATR", "atr2.0": "2.0 ATR", "session": "session"}


def _px(p) -> str:
    return "-" if p is None else f"{float(p):.2f}"


def _notice(kind: str, text: str, fields: dict | None = None) -> dict:
    return {"kind": kind, "text": text, "fields": J.clean(fields or {})}


def send_notices(base_dir, notices: list[dict], notify=None) -> list[dict]:
    """Push notices through `quantlab.alerts.notify` (T-4's transport and its 20 s
    bound). The three kinds are in alerts.PUSH_KINDS. Never raises."""
    real = notify is None
    if real:
        from ..alerts import notify as _n
        notify = _n
    # A token with a CR / LF / space in it makes urllib refuse the URL with a message
    # that spells the token out, and alerts._mask does not catch the escaped form:
    # such a token is "not configured" here — the alert is logged, never pushed.
    log_only = real and telegram_malformed()
    out = []
    for n in notices:
        try:
            if log_only:
                out.append(notify(base_dir, n["kind"], n["text"], n.get("fields") or None, push=False))
            else:
                out.append(notify(base_dir, n["kind"], n["text"], n.get("fields") or None))
        except Exception as exc:                       # notify does not raise; a replacement might
            out.append({"kind": n.get("kind"), "logged": False, "pushed": False, "error": type(exc).__name__})
    return out


class Trade:
    """One setup of one instrument on one day, and the trade it may become.

    phase: observe | signal | skipped | live | closed | dead
    """

    def __init__(self, setup_id: str, instrument: str, day: str, armed: bool,
                 stop_window_s: float = STOP_WINDOW_S):
        self.setup_id, self.instrument, self.day = setup_id, instrument, day
        self.stop_window_s = float(stop_window_s)
        self.armed = bool(armed)
        self.phase = "signal" if armed else "observe"
        self.signal: dict | None = None
        self.entry_at: float | None = None             # epoch of the entry instant, once a trigger is known
        self.skip: dict | None = None
        self.stop_request: dict | None = None
        self.stop_used = False
        self.fill: dict | None = None
        self.provisional = False
        self.side = 0
        self.entry = math.nan
        self.entry_epoch = math.nan
        self.stops: dict = {}
        self.t1: float | None = None
        self.hist: list[dict] = []                     # {"price","mode","source","eff_tick","eff_bar"}
        self.exit_request: dict | None = None
        self.exit: dict | None = None
        self.last_bar_epoch = -math.inf
        self.last_close: float | None = None
        self.last_bar: str | None = None
        self.best: float | None = None                 # favourable extreme over completed minutes (and prices seen)
        self.worst: float | None = None
        self.benchmarked = False
        self.would_fill = False
        self.forming: dict | None = None               # {"epoch", "adv"}: the forming bar's adverse extreme, last seen

    # ---- the stop ----

    def stop_for_bar(self, bar_epoch: float) -> dict:
        return [s for s in self.hist if s["eff_bar"] <= bar_epoch][-1]

    def stop_for_tick(self, t: float) -> dict:
        return [s for s in self.hist if s["eff_tick"] <= t][-1]

    def newest_stop(self) -> dict:
        return self.hist[-1]

    def initial_stop(self, exit_epoch: float) -> dict:
        """What R is measured against: the stop in force at the fill if the trade
        ended in its entry minute, else the human's initial choice (or the default)."""
        firsts = [s for s in self.hist if s["source"] != "move"]
        if exit_epoch < self.entry_epoch + ENTRY_MINUTE_S:
            return firsts[0]
        return firsts[-1]

    def in_stop_window(self, t: float) -> bool:
        return t <= self.entry_epoch + self.stop_window_s

    def unrealised_r(self, price: float | None, t: float) -> float | None:
        if self.phase != "live" or price is None:
            return None
        risk = self.side * (self.entry - self.initial_stop(t)["price"])
        return (self.side * (float(price) - self.entry)) / risk if risk > 0 else None

    def snapshot(self) -> dict:
        return {"setup_id": self.setup_id, "instrument": self.instrument, "day": self.day, "phase": self.phase,
                "armed": self.armed, "side": self.side or None,
                "entry": None if self.fill is None else self.entry,
                "stop": self.newest_stop()["price"] if self.hist else None,
                "stop_mode": self.newest_stop()["mode"] if self.hist else None,
                "t1": self.t1, "provisional": self.provisional, "stop_used": self.stop_used,
                "stop_pending": None if self.stop_request is None else self.stop_request["arg"],
                "exit_requested": self.exit_request is not None,
                "last_bar": self.last_bar}


class TradeManager:
    """Routes events, prices and commands to one `Trade` per instrument per day and journals everything."""

    def __init__(self, journal: J.Journal, armed: bool = False, clock=time.time,
                 stop_window_s: float = STOP_WINDOW_S, max_command_age_s: float = MAX_COMMAND_AGE_S,
                 reason_window_s: float = REASON_WINDOW_S, late_fill_s: float | None = LATE_FILL_S):
        self.journal = journal
        self.late_fill_s = None if late_fill_s is None else float(late_fill_s)
        self.armed = bool(armed)
        self.clock = clock
        self.stop_window_s = float(stop_window_s)
        self.max_command_age_s = float(max_command_age_s)
        self.reason_window_s = float(reason_window_s)
        self._trades: dict[str, Trade] = {}
        self._awaiting: list[Trade] = []               # filled (or would-have-filled) setups without a benchmarks record
        self._last_price: dict[str, float] = {}
        self._last_unspecified: dict | None = None
        self._seen: dict[tuple, dict] = {}
        self._journaled: dict[str, dict] = {}

    # ────────────────────────────── plumbing ──────────────────────────────

    def set_armed(self, armed: bool) -> None:
        """Takes effect at the next signal (section 9.7). Whether disarming is
        allowed at all is the CLI's question: `journal.filled_count` > 0 means no."""
        self.armed = bool(armed)

    def trade(self, instrument: str) -> Trade | None:
        return self._trades.get(J.file_key(instrument))

    def open_trades(self) -> list[Trade]:
        return [t for t in self._trades.values() if t.phase == "live"]

    def needs_fast_polling(self) -> bool:
        """A setup waiting for its entry, or a trade on: the runner keeps its 5 s poll."""
        return any(t.phase in ("signal", "live") for t in self._trades.values())

    def awaiting_benchmarks(self) -> list[str]:
        return sorted({t.instrument for t in self._awaiting if not t.benchmarked})

    def awaiting(self) -> list[tuple[str, str]]:
        """(instrument, day) of every filled or would-have-filled setup still without its benchmarks record."""
        return [(t.instrument, t.day) for t in self._awaiting if not t.benchmarked and t.fill is not None]

    def seen(self, ev: dict) -> bool:
        """After `resume`: is this detector event already in the journal? The runner
        routes an event of an EARLIER day only when it is — such a day is re-fed to
        put a trade back, never to take a new entry after the fact."""
        return (ev.get("setup_id"), ev.get("kind"), ev.get("bar_time")) in self._seen

    def note_price(self, instrument: str, price: float) -> None:
        """Remember an observed price (the `market_price` of the action records, the
        MOVE STOP market test) WITHOUT testing the stop or the target against it.
        The runner uses this, not `on_price`, for the forming bar: closed bars decide
        stops and targets, exactly as they do for the ten mechanical exits, so a
        human who does nothing stays identical to `wick_fixed`."""
        try:
            self._last_price[J.file_key(instrument)] = float(price)
        except (TypeError, ValueError):
            pass

    def snapshot(self) -> dict:
        return {"armed": self.armed, "trades": {k: t.snapshot() for k, t in self._trades.items()},
                "awaiting_benchmarks": self.awaiting_benchmarks()}

    def _now(self, now) -> float:
        return float(self.clock() if now is None else now)

    def _write(self, kind: str, body: dict, tr: Trade | None, now: float, inst: str | None = None) -> dict:
        if tr is not None:
            body = {**body, "setup_id": tr.setup_id, "day": tr.day}
        return self.journal.write(kind, body, instrument=(tr.instrument if tr else inst), now=now)

    def _action_fields(self, tr: Trade | None, t: float) -> dict:
        """Section 3.4, per action: price, seconds since entry, unrealised R at that moment."""
        if tr is None:
            return {"market_price": None, "seconds_since_entry": None, "unrealised_r": None}
        price = self._last_price.get(tr.instrument)
        live = tr.fill is not None and tr.phase in ("live", "closed")
        return {"market_price": price,
                "seconds_since_entry": round(t - tr.entry_epoch, 3) if live else None,
                "unrealised_r": tr.unrealised_r(price, t),
                "phase": tr.phase, "last_bar": tr.last_bar, "best": tr.best, "worst": tr.worst}

    # ────────────────────────────── restart ──────────────────────────────

    def resume(self, records: list[dict] | None = None) -> dict:
        """After a restart: remember what the journal already holds, so the detector
        events the runner re-feeds from its store rebuild today's state SILENTLY
        (no second record, no second alert) and the human's earlier actions — a
        skip, a pending STOP, the stop chosen, every move, an exit — are put back.
        Bars up to the last one a journaled record had seen are not replayed; bars
        after it are, against the stop that was in force: what the market did
        during the gap is applied, not forgotten."""
        records = self.journal.records() if records is None else records
        for r in records:
            ev = r.get("ev")
            if isinstance(ev, dict) and ev.get("setup_id") and ev.get("kind") and not ev.get("provisional"):
                self._seen[(ev["setup_id"], ev["kind"], ev.get("bar_time"))] = r
        state = J.rebuild(records)
        self._journaled = state["setups"]
        if state["armed"] is not None:
            self.armed = True
        return {"seen_events": len(self._seen), "setups": len(self._journaled), "armed": self.armed}

    def _restore(self, tr: Trade) -> None:
        s = self._journaled.get(tr.setup_id)
        if not s:
            return
        if s["skip"] is not None:
            tr.phase, tr.skip = "skipped", s["skip"]
        resolved = {r.get("command_seq") for r in s["stop_sets"] + s["rejects"]}
        for c in s["commands"]:
            if c.get("stage") == "stop_pending":
                tr.stop_used = True
                if c.get("command_seq") not in resolved:
                    tr.stop_request = {"arg": c["arg"], "ts": c["t_cmd"], "seq": c.get("command_seq"),
                                       "reason": c.get("human_reason", UNSPECIFIED)}
        if any(not r.get("void") for r in s["stop_sets"]):
            tr.stop_used = True
        if s["benchmarks"] is not None:
            tr.benchmarked = True

    def _restore_live(self, tr: Trade) -> None:
        s = self._journaled.get(tr.setup_id)
        if not s:
            return
        for r in sorted(s["stop_sets"] + s["moves"], key=lambda r: r["seq"]):
            h = {"price": float(r["price"]), "mode": r.get("mode", "price"), "source": r.get("source", "move"),
                 "eff_tick": float(r["eff_tick"]), "eff_bar": float(r["eff_bar"]),
                 "own_bar": r.get("own_bar"), "base_adv": r.get("base_adv")}
            if r.get("when") == "before_fill":
                tr.hist = [h]                                       # it was the stop in force at the fill
            else:
                tr.hist.append(h)
        seen_bars = [r.get("last_bar") for k in ("stop_sets", "moves", "rejects", "commands") for r in s[k]]
        seen_bars = [J.bar_epoch(b) for b in seen_bars if b]
        if seen_bars:
            tr.last_bar_epoch = max(seen_bars)
            tr.last_bar = J.epoch_bar(tr.last_bar_epoch)
        for r in sorted(s["stop_sets"] + s["moves"] + s["rejects"], key=lambda r: r["seq"]):
            if r.get("best") is not None:
                tr.best, tr.worst = r["best"], r["worst"]
        if s["exit"] is not None:
            tr.exit, tr.phase = s["exit"], "closed"

    # ────────────────────────────── detector events ──────────────────────────────

    def on_event(self, ev: dict, now: float | None = None) -> list[dict]:
        t = self._now(now)
        kind, inst = ev["kind"], J.file_key(ev["instrument"])
        sid, day = ev["setup_id"], ev["day"]
        seen = self._seen.get((sid, kind, ev.get("bar_time")))
        silent = seen is not None
        tr = self._trades.get(inst)
        if tr is not None and tr.day != day:                        # a new day: yesterday's setup is history
            if tr.fill is not None and not tr.benchmarked and tr not in self._awaiting:
                self._awaiting.append(tr)
            tr = None
            self._trades.pop(inst, None)
        lag = round(t - J.bar_epoch(ev["knowable_at"]), 3) if ev.get("knowable_at") else None
        head = {"setup_id": sid, "day": day, "event": kind, "ev": ev, "lag_s": lag}

        if kind in ("signal", "zone"):
            if tr is None:
                armed = bool(seen.get("armed")) if silent else self.armed
                tr = self._trades[inst] = Trade(sid, inst, day, armed, self.stop_window_s)
                if silent:
                    self._restore(tr)
            tr.signal = ev
            if silent:
                return []
            if not tr.armed:
                self._write("observe", head, tr, t)
                return [_notice("signal", self._signal_text(ev, tr, observe=True), self._signal_fields(ev, tr))]
            self._write("signal", {**head, "armed": True}, tr, t)
            return [_notice("signal", self._signal_text(ev, tr), self._signal_fields(ev, tr))]

        if kind == "fill":
            return self._on_fill(ev, t, provisional=False, silent=silent, lag=lag, seen=seen)

        if kind == "entry_trigger" and tr is not None:
            tr.entry_at = J.bar_epoch(ev["entry_at"])

        if silent:
            if kind in ("invalidated", "expired") and tr is not None and tr.phase in ("signal", "observe"):
                tr.phase = "dead" if tr.phase == "signal" else "observe"
            return []

        armed_setup = tr.armed if tr is not None else self.armed
        if not armed_setup:
            self._write("observe", head, tr, t, inst=inst)
            return []
        if kind == "invalidated":
            notices = []
            body = {**head, "skipped": bool(tr and tr.phase == "skipped")}
            void = tr is not None and tr.fill is not None and tr.provisional
            if void:
                body["void"] = True
            self._write("invalidated", body, tr, t, inst=inst)     # journaled BEFORE the state moves: a retry repeats it whole
            if void:
                # the observed open said fill, the closed bar says there was no trade: the closed bar is the record
                notices.append(_notice("trade", f"{inst} entry VOID: the closed bar gives no trade "
                                                f"({ev.get('reason')}); the provisional fill is cancelled"))
                tr.fill, tr.hist, tr.exit = None, [], None
            if tr is not None:
                if tr.phase in ("signal", "live", "closed"):
                    tr.phase = "dead"
                if tr.signal is not None and not body.get("void"):
                    notices.append(_notice("signal", f"{inst} setup INVALIDATED ({ev.get('reason')}): "
                                                     f"price traded {_px(ev.get('traded'))} beyond the wick stop "
                                                     f"{_px(ev.get('wick_stop'))} before the entry. Day done."))
            return notices
        self._write("event", head, tr, t, inst=inst)
        if kind == "expired" and tr is not None:
            had_signal = tr.signal is not None and tr.phase == "signal"
            if tr.phase == "signal":
                tr.phase = "dead"
            if had_signal:
                return [_notice("signal", f"{inst} setup EXPIRED ({ev.get('reason')}): no entry today.")]
        return []

    def on_preview(self, ev: dict | None, now: float | None = None) -> list[dict]:
        """`Detector.preview_fill(observed open)`: the entry as it happens. The
        authoritative `fill` follows when the entry minute has closed."""
        if not ev:
            return []
        t = self._now(now)
        if ev.get("kind") != "fill":
            return []                                   # a provisional invalidation waits for the closed bar
        lag = round(t - J.bar_epoch(ev["knowable_at"]), 3) if ev.get("knowable_at") else None
        return self._on_fill(ev, t, provisional=True, silent=False, lag=lag)

    # ---- the fill ----

    def _on_fill(self, ev: dict, t: float, provisional: bool, silent: bool, lag, seen: dict | None = None) -> list[dict]:
        inst = J.file_key(ev["instrument"])
        tr = self._trades.get(inst)
        head = {"setup_id": ev["setup_id"], "day": ev["day"], "event": "fill", "ev": ev, "lag_s": lag}
        if tr is None or tr.setup_id != ev["setup_id"]:
            if not silent and not provisional:          # a fill with no signal on record: never traded
                self._write("observe", {**head, "why": "fill without a journaled signal"}, None, t, inst=inst)
            return []
        if tr.phase == "observe":
            if not silent and not provisional:
                self._write("observe", head, tr, t)
            return []
        if tr.phase == "skipped":
            if not provisional:
                tr.fill, tr.would_fill = ev, True
                self._set_entry(tr, ev)
                if tr not in self._awaiting and not tr.benchmarked:
                    self._awaiting.append(tr)
                if not silent:
                    self._write("skip", {**head, "stage": "would_fill"}, tr, t)
            return []
        if tr.phase == "dead":
            return []

        if tr.fill is not None:                         # the closed bar confirms (or corrects) a provisional fill
            if provisional or not tr.provisional:
                return self._late_stop_request(tr, t) if not silent else []
            match = (float(ev["entry"]) == tr.entry and ev.get("stops") == tr.fill.get("stops")
                     and ev.get("t1") == tr.fill.get("t1"))
            if not silent:                              # journaled BEFORE the state moves (a failed write is retried whole)
                self._write("fill", {**head, "provisional": False, "preview_match": match,
                                     "preview_entry": None if match else tr.entry}, tr, t)
            tr.provisional = False
            if not match:
                self._rebase(tr, ev)
            tr.fill = ev
            if silent:
                return []
            if match:
                return self._late_stop_request(tr, t)
            return [_notice("trade", f"{inst} fill corrected by the closed bar: entry {_px(tr.entry)}, "
                                     f"stop {_px(tr.newest_stop()['price'])}", {"stops": tr.stops})] \
                + self._late_stop_request(tr, t)

        # first sight of the fill
        if silent and seen is not None and seen.get("kind") == "observe" and seen.get("why") == "late_fill":
            tr.phase = "dead"                           # journaled as not taken by an earlier run: stays not taken
            return []
        if (not silent and not provisional and self.late_fill_s is not None and lag is not None
                and lag - ENTRY_MINUTE_S > self.late_fill_s):       # lag runs from the entry minute's OPEN
            self._write("observe", {**head, "why": "late_fill", "late_fill_s": self.late_fill_s}, tr, t)
            tr.phase = "dead"
            return [_notice("trade", f"{inst} entry NOT TAKEN: the process learnt of the fill {lag - ENTRY_MINUTE_S:.0f} s after the "
                                     f"entry minute closed (limit {self.late_fill_s:.0f} s); journaled as observe")]
        if not silent:                                  # journaled BEFORE the state moves
            self._write("fill", {**head, "provisional": bool(provisional), "armed": True}, tr, t)
        tr.fill, tr.provisional = ev, bool(provisional)
        self._set_entry(tr, ev)
        e0 = tr.entry_epoch
        tr.hist = [{"price": float(tr.stops["wick"]), "mode": "wick", "source": "default",
                    "eff_tick": e0, "eff_bar": e0}]
        tr.phase = "live"
        tr.best = tr.worst = tr.entry
        if tr not in self._awaiting:
            self._awaiting.append(tr)
        if silent:
            self._restore_live(tr)
            return []
        self._last_price.setdefault(inst, tr.entry)
        notices = self._late_stop_request(tr, t)
        stop = tr.stop_for_tick(e0)
        until = e0 + self.stop_window_s
        notices.insert(0, _notice(
            "trade",
            f"{inst} FILL {'long' if tr.side > 0 else 'short'} @ {_px(tr.entry)}"
            f"{' (provisional)' if provisional else ''} — stop in force: {STOP_LABEL.get(stop['mode'], 'price')} "
            f"{_px(stop['price'])}; T1 {_px(tr.t1)}"
            + ("" if tr.stop_used or t > until else f"; STOP accepted until {J.iso_ms(until)[11:19]} UTC"),
            {"setup_id": tr.setup_id, "entry": tr.entry, "stops": tr.stops, "stop": stop["price"],
             "t1": ev.get("t1"), "t2": ev.get("t2"), "target_rule": ev.get("target_rule"),
             "band": sorted(float(tr.stops[k]) for k in BAND)}))
        return notices

    def _late_stop_request(self, tr: Trade, t: float) -> list[dict]:
        """The STOP asked for before the fill, resolved at it. The request is dropped
        only once its record is on disk, so a journal write that failed is retried
        with the fill event and the human's one choice is never lost silently."""
        if tr.stop_request is None or tr.phase != "live":
            return []
        out = self._resolve_stop(tr, tr.stop_request, t)
        tr.stop_request = None
        return out

    @staticmethod
    def _set_entry(tr: Trade, ev: dict) -> None:
        tr.side, tr.entry = int(ev["side"]), float(ev["entry"])
        tr.entry_epoch = J.bar_epoch(ev["bar_time"])
        tr.stops = {k: float(v) for k, v in ev["stops"].items()}
        tr.t1 = float(ev["t1"]["price"]) if ev.get("t1") else None

    def _rebase(self, tr: Trade, ev: dict) -> None:
        """The closed bar's open differs from the observed one. The closed bar is
        the record (it is the entry the ten exits are replayed from): named widths
        are re-resolved, a price the human typed is kept."""
        self._set_entry(tr, ev)
        for h in tr.hist:
            if h["mode"] in tr.stops:
                h["price"] = tr.stops[h["mode"]]

    # ────────────────────────────── prices ──────────────────────────────

    def on_bar(self, instrument: str, bar_time: str, o: float, h: float, l: float, c: float,   # noqa: E741
               now: float | None = None) -> list[dict]:
        """A closed 1-minute bar. Call it after `on_event` for the events that bar produced."""
        t = self._now(now)
        inst = J.file_key(instrument)
        eb = J.bar_epoch(bar_time)
        o, h, l, c = float(o), float(h), float(l), float(c)       # noqa: E741
        tr = self._trades.get(inst)
        if tr is None or tr.phase != "live":
            self._last_price[inst] = c
            return []
        if eb < tr.entry_epoch or eb <= tr.last_bar_epoch:
            return []
        self._last_price[inst] = c
        minutes, day = et_clock(pd.DatetimeIndex([pd.Timestamp(bar_time)]))
        if str(day[0])[:10] != tr.day:                              # the session ended without a 15:55 minute
            px = tr.last_close if tr.last_close is not None else tr.entry
            return self._close(tr, px, "session_end", t, tr.last_bar_epoch if tr.last_bar else tr.entry_epoch, "bar")
        side = tr.side
        entry_bar = eb == tr.entry_epoch
        adverse = l if side > 0 else h
        for mv in tr.hist:                                          # a MOVE STOP made inside this very minute
            if (mv["source"] == "move" and mv.get("base_adv") is not None and mv.get("own_bar") == eb
                    and eb < mv["eff_bar"] and side * (adverse - mv["base_adv"]) < 0
                    and side * (adverse - mv["price"]) <= 0):
                # a new extreme AFTER the move reached the moved stop: only what came after it counts
                return self._close(tr, mv["price"], "trail", t, eb, "bar", stopped=True)
        stop = tr.stop_for_bar(eb)
        sp = stop["price"]
        if (side > 0 and l <= sp) or (side < 0 and h >= sp):
            px = sp if entry_bar else (min(sp, o) if side > 0 else max(sp, o))
            return self._close(tr, px, "trail" if stop["source"] == "move" else "stop", t, eb, "bar", stopped=True)
        if not entry_bar and tr.t1 is not None and ((side > 0 and h >= tr.t1) or (side < 0 and l <= tr.t1)):
            self._extend(tr, h, l)
            return self._close(tr, tr.t1, "target", t, eb, "bar")
        self._extend(tr, h, l)
        out: list[dict] = []
        if tr.exit_request is not None and eb + 60.0 >= tr.exit_request["ts"]:
            out = self._close(tr, c, "exit_now", t, eb, "bar")
        elif int(minutes[0]) >= FLAT_MINUTES:
            out = self._close(tr, c, "flat", t, eb, "bar")
        # only now is the bar "seen": an exit whose journal write failed is retried on the same bar
        tr.last_bar_epoch, tr.last_close, tr.last_bar = eb, c, bar_time
        return out

    def on_forming(self, instrument: str, bar_time: str, o: float, h: float | None, l: float | None,   # noqa: E741
                   last: float, now: float | None = None) -> list[dict]:
        """The FORMING 1-minute bar as the feed shows it now: its open, its running
        high and low (None when the feed cannot know them — the replay exposes the
        open only) and its last price. Three uses, none of which touches a human who
        does nothing (the default and the chosen stop are decided by closed bars):

          * the last price is remembered (`market_price`, the MOVE STOP market test);
          * a MOVED stop is tested: on a bar that opened at or after the move against
            the whole running range (what its closed bar will say, a minute sooner);
            on the minute the move was made in, only against a NEW adverse extreme
            beyond the one seen at the move;
          * a pending EXIT NOW fills at `last` — AFTER the stop in force (and T1) have
            been tested against the running range: a stop that traded earlier in
            the minute is the exit, the later price cannot undo it."""
        t = self._now(now)
        inst = J.file_key(instrument)
        self.note_price(inst, last)
        tr = self._trades.get(inst)
        if tr is None or tr.phase != "live":
            return []
        if h is None or l is None:                                  # no running range: the old reading
            return self.on_price(inst, float(last), t) if tr.exit_request is not None else []
        o, h, l, last = float(o), float(h), float(l), float(last)   # noqa: E741
        fe = J.bar_epoch(bar_time)
        if fe < tr.entry_epoch or fe <= tr.last_bar_epoch:
            return []
        side = tr.side
        adverse, favour = (l, h) if side > 0 else (h, l)
        gap = (lambda sp: min(sp, o)) if side > 0 else (lambda sp: max(sp, o))
        for mv in tr.hist:
            if mv["source"] != "move":
                continue
            if fe >= mv["eff_bar"]:
                if side * (adverse - mv["price"]) <= 0:
                    return self._close(tr, gap(mv["price"]), "trail", t, t, "tick", stopped=True)
            elif mv.get("own_bar") == fe and mv.get("base_adv") is not None:
                if side * (adverse - mv["base_adv"]) < 0 and side * (adverse - mv["price"]) <= 0:
                    return self._close(tr, mv["price"], "trail", t, t, "tick", stopped=True)
        tr.forming = {"epoch": fe, "adv": adverse}
        if tr.exit_request is None:
            return []
        entry_bar = fe == tr.entry_epoch
        stop = tr.stop_for_bar(fe)                                  # the stop this minute's closed bar is tested against
        if side * (adverse - stop["price"]) <= 0:
            px = stop["price"] if entry_bar else gap(stop["price"])
            return self._close(tr, px, "trail" if stop["source"] == "move" else "stop", t, t, "tick", stopped=True)
        if not entry_bar and tr.t1 is not None and side * (favour - tr.t1) >= 0:
            self._extend(tr, h, l)
            return self._close(tr, tr.t1, "target", t, t, "tick")
        self._extend(tr, h, l)
        return self._close(tr, last, "exit_now", t, t, "tick")

    def on_price(self, instrument: str, price: float, now: float | None = None) -> list[dict]:
        """A last-price observation between closed bars (optional)."""
        t = self._now(now)
        inst = J.file_key(instrument)
        price = float(price)
        self._last_price[inst] = price
        tr = self._trades.get(inst)
        if tr is None or tr.phase != "live" or t < tr.entry_epoch:
            return []
        side = tr.side
        stop = tr.stop_for_tick(t)
        if side * (price - stop["price"]) <= 0:
            return self._close(tr, stop["price"], "trail" if stop["source"] == "move" else "stop", t, t, "tick",
                               stopped=True)
        in_entry_minute = t < tr.entry_epoch + ENTRY_MINUTE_S
        if not in_entry_minute and tr.t1 is not None and side * (price - tr.t1) >= 0:
            return self._close(tr, tr.t1, "target", t, t, "tick")
        self._extend(tr, price, price)
        if tr.exit_request is not None:
            return self._close(tr, price, "exit_now", t, t, "tick")
        return []

    @staticmethod
    def _extend(tr: Trade, high: float, low: float) -> None:
        fav, adv = (high, low) if tr.side > 0 else (low, high)
        tr.best = fav if tr.best is None else (max(tr.best, fav) if tr.side > 0 else min(tr.best, fav))
        tr.worst = adv if tr.worst is None else (min(tr.worst, adv) if tr.side > 0 else max(tr.worst, adv))

    def _close(self, tr: Trade, px: float, reason: str, t: float, exit_epoch: float, via: str,
               stopped: bool = False) -> list[dict]:
        side, entry = tr.side, tr.entry
        init = tr.initial_stop(exit_epoch)
        stop0 = float(init["price"])
        risk = side * (entry - stop0)
        cpts = cost_points(tr.instrument, entry)
        if stopped:                                                 # the exit minute's range beyond the fill was not held
            prior = tr.worst if tr.worst is not None else entry
            worst = min(prior, px) if side > 0 else max(prior, px)
        else:
            worst = tr.worst if tr.worst is not None else entry
        best = tr.best if tr.best is not None else entry
        if reason == "target":
            best = px
        fav = lambda p: (p - entry) * side / risk                    # noqa: E731
        voided = [s for s in tr.hist if s["source"] == "choice" and s is not init and
                  exit_epoch < tr.entry_epoch + ENTRY_MINUTE_S]
        req = tr.exit_request if reason == "exit_now" else None
        body = {"exit": float(px), "reason": reason, "via": via,
                "exit_time": J.iso_ms(t if via == "tick" else exit_epoch + 60.0),
                "exit_bar": J.epoch_bar(exit_epoch), "side": side, "entry": entry,
                "stop0": stop0, "stop_mode": init["mode"], "stop_source": init["source"],
                "last_stop": float(tr.newest_stop()["price"]), "cost_points": cpts,
                "r_gross": fav(px), "r": net_r(side, entry, px, stop0, cpts),
                "mae_r": min(0.0, fav(worst)), "mfe_r": max(0.0, fav(best)),
                "seconds_since_entry": round((t if via == "tick" else exit_epoch + 60.0) - tr.entry_epoch, 3),
                "grace": bool(exit_epoch < tr.entry_epoch + ENTRY_MINUTE_S),
                "chosen_stop_void": bool(voided),
                "provisional_fill": tr.provisional,
                "command_seq": req["seq"] if req else None,
                "human_reason": req["reason"] if req else None,
                "requested_ts": J.iso_ms(req["ts"]) if req else None}
        tr.exit = self._write("exit", body, tr, t)
        tr.phase = "closed"
        tr.exit_request = None
        text = (f"{tr.instrument} EXIT {reason} @ {_px(px)} — {body['r']:+.2f} R net "
                f"(against {STOP_LABEL.get(init['mode'], 'price')} stop {_px(stop0)})")
        if voided:
            text += f"; the stop chosen after the fill ({_px(voided[-1]['price'])}) never took effect: " \
                    "the wick stop is live in the entry minute"
        return [_notice("trade", text, {"setup_id": tr.setup_id, "reason": reason, "exit": px, "r": body["r"]})]

    # ────────────────────────────── the human ──────────────────────────────

    def on_command(self, cmd: Command, now: float | None = None) -> list[dict]:
        """Apply one parsed command. Executed first, reason asked after. Every
        outcome — accepted or rejected — is journaled. Never raises."""
        t = self._now(now)
        try:
            return self._command(cmd, t)
        except Exception as exc:                                    # a bug here must not stop the detector
            try:
                self.journal.write("reject", {"why": "internal_error", "error": type(exc).__name__,
                                              "command_seq": cmd.seq, "text": cmd.raw}, now=t)
            except Exception:
                pass
            return [_notice("reply", f"internal error handling {cmd.raw!r}: {type(exc).__name__} — nothing changed")]

    def _command(self, cmd: Command, t: float) -> list[dict]:
        if cmd.seq is None:                                         # not journaled by a transport (a scripted human)
            rec = self.journal.write("command", {
                "stage": "inbound", "source": "direct", "authorized": True, "text": cmd.raw, "sent_ts": cmd.sent_ts,
                "parsed": {"action": cmd.action, "instrument": cmd.instrument, "arg": cmd.arg,
                           "reason": cmd.reason, "error": cmd.error}}, instrument=cmd.instrument, now=t)
            cmd.seq = rec["seq"]
        t_cmd = cmd.sent_ts if (cmd.sent_ts is not None and cmd.sent_ts <= t) else t
        if not cmd.ok:
            return self._reject(None, cmd, t, t_cmd, "unparsable", f"{cmd.error}. {USAGE}")
        if t - t_cmd > self.max_command_age_s:
            return self._reject(self._pick(cmd)[0], cmd, t, t_cmd, "stale",
                                f"{cmd.action} was sent {t - t_cmd:.0f} s ago — too old to act on")
        if cmd.action == "REASON":
            return self._attach_reason(cmd, t)
        tr, why = self._pick(cmd)
        if tr is None:
            return self._reject(None, cmd, t, t_cmd, why,
                                "say which: NQ or ES" if why == "ambiguous_instrument" else "no setup to act on")
        if tr.phase == "observe":
            return self._reject(tr, cmd, t, t_cmd, "observe_mode", "observe mode: the run is not armed, no entries are taken")
        handler = {"SKIP": self._skip, "STOP": self._stop, "MOVE_STOP": self._move, "EXIT_NOW": self._exit_now}
        return handler[cmd.action](tr, cmd, t, t_cmd)

    def _pick(self, cmd: Command) -> tuple[Trade | None, str]:
        if cmd.instrument:
            tr = self._trades.get(J.file_key(cmd.instrument))
            return (tr, "") if tr is not None else (None, "no_setup")
        want = {"SKIP": ("signal",), "STOP": ("signal", "live"), "MOVE_STOP": ("live",), "EXIT_NOW": ("live",)}
        fits = [x for x in self._trades.values() if x.phase in want.get(cmd.action, ())]
        if len(fits) == 1:
            return fits[0], ""
        if len(fits) > 1:
            return None, "ambiguous_instrument"
        rest = [x for x in self._trades.values() if x.signal is not None]
        if len(rest) == 1:
            return rest[0], ""
        if not rest:
            return None, "no_setup"
        rest.sort(key=lambda x: (x.entry_epoch if x.fill is not None else -1.0, x.setup_id))
        return rest[-1], ""

    def _outcome(self, cmd: Command, t_cmd: float) -> dict:
        return {"action": cmd.action, "command_seq": cmd.seq, "human_reason": cmd.reason,
                "t_cmd": t_cmd, "text": cmd.raw}

    def _ask(self, cmd: Command, t: float) -> str:
        if cmd.reason != UNSPECIFIED:
            return ""
        self._last_unspecified = {"seq": cmd.seq, "ts": t, "action": cmd.action}
        return "  reason? " + " / ".join(REASONS)

    def _reject(self, tr: Trade | None, cmd: Command, t: float, t_cmd: float, why: str, say: str,
                extra: dict | None = None) -> list[dict]:
        body = {**self._outcome(cmd, t_cmd), "why": why, "requested": cmd.arg, **self._action_fields(tr, t),
                **(extra or {})}
        self._write("reject", body, tr, t, inst=cmd.instrument)
        tail = self._ask(cmd, t) if cmd.ok and cmd.action != "REASON" else ""
        return [_notice("reply", f"REJECTED {cmd.action or cmd.raw[:40]!s}: {say}{tail}", {"why": why})]

    def _attach_reason(self, cmd: Command, t: float) -> list[dict]:
        last = self._last_unspecified
        if last is None or t - last["ts"] > self.reason_window_s:
            return self._reject(None, cmd, t, t, "no_reason_target",
                                "no action without a reason in the last 10 minutes")
        self.journal.write("command", {"stage": "reason", "action": "REASON", "reason": cmd.reason,
                                       "attaches_to": last["seq"], "command_seq": cmd.seq,
                                       "seconds_after": round(t - last["ts"], 3)}, now=t)
        self._last_unspecified = None
        return [_notice("reply", f"reason {cmd.reason} attached to {last['action']}")]

    # ---- SKIP ----

    def _skip(self, tr: Trade, cmd: Command, t: float, t_cmd: float) -> list[dict]:
        if tr.phase == "skipped":
            return self._reject(tr, cmd, t, t_cmd, "already_skipped", "this setup is already skipped")
        if tr.phase == "dead":
            return self._reject(tr, cmd, t, t_cmd, "setup_dead", "this setup is over; nothing to skip")
        if tr.phase in ("live", "closed") or (tr.entry_at is not None and t_cmd >= tr.entry_at):
            return self._reject(tr, cmd, t, t_cmd, "skip_after_fill", "SKIP is from the signal until the fill; "
                                "the entry has happened. EXIT NOW closes the trade")
        tr.skip = self._write("skip", {**self._outcome(cmd, t_cmd), "stage": "command", "ev": tr.signal,
                                       **self._action_fields(tr, t)}, tr, t)
        tr.phase = "skipped"
        return [_notice("reply", f"{tr.instrument} SKIPPED — followed to the end and journaled with what every "
                                 f"exit would have done.{self._ask(cmd, t)}")]

    # ---- STOP ----

    def _stop(self, tr: Trade, cmd: Command, t: float, t_cmd: float) -> list[dict]:
        if tr.phase == "skipped":
            return self._reject(tr, cmd, t, t_cmd, "setup_skipped", "this setup was skipped")
        if tr.phase in ("dead", "closed"):
            return self._reject(tr, cmd, t, t_cmd, "not_in_trade", "no trade: it is over and stays over")
        if tr.stop_used:
            return self._reject(tr, cmd, t, t_cmd, "stop_already_used", "STOP is once per trade; "
                                "after the window MOVE STOP tightens")
        req = {"arg": cmd.arg, "ts": t_cmd, "seq": cmd.seq, "reason": cmd.reason, "cmd": cmd}
        if tr.phase == "signal":
            self._write("command", {**self._outcome(cmd, t_cmd), "stage": "stop_pending", "arg": cmd.arg,
                                    **self._action_fields(tr, t)}, tr, t)
            tr.stop_used, tr.stop_request = True, req
            what = STOP_LABEL[cmd.arg] if isinstance(cmd.arg, str) else _px(cmd.arg)
            return [_notice("reply", f"{tr.instrument} STOP {what} noted — "
                                     f"{'resolved' if isinstance(cmd.arg, str) else 'validated'} at the fill."
                                     f"{self._ask(cmd, t)}")]
        if not tr.in_stop_window(t_cmd):
            return self._reject(tr, cmd, t, t_cmd, "stop_window_closed",
                                f"the STOP window closed {self.stop_window_s:.0f} s after the fill; MOVE STOP tightens")
        return self._resolve_stop(tr, req, t)

    def _resolve_stop(self, tr: Trade, req: dict, t: float) -> list[dict]:
        """At the fill (a request made earlier) or inside the window (made after it)."""
        cmd: Command = req.get("cmd") or Command(action="STOP", instrument=tr.instrument, arg=req["arg"],
                                                 reason=req.get("reason", UNSPECIFIED), seq=req.get("seq"))
        arg, e0 = req["arg"], tr.entry_epoch
        named = isinstance(arg, str)
        price = float(tr.stops[arg]) if named else float(arg)
        lo, hi = sorted(float(tr.stops[k]) for k in BAND)
        if req["ts"] > e0 + self.stop_window_s:
            tr.stop_used = False
            return self._reject(tr, cmd, t, req["ts"], "stop_window_closed", "the STOP window had closed at the fill")
        if not (lo - 1e-9 <= price <= hi + 1e-9):                  # a named width is held to the band like a price
            tr.stop_used = False                                   # an invalid choice is not the one choice
            what = f"{STOP_LABEL[arg]} = {_px(price)}" if named else _px(price)
            return self._reject(tr, cmd, t, req["ts"], "stop_outside_band",
                                f"{what} is outside the band [{_px(lo)}, {_px(hi)}] (wick stop to 2.0 ATR); "
                                f"the {STOP_LABEL[tr.stop_for_tick(e0)['mode']]} stop stays",
                                {"band": [lo, hi], "requested": price, "requested_name": arg if named else None,
                                 "candidates": tr.stops})
        before = req["ts"] < e0
        eff = e0 if before else e0 + ENTRY_MINUTE_S
        mode = arg if named else "price"
        prev = tr.newest_stop()
        self._write("stop_set", {**self._outcome(cmd, req["ts"]), "mode": mode, "price": price, "source": "choice",
                                 "when": "before_fill" if before else "after_fill", "from": prev["price"],
                                 "seconds_from_fill": round(req["ts"] - e0, 3), "eff_tick": eff, "eff_bar": eff,
                                 "effective_from": J.iso_ms(eff), "candidates": tr.stops, "band": [lo, hi],
                                 "atr_multiple": tr.side * (tr.entry - price) / float(tr.fill["atr"])
                                 if tr.fill.get("atr") else None,
                                 **self._action_fields(tr, t)}, tr, t)
        if before:                                                 # the record is on disk: now the state
            tr.hist = [{"price": price, "mode": mode, "source": "choice", "eff_tick": e0, "eff_bar": e0}]
        else:
            tr.hist.append({"price": price, "mode": mode, "source": "choice", "eff_tick": eff, "eff_bar": eff})
        tr.stop_used = True
        text = f"{tr.instrument} STOP SET {STOP_LABEL.get(mode, 'price')} {_px(price)}"
        if not before:
            text += (f" — in force from {J.iso_ms(eff)[11:19]} UTC; until then the "
                     f"{STOP_LABEL[prev['mode']] if prev['mode'] in STOP_LABEL else 'earlier'} stop "
                     f"{_px(prev['price'])} is live")
        return [_notice("trade" if before else "reply", text + self._ask(cmd, t),
                        {"setup_id": tr.setup_id, "stop": price, "mode": mode})]

    # ---- MOVE STOP ----

    def _move(self, tr: Trade, cmd: Command, t: float, t_cmd: float) -> list[dict]:
        if tr.phase == "signal":
            return self._reject(tr, cmd, t, t_cmd, "move_before_fill", "no trade yet; before the fill it is STOP")
        if tr.phase != "live":
            return self._reject(tr, cmd, t, t_cmd, "not_in_trade", "no open trade")
        if tr.in_stop_window(t_cmd):
            return self._reject(tr, cmd, t, t_cmd, "move_in_stop_window",
                                "MOVE STOP is after the STOP window; in the first 60 s it is STOP")
        new, cur = float(cmd.arg), tr.newest_stop()
        if tr.side * (new - cur["price"]) <= 0:
            return self._reject(tr, cmd, t, t_cmd, "move_widens",
                                f"{_px(new)} is not tighter than the stop {_px(cur['price'])}: "
                                "the stop only moves in the trade's favour", {"stop": cur["price"]})
        last = self._last_price.get(tr.instrument)
        if last is not None and tr.side * (last - new) <= 0:
            return self._reject(tr, cmd, t, t_cmd, "move_through_market",
                                f"{_px(new)} is at or through the last price {_px(last)}; EXIT NOW closes the trade",
                                {"stop": cur["price"]})
        eff_bar = math.ceil(t_cmd / 60.0) * 60.0
        own_bar = math.floor(t_cmd / 60.0) * 60.0
        # the adverse extreme the forming bar had shown when the move was made: inside
        # this minute only a NEW extreme beyond it is evidence against the moved stop
        base = tr.forming["adv"] if tr.forming is not None and tr.forming["epoch"] == own_bar else None
        fields = self._action_fields(tr, t)
        self._write("stop_move", {**self._outcome(cmd, t_cmd), "price": new, "from": cur["price"], "mode": "price",
                                  "source": "move", "eff_tick": t_cmd, "eff_bar": eff_bar, "own_bar": own_bar,
                                  "base_adv": base, **fields}, tr, t)
        tr.hist.append({"price": new, "mode": "price", "source": "move", "eff_tick": t_cmd, "eff_bar": eff_bar,
                        "own_bar": own_bar, "base_adv": base})
        return [_notice("reply", f"{tr.instrument} STOP MOVED {_px(cur['price'])} -> {_px(new)}{self._ask(cmd, t)}",
                        {"setup_id": tr.setup_id, "stop": new})]

    # ---- EXIT NOW ----

    def _exit_now(self, tr: Trade, cmd: Command, t: float, t_cmd: float) -> list[dict]:
        if tr.phase != "live":
            return self._reject(tr, cmd, t, t_cmd, "not_in_trade", "no open trade")
        if tr.exit_request is not None:
            return self._reject(tr, cmd, t, t_cmd, "exit_already_requested", "already exiting at the next price")
        self._write("command", {**self._outcome(cmd, t_cmd), "stage": "exit_requested",
                                **self._action_fields(tr, t)}, tr, t)
        tr.exit_request = {"ts": t_cmd, "seq": cmd.seq, "reason": cmd.reason}
        return [_notice("reply", f"{tr.instrument} EXIT NOW — filling at the next observed price.{self._ask(cmd, t)}")]

    # ────────────────────────────── after the session ──────────────────────────────

    def close_session(self, instrument: str, bars, now: float | None = None) -> list[dict]:
        """Once the session is over: for every filled (or skipped-and-would-have-
        filled) setup of `instrument` without one, write the `benchmarks` record —
        the ten exits and the random stop on that entry, MAE / MFE, whether and
        when T1 / T2 printed — plus, for a traded setup, what the trade would have
        done held under the human's own initial stop. `bars`: the 1-minute store
        (a frame or an `exits.Session`), through the end of that session. Returns
        the records written; a setup whose session is not over in `bars` waits."""
        t = self._now(now)
        inst = J.file_key(instrument)
        todo = [x for x in self._awaiting if x.instrument == inst and not x.benchmarked and x.fill is not None]
        if not todo:
            return []
        sess = bars if isinstance(bars, Session) else Session(bars)
        out = []
        for tr in todo:
            if tr.provisional:
                continue                                           # the closed entry bar has not been seen yet
            try:
                i0 = sess.loc(tr.fill["bar_time"])
            except KeyError:
                continue
            if not _session_over(sess, i0):
                continue
            if tr.phase == "live":                                 # the bars ran out before an exit: last close
                px = tr.last_close if tr.last_close is not None else tr.entry
                self._close(tr, px, "session_end", t,
                            tr.last_bar_epoch if tr.last_bar else tr.entry_epoch, "bar")
            rec = benchmark_record(sess, tr.fill, tr.instrument)
            rec["skipped"] = tr.phase == "skipped"
            if tr.phase != "skipped" and tr.exit is not None:
                stop0 = float(tr.exit["stop0"])
                held = replay_exit(sess, tr.side, tr.entry, i0, stop0, tr.t1, "fixed")
                x = held["exit_idx"]
                rec["human"] = {"stop0": stop0, "stop_mode": tr.exit.get("stop_mode"),
                                "held": {"exit": held["exit"], "exit_time": sess.iso(x),
                                         "exit_et": hhmm(sess.minutes[x]), "reason": held["reason"],
                                         "r_gross": held["r_gross"],
                                         "r": None if held["r_gross"] is None else
                                         net_r(tr.side, tr.entry, held["exit"], stop0,
                                               cost_points(tr.instrument, tr.entry)),
                                         "mae_r": held["mae_r"], "mfe_r": held["mfe_r"]}}
            out.append(self._write("benchmarks", rec, tr, t))
            tr.benchmarked = True
        self._awaiting = [x for x in self._awaiting if not x.benchmarked]
        return out

    # ────────────────────────────── alert text ──────────────────────────────

    @staticmethod
    def _signal_fields(ev: dict, tr: Trade) -> dict:
        """Everything section 3.2 draws, as data (the alert carries it as JSON)."""
        keys = ("kind", "instrument", "day", "setup_id", "bar_time", "et", "side", "direction", "level", "extreme",
                "conf_types", "conf_time", "sweep_bar_time", "zone", "range", "eq", "atr", "ref_price",
                "stops_indicative", "targets", "target_rule", "t1", "t2", "levels")
        out = {k: ev.get(k) for k in keys if k in ev}
        out["armed"] = tr.armed
        return out

    @staticmethod
    def _signal_text(ev: dict, tr: Trade, observe: bool = False) -> str:
        z = ev.get("zone") or {}
        st = ev.get("stops_indicative") or {}
        t1, t2 = ev.get("t1") or {}, ev.get("t2") or {}
        level = ev.get("level") or {}
        head = "OBSERVE (not armed) " if observe else ""
        what = "SIGNAL" if ev["kind"] == "signal" else "ZONE CHANGED"
        lines = [f"{head}{tr.instrument} {what} {ev.get('direction', '')} {ev.get('et', '')} ET",
                 f"swept {level.get('name')} {_px(level.get('price'))}, confirmed {'+'.join(ev.get('conf_types') or [])}",
                 f"zone {z.get('kind')} [{_px(z.get('low'))}, {_px(z.get('high'))}], ref {_px(ev.get('ref_price'))}",
                 "stops (indicative): " + ", ".join(f"{STOP_LABEL[k]} {_px(st.get(k))}" for k in STOP_LABEL if k in st),
                 f"T1 {t1.get('names', '-')} {_px(t1.get('price'))} | T2 {t2.get('names', '-')} {_px(t2.get('price'))}"
                 f" ({ev.get('target_rule')})"]
        if not observe:
            lines.append("SKIP | STOP <wick|1.0|1.5|2.0|session|price>")
        return "\n".join(lines)


def _session_over(sess: Session, i0: int) -> bool:
    """Is the fill's session finished in these bars: a minute stamped >= 15:55 of
    that day, or any bar of a later session?"""
    day = sess.day[i0]
    last = len(sess.c) - 1
    if sess.day[last] != day:
        return True
    return bool(sess.minutes[last] >= FLAT_MINUTES)

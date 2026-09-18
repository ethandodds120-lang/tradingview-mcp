"""The four human controls and the way they come in (DESIGN-tjr-human.md sections 3.3, 9.3).

    [NQ|ES] SKIP [reason]
    [NQ|ES] STOP <wick|1.0|1.5|2.0|session|price> [reason]
    [NQ|ES] MOVE STOP <price> [reason]
    [NQ|ES] EXIT NOW [reason]
    <reason>                       attaches to the last action that had none

Reason codes: noise, structure_changed, news, gut. Parsing is case-insensitive
and tolerant of extra spaces, a leading slash, a bot mention and the
instrument in any position. `parse` only reads the message; whether the control
is allowed right now is `trade.TradeManager.on_command`'s decision.

The inbound transport is Telegram's getUpdates with long polling and an offset.
It accepts commands ONLY from TELEGRAM_CHAT_ID, journals every inbound message
whether accepted or not, never raises into the runner and is bounded by the
clock the way `quantlab.alerts` is: the request runs in a daemon thread that is
joined for its budget and abandoned if it has not come back. The token is read
from the process environment at the moment of the request, used only to build
the URL handed to the transport, and never printed, logged, journaled or put in
an error message. With either variable unset nothing touches the network.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from ..alerts import CHAT_VAR, TOKEN_VAR, _flat
from ..alerts import _mask as _alerts_mask
from ..alerts import credentials, token_problem

REASONS = ("noise", "structure_changed", "news", "gut")
UNSPECIFIED = "unspecified"
ACTIONS = ("SKIP", "STOP", "MOVE_STOP", "EXIT_NOW", "REASON")
#: what STOP accepts by name -> the key of the fill's `stops`
STOP_NAMES = {"WICK": "wick", "1": "atr1.0", "1.0": "atr1.0", "1.5": "atr1.5", "2": "atr2.0", "2.0": "atr2.0",
              "SESSION": "session"}
INSTRUMENTS = ("NQ", "ES")
REASON_WINDOW_S = 600.0
USAGE = ("commands: SKIP | STOP <wick|1.0|1.5|2.0|session|price> | MOVE STOP <price> | EXIT NOW"
         "  (+ optional NQ/ES, + optional reason: noise, structure_changed, news, gut)")

LONG_POLL_S = 20              # getUpdates' own timeout: how long Telegram holds the request open
ATTEMPT_MARGIN_S = 10.0       # wall-clock budget of one request = long poll + this
BACKOFF_S = 5.0               # after a failed request, before the next one (background loop only)
MAX_TEXT = 500                # of an inbound message, as journaled

_PRICE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+(?:\.\d+)?$")


@dataclass
class Command:
    action: str | None                 # one of ACTIONS, or None when the message is not a command
    instrument: str | None = None      # NQ | ES | None (the manager resolves it)
    arg: str | float | None = None     # STOP: a key of the fill's stops ("wick", "atr1.5", ...) or a price; MOVE_STOP: a price
    reason: str = UNSPECIFIED
    error: str | None = None           # why it is not a command
    raw: str = ""
    sent_ts: float | None = None       # Telegram's own stamp of the message (epoch s), when it came that way
    seq: int | None = None             # the journal sequence number of the inbound `command` record
    update_id: int | None = None

    @property
    def ok(self) -> bool:
        return self.action is not None and self.error is None

    def as_dict(self) -> dict:
        return asdict(self)


def _price(tok: str) -> float | None:
    if not _PRICE.match(tok):
        return None
    try:
        v = float(tok.replace(",", ""))
    except ValueError:
        return None
    return v if v > 0 else None


def parse(text: str) -> Command:
    """Read one message. Never raises; an unreadable message comes back with
    `action=None` and `error` saying why."""
    raw = "" if text is None else str(text)
    cmd = Command(action=None, raw=raw[:MAX_TEXT])
    up = raw.strip().upper()
    for joined in ("MOVE_STOP", "EXIT_NOW", "STRUCTURE_CHANGED"):
        up = up.replace(joined, joined.replace("_", " "))
    toks = up.split()
    if not toks:
        cmd.error = "empty message"
        return cmd
    toks[0] = toks[0].lstrip("/").split("@")[0]                 # /skip, /skip@the_bot
    toks = [t for t in toks if t]
    # the instrument, anywhere
    rest = []
    for t in toks:
        root = t.split(":")[-1].rstrip("!").rstrip("0123456789")
        if root in INSTRUMENTS and (t == root or t.endswith("!") or ":" in t):
            if cmd.instrument and cmd.instrument != root:
                cmd.error = "two instruments in one command"
                return cmd
            cmd.instrument = root
        else:
            rest.append(t)
    toks = rest
    # the reason: the last token(s); "structure changed" arrives as two after the underscore fold
    if len(toks) >= 2 and toks[-2:] == ["STRUCTURE", "CHANGED"]:
        cmd.reason, toks = "structure_changed", toks[:-2]
    elif toks and toks[-1].lower() in REASONS:
        cmd.reason, toks = toks[-1].lower(), toks[:-1]
    if not toks:
        if cmd.reason != UNSPECIFIED:
            cmd.action = "REASON"
        else:
            cmd.error = "no command"
        return cmd
    head = toks[0]
    if head == "SKIP":
        if len(toks) > 1:
            cmd.error = f"SKIP takes nothing but a reason; got {' '.join(toks[1:])!r}"
        else:
            cmd.action = "SKIP"
        return cmd
    if head in ("EXIT", "EXITNOW"):
        tail = toks[1:]
        if tail and tail[0] == "NOW":
            tail = tail[1:]
        if tail:
            cmd.error = f"EXIT NOW takes nothing but a reason; got {' '.join(tail)!r}"
        else:
            cmd.action = "EXIT_NOW"
        return cmd
    if head in ("MOVE", "MOVESTOP"):
        tail = toks[1:]
        if head == "MOVE" and tail and tail[0] == "STOP":
            tail = tail[1:]
        if len(tail) != 1 or _price(tail[0]) is None:
            cmd.error = "MOVE STOP needs one price"
        else:
            cmd.action, cmd.arg = "MOVE_STOP", _price(tail[0])
        return cmd
    if head == "STOP":
        tail = toks[1:]
        if len(tail) == 2 and tail[1] == "ATR":                 # "STOP 1.5 ATR"
            tail = tail[:1] if tail[0] in ("1", "1.0", "1.5", "2", "2.0") else tail
        if len(tail) != 1:
            cmd.error = "STOP needs one of wick, 1.0, 1.5, 2.0, session, or a price"
            return cmd
        t = tail[0]
        for strip in ("ATR",):
            if t.startswith(strip) and t[len(strip):] in STOP_NAMES:
                t = t[len(strip):]
            elif t.endswith(strip) and t[:-len(strip)] in STOP_NAMES:
                t = t[:-len(strip)]
        if t in STOP_NAMES:
            cmd.action, cmd.arg = "STOP", STOP_NAMES[t]
        elif _price(t) is not None:
            cmd.action, cmd.arg = "STOP", _price(t)
        else:
            cmd.error = f"STOP: cannot read {tail[0]!r} as a width or a price"
        return cmd
    cmd.error = f"not a command: {' '.join(toks)[:60]!r}"
    return cmd


# ────────────────────────────── the Telegram inbound transport ──────────────────────────────

_CHAT_SHAPE = re.compile(r"-?[0-9]+|@[A-Za-z0-9_]+")


def telegram_problem() -> str | None:
    """None when Telegram can be used; otherwise why not, in words that never
    carry a value. A token or chat id with whitespace or a control character in
    it (a trailing CR from a CRLF launcher is the usual one) is MALFORMED and
    Telegram is treated as not configured: urllib would refuse the URL with an
    error message that spells the token out."""
    token, chat = os.environ.get(TOKEN_VAR), (os.environ.get(CHAT_VAR) or "").strip()
    if not (token or "").strip() or not chat:
        return f"{TOKEN_VAR} / {CHAT_VAR} not set"
    problem = token_problem()                      # quantlab.alerts' validation: one rule for both directions
    if problem is not None:
        return problem + " — Telegram is OFF"
    if not _CHAT_SHAPE.fullmatch(chat):
        return (f"{CHAT_VAR} is malformed (whitespace or a control character, or not a number): "
                "Telegram is OFF until the launcher is fixed")
    return None


def malformed() -> bool:
    """Both variables are set but one of them cannot be used (see `telegram_problem`)."""
    both = bool((os.environ.get(TOKEN_VAR) or "").strip()) and bool((os.environ.get(CHAT_VAR) or "").strip())
    return both and telegram_problem() is not None


def configured() -> bool:
    """Both variables set AND well-formed. Says nothing about their values."""
    return telegram_problem() is None


def _mask(text: str, token: str | None = None) -> str:
    """No error message leaves here carrying the token — raw, stripped or escaped."""
    return _alerts_mask(text, token)               # raw, stripped, percent-encoded, escaped — and the environment's always


class _JournalFailed(RuntimeError):
    """The inbound record of an update could not be written: the update is NOT consumed."""


def _fetch(url: str, body: dict, timeout: float) -> tuple[int, str]:
    """The transport: one HTTPS POST of a JSON body to getUpdates. Module-level so a
    test replaces it (`commands._fetch = fake`) and nothing reaches the network."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read().decode("utf-8", errors="replace")


def _bounded(url: str, body: dict, budget: float) -> tuple[int, str]:
    """One request, bounded by the clock and not by urllib (see quantlab.alerts)."""
    box: dict = {}

    def work():
        try:
            box["value"] = _fetch(url, body, budget)
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=work, name="tjr-human-getupdates", daemon=True)
    worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise TimeoutError(f"no response within {budget:.1f} s — request abandoned")
    if "error" in box:
        raise box["error"]
    return box["value"]


class TelegramInbound:
    """getUpdates, long polling, with an offset.

    `poll(timeout_s)` makes one request and returns the commands it brought —
    only those from TELEGRAM_CHAT_ID. Every message, from anyone, is journaled
    as a `command` record first (`authorized`, `parsed`), and a message from
    another chat also as a `reject` (`unauthorized_chat`); nobody else gets an
    answer. `start()` runs the same thing in a daemon thread and `drain()` hands
    over what has arrived, so a 20 s long poll never sits in the runner's loop.

    Never raises. `last_error` is one line with the token masked.
    """

    def __init__(self, journal, clock=time.time, long_poll_s: int = LONG_POLL_S):
        self.journal = journal
        self.clock = clock
        self.long_poll_s = int(long_poll_s)
        self.offset: int | None = None
        self.last_error: str | None = None
        self.error_count = 0                       # consecutive requests that failed (the runner watches this)
        self.requests = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        try:                                       # a restart must not act twice on a message it already journaled
            seen = [r.get("update_id") for r in journal.records()
                    if r.get("kind") == "command" and isinstance(r.get("update_id"), int)]
            if seen:
                self.offset = max(seen) + 1
        except Exception:
            pass

    # ---- one request --------------------------------------------------------------------------------------

    def poll(self, timeout_s: float | None = None) -> list[Command]:
        try:
            out = self._poll(self.long_poll_s if timeout_s is None else timeout_s)
        except Exception as exc:                   # belt and braces: never into the runner
            self.last_error = _flat(_mask(f"{type(exc).__name__}: {exc}"))
            out = []
        self.error_count = self.error_count + 1 if self.last_error else 0
        return out

    def _poll(self, timeout_s: float) -> list[Command]:
        problem = telegram_problem()
        if problem is not None:                    # unset, or malformed: no request is ever built from it
            self.last_error = problem
            return []
        token, chat = credentials()                # the chat id stripped, as quantlab.alerts sends to it
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        body = {"timeout": int(max(0, timeout_s)), "allowed_updates": ["message"]}
        if self.offset is not None:
            body["offset"] = self.offset
        self.requests += 1
        try:
            status, raw = _bounded(url, body, float(max(0, timeout_s)) + ATTEMPT_MARGIN_S)
            data = json.loads(raw)
            if not (200 <= status < 300) or not data.get("ok"):
                self.last_error = _flat(_mask(f"HTTP {status}: {raw[:200]}", token))
                return []
        except urllib.error.HTTPError as exc:
            self.last_error = _flat(_mask(f"HTTP {exc.code}: {exc.reason}", token))
            return []
        except Exception as exc:
            self.last_error = _flat(_mask(f"{type(exc).__name__}: {exc}", token))
            return []
        self.last_error = None
        out: list[Command] = []
        for upd in data.get("result") or []:
            try:
                cmd = self._one(upd, str(chat))
            except _JournalFailed as exc:          # not journaled = not received: asked for again, offset not moved
                self.last_error = _flat(_mask(f"journal write failed, update kept for the next request: {exc}", token))
                break
            except Exception as exc:               # a malformed update is skipped, not fatal
                self.last_error = _flat(_mask(f"update skipped: {type(exc).__name__}: {exc}", token))
                cmd = None
            if isinstance(upd, dict) and isinstance(upd.get("update_id"), int):
                self.offset = max(self.offset or 0, upd["update_id"] + 1)
            if cmd is not None:
                out.append(cmd)
        return out

    def _one(self, upd: dict, chat: str) -> Command | None:
        uid = upd.get("update_id")
        if self.offset is not None and isinstance(uid, int) and uid < self.offset:
            return None                            # already journaled by an earlier run
        msg = upd.get("message") or {}                  # an edited message is not a command: it would act twice
        text = msg.get("text")
        from_chat = str((msg.get("chat") or {}).get("id", ""))
        authorized = bool(from_chat) and from_chat == chat
        cmd = parse(text if isinstance(text, str) else "")
        cmd.update_id = uid if isinstance(uid, int) else None
        cmd.sent_ts = float(msg["date"]) if isinstance(msg.get("date"), (int, float)) else None
        now = float(self.clock())
        # Journal.write is locked: this thread and the runner's both write
        try:
            rec = self.journal.write("command", {
                "stage": "inbound", "update_id": cmd.update_id, "chat_id": from_chat or None,
                "authorized": authorized, "text": cmd.raw, "sent_ts": cmd.sent_ts,
                "parsed": {"action": cmd.action, "instrument": cmd.instrument, "arg": cmd.arg,
                           "reason": cmd.reason, "error": cmd.error}},
                instrument=cmd.instrument if authorized else None, now=now)
        except OSError as exc:
            raise _JournalFailed(type(exc).__name__) from None
        cmd.seq = rec["seq"]
        if not authorized:
            try:
                self.journal.write("reject", {"why": "unauthorized_chat", "command_seq": rec["seq"],
                                              "chat_id": from_chat or None, "action": cmd.action}, now=now)
            except OSError:
                pass                               # the command record already says authorized: false
            return None
        return cmd

    # ---- the background loop ------------------------------------------------------------------------------

    def start(self) -> bool:
        """Long-poll in a daemon thread. False (and nothing started) when Telegram is not configured."""
        if not configured() or (self._thread and self._thread.is_alive()):
            return bool(self._thread and self._thread.is_alive())
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                got = self.poll()
                for c in got:
                    self._queue.put(c)
                if self.last_error:
                    self._stop.wait(BACKOFF_S)

        self._thread = threading.Thread(target=loop, name="tjr-human-inbound", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def drain(self) -> list[Command]:
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out

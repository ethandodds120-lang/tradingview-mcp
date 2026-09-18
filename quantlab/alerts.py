"""Alerts from the box — never from a session (DESIGN-risk.md §4).

`notify` always writes one line to stderr and one to `paper_runs/alerts.log`,
in the format `book.alert` has always used plus a kind in brackets, so anything
that tails that file keeps working. When `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` are in the environment it also pushes the kinds that are
meant to be pushed — a fill, a stale run, a kill-rule trip, and the test line —
through urllib, with a 10 s timeout and one retry. A push that fails is one more
line in alerts.log; it never raises into a poll and never hangs past 20 s.

"Never past 20 s" is enforced by the clock, not by urllib: its `timeout` bounds
each socket operation, not the attempt — DNS resolution is outside it entirely
(a box whose resolvers are unreachable spends 10-30 s in getaddrinfo before the
socket timeout is consulted) and a server that drips a header every few seconds
never trips it. So each attempt runs in a worker thread that is joined for at
most its budget and abandoned if it has not come back, and the retry is skipped
when the first attempt has used the 20 s. The thread is a daemon: an abandoned
attempt cannot keep the process alive.

The token is read from the environment by the send itself and used only to
build the URL that goes to the transport. It is never logged, printed, stored,
or put in an error message: anything that mentions the URL has the token
masked first. The environment file that carries it (`/etc/quantlab/telegram.env`)
is written by the user, by hand — nothing here writes it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ALERTS_FILE = "alerts.log"
TOKEN_VAR = "TELEGRAM_BOT_TOKEN"
CHAT_VAR = "TELEGRAM_CHAT_ID"
TIMEOUT_S = 10.0          # per attempt: urllib's socket timeout AND the thread's wall-clock budget
RETRIES = 1
DEADLINE_S = 20.0         # per notify, every attempt included — the tick's "never past 20 s"
RETRY_MIN_S = 1.0         # no retry with less than this left of the deadline

#: the kinds that leave the box (§4: exactly three events, plus the test)
PUSH_KINDS = frozenset({"fill", "stale", "halt", "test"})


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configured() -> bool:
    """Both variables set and non-empty. Says nothing about their values."""
    return bool(os.environ.get(TOKEN_VAR)) and bool(os.environ.get(CHAT_VAR))


def _post(url: str, body: dict, timeout: float) -> tuple[int, str]:
    """The transport: one HTTPS POST of a JSON body. Module-level so a test can
    replace it (`alerts._post = fake`) and nothing ever reaches the network."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read().decode("utf-8", errors="replace")


def _mask(text: str, token: str | None) -> str:
    """No error message leaves here carrying the token."""
    text = str(text)
    return text.replace(token, "<token>") if token else text


def _log(base_dir: str | Path, line: str) -> None:
    print(line, file=sys.stderr)
    try:
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        with (base / ALERTS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass                                      # stderr already has it


def _text(message: str, fields: dict | None) -> str:
    if not fields:
        return message
    lines = [message] + [f"{k}: {v}" for k, v in fields.items()]
    return "\n".join(lines)


def _attempt(url: str, body: dict, budget: float) -> tuple[int, str]:
    """One attempt, bounded by the clock: `_post` runs in a daemon thread that
    is joined for `budget` seconds and abandoned if it is still going — urllib's
    timeout does not cover DNS or a server that keeps the socket just alive.
    Re-raises whatever the transport raised. `_post` is looked up when the
    thread runs, so a test's replacement is the one that is called."""
    box: dict = {}

    def work():
        try:
            box["value"] = _post(url, body, min(TIMEOUT_S, budget))
        except BaseException as exc:              # handed back to the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=work, name="alerts-post", daemon=True)
    worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise TimeoutError(f"no response within {budget:.1f} s — attempt abandoned "
                           "(DNS and a slow server are outside the socket timeout)")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _flat(message: str) -> str:
    """The log line is one line: a multi-line message (the heartbeat's "N things
    are stale:" list, an HTML error page from Telegram's edge) has its newlines
    folded to " | ", so alerts.log stays one entry per notify and anything that
    tails or greps it keeps working. The Telegram text keeps the newlines."""
    return " | ".join(part.strip() for part in str(message).splitlines() if part.strip())


def _send(text: str) -> dict:
    """POST sendMessage, one retry, never raises, never past DEADLINE_S of wall
    clock. The dict says what happened and carries no token and no URL; its
    `error` is one line (a 502 page from the edge is several — folded)."""
    token, chat_id = os.environ.get(TOKEN_VAR), os.environ.get(CHAT_VAR)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = {"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True}
    last = None
    attempts = 0
    t0 = time.monotonic()
    for attempt in range(1 + RETRIES):
        remaining = DEADLINE_S - (time.monotonic() - t0)
        if attempt and remaining < RETRY_MIN_S:
            last = f"{last} — no retry, the {DEADLINE_S:.0f} s budget is spent"
            break
        attempts += 1
        try:
            status, raw = _attempt(url, body, max(0.0, min(TIMEOUT_S, remaining)))
            ok = 200 <= status < 300
            if ok:
                try:
                    ok = bool(json.loads(raw).get("ok", True))
                except ValueError:
                    pass
            if ok:
                return {"sent": True, "status": status, "attempts": attempts,
                        "elapsed_s": round(time.monotonic() - t0, 3)}
            last = f"HTTP {status}: {_mask(raw[:200], token)}"
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            last = _mask(f"HTTP {exc.code}: {detail or exc.reason}", token)
        except Exception as exc:                  # URLError, timeout, anything
            last = _mask(f"{type(exc).__name__}: {exc}", token)
    return {"sent": False, "error": _flat(last) or "push failed", "attempts": attempts,
            "elapsed_s": round(time.monotonic() - t0, 3)}


def notify(base_dir: str | Path, kind: str, message: str, fields: dict | None = None,
           push: bool | None = None) -> dict:
    """One alert. Always logged, as exactly one line of the form
    `ALERT <utc> [<kind>] <message> [<compact json fields>]`; pushed when
    `push` (default: kind in PUSH_KINDS) and Telegram is configured. Never
    raises."""
    line = f"ALERT {_now()} [{kind}] {_flat(message)}"
    if fields:
        line += " " + json.dumps(fields, default=str, sort_keys=True)
    _log(base_dir, line)
    out = {"kind": kind, "logged": True, "pushed": False}
    if push is None:
        push = kind in PUSH_KINDS
    if not push:
        out["why_not_pushed"] = "log-only kind"
        return out
    if not configured():
        out["why_not_pushed"] = f"{TOKEN_VAR} / {CHAT_VAR} not set"
        return out
    try:
        res = _send(_text(message, fields))
    except Exception as exc:                      # belt and braces: never into a poll
        res = {"sent": False, "error": _flat(_mask(f"{type(exc).__name__}: {exc}",
                                                   os.environ.get(TOKEN_VAR)))}
    out.update(pushed=bool(res.get("sent")), telegram=res)
    if not res.get("sent"):
        # one physical line, whatever the transport said (folded again here so
        # the guarantee does not depend on _send having done it)
        _log(base_dir, f"ALERT {_now()} [telegram] push of [{kind}] failed after "
                       f"{res.get('attempts', 1 + RETRIES)} attempt(s) in "
                       f"{res.get('elapsed_s', 0.0):.1f} s: {_flat(res.get('error') or '')}")
    return out

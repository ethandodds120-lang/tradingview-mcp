# DESIGN-risk — kill rules, halt, heartbeat and push alerts (T-1 + T-4)

Contract written 2026-09-12 before the code, per the standing rules. Plain
code on every poll; no model anywhere in the risk loop. Paper only. Nothing
here is installed on the VPS until the dry-run in §7 has been shown and the
user has said go.

## 0. Why

The box runs unattended; that is the point. Two things follow. A run that is
losing in a way its own backtest said it would not must stop deciding on its
own, before a person notices — that is T-1. And the box must be the one that
says so, and says when it has gone quiet, and says when an order has filled —
that is T-4. Both are deterministic code evaluated on every poll, journaled
like everything else, and reversible only by a person.

## 1. Kill rules (T-1)

Evaluated on **every poll** of every run that is neither `STOPPED` nor
`HALTED`, after the poll has processed its bars (or, on a poll with no new
bar, after the broker mirror), on the numbers as of that poll. Any rule that
trips halts the run (§2). The evaluation and its numbers are written to
`state["risk"]` on every poll and journaled only when something changes
(a rule arms, or trips).

### 1.1 Epoch

Every rule reads the run **since its epoch**: the forward start, or the most
recent `resume` (§2.3). A resume is a person's decision to start the rules
over; without an epoch a rule that tripped on the last 60 trades would trip
again on the next poll with the same 60 trades. The epoch is recorded in
state as `{"bar": <last_bar at epoch start>, "equity": <equity then>,
"closed_trades_before": <count>}`.

### 1.2 The three rules

| rule | statistic | armed when | trips when |
|---|---|---|---|
| **R1 profit factor** | over the last **60 closed round trips** of the epoch (`PaperRun.round_trips()`, `exit != "OPEN"`): `PF = Σ pnl⁺ / Σ \|pnl⁻\|`; no losses → `PF = +∞` | 60 closed trades in the epoch | `PF < 1.0` |
| **R2 equity band** | `σ_bar` = standard deviation of the engine's per-bar net returns for this run's strategy and config over the run's **seed history** (bars with index ≤ `forward_start`, through `engine.run` with the run's `cost_bps`, `vol_target`, `vol_lookback`, `max_leverage`, `rebalance_band`, taking the bars after the vol lookback is populated). `n` = bars in the epoch; `E_0` = equity at the epoch start; `E` = equity now (state, which for routed runs is the mirrored broker equity, so intrabar). The band is **zero-drift**: `ln(E / E_0) < −2 · σ_bar · √n` | `n ≥ 20` | below the band |
| **R3 realised vol** | annualised standard deviation of the epoch's per-bar equity returns (`journal` bar records, `equity` field) over the last `vol_lookback` bars, or all epoch bars if fewer; annualised with `data.periods_per_year` of those bars' index | 20 bars in the epoch | `> 2 × vol_target`; disabled, and said so, when `vol_target` is `None` |

Why zero drift in R2: a backtest mean is the number the gauntlet exists to
distrust, and a band built on it would move against the run every quiet
week. `σ` from the backtest is the honest part of the distribution; the band
asks only "is this losing faster than that noise allows?". The multiple
testing this implies — a 2σ band checked at every `n` is crossed by a
driftless walk far more often than 2.3 % of the time — is reported, not
hidden: `paper.py risk` prints the fraction of 1000 stationary-block-bootstrap
paths of the seed returns (block 10, 250 bars) that would cross the band at
some point, as the rule's own false-trip rate.

Why arming thresholds: a volatility estimate on fewer than 20 bars is noise,
and the ticket's 60-trade window is its own threshold. Each is a deviation
from a literal reading of the ticket and is recorded in §8.

### 1.3 What a trip does

1. Writes `HALTED` in the run directory: `{"at", "rule", "value",
   "threshold", "inputs": {...every number the rule read...}, "epoch",
   "unhalt": "human — paper.py resume"}`.
2. Journals `{"type": "halted", ...the same...}`.
3. Pushes a `halt` alert (§4) with the rule and the numbers.
4. Clears `pending` and journals it as `unfilled` with `why_not: "halted"` if
   an order was planned and not yet sent.

## 2. Halt (T-4's flag, honoured like STOPPED)

### 2.1 What a halted run still does

Polls, on every tick, exactly as before **up to the decision**: refreshes the
broker, reconciles and settles anything already in flight (a leg that was
sent before the trip is real and must be recorded), fetches and stores bars,
marks equity, journals a bar record for every new bar with `"halted": true`.
It does **not** decide (`pending` stays `None`), does not route, does not send
a deferred on-open order, does not retry a failed send. The kill rules are
not re-evaluated while halted. The book still counts its exposure (a halted
run's position is real) and `tick-wants` still lists it (it must keep
polling); `book.json`'s `k` is applied to nothing of its.

The position is left where it is. Flattening is a person's decision, made at
the broker or by `paper.py stop`.

### 2.2 Manual halt

`paper.py halt --id <run> --reason "<why>"` writes the same marker with
`rule: "manual"` and journals it. Same semantics from then on.

### 2.3 Resume

`paper.py resume --id <run> --reason "<why>"` removes the marker, journals
`{"type": "resumed", "at", "reason", "halt": <the marker it removed>}`, and
starts a new epoch (§1.1) at the run's current bar and equity. Un-halting is
never automatic and never done by the tick, the heartbeat, or a chat session.

### 2.4 STOPPED and HALTED together

`STOPPED` wins: a stopped run is not polled at all. `halt` on a stopped run
is refused; `stop` on a halted run works (the marker stays as history).

## 3. Heartbeat (T-4)

`paper.py heartbeat [--dry-run]`, run by **its own timer**
(`quantlab-heartbeat.timer`, `*:2/5`, `Persistent=true`), not by the tick's
`Wants=`: a dead tick cannot report itself, so the watcher must not depend on
the thing it watches. What it checks, every five minutes:

| thing | fresh means | else |
|---|---|---|
| each run not `STOPPED` (halted runs included — they still poll) | `state.updated` no older than **2 × tick_s** (`execution.tick_s` or 300 → 600 s) | stale |
| `paper_runs/book.json` | `as_of` no older than 2 × tick_s | stale |

State in `paper_runs/heartbeat.json`: per item `{stale_since, last_alert}`.
Alerts (§4, kind `stale`) on the transition fresh → stale, then **at most
once every 6 hours** while it stays stale — a dead network does not send four
messages every five minutes. Recovery (stale → fresh) is written to
`alerts.log` and clears the state; it is **not pushed**, because the ticket
names exactly three push events. `--push-recovery` turns that push on if the
user wants it.

Limit, stated plainly: if systemd or the box itself is down, nothing on the
box can say so. An external check of the box is outside this ticket.

## 4. Push alerts — from the VPS, never from a session

`quantlab/alerts.py`: `notify(base_dir, kind, message, fields=None) -> dict`.

- **Always**: one line to stderr and one appended to `paper_runs/alerts.log`
  (`ALERT <utc> [<kind>] <message>`), which is the existing `book.alert`
  format with a kind; `book.alert` delegates here with kind `book`.
- **Telegram**, when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are in the
  environment: `POST https://api.telegram.org/bot<token>/sendMessage` with
  the message as text, via `urllib` (no new dependency), **10 s timeout, one
  retry**, and a failure is one more line in `alerts.log` — never an
  exception into a poll, never a hang past 20 s inside the tick.
- The environment comes from `EnvironmentFile=-/etc/quantlab/telegram.env`
  (mode 600, owned by the unit's user) on the poll, book and heartbeat units.
  **That file is written by the user, by hand.** No code here and no chat
  session ever reads, writes, prints or transmits the token; `paper.py alert
  --test` only reports whether the two variables are set and whether a test
  message was accepted.

Exactly three events push (the ticket's list), plus the test:

| kind | sent by | content |
|---|---|---|
| `fill` | the poll, routed runs only, when a fill is recorded | run id, side, units, fill price, the reference price, slip bps, latency s, order id — the execution fields of `DESIGN-execution.md` §4 |
| `stale` | the heartbeat | what is stale, how old, the limit |
| `halt` | the poll, at the trip | run id, rule, value, threshold, the inputs |
| `test` | `paper.py alert --test` | a fixed line with the host name and time |

The book-stale alert (`k` not applied) stays log-only: it fires on every
decision while the book is stale, and the heartbeat's `stale` on `book.json`
already carries the cause.

## 5. Files and commands

| where | what |
|---|---|
| `quantlab/risk.py` | `evaluate(run) -> dict` (the three rules, with every input), `false_trip_rate(seed_returns)`, `HALTED` marker read/write, epoch helpers |
| `quantlab/alerts.py` | `notify`, the Telegram transport, `configured() -> bool` |
| `quantlab/paper.py` | `HALTED_FILE`, `is_halted`, `halt()`, `resume()`; `poll()` honours halt per §2.1; `_poll` calls `risk.evaluate` after bar processing / mirror and halts on a trip; the `fill` push on routed fills; `book.alert` → `alerts.notify` |
| `paper.py` | `risk --id X \| --all` (read-only, loads neither feed nor broker), `halt`, `resume`, `heartbeat [--dry-run] [--push-recovery]`, `alert --test` |
| `deploy/quantlab-heartbeat.service`, `.timer` | oneshot + its own timer, `EnvironmentFile=-/etc/quantlab/telegram.env` |
| `deploy/quantlab-poll@.service`, `quantlab-book.service` | gain the same `EnvironmentFile=-` line (no other change) |
| `deploy/README.md` | §8 rewritten around the heartbeat; new §9: creating the bot, the chat id, the env file (user-side steps), `paper.py alert --test` |
| `tests/test_risk.py` | standalone runner like the others: each rule arms and trips on a constructed journal; the epoch after resume; halt honoured by `poll` on a replay-feed run (journals bars, no decision, no route); heartbeat transitions and the 6-hour limit with an injected clock; alerts always log, Telegram transport mocked (unset env → no call; set → the right payload; transport failure → logged, no raise) |

## 6. Acceptance

- A legacy run (no `execution` block) and a routed run journal **byte-identical
  bar records** when no rule trips. `state.json` gains the `risk` key and
  nothing else; `config.json` is untouched (the fingerprint is not affected).
- `poll --dry-run` still writes nothing and sends nothing; it prints what the
  rules read and whether the poll would halt.
- `risk --all` on copies of the four VPS runs completes in under 5 s and
  loads no feed and no broker.
- A run that trips halts on that poll: the `HALTED` marker, the `halted`
  journal line and the alert line all carry the same numbers, and the next
  poll journals a bar record with `halted: true` and no `order`.
- `resume` then `poll`: a new epoch, no immediate re-trip on the old numbers.
- The heartbeat on a directory with one fresh, one stale and one stopped run
  alerts once for the stale one, not again within 6 hours, and not for the
  stopped one.
- With the two Telegram variables unset, nothing tries the network.
- Both existing test files still pass; `run.py --strategy tjr --synthetic
  --quick` is unchanged.

## 7. Dry-run on the VPS's state (read-only copies) — filled by the build

## 8. Deviations from the tickets — filled by the build

- Arming thresholds (20 bars for R2 and R3) and the epoch are additions.
- R2 uses zero drift, not the backtest mean.
- The heartbeat runs on its own timer, not inside the tick.
- Recovery from stale is log-only unless `--push-recovery`.

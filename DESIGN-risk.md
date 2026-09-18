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
| **R2 equity band** | `σ_bar` = standard deviation of the engine's per-bar net returns for this run's strategy and config over the run's **seed history** (bars with index ≤ `forward_start`, through `engine.run` with the run's `cost_bps`, `vol_target`, `vol_lookback`, `max_leverage`, `rebalance_band`, taking the bars after the vol lookback is populated). `n` = bars in the epoch; `E_0` = equity at the epoch start; `E` = equity now (state, which for routed runs is the mirrored broker equity, so intrabar). The band is **zero-drift**: `ln(E / E_0) < −2 · σ_bar · √n` as first written, **−3.0 · σ_bar · √n since 2026-09-18** (amendments below) | `n ≥ 20` | below the band |
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

**Amendment, 2026-09-18, the user's decision on §8's open question — σ_bar is
floored at what the sizing itself aims for.**

```
σ_bar = max( σ_seed , vol_target / √ppy )
```

`σ_seed` is the seed standard deviation as defined above; `ppy` is
`data.periods_per_year` of the run's bar store, the figure `vol_scale` sizes
the run to `vol_target` with, so the floor is the per-bar σ of a run that is
invested and on target. No floor when `vol_target` is `None`. R2 stays
disabled where §8 says it is (no usable seed σ); the floor does not arm it on
its own. R2's inputs record `sigma_seed`, `sigma_floor` and which one the band
used. Reason, in the user's words: a kill rule that trips on healthy behaviour
is worse than none — a long-only filter that sat out much of its seed has a
seed σ well under what it runs at while invested (SOL: 0.0048 against 0.0078),
and a 2σ band built on it is nearer 1σ in the stretches that matter. The floor
cannot tighten any band. `paper.py risk` prints the false-trip rate of the band
actually used, twice: on every seed return, as before, and on the
**invested-only** seed returns, which is the honest figure for a run that is
holding a position. If that figure is above about 10 % over 250 bars the band
needs widening, not only flooring — `BAND_SIGMAS` is then the user's next
decision, with the numbers in §7.

**Second amendment, 2026-09-18 — the band is 3.0 σ, by the user's own test.**
With the floor in, SOL's chance-trip rate was 2.4 % on all seed returns and
**39.2 % on the invested-only returns** — still far above the 10 % the user
set, so by the rule just above the band is widened as well:

```
trip when  ln(E / E_0) < −3.0 · σ_bar · √n        (BAND_SIGMAS = 3.0)
```

3.0 is the first round multiple that puts every live run under 10 %, SOL
included (§7). The reason 2.0 could not: a perfectly healthy zero-drift walk
whose true σ *is* the band σ crosses a 2 σ√n band at some n between 20 and
250 **14.6 %** of the time (9.0 % at 2.25, 5.2 % at 2.5, 1.5 % at 3.0) — the
equity runs only looked fine at 2.0 because the bootstrap kept their seeds'
strongly positive drift, the very mean this rule refuses to trust. What 3.0
costs, stated plainly: for a run at 15 % vol the band is about −13 % at 20
bars, −21 % at 50, −30 % at 100 and −45 % at 250, so R2 is now a brake on a
run that is badly broken, not a detector of one that is merely not working;
R1 (profit factor) and R3 (realised vol) are what watch the rest. 2.5 is
the alternative if more sensitivity is wanted, at about 19 % chance trips
for SOL. The printed false-trip rate now counts a crossing only from the
arming bar on — the rule cannot trip before n = 20, so an earlier crossing is
not a trip.

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

## 7. Dry-run on the VPS's state (read-only copies) — 2026-09-12

Copies of the five run directories, `book.json` and `alerts.log` were taken
from the box at about 16:55 UTC (scp, read-only) and every command below ran
against the copies; the md5 of all 24 files was identical before and after.
Nothing was installed, started or written on the VPS. The build was reviewed
by three adversarial lenses with one fix round, then the remaining findings
were fixed in three further rounds, each re-verified by a regression lens and
an operations lens (§8); 28 → 36 tests in `tests/test_risk.py`; the two other
test files and `run.py --strategy tjr --synthetic --quick` unchanged; bar
records of a run that never trips byte-identical to HEAD over 600-bar replay
rehearsals (the wall-clock `at` masked).

**`paper.py risk --all`** — 1.07 s, no feed, no broker. Every rule is unarmed
on every live run (one or zero forward bars); nothing would halt.

| run | strategy | R2 `ln(E/E₀)` | band at n | σ_bar | seed returns (warm-up dropped) | R2 false-trip rate |
|---|---|---|---|---|---|---|
| gld-trend-20260912 | trend_filter | −0.00578 | −0.02011 (n=1) | 0.01005 | 339 (0) | 5.7 % |
| qqq-tsmom-20260912 | tsmom | −0.00097 | −0.02113 (n=1) | 0.01057 | 146 (193) | 9.0 % |
| slv-trend-20260912 | trend_filter | −0.00384 | −0.02116 (n=1) | 0.01058 | 338 (1) | 6.6 % |
| sol-trend-20260913 | trend_filter, routed | +0.00100 | — (n=0) | 0.00483 | 339 (0), ppy 365.25 | 26.8 % |
| sol-trend-20260906 | STOPPED | skipped | | | | |

R1 has no closed trades anywhere; R3 needs 20 epoch bars. The SOL false-trip
rate is high because its seed returns carry a negative drift and the
bootstrap resamples them as they are (§8). Before the warm-up fix (§8, first
item) the QQQ σ was 0.0069 and its band a third too tight.

**`paper.py heartbeat --dry-run`** — 0.67 s; the copies were about 65 minutes
old, so all five items read STALE against the 600 s limit and the report
showed the one combined message they would share; no `heartbeat.json`
written, `alerts.log` untouched, nothing pushed.

**A forced halt** on a throwaway replay run in the scratch directory
(`trend_filter` on `data/NQ_5min_60d.csv`, 36 forward bars: R2 armed with
σ_bar 0.000545 and band −0.00654, R3 armed at 0.100 vs 0.300). Equity was
hand-lowered to 99,000 (no threshold touched); the next poll tripped R2 at
n = 39, value −0.01005 vs −0.00681. The `HALTED` marker, the `halted` journal
line and the `[halt]` alert line carry the same numbers. The following poll
journaled three bar records with `"halted": true`, `signal`, `target` and
`order` null, and no decision. `resume` removed the marker, journaled the
marker it removed and opened a new epoch at 99,000 with two closed trades
left behind; the next poll decided again and did not re-trip.

**`paper.py alert --test`** with both variables unset: reports `NOT set`
for each, sends nothing, tries no network, exit 0.

**Also read, not written:** the local rehearsal run `paper_runs/rehearsal-tjr`
already evaluates as R2 tripped and will halt on its next poll — expected on
a replay rehearsal, not a live event.

**Re-run 2026-09-18, after the σ floor and the 3.0 band** — fresh read-only
copies taken the same morning (five or six forward bars per run; still
nothing armed, nothing would halt; md5 of the copies unchanged). 42 tests.

| run | σ seed | σ floor | σ used | chance trip, all seed returns | chance trip, invested-only | invested-only at 2.0 σ, for comparison |
|---|---|---|---|---|---|---|
| gld-trend-20260912 | 0.01005 | 0.00945 | seed | 0.1 % | 0.3 % | 7.0 % |
| qqq-tsmom-20260912 | 0.01057 | 0.00945 | seed | 0.0 % | 0.0 % | 9.0 % |
| slv-trend-20260912 | 0.01058 | 0.00945 | seed | 0.2 % | 0.4 % | 9.8 % |
| sol-trend-20260913 | 0.00483 | 0.00785 | **floor** | 0.0 % | **7.1 %** | 39.2 % (71.2 % before the floor) |

Over 250 bars, 1000 block-bootstrap paths, crossings counted from the arming
bar. SOL at 2.5 σ would be about 19 %. Reference for any honest invested
run — a driftless Gaussian walk at exactly the band σ, n from 20 to 250:
14.6 % at 2.0, 9.0 % at 2.25, 5.2 % at 2.5, 2.9 % at 2.75, 1.5 % at 3.0. With
the seed means removed the equity runs' invested-only rates at 3.0 are 5.0 %
(GLD), 1.7 % (QQQ) and 10.7 % (SLV) when counted from the first bar, lower
from the arming bar.

**To install, after the user's go** (deploy/README.md §8–§9): pull on the
box (the polls start evaluating the rules on the next tick; nothing can arm
for 20 bars), write `/etc/quantlab/telegram.env` by hand (mode 600), copy
the heartbeat unit and timer, `daemon-reload`, `enable --now
quantlab-heartbeat.timer`, run `paper.py alert --test` and `paper.py risk
--all` on the box, and read the first tick's journal lines.

## 8. Deviations from the tickets and from §1–§6, as built

Written back by the build and its reviews. None changes a number a run
journals when no rule trips.

**Decided 2026-09-18: option (b), the floor — applied; see the amendment under
§1.2 and the numbers in §7.** The question as it stood: R2's σ on a strategy
that is often flat. As first built, σ_bar is the standard deviation of every seed return
after the strategy's first position, flat bars included. A long-only filter
that sits out for much of its seed therefore gets a σ well below what it
runs at while invested, and its 2σ band is closer to 1σ in the stretches
that matter. Measured: `sol-trend-20260913` — the one routed run, holding a
position now — has a seed σ_bar of 0.00483 against the 0.00785 its own 15 %
vol target implies on a 365-day calendar, so its band is 1.6× too tight
while invested and its false-trip rate prints 26.8 %. A reviewer's intraday
`trend_filter` replay tripped R2 on the first armed bar on a −0.50 % move
for the same reason (σ_bar 0.0004). The three daily equity runs are
unaffected (σ_bar 0.0101–0.0106 against 0.0094 implied). Options: **(a)**
keep the ticket's literal "backtest-implied" σ and accept early halts on
often-flat strategies; **(b)** floor σ_bar at `vol_target / √ppy`, the
per-bar σ the sizing itself aims at — recommended: it is one line, it cannot
tighten any band, and it leaves the three equity runs' numbers unchanged;
**(c)** an invested-bars-only σ. Not changed here: it is the rule's
definition, and nothing is installed until the user has seen it.

**Rules**

- R2's σ_bar is the standard deviation of the engine's seed returns from
  `max(vol_lookback, the strategy's first non-zero position)`: a strategy's
  own warm-up contributes exact zeros that are not its noise (QQQ tsmom: 193
  of 339 seed bars). Flat bars *after* the strategy has started are kept
  (the band counts every epoch bar); for a run that is continuously
  invested the invested-only σ is 1.2–1.7× larger and the printed false-trip
  rate is a lower bound in those stretches. R2 is disabled with a note when
  fewer than `max(vol_lookback, 20)` returns survive the warm-up, when the
  strategy never holds a position over the seed, when σ is zero, or for a
  panel strategy; its inputs record the seed length, the warm-up dropped and
  the first position bar.
- R3 is annualised with `periods_per_year` of the run's whole bar store —
  the figure `vol_scale` sizes to `vol_target` with — not of the epoch
  window, which misreads bars-in-window as bars-per-day on intraday runs.
  The first epoch bar's return is measured from the epoch's own equity.
- R1 with no losing trades stores `null` with `no_losses: true` (Infinity is
  not JSON); the trip logic uses the raw float.
- `false_trip_rate` bootstraps the seed returns as they are, drift included.
- Evaluation point on a new-bar poll: after the newest bar's record is
  journaled and before its order is routed. Catch-up bars are not evaluated
  one by one. On a poll with no new bar the deferred on-open send and the
  route retry run first, then the broker mirror, then the rules; only R2 on
  intrabar equity can trip there. The bootstrap poll evaluates nothing.
- The trip poll's own bar record is a normal record carrying the decision;
  the pending it produced is written off as `unfilled` with `why_not:
  "halted"` unless nothing was ever sent for it (then it is simply cleared);
  the next poll's record is the one with `halted: true`.
- An exception inside `risk.evaluate` fails **closed**: the error is journaled
  when its text changes (not every poll), printed as an ALERT, and after
  three consecutive failures the run halts with rule `risk_error`, pushed
  like any trip; the count lives in `state["risk"]`. A malformed journal
  line (a crash mid-write) is skipped with one ALERT rather than raised.
- R2's `E` on a routed run is the whole paper account, as §1.2 says — so one
  routed run per account and symbol (today: SOL only).
- The σ floor (§1.2) uses `periods_per_year` of the whole bar store, the same
  figure R3 uses. It follows the calendar inference: one skipped day in a
  7-day store reads as a 252-day calendar for a while and steps the floor up
  about 20 % — which only widens the band. `sigma_source` is `None` where R2
  has no usable seed σ; the floor never arms the rule. The `risk` journal
  line written when a rule arms carries `sigma_seed`, `sigma_floor` and
  `sigma_source`, and on a floored run a different `sigma_bar` and threshold
  than before; bar records are untouched. Deploying the floor or a new band
  multiple onto a run that is already armed changes its band with no journal
  line of its own — not the case for any live run today.
- The invested-only false-trip rate splices the seed's invested bars into one
  series, resamples it as if the run were invested for all 250 bars, keeps
  the seed's drift, and leaves out exit bars (entry costs in, exit costs
  out). It is an upper-ish bound with the precision of a few hundred bars.

**Halt and resume**

- `resume` is the only writer of `risk_epoch`. The HALTED marker and the
  epoch are re-read from disk immediately before every evaluation, and every
  save from any process other than the resuming one adopts the epoch (and the
  cleared `risk`) found on disk, so a `resume` issued while a poll is in
  flight, or while a `paper.py run` loop holds the run — a first resume or a
  later one — is honoured rather than clobbered. One window is left open and
  recorded: a resume landing inside a single evaluation, between its sync and
  the save of a trip. Run `resume` between ticks all the same. The mirror
  image is recorded and not closed: a manual `halt` issued from another
  process while a poll in flight or a `run` loop holds a pending order
  journals that order as unfilled twice, once by each process. Run `halt`
  between ticks too.
- A manual halt is journaled and logged, not pushed (three push events); its
  marker carries an extra `reason`. `resume` on a STOPPED run is refused. A
  HALTED marker present at the very first poll seeds history with no
  decision. The default epoch is derived (forward start, start equity, zero
  trades) and never stored; `risk_epoch` appears in state only after a
  resume. A trade open at the moment of resume is counted in the new epoch
  when it closes, with its whole P&L.

**Heartbeat**

- Runs on its own `Persistent` timer at `*:2/5`; `After=quantlab-book.service`
  as ordering only (no `Wants=`), so a replay on boot reads the state the
  tick just wrote and the heartbeat still fires when the tick's timer is
  gone. The logic lives in `quantlab/risk.py`.
- One combined `stale` message per pass for every item that is due;
  `heartbeat.json` is written before the push; `last_alert` is stamped only
  when the alert was delivered (pushed, or log-only because Telegram is not
  configured). A failed push sets a 15-minute `retry_not_before` on the item;
  a delivered push starts the 6-hour clock. Both clocks are read with 30 s of
  slack, because the timer fires with up to 5 s of random delay against
  whole-second stamps and a pass a few seconds early would otherwise wait a
  whole extra pass.
- What the `After=` ordering costs: a poll still running delays the watcher
  (never cancels it) by at most the poll's and the book's own timeouts; and on
  boot the ordering is best effort, the worst case being one `stale` push
  followed by a recovery line.

**Alerts**

- The 20 s bound is enforced by the clock: each attempt runs in a worker
  thread joined for its budget and abandoned if it has not returned (DNS and
  a dripping server are outside urllib's socket timeout); the retry is
  skipped when the budget is spent. An abandoned attempt can still deliver
  late, in which case the next pass may duplicate it on Telegram — accepted.
- `alerts.log` is strictly one line per `notify`: `ALERT <utc> [<kind>]
  <message>` with newlines flattened to ` | ` and `fields` as one compact
  JSON object; the `[telegram] push failed` line that follows a failed push
  is one line too, whatever the transport returned. The Telegram text keeps
  the newlines and is capped at 4000 characters. Kinds outside `{fill, stale, halt, test}` are log-only
  regardless of configuration (`book`, `recovered`).
- The `fill` push is not unit-tested (it needs a routed fill); it goes
  through `notify`, which never raises and is bounded.
- The token must look like a bot token exactly as it stands — digits, a
  colon, then the secret; no space, no quote, no trailing CR from an editor
  that saved CRLF. Anything else counts as not configured, with one
  `alerts.log` line saying it is malformed and never the value, so a
  malformed token can no longer reach a URL or an error text. `paper.py
  alert --test` says `set but UNUSABLE - malformed` for it. Masking covers
  the raw, stripped, percent-encoded and escaped forms.
- The pushed set in code is wider than T-4's three since 2026-09-18:
  `signal`, `trade`, `reply` and `status` belong to the `tjr_human` package
  (`DESIGN-tjr-human.md` §9.3). Nothing in `paper.py` or `risk.py` uses them.

**From the contract as first written**

- Arming thresholds (20 bars for R2 and R3) and the epoch are additions.
- R2 uses zero drift, not the backtest mean.
- The heartbeat runs on its own timer, not inside the tick.
- Recovery from stale is log-only unless `--push-recovery`.
- §6's byte-identity claim masks the bar record's `at` (a wall clock).
- `poll --dry-run` on a *replay* feed advances the replay cursor (pre-existing
  rehearsal behaviour), so its numbers are not comparable to the real poll
  that follows; on the csv and live feeds they are.

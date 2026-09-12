# Execution and book allocator — the contract

Paper only. This is the specification the code is written to; if code and this
file disagree, the file is wrong and must be fixed first.

## 0. Why

The first routed fill on sol-trend-20260906 paid 114.8 bps against the price the
backtest assumed: 83.9 bps of that was delay, 30.7 slippage. The delay was not
poll cadence. A decision is made at the close of bar t and marked "fills at next
bar open", but the run only sees bar t+1 after it has *closed* — the feed drops
the newest bar unconditionally as "forming" — so the order goes out roughly a
full bar late, benchmarked to an open that is already history. Every routed
order in the old model has this shape. This document fixes it, adds a portfolio
governor, and makes every fill measurable.

## 1. Feed: closed-bar rule (unconditional, all runs)

`Feed.bars()` must drop the newest bar only if it is still forming. A bar stamped
T on timeframe Δ is closed at `closed_at(T)`:

| asset class | daily bar closed at |
|---|---|
| crypto | T + 24h (UTC) |
| US equity | 16:00 America/New_York on T's calendar date, in UTC |
| intraday, any class | T + Δ |

For crypto this is a no-op relative to today (the partial current-day bar is
dropped, yesterday's kept). For equities it removes a one-session lag: Friday's
bar is complete at 20:00 UTC Friday and visible to the 20:05 poll, instead of
appearing only when Monday's bar exists. Half-days are not modelled (flag).
Feeds other than Alpaca keep `drop_last` as-is.

The equity close is a wall-clock time, not an offset: build 16:00 on the naive
New York date and localize it, never add 16h to a tz-aware midnight. On a DST
transition day the two differ by an hour (2026-03-08 closes 20:00 UTC,
2026-11-01 closes 21:00 UTC; the offset form gives 21:00 and 20:00).

## 2. Run config: `execution` block (new runs; legacy runs unchanged)

Top-level key in config.json, written by `paper.py start`. NOT fingerprinted
(`_FINGERPRINTED` is a fixed tuple; adding a name there would ParamDrift every
existing run). Absent ⇒ legacy behaviour byte-for-byte.

```json
"execution": {
  "route_at": "decision",          // "decision" (new) | "next_close" (legacy)
  "style": "marketable_limit",     // crypto: marketable_limit | market
                                   // equity: market_on_open | market
  "limit_bps": 5,
  "limit_wait_s": 60,
  "fallback": "market",
  "book": true                     // apply the book multiplier
}
```

Defaults at `start`: `route_at=decision`; `style=marketable_limit` if the broker
symbol is crypto else `market_on_open`; `book=true`. Flags: `--route-at`,
`--exec-style`, `--limit-bps`, `--limit-wait`, `--no-book`.

## 3. Routing at decision (`route_at == "decision"`, routed runs only)

In `_decide`, once `pending` is set for bar t (and only if `move`):

1. `arrival = broker.quote()` → `{bid, ask, mid, at, source}`; source is
   `"quote"` or `"trade"` (fallback to last trade) or `"unavailable"`.
2. Build `client_order_id = f"{run_id}:{bar_t:%Y%m%dT%H%M}:{leg}"` (≤48 chars;
   run_id sanitised to `[A-Za-z0-9-]`).
3. Send now:
   - crypto: marketable limit at `mid*(1 ± limit_bps/1e4)` (buy +, sell −),
     TIF GTC, wait `limit_wait_s`; then **cancel and confirm** (re-query until
     status ∈ {canceled, filled, expired, rejected}; never send the fallback
     while status is `pending_cancel`); if remaining > 0 send market for the
     remainder (`fallback`). Fill = VWAP across legs.
   - equity: market-on-open, `TimeInForce.OPG`, whole units (Alpaca fractional
     orders are DAY-only; `_whole_units_only` must return True for OPG). The
     poll does not wait for the fill. **Not sent at decision time**: Alpaca
     rejects an OPG that arrives between 09:28 and 19:00 New York, and the
     decision poll for an equity runs at ~16:05. The decision poll *plans* the
     legs (`broker._plan` / `_client_id` / `_leg_qty`, no send), journals
     `{"type": "order_planned", ...}` and persists `state["inflight"]` with
     `deferred: true`, legs `status: "deferred"`, and `send_not_before` /
     `send_not_after` from `broker.opg_window()` (19:00 NY on the day before the
     next open; next open − 2 min). A later poll sends it — see §5. `place()`
     re-sizes off the mark at send time, against the same target and prefix.
4. Journal `{"type": "order_sent", ...}` immediately with arrival fields,
   client_order_ids, style, limit_price, then persist `state["inflight"]` =
   the same payload **before** waiting on anything (crash safety). Before any
   of this — before `quote()` — the bar record is journaled **and state is
   saved** with `pending` set, so a failure inside the routing step cannot get
   the bar decided and journaled twice. That failure is journaled as
   `{"type": "error", "stage": "route", "bar", "error"}`, the reason is kept on
   `pending.route_error`, and `inflight` is left unset (or, if the intent was
   already persisted, marked with `error`); the next poll recovers per §5.
   `order_sent` is written only when a leg actually went out.
5. Accounting stays where it is: `_process_bar` step 1 for bar t+1 consumes
   `state["inflight"]` (already filled, or resolves it via §5) instead of
   calling `rebalance()` at t+1's close. The `fill` record is written there, so
   it can include `bar_price` (t+1's open, now known) and every bps field.

Legacy (`next_close`): unchanged — `rebalance()` at t+1 close with a market
order. sol-trend-20260906 stays on this path until stopped.

## 4. Execution fields on every routed fill

`side_sign = +1 buy / −1 sell`. Every bps = `side_sign * (later − earlier) / earlier * 1e4`
so positive always means the run paid more (buy) or received less (sell).

| key | meaning |
|---|---|
| `bar_price` | open of the fill bar — what the backtest assumed (existing) |
| `arrival_mid`, `arrival_bid`, `arrival_ask`, `arrival_at`, `arrival_source` | NBBO at the instant pending was set |
| `arrival_spread_bps` | (ask − bid)/mid |
| `poll_price` | mark when the first leg was sent (== legacy `ref_price`; both keys kept, `poll_price` is canonical) |
| `sent_at`, `lag_s` | wall clock of first send; `sent_at − arrival_at` |
| `fill_price` | VWAP across legs (existing) |
| `delay_bps` | poll_price vs bar_price |
| `arrival_drift_bps` | poll_price vs arrival_mid |
| `slippage_bps` | fill_price vs poll_price |
| `total_bps` | fill_price vs bar_price |
| `assumed_bps` | cost_bps / 2 |
| `excess_bps` | total_bps − assumed_bps |
| `order_style` | market / marketable_limit / limit+market / market_on_open / simulated |
| `legs` | `[{client_order_id, order_id, type, limit_price, status, filled_qty, avg_price, latency_s, cancelled}]` |

Constraints (from `fills_frame`/`round_trips`): never put keys named `bar`,
`equity`, `held` inside the fill dict; keep `units`, `fill_price`,
`commission`, `ref_price` numeric. New record types (`order_sent`,
`order_planned`, `error`, `execution_backfill`, `book`, `stopped`,
`execution_change`) must not carry a dict under `fill` or `unfilled`.

`PaperRun.execution_frame()` = routed fills with these columns.
`paper.py report --execution --id X` prints count, mean/median of delay,
slippage, total, excess, arrival spread, and lag, plus per-style breakdown.

## 5. Resume (client_order_id)

At the start of every poll, if `state["inflight"]` exists: for each leg,
`get_order_by_client_id`. filled ⇒ record; open ⇒ keep waiting (crypto) or
leave until the open (OPG); canceled/expired/rejected ⇒ journal `unfilled`
with `why_not` **and save state at once**, so a feed failure later in the
poll cannot leave the journal and state disagreeing. Never re-send a leg whose
client_order_id already exists.

An `unfilled` written here carries `bar` = the **decided** bar (the fill bar
has not arrived) and `fill_bar: null`; one written at the fill bar (step 1 of
`_process_bar`) carries `bar` = the fill bar and `fill_bar` = the same. Both
carry `decided_at`. `unfilled_frame()` documents this.

Legs empty but the intent persisted (`routed: true`, no legs — the poll died
inside `place()`): look the legs up by prefix. Found ⇒ adopt. Not found ⇒
drop the intent, keep `pending` with `route_error`, and — only while the fill
bar has still not closed (the poll fetched and found no new bar) — route the
decision again, after one more lookup by client id (`_retry_route`). Never a
blind re-send; a decision whose fill bar has closed is settled as `unfilled`
with the failure as `why_not`. `pending` with no `inflight` at all (the send
failed before the intent was saved) takes the same path.

Deferred market-on-open (§3.3): untouched by the top-of-poll reconcile. Once
the poll has fetched and found the fill bar still ahead, `_send_deferred`:
`now < send_not_before` ⇒ wait; `send_not_before ≤ now < send_not_after` ⇒
`broker.place(style="market_on_open")` with the planned prefix, journal
`order_sent`; `now ≥ send_not_after` with nothing sent (no poll ran) ⇒ when
`market_open()`, `place(style="market")` with `late: true` on the inflight,
the `order_sent` line and the eventual fill; while closed, keep waiting. If
the fill bar closes with the plan still unsent it is settled as `unfilled`
("planned but never sent", window quoted). The decision poll itself sends
immediately when the window is already open. A poll that dies inside the send
finds its legs by client id on the next one (`_send_leg` adopts an existing
client_order_id).

## 6. Book governor

`paper.py book [--dir D] [--write]` — default is read-only.

Per run i (skips `STOPPED` runs and runs without a symbol — the symbol is
`broker.symbol`, else `feed.symbol`, else `feed.ticker` for yahoo runs):
- `E_i = state.equity`; `raw_i = state["target_raw"]` if present else
  (`pending.target` if pending else held). A run with no `execution` block
  never writes `target_raw`; its `raw_source` is `legacy:pending` or
  `legacy:held` so it cannot be mistaken for a new run that has not decided.
- `w_i = (E_i / ΣE) · raw_i`.
- Vol on the native calendar: `D_i = std(last 60 native daily returns from
  bars.csv) · sqrt(ppy_i)`, `ppy_i` from `book.native_ppy(index)`: 365.25 when
  the index is daily and the last 28 bars have no gap of 2 days or more (a
  seven-day crypto calendar), otherwise `periods_per_year(index)`. The
  engine's `periods_per_year` alone says 252 for every daily index, which
  understates seven-day vol by sqrt(365.25/252) ≈ 1.204.
- Correlation `R` from inner-joined single-day returns: compute each run's
  daily returns on its own calendar first, then inner-join on normalised date,
  last 60 joined rows.
- `Σ = D R D`, `book_vol = sqrt(wᵀ Σ w)`, `k_uncapped = 0.15 / book_vol`,
  `k = min(1.0, k_uncapped)`. Governor only.
- `after_vol = k · book_vol` (valid because k applies to all runs).

`--write` atomically writes `paper_runs/book.json`:
`{version:1, as_of, tick_s:300, vol_target, cap:1.0, k, k_uncapped,
book_vol, after_vol, applies_to:[...], rows:[...], cov:{basis, window, D, ppy,
R, symbols, runs}}` and appends the same document as one line to
`paper_runs/book.jsonl` (the tick history the user asked for). Each row also
carries its `ppy`, and the table prints a `ppy` column plus the `D` vector
under the correlation matrix, so the basis of `book_vol` is on the page.

Run side (`_decide`, right after `target = sig * scale`, before the band):
`target_raw = target`; `k, meta = _book_multiplier()`; `target = target_raw * k`.
State gets `target_raw`, `book_k`. The bar record gets `target_raw`, `book_k`,
`book_k_uncapped`, `book_as_of`, `book_stale` (bool), `book_reason`.

Freshness: if `book.json` is missing, unreadable, or `as_of` older than
`2 · tick_s`, or the run is not in `applies_to`: `k = 1.0`, `book_stale = true`,
`book_reason` set, and an ALERT line is printed to stderr and appended to
`paper_runs/alerts.log`. Never silent.

### Scope: what the governor does not do

The allocator governs **total book volatility only**. It scales every run by the
same k, so it never changes the *shape* of the book — it does not reduce
concentration and it does not know that two runs are the same bet. Today GLD and
SLV correlate at 0.90 and together are ~24% of the book; after the governor they
are still, effectively, one position held twice. Redundancy and concentration
limits are out of scope for this change and belong to a later allocator that can
re-weight across runs, not just scale them.

## 7. State on every poll

`_save_state()` at the end of every poll for every run, including simulated
runs with no new bar. `updated` becomes a real heartbeat. On a run routed at
decision it is also saved right after the bar record is journaled and
`pending` set (before any routing I/O), right after `inflight` is persisted,
and right after a reconcile writes `unfilled` (§3.4, §5).

## 8. Stop

`paper.py stop --id X --reason "..."` writes `paper_runs/X/STOPPED`
(`{at, reason}`), journals `{"type":"stopped"}`, and exits. `poll` on a stopped
run prints one line and exits 0. `book` and the tick generator skip it. The
journal, bars and state are kept untouched.

## 9. systemd: one tick transaction

- `quantlab-tick.timer`: `OnCalendar=*:0/5`, `Persistent=true`,
  `RandomizedDelaySec=5`, `Unit=quantlab-tick.service`.
- `quantlab-tick.service`: `Type=oneshot`, `ExecStart=/bin/true`, plus a
  generated drop-in `/etc/systemd/system/quantlab-tick.service.d/wants.conf`
  containing `Wants=quantlab-poll@<id>.service …` for every non-stopped run and
  `quantlab-book.service`.
- `quantlab-poll@.service` gains `Before=quantlab-book.service`,
  `TimeoutStartSec=600` (a two-leg crypto flip is 2 × (60 limit + 30 cancel
  confirm + 90 market + 30 confirm) ≈ 420 s, so a poll can outlive one 300 s
  tick; systemd merges the next start job into the running one — no double
  run, one lost tick — and `Before=` makes the book wait), `TimeoutStopSec=150`
  (> limit_wait 60 + cancel confirmation).
- Run ids are systemd instance names: `paper.py start` refuses an `--id` (or a
  generated default) that does not match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`,
  and `tick-wants` leaves any run dir that fails it out of `Wants=` with a
  warning on stderr.
- `quantlab-book.service`: oneshot, `paper.py book --write`,
  `After=network-online.target`.
- `deploy/sync-tick.sh`: runs `paper.py tick-wants` to regenerate the drop-in,
  `daemon-reload`. Adding a run = `paper.py start … && deploy/sync-tick.sh`.
- Per-run timers are disabled and removed by the migration. Poll services stay
  templated.
- SIGTERM during a wait: broker cancels the open leg(s), poll saves state
  (try/finally), exits non-zero. Resume (§5) picks up on the next tick.
  `poll()` installs a SIGTERM handler for its own duration that raises
  `SystemExit(143)`, so a stop anywhere in the poll — not only inside a broker
  wait — takes the same save path; the broker's handler cancels first and
  then defers to it.

## 10. Backfill of the SOL fill

Append (never edit) one `execution_backfill` record to
sol-trend-20260906/journal.jsonl, computed from journal line 5 and bars.csv:
`bar_price 103.75`, `arrival_source "backfill:decision_close"` with
`arrival_mid` = close of the decided bar (2026-09-07) from bars.csv,
`arrival_at` = the instant that bar **closed** (`feed.closed_at`: the
2026-09-07 00:00 bar closes `2026-09-08T00:00:00+00:00`), not the wall clock
of the poll that processed it — that goes under `processed_at`
(`2026-09-09T06:50:43+00:00`), `poll_price 104.62` (last trade at send,
labelled `"backfill:last_trade"`), `fill_price 104.941064`, `sent_at` =
record `at` − `latency_s` (`2026-09-09T06:50:43.84+00:00`), `lag_s` =
`sent_at − arrival_at` = 111043.84, and every bps field per §4: delay 83.9,
slippage 30.7, total 114.8, assumed 25.0, excess 89.8. `report --execution`
includes it.

## 11. Dry-run

`paper.py poll --id X --dry-run` fetches bars, applies §1, processes new bars in
memory, prints decisions and the orders it *would* send — for a market-on-open
that is the planned legs with `deferred: true` and the `send_not_before` /
`send_not_after` window — and writes nothing:
no bars.csv append, no journal, no state, no orders. `book` without `--write`
is already read-only. `paper.py backfill-execution --id X` prints the record;
`--append` is required to write it.

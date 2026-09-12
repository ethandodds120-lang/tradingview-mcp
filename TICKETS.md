# Tickets

Work that is agreed, sequenced, and not started. One entry per item; close by
deleting it when the commit that fixes it lands. Order matters and is stated.

## Sequence after the 2026-09-12 deploy has run quietly

1. **T-1 kill rules** (plain code, no LLM in the loop)
2. **T-3 bootstrap ignores the rebalance band** — fix here, before any new
   routed runs are started in number
3. **T-2 incubation gate**
4. **T-4 heartbeat, halt flag, and push alerts from the VPS**
5. strategy search

---

## T-1 — Deterministic kill rules

Halt a run and alert when any of these trips. Plain code, evaluated on every
poll, no model anywhere in the risk loop:

- rolling 60-trade profit factor < 1.0
- equity below a 2-sigma band on the backtest-implied distribution
- realised vol > 2x the run's target

Halt means: stop deciding, stop routing, keep polling and journaling, write a
`HALTED` marker with the rule and the numbers that tripped it. Un-halting is a
human action, never automatic.

## T-2 — Incubation gate

A run cannot route until N forward bars have accumulated with no gauntlet
regression. Until then it runs on the simulated ledger regardless of what
`--broker` says, and the report says so. N and "regression" to be defined
against the gauntlet's own gates.

## T-3 — Bootstrap ignores the rebalance band

**Status:** open. **Severity:** every new run pays one unnecessary trade.
**Order:** after T-1, before the strategy search.

**Observed** — sol-trend-20260913, started 2026-09-12 05:22 UTC adopting the
account's 325.49 SOL/USD:

```
adopted position          33.6% of equity
bootstrap target          27.9%   (365-calendar sizing)
difference                 5.7%   < rebalance_band 0.10
what it did               sold 53.975 SOL at 101.404 on the bootstrap poll
what it should have done  held — the move is inside the band
```

**Cause** — `PaperRun._bootstrap` calls `_decide(store, ts)` without `held`, so
the band test compares the target against 0.0 and always sets a pending. That
was harmless when a pending only became an order at the next bar close; under
`route_at == "decision"` the pending is routed in the same poll, so the
missing band check is now a live order. Simulated runs journal a phantom trade
for the same reason.

**Compounding effect** — a run started mid-bar benchmarks that first order
against a bar open that is hours old. This morning's was 5 h 22 m after the
close and about 100 bps of drift. Steady state is under 5 minutes; the first
order never is.

**Fix** — pass the adopted position into the bootstrap decision: for a routed
run `held = broker.fraction(price)`, for a simulated run the ledger's
`(equity - cash) / equity`, and let the existing band test decide whether a
pending is set at all. Gate it on the `execution` block so legacy runs keep
their bootstrap semantics byte for byte.

**Acceptance**
- a new routed run adopting a position inside the band journals `bootstrap`
  with no pending and sends nothing
- the same run adopting a position outside the band sends exactly one order
- a simulated run with no position behaves as today
- a legacy run (no `execution` key) is byte-identical to today
- the six-truncation causality check still passes for every registry strategy

## T-4 — Heartbeat, halt flag, push alerts from the VPS

The box runs unattended; that is the point. So the box does the telling.

- **Heartbeat** — `state.updated` already moves on every poll of every run.
  Add one process on the box that reads them and knows what "stale" means per
  run (a stopped run may be stale; a live one may not be older than two ticks).
- **Halt flag** — the `HALTED` marker from T-1, honoured by `poll` the way
  `STOPPED` is, plus a manual `paper.py halt` / `paper.py resume`.
- **Push alerts FROM the VPS**, Telegram or email, no interactive session
  involved, on exactly three events:
  1. a fill (routed runs; include the execution fields)
  2. a run gone stale
  3. a kill rule tripping (T-1), with the rule and the numbers

Alerts are sent by the tick or by the heartbeat process, never by anything
that needs a person or a chat session to be open. Rate-limit the stale alert
so a dead network does not send four messages every five minutes.

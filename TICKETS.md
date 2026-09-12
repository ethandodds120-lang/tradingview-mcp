# Tickets

Work that is agreed, sequenced, and not started. One entry per item; close by
deleting it when the commit that fixes it lands. Order matters and is stated.

## Sequence (re-ordered 2026-09-12, after the TJR intraday round 2 closes)

1. **T-1 kill rules** together with **T-4 heartbeat, halt flag, push alerts**
   (plain code, no LLM in the loop)
2. **T-7 indicator + human-exit experiment** — build only after T-1 and T-4
   are live; its Part 1 (trailing benchmark) and Part 2 (wide funnel) are
   analysis and run now
3. **T-3 bootstrap ignores the rebalance band** — before any batch of new
   routed runs
4. **T-2 incubation gate** (carried; not re-confirmed in the latest ordering)
5. **T-5 Turtle / Donchian on a diversified basket** (from the traders report)
6. **T-6 ORB Stocks-in-Play** (from the traders report)
7. **T-8 POC / HVN standalone test** — whenever there is budget; cheap

**TJR intraday, mechanical: closed 2026-09-12.** Two pre-specified rounds;
verdict in `DESIGN-tjr-intraday.md` §12 (best of 192 trials deflates to 0.41;
every 5-minute run fails every edge gate; the 1-minute data runs to 30
sessions). No round three on the rules. T-7 is not a third round: it measures
the discretionary exit layer the mechanical test could not, as an experiment
on the human, with its own pre-registration in `DESIGN-tjr-human.md`.

## T-7 — Indicator + human-exit experiment (`tjr_human`)

**Why.** Two round-1 numbers point somewhere the rules cannot go: T1 printed
later on 4 of 8, and the median favourable excursion *after* the fixed stop
fired was about 3 R (raised as "5 of 8, 3.4 R"; corrected by Part 1 —
`DESIGN-tjr-intraday.md` §9). Round 2 showed a wider initial stop makes total
R worse. The only
version that could harvest that excursion is the one TJR actually trades:
signals mechanical, exit human. This measures that layer instead of assuming
it.

**What.** A paper run built to the wider spec (`DESIGN-tjr-human.md` §2):
the bot finds and enters every signal; the human may only skip a signal
before it fills, tighten the stop, or exit at market. Everything is journaled
by the bot. Pre-registered sample of 100 filled trades, pass bar fixed before
trade one — `DESIGN-tjr-human.md` §5.

**Gates.** Part 1 first: if a mechanical trailing rule harvests the excursion
on its own, deploy the rule and skip the human. Part 2 next: the funnel on
the wider spec, no gauntlet. Build after T-1 and T-4 are live — the run needs
the kill rules and the push channel, and its human controls ride the same
Telegram bot. Never goes live on the box before a dry run is shown.

**Open cost item.** The signal layer runs on live ES/NQ 1-minute and 5-minute
bars. The VPS has no futures feed (Alpaca serves none). Options and their
reliability are in `DESIGN-tjr-human.md` §7; the paid live feed is the same
line item T-5 flags.

## T-8 — POC / HVN standalone test

Separate from TJR. Session point of control and fixed 20-session composite
high-volume nodes as a standalone level strategy on ES/NQ 5-minute bars,
tagged `folklore`: first touch of a level from above/below, entry on a
5-minute close back away from it, 1 ATR stop, target the next level. Full
gauntlet at real futures costs; every grid cell counted as a trial in the
cumulative count. Says whether the level type carries anything on its own,
independent of the TJR structure. Uses the volume-profile primitives Part 2
of T-7 adds.

## T-5 — Turtle / Donchian on a diversified basket

The published trend-following result is a *diversified futures* result
(Moskowitz-Ooi-Pedersen: 58 contracts across equity indices, currencies,
commodities, bonds). `donchian` exists in the registry as a single-instrument
strategy; the basket version is a panel strategy (`kind="panel"`, book-level
vol targeting) over a universe wide enough to mean something.

**Data flag, to be costed when we get there:** a diversified futures basket
needs a paid CME (and ideally ICE/Eurex) history feed — Databento or Norgate
are the usual retail-priced options. TradingView's continuous contracts cover
the CME index futures we have used so far but not a 20–50 contract universe
across asset classes, and Yahoo's futures history is 60 days at intraday and
inconsistent at daily. An ETF proxy basket (sector, bond, commodity, currency
ETFs from Alpaca) is the free alternative and is what `xs_momentum` already
runs on; it is a different instrument set with different costs and no
overnight session, and the report should say which was used.

## T-6 — ORB Stocks-in-Play

Opening-range breakout on stocks selected for relative volume ("in play"),
from the traders report. Needs: a daily universe scan (relative volume at the
open vs a trailing baseline), 1- or 5-minute equity bars from Alpaca (free tier
is IEX; fine for liquid names, thin for the tail), and the panel machinery
because the selection is cross-sectional. Costs are equity costs (near zero
commission, spread plus slippage); the interesting failure mode is fills at
the open on names that are in play *because* they are gapping.

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

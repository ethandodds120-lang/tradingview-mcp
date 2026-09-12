# Blueprint — multi-strategy, multi-asset paper trading

The goal is infrastructure: many strategies, several asset classes, running
unattended, all on paper. Strategy quality is a separate question the gauntlet
already answers. This document is only about the scaffolding.

## What already exists — do not rebuild it

| capability | where | state |
|---|---|---|
| many independent runs | `quantlab/paper.py` `PaperRun` | **done.** Each run is a directory with its own `config.json`, `state.json`, `bars.csv`, `journal.jsonl`. Four coexist today. |
| pluggable bar sources | `quantlab/feeds.py` `build()` | **done.** csv, yahoo, tradingview, alpaca, replay. |
| pluggable execution | `quantlab/broker.py` `build()` | **partly.** The dispatch exists; there is exactly one real adapter. |
| strategy registry + families | `quantlab/strategies/` | **done.** |
| validation gauntlet | `quantlab/validate.py` | **done.** |
| unattended execution | `deploy/` + systemd timer | **done for one run.** Needs templating for N. |

Two runs with different feeds and different brokers already coexist on disk. The
data model is not the bottleneck.

## The actual gap: asset class coverage

Alpaca trades US equities, ETFs, options and crypto. It does **not** trade futures
or forex — `NQ1!` returns `asset not found for NQ1!` from the live API, and there
is no forex endpoint at all.

| asset class | bar feed | execution | status |
|---|---|---|---|
| crypto | Alpaca | Alpaca | **works today** |
| US stocks / ETFs | Alpaca | Alpaca | **works today** — never yet run |
| futures | — | — | needs a new feed + broker adapter |
| forex | — | — | needs a new feed + broker adapter |

So the blueprint is mostly: *write two more broker adapters, and one more feed
each.* Everything upstream of the adapter already works.

## The broker adapter contract

This is smaller than it looks. `quantlab/paper.py` only ever touches seven things
on a broker:

```
attributes   cash, units, account_equity
methods      equity(price), fraction(price), rebalance(ref_price, target_fraction), refresh()
```

Plus two used at run creation and for display:

```
prepare() -> dict      validate credentials/symbol, return config metadata
market_open() -> bool  is the venue accepting orders right now
describe() -> str      one line for the banner
```

`rebalance()` is the only hard one. It takes a *target fraction of equity*, works
out the delta in units, sends the order, waits for the fill, and returns a fill
record — or `{"routed": False, "why_not": ...}` if the venue refused. Read
`AlpacaBroker.rebalance` before writing a second one; the awkward cases it already
handles (flip-through-flat needing two orders, fees taken in kind, fractional vs
whole units, market closed) are the same everywhere.

Register a new adapter by adding one branch to `broker.build()`.

### Candidate venues

Worth verifying before committing — I have not tested these APIs:

- **Forex** — OANDA has a REST API and a free practice account, and is the least
  painful option. IBKR also does forex.
- **Futures** — Interactive Brokers, Tradovate, or AMP. IBKR has a paper account.
- **One venue for all three** — IBKR covers stocks, futures, forex and options
  under one API, which is architecturally tidy. The cost is that its API needs
  TWS or IB Gateway running as a desktop process with a socket connection, which
  is meaningfully harder to keep alive on a headless VPS than a REST endpoint.

If the aim is a working blueprint fastest: **stocks on Alpaca** (already possible,
zero new code), then **forex on OANDA** (clean REST), then **futures on IBKR**
(hardest, do last).

## The other gap: every strategy here is single-instrument

`Strategy` is `(df, **params) -> position Series` for one asset, and `engine.run`
backtests one position series against one price series. That shuts out an entire
family: cross-sectional momentum, pairs, ETF flow reversal, anything that ranks a
universe and holds many names at once. You cannot express "rank 20 ETFs, long the
top quintile, short the bottom" in the current contract at all.

This was scoped with a working prototype on a real 20-ETF, 1000-bar panel, not
estimated. Findings:

**`engine.run_panel` is about 35 lines** and can return the same `BacktestResult`
the package already consumes. Every rule generalises cleanly:

| single-asset | panel |
|---|---|
| `held = desired.shift(1)` | `held = weights.shift(1)` — same lag, whole row |
| cost on `\|position.diff()\|` | cost on L1 turnover of the row |
| vol target off asset vol | vol target off the **book's** realised vol |

That last row matters beyond this feature: scaling each leg to its own vol target
is exactly the stacking bug `paper.py portfolio` exists to report. A panel engine
has to size the book, which is the right behaviour everywhere.

**The three newest gates work on panel output unchanged.** Verified, not assumed —
`deflated_sharpe`, `probability_of_backtest_overfitting` and
`drawdown_distribution` only ever see a return series, so they neither know nor
care how many instruments produced it.

**`causality_check` generalises for free.** Its diff works on a DataFrame signal
as-is; the prototype's cross-sectional momentum returned a max weight difference
of 0.00e+00 across three truncations.

**The coupling is five lines.** `engine.run(...)` is called at exactly five places
in `validate.py` — twice in `walk_forward`, once each in `trial_matrix`,
`random_benchmark` and `cost_sweep`. Give `Strategy` a `kind` field and a
`backtest()` method that dispatches to `run` or `run_panel`, replace those five
call sites, and four of the five validators become panel-capable with no other
change. Date-based slicing already works, because a wide price panel has dates as
rows.

### What actually needs new code

1. **`engine.run_panel`** — written and tested in the prototype.
2. **A panel random benchmark.** `random_entry` returns one Series; the panel
   version needs random weight rows with matched turnover. This is the only
   genuinely new logic, maybe 20 lines.
3. **`metrics.summary`'s trade count is wrong for panels.** It counts sign flips of
   `position`, but a panel's `position` is gross exposure and never goes negative.
   On the prototype it reported **1 trade for a strategy that rebalanced 47 times**.
   Needs a turnover-based count when the result came from a panel.
4. **`run.py` needs a different benchmark row.** `buy_hold` on "the strategy's
   instrument" is meaningless when there are twenty; use an equal-weight universe.

Roughly 100 lines of new code plus mechanical edits. Worth doing before adding a
third broker adapter — it unlocks more strategy families than futures access does,
and it needs no new venue, credentials, or money.

### Prototype result, for calibration

Cross-sectional momentum on 20 liquid ETFs, 2bps per side, 2022-2026:

```
  Sharpe 0.43   CAGR 5.08%   MaxDD -19.75%   t 0.86
  DSR 0.964 over 6 trials     PBO 6%
```

That DSR and PBO are better than anything currently in the registry — it is the
first thing tested here to clear the deflation gate. Do not get excited: t is 0.86
on four years of data, which is not evidence of anything. It is a reason to build
the panel engine properly and test the family honestly, not a reason to trade it.

## Infrastructure changes for N runs

### 1. One systemd unit for all runs

The current unit hardcodes the run id. Replace it with a template so one file
serves every run — see `deploy/quantlab-poll@.service`:

```bash
systemctl enable --now quantlab-poll@sol-trend-20260906.timer
systemctl enable --now quantlab-poll@qqq-tsmom-20260912.timer
```

### 2. Poll cadence per asset class

Cadence should track the bar size, not the asset. The rule that matters: the order
is a market order sent whenever the poll runs, while the backtest assumed a fill at
the bar open, so **everything between those two moments is pure delay cost**. On
the first live fill at hourly cadence that was 84 bps, against a strategy with zero
headroom at 50.

| bar size | poll every | why |
|---|---|---|
| 1D | 5 min | cheap; caps delay near the close |
| 1H | 2-5 min | same reasoning, tighter |
| 5-15 min | 1 min | delay starts to dominate |

Market hours only change whether a poll can *fill*, not whether it should run.
`require_open` already handles that; crypto ignores it.

### 3. Account isolation — decide this before adding the second routed run

A run computes its equity as **cash + the value of its own symbol**. Two routed
runs sharing one brokerage account therefore both see the same cash and neither
sees the other's position, so both compute the wrong equity and sized positions
drift from what was tested. They also compete for the same buying power.

The Alpaca broker already warns about this:

> `account already holds SOLUSD — this run's equity is computed from cash + this symbol only, so those positions will make it disagree with the account`

Three options, in order of preference:

1. **One brokerage account per routed run.** Cleanest, and what the code assumes.
   Check how many paper accounts each venue allows.
2. **One run per venue.** Stocks on Alpaca, forex on OANDA, futures on IBKR — the
   isolation falls out for free until you want two stock strategies.
3. **Build a portfolio allocator** that owns the account and divides equity between
   strategies. This does not exist and is a real piece of work. Do not drift into
   it by accident.

### 4. Monitoring across N runs

The failure mode is silence. Per run, check `state.json`'s `updated` field; alert
if it is older than a few times the poll interval. `deploy/README.md` has the
one-liner. Wrap it in a loop over `paper_runs/*` once there is more than one.

## Suggested order of work

1. **Stand up a second run on stocks.** Zero new code — Alpaca already does
   equities. This proves multi-run on the VPS and forces the account-isolation
   decision while it is still cheap.
2. **Template the systemd unit.** Small, and blocks everything after it.
3. **Add the OANDA adapter for forex.** First real test of whether the broker
   contract is actually general or accidentally Alpaca-shaped.
4. **Add futures last.** Hardest venue, and the contract will be better understood
   by then.

## What not to let slide

Paper trading makes losses free. It does not make these free:

- **Every routed run still needs a strategy that passed the gauntlet.** Not because
  the money is real, but because infrastructure tested against a strategy that
  never fires, or fires constantly, will not be tested at all.
- **Do not enable `--live` on anything** while the blueprint is being built. The
  gate exists precisely so that flipping it has to be deliberate.
- **One machine polls a given run.** Two pollers means two divergent state files
  and duplicate orders.

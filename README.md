# quantlab

A quantitative strategy research harness. Backtest engine, nine strategies across
two explicitly named families, and a four-part validation gauntlet designed to kill
strategies that only *look* good.

There is no live execution layer. Forward tests can route their orders to an Alpaca
paper account, which is a better fill model, not live trading — see "Paper trading"
and "Why no bot yet" below.

```
run.py                  CLI entry point — the gauntlet
paper.py                CLI entry point — forward testing (simulated or routed fills)
quantlab/data.py        CSV/TradingView loaders, synthetic random walk, annualization
quantlab/engine.py      backtest core: cost model, vol targeting, one-bar lag
quantlab/metrics.py     Sharpe, drawdown, t-stat, summary
quantlab/validate.py    causality, walk-forward, sensitivity, random benchmark, cost sweep
quantlab/feeds.py       bar sources: TradingView chart, yfinance, csv, replay
quantlab/paper.py       forward-test runner: append-only decision log, paper account
quantlab/broker.py      optional order routing to an Alpaca paper account
quantlab/tjr.py         shim — the model moved into strategies/predictive/

quantlab/strategies/
  __init__.py           Strategy dataclass, metadata, REGISTRY, family helpers
  predictive/
    primitives.py       shared pattern vocabulary — swings, sweeps, BOS, gaps
    tjr.py              sweep -> BOS -> FVG state machine (path-dependent)
    fvg.py              the three-candle imbalance on its own
  systematic/
    momentum.py         tsmom, ma_cross, donchian, trend_filter
    meanrev.py          rsi_meanrev
  benchmarks.py         buy_hold, random_entry
```

## Install

```bash
pip install pandas numpy
# optional, for --yahoo
pip install yfinance
```

## Run

```bash
# the null hypothesis: a random walk with no structure
python run.py --strategy tsmom --synthetic

# all strategies side by side
python run.py --compare --synthetic

# your own data — TradingView chart export works directly
python run.py --strategy tjr --csv NQ_5min.csv --cost-bps 4

# real daily data
python run.py --strategy tsmom --yahoo SPY --start 2005-01-01

# one family only
python run.py --family predictive --csv NQ_5min.csv

# the two families against each other on the same data
python run.py --head-to-head --csv NQ_5min.csv
```

Flags: `--cost-bps` (round trip), `--vol-target`, `--folds`, `--trials`, `--quick`,
`--family`, `--head-to-head`.

## What's in it

**Engine** (`engine.py`) — vectorized, deliberately pessimistic:
- Signals lagged one full bar. You cannot trade on a bar you haven't finished.
- Costs charged on every unit of position change, commission and slippage separately.
- Volatility targeting from trailing data only, with a no-trade band so you aren't
  charged for micro-rebalances.

**Strategies** (`quantlab/strategies/`) — split into two families that make
different kinds of claim:

**Predictive** — a chart pattern or market structure tells you what price does
next. A liquidity sweep implies a reversal; a fair value gap implies a
retracement. The claim is about a *specific setup*, and underneath it is a story
about intent: someone's stops were taken, someone has an order to defend. This is
discretionary logic made mechanical, which is what makes it worth testing — the
discretionary version cannot be falsified and this version can.

| name | evidence | what it is |
|---|---|---|
| `tjr` | folklore | ICT sweep -> break of structure -> FVG entry, with stops and targets |
| `fvg` | folklore | fair value gaps, same logic as the Pine indicator |

**Systematic** — no view on any individual setup. These harvest a statistical
property measured across the whole dataset: return autocorrelation, volatility
clustering, drift. There is no narrative about why price *should* move, and on any
given bar the rule is as likely to be wrong as right; the claim was only ever
about an average.

| name | evidence | what it is |
|---|---|---|
| `tsmom` | published | time-series momentum, the Moskowitz/Ooi/Pedersen rule — your baseline |
| `ma_cross` | published | moving average crossover |
| `donchian` | published | turtle-style channel breakout |
| `trend_filter` | published | long while price is above its own MA — a drawdown overlay |
| `rsi_meanrev` | folklore | short-term mean reversion — included to show how costs kill things |

**Benchmark** — neither family. What both have to beat.

| name | evidence | what it is |
|---|---|---|
| `buy_hold` | published | the opportunity cost you actually have to beat |
| `random_entry` | untested | coin flips at your trade frequency — the null hypothesis |

Both families make predictions, and both can be wrong. The distinction is what the
claim rests on: **a pattern implying intent, versus a statistical property
persisting.** A systematic strategy is not "not predicting" — a momentum rule says
next period's return is related to the last one, which is a prediction and a
falsifiable one.

`evidence` is the honest state of what is known about the effect, not a rating of
how well it backtests. `folklore` is not an insult; it means *untested here*, which
is the entire reason this harness exists. Published pedigree is a reason to take a
prior seriously and not a reason to skip the gauntlet — an effect documented on
equity indices from 1965 is not thereby true of SOL/USD on daily bars.

`python run.py --head-to-head` runs both families on the same data and compares
them on out-of-sample Sharpe, gate pass rates, trade count and **cost drag**. Cost
drag is the column to read: predictive strategies trade setups rather than
averages, so they trade far more often and need a proportionally larger gross edge
to finish level. On the synthetic random walk they take ~2x the trades per
strategy and pay away ~15% of equity in costs against ~6% for systematic.

**Predictive primitives** (`strategies/predictive/primitives.py`) — the pattern
vocabulary is shared rather than copy-pasted: `confirmed_swings`, `detect_sweep`,
`structure_level`, `detect_bos`, `find_fvg`, plus `is_displacement` and
`in_killzone` for models not yet written. Adding order blocks or turtle soup should
be composition over that vocabulary, not another copy of the pivot loop.

`tjr` is path-dependent — where it enters, where it stops out and where it takes
profit all depend on the order things happened in, so it cannot be written as a
vectorized signal. It keeps its own module and exposes `simulate()` (trades +
position series) and `signal()` (adapter for the engine). Any model with stops and
targets belongs in a module like that rather than as a plain signal function.

The pivot handling is the part worth reading, and it now lives in
`confirmed_swings`. A swing high at bar `p` is not knowable at bar `p` — it needs
`swing_right` bars to its right before it is confirmed, so it only becomes
admissible at bar `p + swing_right`, which is what the `confirmed_at` tag records.
Detecting pivots with a centered window instead is the single most common way an
ICT backtest produces results that cannot be traded.

**Validation** (`validate.py`) — the point of the whole thing:

0. **Causality.** Recompute the signal on truncated history. If a strategy only
   uses bars up to `t`, its signal on `df[:k]` must match its signal on the full
   frame across every overlapping bar. A mismatch means information from after the
   cut leaked backwards, and every number after it is fiction. This catches
   `.shift(-1)`, centered rolling windows, and pivots marked at the bar they
   printed rather than the bar they were confirmed.
1. **Walk-forward.** Parameters are chosen on in-sample data, then applied unchanged
   to data the optimizer never saw. Only out-of-sample returns count. This is the
   difference between a backtest and evidence.
2. **Parameter sensitivity.** A real effect is a plateau — neighbouring parameters
   behave similarly. A lone spike surrounded by garbage is a coincidence you found
   by looking hard enough.
3. **Random-entry benchmark.** Hundreds of coin-flip strategies with your trade
   frequency. If you can't beat the 95th percentile, your rules haven't been
   distinguished from randomness.
4. **Cost sweep.** At what commission does the edge die? If the answer is near where
   you actually trade, you don't have an edge, you have a rounding error.

## Reading the output

Start with `--synthetic`. That data is a random walk with mild positive drift and
**no predictable structure whatsoever**. It is your calibration.

Two things to notice there:

- `fvg` shows a gross Sharpe near zero and a *negative* net Sharpe. That's costs
  eating a non-existent edge across 600+ trades. This is what a fake strategy looks like.
- `tsmom` scores positively and beats 99% of random entries — but compare it to
  `buy_hold`, not to the random benchmark. The random entries go long and short
  50/50, so they lose to the drift automatically. Trend following on drifting data
  is partly just *being long*. **Always read the random benchmark next to buy_hold.**

That second one is a real trap and the harness is built to expose it rather than
hide it.

`tjr` gives you a third calibration number. On the random walk it produces a
walk-forward OOS Sharpe around 0.3 from ~30 trades — on data with no structure at
all. That is the level a real-data result has to clear before it means anything,
and it is also a reminder that 30 trades is not a sample. Read the trade count
before the Sharpe.

## Adding your own strategy

Put it in the module for its family — `systematic/momentum.py`,
`systematic/meanrev.py`, or a new module under `predictive/` — then register it:

```python
def my_signal(df, threshold=1.5):
    # return desired position at each bar: -1 short, 0 flat, +1 long
    # you may use df up to and including row t. never .shift(-n).
    z = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
    return -np.sign(z) * (z.abs() > threshold)

REGISTRY["my_strat"] = Strategy(
    "my_strat", my_signal,
    {"threshold": 1.5},
    {"threshold": [1.0, 1.5, 2.0, 2.5]},
    family="systematic",
    thesis="Short-horizon returns revert, so a 2-sigma stretch is partly given back.",
    evidence="folklore",
    source="",
)
```

`family` and `evidence` are required and validated — construction raises if either
is outside its allowed set, so a strategy cannot reach the registry without
declaring what kind of claim it is making. Fill `thesis` in as one sentence stating
what has to be *true* for the strategy to work, not what it does mechanically; if
that sentence is hard to write, that is information.

Be honest about `evidence`. Marking your own idea `published` because it resembles
something published defeats the purpose of the field.

The grid is what walk-forward optimizes over and what sensitivity sweeps. Give
every tunable parameter a grid or the validation is measuring nothing.

Predictive strategies should be assembled from `predictive/primitives.py` rather
than re-deriving swing detection. If you need a primitive that isn't there, add it
there — and if it changes an existing strategy's numbers, that is a separate
change with its own before/after, not a refactor.

Models with stops and targets go in their own module like `tjr.py`, exposing
`simulate()` and a `signal()` adapter. One rule those must respect: do not flatten
the position on the final bar of the input. That value never reaches the engine
anyway (it lags by one, and there is no next bar), and flattening it makes the
signal on a truncated history differ from the signal on the full history, which is
exactly what the causality check reads as a leak.

## Paper trading

`paper.py` watches bars arrive, runs the same strategy code the backtester runs, and
writes down what it would have done at the moment it would have done it. By default
the fill is simulated in-process and nothing leaves the machine.

```bash
# forward test against the live TradingView chart
python paper.py start --strategy tjr --tv-symbol NQ1! --tv-timeframe 5
python paper.py run --id tjr-NQ1-20260905 --interval 60     # or `poll` from cron
python paper.py report --id tjr-NQ1-20260905   # equity, position, trade summary
python paper.py trades --id tjr-NQ1-20260905   # flat-to-flat round trips with P&L
python paper.py fills  --id tjr-NQ1-20260905   # every fill: price, size, commission
python paper.py journal --id tjr-NQ1-20260905  # every bar, fill or not
python paper.py list

# rehearse the machinery on a stored file — NOT a forward test
python paper.py start --strategy tjr --replay NQ_5min.csv --id rehearsal
python paper.py run --id rehearsal --interval 0 --max-polls 3200
```

### Routing the orders to a broker

The same decisions can be sent to an Alpaca paper account instead of being filled
by the simulation. Put `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in `.env`, then:

```bash
python paper.py broker --symbol QQQ          # what the account looks like, reads only

# bars off the NQ1! chart, orders for QQQ
python paper.py start --strategy tjr --tv-symbol NQ1! --tv-timeframe 5 \
    --broker alpaca --broker-symbol QQQ --allow-short
python paper.py run --id tjr-NQ1-20260905 --interval 60
python paper.py unfilled --id tjr-NQ1-20260905   # orders the broker would not fill
```

The bar feed and the broker symbol are separate on purpose: the chart you develop
on is often not something you can trade. Because they are different instruments,
the broker sizes and marks off the traded symbol's own price, and records the bar's
price beside it for the audit trail.

This changes exactly one thing — where the fill price comes from. It is worth doing
for two reasons: the realized slippage stops being the `--cost-bps` guess and starts
being a measurement (`report` prints one against the other), and orders that get
rejected, clamped or missed are recorded as missing trades instead of being filled
for free by a simulation that always says yes. It is **not** live trading. An Alpaca
paper account is their simulator: no queue position, no market impact, no borrow.

Guardrails, because this is the part that can lose money if it is wrong:

- `--live` is required to touch a real account, and the base URL is checked against it.
- A symbol the account cannot trade, or cannot short while `--allow-short` is set,
  fails at `start` — not at 3am on the first signal.
- Market orders are not sent while the market is closed (`--queue-when-closed` to
  override). A market order sent Saturday fills at Monday's open on a Friday decision.
- When a poll delivers several bars at once the run is catching up, and the opens
  those older decisions would have filled at are gone. Only the newest bar routes;
  the rest are journaled as unfilled.
- The account is the authority on its own money: `start_equity` comes from the real
  balance, and cash, position and mark are read back from the broker every poll.
- The broker spec is part of the run fingerprint, so it cannot be swapped mid-run.

Feeds: `--tv-symbol` (the live chart, read-only, via this repo's `tv` CLI),
`--yahoo`, `--csv` (a file something else appends to), `--replay` (rehearsal).

A forward test is worth more than a backtest for exactly one reason: the data did
not exist when the strategy did. Three rules protect that, and they are the whole
point of the module.

1. **Bars that already existed at run creation are history.** They seed the
   lookback and are never counted as paper trades. You cannot start a run today and
   claim last month's bars as forward results.
2. **Parameters are fingerprinted at creation and checked on every poll.** Edit the
   params of a live run and it refuses to continue and tells you to start a new one.
   Retuning after seeing forward results turns the forward test back into a
   backtest, and it happens by accident unless something stops it.
3. **The forming bar is dropped.** Decisions are made on closed bars only, and the
   order fills at the *open of the next bar* — not the close of the current one.

That last point is also where paper trading is stricter than the backtester. The
vectorized engine fills close-to-close; the paper account fills at the next open
plus slippage and commission. `report()` prints both over the same bars, so the gap
between the rows tells you what the fill assumption was worth. On a rehearsal of
`tjr` over `NQ_5min.csv` the two came out within five basis points of each other,
which says the bar-by-bar loop and the vectorized engine agree — not that either
one is right about real fills.

Everything is on disk under `paper_runs/<id>/`: `config.json` (frozen), `bars.csv`
(append-only), `journal.jsonl` (append-only, one record per bar), `state.json`
(rebuildable). Kill the process whenever; it resumes where it stopped.

`journal.jsonl` is the raw record and every view above is derived from it — one JSON
object per bar with the signal, the target position, the fill if there was one, and
the equity mark. `trades` collapses fills into flat-to-flat round trips; because
positions are vol-targeted and resized while open, a trade is not one buy and one
sell, so P&L is accumulated from the cash side of the ledger and reconciles with the
account to the cent.

Read the trade count before the equity curve, and give it months. Two weeks of
forward data on an intraday strategy is a few dozen trades, which is noise wearing
a P&L curve as a costume.

## Why no bot that trades

Order routing is a weekend of work — `quantlab/broker.py` is that weekend, and it
does not make this a trading bot. Knowing whether a signal is real is the entire
job, and it's the part that decides whether the bot makes or loses money. Routing
to a paper account improves the fill model and nothing else: it cannot tell you the
edge is real, and it is easy to mistake a working order path for a working strategy.

The published research is blunt about this. McLean and Pontiff found that
peer-reviewed, statistically validated return predictors decline about 26%
out-of-sample and 58% post-publication. Those are the *good* ideas — the ones that
survived academic refereeing. Whatever survives your own search will be weaker than
that, not stronger.

So the sequence is: pass the gauntlet, then paper trade for months with `paper.py`,
then talk about capital. If a strategy passes all four tests and holds up on forward
data you collected after building it, that's the point to point `--live` at it — and
the guardrails in `broker.py` are there to make that a decision rather than a typo.

Note also that the TradingView MCP bridge explicitly excludes automated trading in
its terms; live execution would go through a broker API, not through the chart.

## Honest limitations

- Close-to-close fills. Real fills are worse, especially on gaps and at the open.
- Single instrument. Real trend following gets most of its Sharpe from diversifying
  across dozens of uncorrelated markets — a single-asset test understates it.
- No borrow costs, no margin interest, no overnight financing, no tax.
- Survivorship bias if you feed it a ticker that exists today because it did well.

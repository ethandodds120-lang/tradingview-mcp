# TJR intraday — his rules, made mechanical (ES / NQ, 5-minute bars)

Tagged `folklore`. This is the session-based model TJR (Trades by TJR) teaches,
written as rules a program can follow with no discretion left in them. Where his
definitions are precise they are used verbatim. Where he leaves a choice to the
trader, the choice is stated here, and if it plausibly matters it is in the grid
so the search cost is priced in. This is not the existing `tjr` strategy, which
has no sessions and no clock; that one is untouched.

The daily-bar version of the sweep → BOS → FVG sequence did not survive the
gauntlet. This asks whether the intraday version, with his time-of-day rules,
does. The honest prior, from the literature in `RESEARCH-strategies.md`: no —
no peer-reviewed test of any ICT component exists, and the best independent
mechanical backtest found nothing significant.

## 0. Data contract

- 5-minute OHLCV, index tz-naive **UTC** (the codebase convention). Converted to
  `America/New_York` inside the strategy with `zoneinfo`; DST-correct.
- Bar stamp = bar **open** time (yfinance and TradingView both do this). A bar
  stamped 09:45 covers 09:45–09:50.
- The full CME session is expected: 18:00 ET → 17:00 ET, ~276 bars a day. If the
  frame has no bar stamped 09:30 ET on any day (daily data, an equity RTH file),
  the strategy returns an all-flat series and never raises — it has nothing to
  do, and `--compare --synthetic` must keep working.
- **SMT variant** needs the other index. `data.load_futures_pair(traded_csv,
  pair_csv)` returns the traded OHLCV plus `pair_open/high/low/close`, inner-
  joined on timestamp. `run.py --pair-csv PATH` uses it. `Bars.from_frame`
  ignores the extra columns; the strategy reads them directly.
- Sources, two of them:
  - yfinance `ES=F` / `NQ=F`, 5m, 60 days (Yahoo's intraday limit) →
    `data/ES_5min_60d.csv`, `data/NQ_5min_60d.csv`. 50 sessions with a 09:30
    bar. The first pass ran on these.
  - TradingView `CME_MINI:ES1!` / `NQ1!`, 5m, paged back with
    `scripts/dump_bars.js` until the feed ran out → `data/ES_5min_tv.csv`,
    `data/NQ_5min_tv.csv`: 2026-02-15 → 2026-09-11, ~41k bars, **149 sessions**
    with a 09:30 bar. Checked against the Yahoo file on their 13,665-bar
    overlap: 98.6% of closes identical, no timezone offset, no data holes (the
    only multi-hour gaps are weekends and CME holidays). Unadjusted continuous
    contract: the quarterly roll shows as a jump on the Sunday-evening open
    (161 bps on 2026-06-14), which distorts that one day's PDH/PDL.
  The gauntlet's verdict is the one on the TradingView data. 149 sessions is
  still **the binding constraint on this test**: one setup a day at most, so
  every gate is scored on a few dozen trades, and every number it produces
  must be read against that.

## 1. Clock (everything below is ET)

| name | bars stamped | notes |
|---|---|---|
| trading day D | 18:00 (D−1) … 16:55 (D) | the CME session that *ends* on D |
| Asia | 18:00 (D−1) … 02:55 | ends 03:00 |
| London | 03:00 … 08:25 | ends 08:30 |
| sweep window | 09:30, 09:35, 09:40, 09:45 | "09:30–09:50" |
| entry window | 09:50, 09:55, 10:00, 10:05 | "09:50–10:10" |
| flat-by | the 15:55 bar's close | no overnight holds |

## 2. Liquidity — six levels per trading day, all known before 09:30

| level | definition |
|---|---|
| `ASIA_H`, `ASIA_L` | high / low of the Asia bars of day D |
| `LON_H`, `LON_L` | high / low of the London bars of day D |
| `PDH`, `PDL` | high / low of the **previous trading day's full session**, 18:00 (D−2) … 16:55 (D−1) |

PDH/PDL is the prior daily candle as a futures chart draws it. The RTH-only
alternative (09:30–16:00) is a different level; it is not in the grid and not
implemented. Every level is final before the window that uses it (Asia at 03:00,
London at 08:30, PD at 17:00 the day before), which is what makes them causal.

## 3. Swings

`confirmed_swings(bars, left=2, right=2)` over the frame. At bar i only swings
with `confirmed_at <= i` exist. The lookahead guard in that primitive is the
whole reason it is used here rather than a rolling max.

## 4. The setup, one per trading day, first valid one wins

**1. Sweep** — a bar in the sweep window whose wick goes through a level and whose
close comes back inside:

- long bias: `low < L` and `close > L` for some `L ∈ {ASIA_L, LON_L, PDL}`.
  Sweep extreme = that bar's low. If the bar breached more than one low, the
  swept level is the **lowest** breached (most liquidity taken).
- short bias: the mirror on `{ASIA_H, LON_H, PDH}`.
- a bar that sweeps both a high and a low is ignored (ambiguous).
- the first qualifying bar in the window fixes the direction for the day.

**2. SMT filter** (variant only) — at the sweep bar the *pair* must **not** have
swept its own level of the same kind: for a long, `pair_low >= pair_L` where
`pair_L` is the pair's Asia low / London low / PDL matching the swept one. If
the pair also swept, there is no setup that day. Divergence is the tell that the
sweep was a manipulation rather than a breakdown; agreement says it was a real
move.

**3. BOS** — after the sweep bar and no later than the last entry-window bar
(10:05), a bar **closes** beyond the most recent confirmed opposing swing that
existed at the sweep bar: for a long, the last admissible swing high with
`index <= sweep_idx`. `detect_bos(bars, i, level, dir)`. No admissible swing →
no setup.

**4. Dealing range and equilibrium** — range = [sweep extreme, BOS level].
`EQ` = midpoint. A long may only enter in **discount**: the entry zone's top must
be `<= EQ`. Shorts mirror (premium). This is his rule, not a filter, so it is
fixed on.

**5. Entry zone** — one of:

- **FVG**: a bullish three-candle gap (`find_fvg(bars, k, +1, min_fvg_atr)`) that
  completed on the displacement leg, `sweep_idx + 2 <= k <= bos_idx`, or later
  inside the entry window. Freshest wins.
- **iFVG** (if `allow_ifvg`): a **bearish** FVG that completed at any bar of day
  D up to the BOS bar, which a later bar `j >= sweep_idx` closes **above**
  (`close > gap.high`). At `j` it inverts and its `[low, high]` becomes a long
  entry zone. Most recent inversion wins.

Both must satisfy the EQ rule. If both kinds qualify, the more recently formed
or inverted zone is used.

**6. Fill** — on an entry-window bar strictly after both the BOS bar and the
zone's bar: for a long, if `bar.low <= zone.high` the entry is
`min(zone.high, bar.open)` and the position opens on that bar. If that same bar's
low is also at or below the stop, the trade is stopped on the entry bar
(the tjr rule). No fill by 10:05 → the setup expires, no trade.

**7. Stop** — sweep extreme − `stop_buffer_atr × ATR(atr_len)` for a long. Risk
must be positive.

**8. Target** — the nearest liquidity level **above** the entry among
`{ASIA_H, LON_H, PDH}` (strictly above entry). None above → no trade.
Mirror for shorts. There is no R-multiple target: his target is the next pool.

**9. Exit** — checked in this order on every bar after entry: stop
(`low <= stop`), target (`high >= target`), the 15:55 bar (close at its close,
reason `flat`). On the final bar of the input the position is **held**, not
flattened — the eod rule from `tjr.py`, which is what lets the causality check
compare the last bar.

**10.** One trade per trading day. After the trade, or after the entry window
expires, nothing else happens until the next day's sweep window.

## 5. Position series

As `tjr`: `pos[i] = side` from the entry bar through the bar before the exit
bar. The engine's `shift(1)` is what makes bar i's position tradable on i+1.

## 6. Parameters

```
swing_left=2, swing_right=2, atr_len=14,
stop_buffer_atr=0.25, min_fvg_atr=0.0, allow_ifvg=True,
smt=False, flat_at="15:55"
```

Grid — deliberately small, every parameter raises the deflation bar:

```
stop_buffer_atr: [0.0, 0.25]
min_fvg_atr:     [0.0, 0.25]
allow_ifvg:      [True, False]        -> N = 8
```

Cumulative trial count for the deflated Sharpe: 8 × 2 variants × 2 instruments
= **32**, plus the run reported. Report DSR against 8 (the run's own grid) and
against 32 (everything tried). The 32 is the honest one.

## 7. Registry

| name | fn | family | evidence | kind |
|---|---|---|---|---|
| `tjr_intraday` | `tjr_intraday.signal` | predictive | folklore | single |
| `tjr_intraday_smt` | same, `smt=True` | predictive | folklore | single |

Thesis (base): "The 09:30 sweep of an overnight or prior-day extreme is
engineered to take resting stops, and the reversal it starts runs to the
opposite pool of liquidity." SMT adds: "…and ES/NQ divergence at the sweep marks
it as a manipulation rather than a breakdown." Source: "TJR / ICT trading
community, session-based model. No peer-reviewed support; see RESEARCH.md."

`Strategy` gains an optional `simulate` field so `run.py` can print trade detail
for any path-dependent model, not just the one named `tjr`.

## 8. Cost model (real futures, retail)

| | price | notional | commission RT | 1-tick slip RT | total RT |
|---|---|---|---|---|---|
| ES | 7,660 | $383k | $4.50 | $25.00 | **0.77 bps** |
| NQ | 29,392 | $588k | $4.50 | $10.00 | **0.25 bps** |

At 2 ticks: ES 1.42, NQ 0.42. Run at `--cost-bps 1` (ES) and `0.5` (NQ) as
"real", and `2` for both as "conservative". The cost sweep covers the rest.

## 9. Findings (2026-09-12, TradingView data, 149 sessions)

**The funnel.** Of 149 sessions, the sweep window produced a qualifying sweep on
roughly 93 (NQ) and 97 (ES). BOS followed within the window on about half of
those. A zone in discount/premium existed on a third of the BOS days. Fills:

| | sweeps | BOS | zone | filled | wins |
|---|---|---|---|---|---|
| NQ base | 93 | 50 | 34 | **7** | 0 |
| NQ SMT | 93 | 22 | 13 | **3** | 0 |
| ES base | 97 | 46 | 24 | **1** | 0 |
| ES SMT | 97 | 15 | 9 | **1** | 0 |

Every fill was an **inverted FVG**. Across 149 sessions on two instruments, not
one three-candle gap formed on the displacement leg below equilibrium — the
leg from the sweep wick to the BOS is two or three 5-minute bars and rarely
leaves a gap at all, let alone one in the cheap half of the range. The 16 grid
trials with `allow_ifvg=False` therefore never trade; the 16 that do are all
between −1.0 and −2.3 Sharpe. The best of 32 trials is the one that never fires.

**Why every trade stopped out.** The geometry of the eight base-variant trades:

```
stop distance      median 14.3 bps  = 0.94 ATR of one 5-minute bar
stopped on the entry bar             3 of 8
target printed later the same day    5 of 8
best excursion in the trade's favour median 3.4 R — after the stop had fired
```

The direction call — sweep, then a run to the next pool — was right five times
in eight. The stop rule, "beyond the sweep wick", places the stop inside the
range of a single 5-minute bar, so the retrace that fills the zone routinely
runs through it before the move the setup predicted. This is precisely the
step a discretionary trader "manages" by hand, and it is why discretionary
win-rate claims for this model cannot be checked: the rule that kills the
mechanical version is the one the discretionary version does not commit to.

**The SMT filter** removes 55–59 sweep days as "pair also swept" and changes
nothing about the outcome: the survivors are a subset of the same losers.

**The gauntlet**, full run, real costs (NQ 0.5 bps, ES 1 bps round trip) and a
conservative 2 bps; walk-forward 5 folds; 300-trial random benchmark with the
coin flips holding as long as the strategy does; gates 8 (7 where the grid best
never trades and DSR is undefined). Raw printouts in `results/tjr_intraday/`.

| run | trades | wins | IS Sharpe | buy&hold | OOS Sharpe | folds+ | vs random | breakeven | gates |
|---|---|---|---|---|---|---|---|---|---|
| NQ base, 0.5 bps | 7 | 0 | −2.29 | +0.82 | −1.44 | 0% | 12th pct | 0 bps | 2/8 |
| NQ SMT, 0.5 bps | 3 | 0 | −2.31 | +0.82 | −1.53 | 0% | 10th pct | 0 bps | 2/8 |
| ES base, 1 bps | 1 | 0 | −1.51 | +0.43 | −1.65 | 0% | 85th pct | 0 bps | 1/7 |
| ES SMT, 1 bps | 1 | 0 | −1.51 | +0.43 | −1.65 | 0% | 85th pct | 0 bps | 1/7 |
| NQ base, 2 bps | 7 | 0 | −2.49 | +0.31 | −1.49 | 0% | 67th pct | 0 bps | 2/8 |
| NQ SMT, 2 bps | 3 | 0 | −2.47 | +0.31 | −1.70 | 0% | 68th pct | 0 bps | 2/8 |
| ES base, 2 bps | 1 | 0 | −1.62 | +0.07 | −1.78 | 0% | 98th pct | 0 bps | 2/7 |
| ES SMT, 2 bps | 1 | 0 | −1.62 | +0.07 | −1.78 | 0% | 98th pct | 0 bps | 2/7 |

Deflated Sharpe against the cumulative 32 trials: **undefined** — the best trial
is one that never trades (Sharpe exactly 0), and every trial that does trade is
negative. The expected best-of-32 from pure noise would have been +1.82.

The gates that "pass" are artefacts of not trading, not of merit: PBO reads 50%
because half the grid columns are identical zeros; the ruin gate passes because
seven one-R losses on a vol-targeted 5-minute book draw down about 1%. The ES
runs' "beats 85–98% of random entries" is the same artefact from the other side:
a one-trade strategy against coin flips that hold one bar and pay costs on
every flip. Every gate that measures an edge fails, on both instruments, both
variants, both cost levels.

Costs did not decide this. Breakeven is 0 bps on every run — the gross Sharpe
is already negative — and the difference between real and conservative costs is
a rounding error against a strategy losing 1R on every fill.

The daily-bar version of this sequence failed by producing an edge too small to
survive its search and its costs. The intraday version fails differently: the
rules, followed exactly, produce a trade on roughly one session in twenty, and
that trade is stopped before the move it predicted arrives. Not one of the 12
won. Twelve trades is not a sample, and none of the above is evidence about the
market; all of it is a description of what his rules do when a program follows
them.

## 11. Round 2 — the final round: 1-minute entries and the stop-width variable

Pre-specified before any data was looked at. Two changes, one cumulative
trial count, and whatever comes back is the verdict on TJR. There is no round
three.

### 11.1 Why these two

He specifies 1-minute bars for confirmation and entry; round 1 ran everything
on 5-minute bars and found that the displacement leg (two or three 5-minute
bars) never leaves a gap, so every entry was an inverted FVG. And round 1's
geometry — direction right 5 of 8, median excursion 3.4 R *after* the stop
fired, stop inside one bar's range — points at exactly one variable: how far
the stop sits. Both are tested here; nothing else changes.

### 11.2 Data

TradingView `CME_MINI:ES1!` / `NQ1!` at **1 minute**, paged back with
`scripts/dump_bars.js` until the feed runs out → `data/ES_1min_tv.csv`,
`data/NQ_1min_tv.csv`. The session count this yields is reported before the
gauntlet is read, and if it is under ~100 sessions the 1-minute result is
labelled as such: it is still run, because the contract says so, but it is
not evidence either way.

### 11.3 The two-timeframe model (`entry_tf = 1`)

Input is the 1-minute frame. Everything in §1–§4 steps 1–4 (levels, swings,
sweep, BOS, equilibrium) stays on **5-minute context bars**, which the
strategy builds itself by resampling the 1-minute OHLC to 5-minute bars
labelled by open time. A 5-minute bar exists at 1-minute bar *i* only once its
last constituent minute is `<= i`; nothing about a partial 5-minute bar is ever
read. Sweep and BOS are therefore detected on the 1-minute bar that
*completes* the 5-minute bar they occur on. Swings are `confirmed_swings` on
the 5-minute context and admitted at the 1-minute bar completing their
`confirmed_at` bar.

Steps 5–9 move to 1-minute bars:

- **Displacement leg** = 1-minute bars from the bar that printed the sweep
  extreme (the 5-minute sweep bar's wick) to the bar completing the BOS. An
  FVG qualifies if it completes at `k` with `extreme_1m + 2 <= k <= bos_1m`, or
  later inside the entry window. Same EQ rule.
- **iFVG** = a counter-trend 1-minute gap completed at any bar of day D up to
  the BOS bar, inverted by a 1-minute close beyond it at `j >= first minute of
  the 5-minute sweep bar`.
- **Fill** on a 1-minute bar stamped in 09:50:00–10:09:59, strictly after both
  the BOS bar and the zone bar; `entry = min(zone.high, open)` for longs.
- **Stop / target / exit** evaluated on every 1-minute bar; flat at the 15:55
  1-minute bar's close; eod hold on the last bar of the input.
- One trade per day, first valid setup wins, as before.

The 5-minute model (`entry_tf = 5`) is untouched except for §11.4.

### 11.4 The stop-width variable, both timeframes

`stop_mode`, applied at fill time; `risk = |entry − stop|` must be positive:

| `stop_mode` | long stop | notes |
|---|---|---|
| `wick` | sweep extreme − `stop_buffer_atr` × ATR | round 1's rule, unchanged |
| `atr1.0` | entry − 1.0 × ATR | |
| `atr1.5` | entry − 1.5 × ATR | |
| `atr2.0` | entry − 2.0 × ATR | |
| `session` | (lowest low of trading day D up to the entry bar) − `stop_buffer_atr` × ATR | "beyond the session extreme"; coincides with `wick` whenever the sweep set the day's low, which is often |

ATR is **ATR(14) of the 5-minute context** in both models, at the 5-minute bar
containing the entry, so the multiples mean the same thing on both timeframes
and are comparable to round 1's 3.4 R figure. Targets do not change — the next
pool — so wider stops mean lower reward-to-risk; that trade-off is the test.

### 11.5 Grid and the cumulative trial count

```
stop_mode:    [wick, atr1.0, atr1.5, atr2.0, session]
min_fvg_atr:  [0.0, 0.25]
allow_ifvg:   [True, False]           -> 20 per (timeframe, instrument, variant)
stop_buffer_atr fixed at 0.25 (round 1 found 0 vs 0.25 immaterial)
```

2 timeframes × 2 instruments × 2 variants × 20 = **160 new trials**. Round 1's
32 are counted in full even though eight of them recur here, so the
cumulative count is **192**. The deflated Sharpe of the best trial is reported
against 192; `tjr_intraday_search.py` pools everything.

### 11.6 Registry

| name | frame | change |
|---|---|---|
| `tjr_intraday`, `tjr_intraday_smt` | 5-minute | gain `stop_mode` (default `wick`); grid becomes the 20 above |
| `tjr_intraday_1m`, `tjr_intraday_1m_smt` | 1-minute | new; same module, `entry_tf=1` |

### 11.7 Acceptance

- With `stop_mode=wick`, `stop_buffer_atr=0.25`, the 5-minute model reproduces
  round 1's twelve trades exactly (entry, stop, target, exit, bar for bar).
- `causality_check` clean on the 1-minute frames for both variants, **and** a
  truncation at a minute that is not a 5-minute boundary gives an identical
  signal over the overlap — that is the test that the resample does not read a
  partial context bar.
- All §10 invariants hold on 1-minute trades, with "sweep bar" and "BOS bar"
  meaning the completing 1-minute bar.
- Unit tests: a hand-built 1-minute day where every rule fires; each
  `stop_mode` produces the stop the table says on that day; a mid-5-minute-bar
  truncation test.
- `tjr` byte-identical; `run.py --strategy tjr --synthetic --quick` identical.

## 10. Acceptance

- `causality_check` clean (6 truncations) on both ES and NQ frames, both variants.
- On `--synthetic` daily data: all-flat, no exception, `--compare` still runs.
- Every trade in the frame satisfies: sweep bar stamped in the sweep window;
  entry bar in the entry window; entry bar > BOS bar > sweep bar; stop beyond
  the sweep wick; target equals one of the day's six levels and is beyond
  entry; exit reason ∈ {stop, target, flat, eod}; at most one trade per day.
- A hand-built synthetic day where every rule fires produces exactly the
  expected trade, and a hand-built day violating each rule in turn produces
  none (unit tests, in the repo under `tests/`).
- The trades frame carries the `tjr` columns plus `sweep_level` (name),
  `target_level` (name), `zone_kind` (`fvg`/`ifvg`), `smt` (bool),
  `session_day`.
- Existing `tjr`, `run.py --strategy tjr --synthetic --quick`, untouched and
  byte-identical.

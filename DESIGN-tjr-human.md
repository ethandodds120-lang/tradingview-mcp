# TJR, the human exit layer — pre-registration and contract

Paper only. Written before any trade, and before Part 2's funnel was run. §5
is the pre-registration: once trade one has filled, nothing in it changes.

## 0. Why this exists

The mechanical verdict stands (`DESIGN-tjr-intraday.md` §12): two rounds, 192
trials, best +3.68 against an expected best-of-noise of +4.29, closed. But two
round-1 numbers point somewhere the rules cannot go. T1 printed later in the
session on 4 of the 8 trades, and the median favourable excursion *after* the
fixed stop had fired was about 3 R. (The ticket was raised on "5 of 8 and
3.4 R"; Part 1 found that measurement had leaked into the next session's
evening bars, and `DESIGN-tjr-intraday.md` §9 is corrected. The question
survives the correction.) No fixed stop harvests that, and round 2 showed
widening the initial stop makes total R worse. The one version that could harvest it is the
one TJR actually trades — signals mechanical, exit discretionary. This
measures that layer rather than assuming it. It is an experiment on the human,
not a third round on the rules.

## 1. Part 1 — the trailing-stop benchmark (gates everything after it)

"Manage the stop" encoded as mechanical rules and run on the **existing**
round-1 and round-2 entries — no new data, no new entries. Five rules,
pre-specified, added to the cumulative trial count (192 → **197**):

| rule | long stop at bar j, from bars ≤ j−1 |
|---|---|
| `be_1r` | initial until the favourable excursion reaches +1 R, then entry, then hold |
| `atr1.0` | max(initial, highest high since entry − 1.0 × ATR) |
| `atr1.5` | max(initial, highest high since entry − 1.5 × ATR) |
| `swing` | max(initial, the most recent confirmed 5-minute swing low) |
| `hybrid` | initial until +1 R, then max(entry, highest high since entry − 1.0 × ATR) |

Every rule starts from the round-1 initial stop (sweep wick − 0.25 ATR), only
ever tightens, keeps the round-1 target (T1, the next opposing pool) and the
15:55 flat, and is evaluated causally: the stop that applies to bar j is
computed from bars through j−1, and a bar that opens beyond it exits at the
open. ATR is ATR(14) of the 5-minute context at the newest completed context
bar. Swings are `confirmed_swings(left=2, right=2)` on the context, admitted
by `confirmed_at`.

Reported per trade and in total, alongside the fixed-stop baseline, by
`tjr_trailing_benchmark.py` → `results/tjr_intraday/part1_trailing.txt`.
The best of the five is the benchmark the human must beat in §5. **If any rule
captures the excursions on its own, the rule is deployed and the human is
skipped** — that outcome ends T-7 before it starts.

## 2. Part 2 — the wider spec (funnel only, no gauntlet, no verdict)

Round 1 encoded a narrow reading. His own short-form description is wider.
The funnel is re-run on the same 149 sessions with these definitions, which
are also the definitions `tjr_human` (§3) is built to.

**Key levels** — sweep targets, all of these, all final before 09:30 ET:

| level | definition |
|---|---|
| `ASIA_H/L`, `LON_H/L`, `PDH/PDL` | as round 1 (§2 of the intraday contract) |
| `H1_SH`, `H1_SL` | confirmed swing highs/lows (`left=right=2`) on the 1-hour resample of the 5-minute bars; the most recent above/below price at 09:30 |
| `H4_SH`, `H4_SL` | the same on the 4-hour resample |
| `POC_PREV` | point of control of the **previous session's** volume profile (18:00 D−2 … 16:55 D−1), volume at price in bins of one tick × `poc_bin_ticks` |
| `HVN_*` | high-volume nodes of a **fixed 20-session composite** profile (the 20 completed sessions before D): local maxima of the smoothed profile whose volume ≥ `hvn_frac` × the profile's maximum. Fixed, not visible-range: VRVP changes with zoom and is not reproducible |

Resample boundaries are by the clock (an hourly bar exists once its last
5-minute bar has printed); swings are admitted by `confirmed_at` on the
resampled series, so a 4-hour swing confirmed at 08:00 with `right=2` is not
admissible until 16:00 and will not be in the 09:30 set — that is the rule,
not a bug.

**Order-flow change** — confirmation, any of, on a bar after the sweep and no
later than 10:05:

| confirmation | long definition |
|---|---|
| `bos` | close above the most recent confirmed opposing swing existing at the sweep (round 1) |
| `ifvg` | close above a bearish FVG of day D (round 1's inversion, here as confirmation rather than zone) |
| `ote` | close above the 79% retracement of the sweep leg, the leg being [the pre-sweep swing high existing at the sweep, the sweep extreme] |

**Continuation zones** — any of, and it must sit in discount (long) /
premium (short) of the dealing range [sweep extreme, highest high from the
sweep bar to the confirmation bar]:

| zone | definition |
|---|---|
| `fvg` | bullish three-candle gap on the displacement leg, sweep_idx+2 … confirmation bar |
| `eq` | the 50% level of the dealing range, as a touch |
| `ob` | order block: the last bearish candle before the displacement leg, [low, high] |
| `breaker` | a bearish order block above the sweep that a bar since the sweep closed above, now [low, high] as support |

**Fill** as round 1: an entry-window bar strictly after the confirmation and
the zone bar touches the zone. One setup a day, first valid one wins; where
more than one level was swept, or more than one confirmation or zone
qualifies, every combination is *counted* in the funnel but only the first
by time fills.

**Output.** sessions → sweeps → confirmations → zones in discount/premium →
fills, broken out by level type, confirmation type and zone type, for NQ and
ES. No Sharpe, no gates: the question is whether the narrow reading starved
the test or the wide one produces more of the same. Part 2 adds **no trials**
to the count because nothing is scored — but it opens a 7 × 3 × 4 branch
space that §3 inherits, and §5's deflation is what pays for it.

Primitives that are general — volume profile, POC, HVN, order block, breaker
— go in `primitives.py`. T-8 reuses them.

## 3. Part 3 — `tjr_human` (T-7)

### 3.1 Signal layer

- Windows and instruments as round 2: sweep 09:30–09:50 ET, entry
  09:50–10:10 ET, flat by 15:55; NQ and ES; 5-minute structure with 1-minute
  entry confirmation (the two-timeframe model of `DESIGN-tjr-intraday.md`
  §11.3).
- Sweep: wick through a §2 key level, close back inside.
- Confirmation: any of the three §2 confirmations, on 1-minute bars, after
  the sweep.
- Zone: any of the four §2 zones, in discount / premium of the dealing range.
- Entry: price returns to the zone **and** a 1-minute candle closes out of
  the zone in the trade direction. Enter at the next bar's open.
- Initial stop: beyond the sweep wick (round 1's definition), the fixed
  baseline.
- Targets: next opposing key level (T1), then the one beyond it (T2).
- The bot takes every paper entry itself. The human does not pick entries.

### 3.2 Chart output

At signal time, draw on the TradingView chart: every active key level
labelled by type; the sweep bar; the confirmation bar and its type; the zone
box and its type; the stop line; T1 and T2. Fire an alert carrying the same
information as JSON. The chart is a display layer; the alert path (§7) must
not depend on it.

### 3.3 Human controls, paper only

| control | when | rule |
|---|---|---|
| `SKIP` | from signal until fill | logged with the signal's full state, and later with what the trade would have done |
| `MOVE STOP` | while in the trade | **only in the trade's favour**; a widen is rejected and logged as an attempt with the requested price |
| `EXIT NOW` | while in the trade | closes at market; logged |

No other input. No sizing, no target changes, no entry choice.

### 3.4 Journaling — the bot writes everything, the human writes nothing

Per action: timestamp (ms), price, seconds since entry, unrealised R at that
moment, the action, a reason code from `{noise, structure_changed, news,
gut}`. Per trade: level type, confirmation type, zone type, instrument,
entry, initial stop, every stop move, exit, exit reason, realised R, MAE,
MFE, whether T1 / T2 printed and when. And per trade, on the same entry: what
the fixed stop, each §1 rule, and a uniformly random stop in [0.5, 2.0] ATR
would have done.

### 3.5 → §5. Pre-registration.

### 3.6 Weekly report (Sunday, from the journal, by the box)

- human R/trade vs fixed stop, each §1 rule, and the random stop
- behaviour: stop moves within 2 minutes of entry; exits before any rule
  would have exited; widen attempts; signals skipped and what they did
- branches: R by level type, confirmation type, zone type
- trades to go; current trajectory against the §5 bar

## 4. Part 4 — POC / HVN standalone (T-8)

Separate ticket, separate contract when it starts. Uses the §2 primitives.
Counted as trials.

## 5. Pre-registration — fixed once trade one has filled

- **Sample: 100 filled trades.** Not fewer because it is going well, not more
  because it is not. Skipped signals do not count toward the 100; they are
  reported alongside.
- **Paper only for the entire sample.**
- **Benchmark:** the best §1 trailing rule, chosen on the *same* 100 entries
  (in-sample for the rules, which favours them — that is the intended
  direction of the bias).
- **Pass:** the human's realised R per trade beats that benchmark after
  deflation. Operationally: the per-trade paired difference `d_i = R_human,i −
  R_best_rule,i` over the 100 trades is run through `validate.deflated_sharpe`
  with `n_trials = 9` — the six mechanical exits (fixed + five rules) plus the
  three human controls, each counted as one adjustable choice — and the
  result must be **≥ 0.95**, with `mean(d) > 0`. Deflation is the price of
  the §2 branch space and of the human's within-trade freedom; the fixed
  count of nine is a floor chosen before the data, not a fit to it.
- **Fail:** anything else. The discretionary layer is then noise on this
  sample; T-7 closes; every number stays in the record.
- The random-stop and fixed-stop comparisons in §3.4 are reported, not
  scored.
- Neither the sample size nor the bar changes after trade one.

## 6. What "beating the rule" would and would not show

A pass says one person, on 100 trades, over the weeks it takes to collect
them, exited better than the best simple rule on those entries, by enough to
survive a nine-way selection. It does not say the entries have an edge — the
mechanical test already said they do not on a fixed stop — and it does not
say the pass would repeat. It would justify a second pre-registered sample,
nothing more.

## 7. Alert path — recommendation and costs

Requirement: the signal reaches the phone by 09:50 ET every trading day; the
human is at the screen 09:30–10:30 ET; SKIP / MOVE STOP / EXIT NOW must get
back to the run.

| path | what it depends on | verdict |
|---|---|---|
| **VPS detects the signal and pushes to Telegram** (same bot as T-4); the three controls are Telegram commands to the same bot | the VPS, Telegram, **and a live ES/NQ 1-minute feed on the VPS, which it does not have** — Alpaca serves no futures | **recommended for the run**, once the feed exists. One implementation of the signal, the one the gauntlet tested; no desktop in the loop; the controls come back the same way the alert went out |
| TradingView webhook → VPS | a Pine re-implementation of the signal, a paid TradingView plan, TradingView's alert servers, webhook delivery | **no.** Two implementations of one signal drift, and the Pine one cannot be gauntlet-tested. The drawing (§3.2) still happens in Pine or via the bridge, but not the decision |
| TradingView MCP bridge as the alert path | the Windows desktop app running with CDP, and a session | **no, for alerts.** It is the fragility the VPS move removed. **Yes, for the §3.2 chart drawing**, as a display layer that can be down without the experiment stopping |
| **Interim: the Windows PC runs the detector on the existing `TradingViewFeed` and pushes to Telegram** | TradingView Desktop up on the PC during 09:25–10:15 ET | acceptable **for the 100-trade sample only**, because the human is at that screen in that window anyway and would see the PC down; it is the same signal code, not a re-implementation. Not acceptable unattended |

The live futures feed is the same line item `TICKETS.md` flags for T-5:
Databento live (pay-as-you-go, CME) is the usual retail-priced option. It is
costed when T-7's build starts, with T-5 in mind; if the interim path is
used for the sample, the experiment does not wait on it.

## 8. What is not in scope

Sizing, target changes, entry discretion, a live account, a second instrument
set, and any change to §5 after trade one.

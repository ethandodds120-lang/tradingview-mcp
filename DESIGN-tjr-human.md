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

### 2.1 Definitions as built (Part 2, 2026-09-12)

Written back by the funnel build — `tjr_wide_funnel.py`, the general
primitives in `primitives.py` (`resample_context`, `volume_profile`, `vwap`,
`poc`, `hvn`, `order_block`, `breaker_block`). Where §2 above left a choice,
this is the choice; the numbers in `results/tjr_intraday/part2_wide_funnel.txt`
and the day-level `part2_funnel_days_{NQ,ES}.csv` are under these definitions.

**Levels — every one frozen at the 09:25 bar's close** (the last bar of D
before the sweep window) from bars ≤ that bar:

| item | as built |
|---|---|
| resampled bins | `resample_context(df, 60 / 240)`: bins on the New York wall clock anchored at 18:00 ET, labelled by nominal open time in tz-naive UTC. A bin exists at 5-minute bar *i* once its last slot, by minute arithmetic, is ≤ *i* — never by looking at the next row. The last bin of a session is **clipped at the 17:00 close** (the 14:00 4-hour bin runs 14:00–16:55 and completes at the 16:55 bar; 1-hour bins complete at :55, 23 a session). A bin whose last slot never prints (a CME early close) never completes at its own time — `last` skips it — but it **stays a bar of the resampled series**, readable once the next bin completes, and a swing may pivot on it (the bar a chart shows; causal, since at any input bar the readable bins are the same whether the frame is cut there or runs on). On the TradingView files: one 1-hour bin (2026-04-03) and six 4-hour bins (02-16, 04-03, 05-25, 06-19, 07-03, 09-07). Dropping them instead would move H4_SH on 7 days and no funnel number (reviewer's check). An earlier draft of this row said "never a bar"; corrected 2026-09-12. Empty bins do not exist. |
| `H1_SH/SL`, `H4_SH/SL` | `confirmed_swings(left=2, right=2)` on each resampled series; a swing is admissible at bar *i* iff `confirmed_at <= ctx.last[i]`, the newest *completed* bin. `*_SH` = the admissible swing high nearest **in price** above the 09:25 close, `*_SL` the nearest below; none → absent for the day (H4_SH absent on 15 NQ / 14 ES days). §2's "most recent" was read as nearest in price, the same sense the HVN naming uses. |
| volume profile | `volume_profile`: each bar's volume spread uniformly over the bins its [low, high] covers; bin width = tick 0.25 × `poc_bin_ticks` (default 4 → $1.00), grid anchored at multiples of the width so sessions line up bin for bin. Level prices are bin **centres** (x.50). |
| `POC_PREV` | POC of the previous session in the frame (its 18:00 (D−2) … 16:55 (D−1) bars): the bin with the most volume, ties → the bin nearest the session's VWAP (typical price (h+l+c)/3, volume-weighted). Absent on the first session of the file, present on the other 148. **"Previous session" is the previous clock day, whatever its length.** The frame has 150 clock days; 149 have a sweep-window bar. The 150th is 2026-04-03 (Good Friday): 183 bars, last stamp 09:10, 42k contracts against ~566k on a normal day. It has no sweep window and no funnel row, but it is 04-06's previous session — its PDH / PDL (24255 / 24106, as in round 1) and its POC_PREV come from those 183 bars — and it is one of the composite's 20 sessions from 04-06 through 05-01. The five 12:55 early closes (02-16, 05-25, 06-19, 07-03, 09-07) are ordinary sessions for every purpose here. Flagged by review; kept as the literal definition (a trader on 04-06 reads Friday's profile, thin as it is). |
| `HVN_1..6` | the composite is the last **20 completed sessions before D**, summed bin for bin (fewer on the first twenty days; the depth is reported per day). Smoothed with a centred 5-bin moving average on the **price** axis — a profile of past bars, not a window on time, so not lookahead; at the grid ends the mean is over the bins covered. Nodes = local maxima (`>=` the bin below, `>` the bin above, so a plateau gives one node) whose smoothed volume ≥ `hvn_frac` (0.5) × the smoothed maximum; at most `hvn_max` (6), named HVN_1.. nearest to the 09:25 close first. NQ yields six on every day; ES four to six on most. |
| which side | ASIA/LON/PD/H1/H4 by name. `POC_PREV` and `HVN_*` rest on the side of the 09:25 close: ≤ the close → a low (a long's target), > the close → a high. |

**Sweep** — round 1's test (`tjr_intraday.sweep_of_levels`) against the union
of the day's levels: wick through, close back inside; several lows in one wick →
the **lowest** is the setup's level, several highs → the highest; a wick through
a low *and* a high is ambiguous and skipped, and the next bar in the window may
still sweep; the first qualifying bar fixes the direction. Every level the sweep
bar breached is recorded alongside the deepest. With up to seventeen levels the
ambiguity rule bites: 29 (NQ) / 21 (ES) of round 1's sweep days are lost to it,
14 / 19 days sweep only under the wide set. Restricted to the six round-1 levels
the same code reproduces round 1's 93 / 97 sweeps day for day, bar for bar (the
printout checks this against `tjr_intraday.day_log`).

**Confirmations** — on a bar strictly after the sweep bar, stamped ≤ 10:05:

| | long definition as built |
|---|---|
| `bos` | close > the most recent confirmed 5-minute opposing swing (left=right=2) admissible at the sweep bar (`confirmed_at <= sweep_idx`), frozen there — round 1's level |
| `ifvg` | close > `gap.high` of a bearish 5-minute FVG completed on day D (any bar from 18:00 D−1) at or before this bar. **As written this admits a gap that a bar had already closed through before the sweep**, so it fires on the first bar after most sweeps: on 65 of 66 (NQ) and 76 of 76 (ES) confirmation bars, of which only 26 / 25 rest on a gap first closed through *after* the sweep bar. The printout carries a "fresh inversion only" sensitivity (NQ 49 confirmations → 20 fills, ES 53 → 26) and the CSV flags each confirmation bar |
| `ote` | close > extreme + 0.79 × (pre-sweep swing high − extreme), the swing being `bos`'s; no admissible swing → neither `bos` nor `ote` (never happened). Implemented literally: a swing on the wrong side of the extreme (one NQ day) is not guarded against |

The first bar on which any fires is the confirmation bar; the types on it, and
every type that fires on any bar by 10:05, are both recorded.

**Dealing range** at confirmation: [sweep extreme, highest high from the sweep
bar through the confirmation bar] (shorts mirror). EQ = midpoint; discount =
zone.high ≤ EQ, premium = zone.low ≥ EQ.

**Zones** — each must be in discount / premium:

| | long definition as built | freshness stamp |
|---|---|---|
| `fvg` | bullish 5-minute FVG completed at *k*, sweep_idx+2 ≤ *k* ≤ confirmation bar, or later inside the entry window | *k* |
| `eq` | [EQ, EQ], touched when low ≤ EQ. In discount by construction, so **the zone stage equals the confirmation stage**; the printout also reports the count without it (NQ 30, ES 41) | the confirmation bar |
| `ob` | `order_block`: the last bearish candle (close < open) with index ≤ sweep_idx, no lower bound; [low, high] | its candle |
| `breaker` | `breaker_block`: for a confirmed 5-minute swing high of day D before the sweep (confirmed by the bar being read), the last close > open candle strictly before the swing bar and on or after the day start, once a bar *j* ≥ sweep_idx has closed above its high (the sweep bar itself counts, as round 1's inversion did); several → the most recently violated; [low, high] | *j* |

The setup uses the freshest stamp; on the same bar, fvg over breaker over eq
over ob. A zone's bar for the fill rule is the bar it became knowable on — the
later of its stamp, the confirmation bar and, for a breaker, the bar its swing
confirmed on. Because `eq` is stamped at the confirmation bar it outranks every
leg gap, order block and earlier breaker; only a gap or violation on or after
the confirmation bar displaces it (zone used: eq 57 of 66 NQ, 64 of 76 ES; ob
never).

**Fill** — an entry-window bar (09:50–10:05 stamps) strictly after both the
confirmation bar and the zone's bar, low ≤ zone.high (long); entry =
min(zone.high, open). One setup a day, first valid by time. Fills are counted;
no stop, target or exit exists in the funnel.

**Causality.** Truncating each frame at six mid-session bars (inside the sweep
window, inside the entry window, mid-afternoon) reproduces every earlier day's
row and the cut day's row as of the cut, on both instruments; daily synthetic
data (no 09:30 bar) is all-flat with no exception. Both are in the printout.

**Not built here, deliberately:** no SMT, no pair columns, nothing registered,
no trial counted.

### 2.2 Part 2 findings (2026-09-12, 149 sessions each, no outcomes measured)

Two adversarial reviews: one not refuted (four low notes, all intrabar-order or
definitional, none moving a number); one refuted on the *recorded* definition
of two items — never-completed resample bins, and the Good Friday session —
which are corrected in §2.1 above. The funnel numbers are unchanged by either.
Causality: the script's six cuts per instrument reproduce the full run, and
both reviewers' own truncations did the same. Printout: `results/tjr_intraday/part2_wide_funnel.txt`;
day-level rows: `part2_funnel_days_{NQ,ES}.csv`.

**The funnel, round 1 beside the wide spec, and beside the wide spec with the
one obvious degeneracy removed (`ifvg` counted only on a fresh inversion):**

| stage | NQ r1 | NQ wide | NQ wide, fresh ifvg | ES r1 | ES wide | ES wide, fresh ifvg |
|---|---|---|---|---|---|---|
| sessions | 149 | 149 | 149 | 149 | 149 | 149 |
| sweeps | 93 | **78** | 78 | 97 | 95 | 95 |
| confirmations | 50 | 66 | 49 | 46 | 76 | 53 |
| zones in discount | 34 | 66 | 49 | 24 | 76 | 53 |
| fills | **7** | **41** | **20** | **1** | **51** | **26** |

**Where it widened, stage by stage.**

1. **Sweeps went down on NQ and flat on ES.** Two effects pull against each
   other. The new levels alone made sweeps on 8 (NQ) and 16 (ES) days that
   round 1 had no level for. But with fifteen to seventeen levels on the board
   — 4.9 breached per NQ sweep bar — the opening bars straddle a high *and* a
   low far more often, and §2's ambiguity rule threw out 226 sweep-window bars
   on 100 NQ days, removing 29 of round 1's 93 sweep days (21 of 97 on ES).
   Every level type is the deepest level on some days; among the new ones HVN,
   H1 and H4 are the largest sources (NQ 15 / 12 / 8 days, ES 16 / 15 / 15).
2. **Confirmation is where the literal spec stops filtering.** `ifvg` fires on
   the confirmation bar on 65 of 66 NQ setups and 76 of 76 ES. Read literally
   — a close above any bearish gap of day D completed at or before this bar —
   it counts gaps price closed through hours before the sweep, so the first
   bar after almost any sweep "confirms". One hand-checked case (NQ 03-09)
   confirmed on a close *lower* than the sweep bar's close. `bos` fires by
   10:05 on 25 / 34 days and `ote` on 33 / 40; neither is ever the sole
   first-to-fire because `ifvg` has already fired. Requiring a **fresh**
   inversion (no close through the gap between its completion and the sweep)
   cuts confirmations to 49 / 53.
3. **The `eq` zone cannot fail.** The 50% level sits in discount by
   construction, so the zone stage equals the confirmation stage on every day
   (66 = 66, 76 = 76), and `eq` is the zone used on 57 of 66 NQ and 64 of 76
   ES setups. Excluding it, zones in discount are 30 / 41. `ob` qualifies on
   12 / 14 days and is never used (always staler than `eq`); `breaker`
   qualifies on 16 / 27 and is used on 4 / 8; a leg `fvg` exists on 5 / 6.
4. **Fills follow.** The midpoint of a two-to-three-bar dealing range is
   touched inside the entry window most of the time. Of the 41 NQ fills, 28
   are (some level, `ifvg` alone, `eq`); of 51 ES, 34.

**The answer to the question.** The narrow reading did not starve the test at
the sweep — it starved it at confirmation → zone → fill, where round 1 asked
for a break of structure and a gap-shaped zone in discount. The wide reading
produces roughly three to six times the fills per instrument even after the
`ifvg` degeneracy is removed (20 and 26 against 7 and 1). But the bulk of the
literal spec's extra fills arrive through two definitions that barely filter:
an already-inverted gap as "confirmation", and a midpoint that is in discount
by construction as a "zone". Whether the extra fills are more of the same is an
outcome question, and Part 2 measures no outcomes, by design. That is T-7's.

**Three definitions T-7 must settle before trade one** — flagged, not changed;
§5 is the pre-registration and §2 is his spec, so these are the user's calls:

- `ifvg` as confirmation: **fresh inversion only** is the recommendation. The
  literal reading is not a confirmation of anything.
- `eq` as a zone: as specified it never fails the discount rule and is always
  the freshest. Keep it as his definition and accept that "zone" then filters
  nothing, or require the touch to come on a completed bar after a pullback
  (i.e. price must first leave the midpoint), or pair it with a second zone
  type. Any of the three is defensible; the funnel cannot choose.
- the ambiguity rule against sixteen levels: as specified it removes more
  round-1 sweeps than the new levels add. Options: judge ambiguity within a
  level class (an HVN low and an H1 high in one wick is not the round-1
  ambiguity), or keep POC/HVN as *targets* only and not as sweep levels.

**Branch space, realised.** The 7 × 3 × 4 = 84 combinations collapse in
practice: (any level, `ifvg` alone, `eq`) is about two-thirds of fills. §5's deflation count
of nine stands; it was set before this and is not raised by it.

**Levels at 09:25.** H4 swing highs are absent on 15 NQ / 14 ES days (no
admissible 4-hour pivot above the close), H1 swing highs on 7 / 4, H4 lows on
3 / 3, H1 lows on 2 / 0; the session, PD, POC and HVN levels are present on
148 or 149. NQ's 20-session composite yields six HVNs on 148 days; ES's is
flatter (six on 100 days, three to five on 46), so HVN is a thinner level set
on ES.

### 2.3 Decisions on §2.2 (2026-09-12) — the spec for the build and for one re-run

Three definitions, decided by the user on the Part 2 findings. They replace
the literal readings of §2.1 for `tjr_human` and for the single re-run of the
funnel recorded in §2.4. They were chosen on funnel counts; no outcome has
been measured, so §5's deflation count does not move for them.

1. **`ifvg`: fresh inversion only.** A bearish gap of day D confirms a long
   only if no bar between its completion and the sweep bar (inclusive) closed
   above it; the inverting close is on a bar strictly after the sweep bar.
   Shorts mirror. This is the "fresh" reading of §2.1; the literal reading is
   dropped.
2. **`eq`: valid only as a retrace into it.** The midpoint is frozen at the
   confirmation bar, as before. A touch counts only if price has first been at
   least 0.5 × ATR beyond the midpoint on the far side: for a long, some
   completed bar *j* with sweep bar ≤ *j* < touching bar has high ≥ EQ + 0.5 ×
   ATR; for a short, low ≤ EQ − 0.5 × ATR. ATR is ATR(14) of the 5-minute
   frame at the confirmation bar (on 1-minute entries, at the newest completed
   5-minute context bar — `DESIGN-tjr-intraday.md` §11.4's convention). The
   range extreme through the confirmation bar is already on the far side, so
   the condition is met at confirmation whenever the dealing range is at least
   one ATR wide, and otherwise waits for a later bar to extend that far. The
   zone's `formed` stamp stays the confirmation bar (its rank among zones is
   unchanged); it becomes *usable* at the excursion bar. No second zone type is
   required; co-occurrence — which other zone types were in discount when an
   `eq` fill was taken — is logged per trade and reported.
3. **Ambiguity within a level class; POC and HVN are targets only.** The sweep
   levels are the directional set — ASIA_H/L, LON_H/L, PDH/PDL, H1_SH/SL,
   H4_SH/SL — in five classes by type. POC_PREV and HVN_1..6 are computed and
   logged every day and serve as T1 / T2 candidates (the next opposing level of
   any type); they are not swept — they are not TJR's, and T-8 tests them on
   their own. Round 1's test (`sweep_of_levels`) runs per class: a class whose
   wick took one of its lows *and* one of its highs abstains for that bar;
   among the classes that return a sweep (breached, closed back inside), if all
   give the same side the bar sweeps on that side and the setup's level is the
   deepest breached level of that side across those classes; if two classes
   give opposite sides the bar is ambiguous and skipped, and the next bar in
   the window may still sweep. A bar on which no class returns a sweep is not
   a sweep. Every level the wick took, in any class, is recorded.

### 2.4 The one re-run under §2.3 (2026-09-12) — the fill count the experiment will produce

`tjr_wide_funnel.py --mode decided` (the default now; `--mode literal`
reproduces §2.1's printout and CSVs byte for byte). Printout
`results/tjr_intraday/part2_decided_funnel.txt`, day rows
`part2_decided_days_{NQ,ES}.csv` (the 29 literal columns plus 14: class
bookkeeping, the eq excursion, co-occurrence, targets). Two adversarial
reviews, neither refuted, no finding that moves a number; causality clean at
seven cuts per instrument in the script's proof, 49 hand-chosen cuts plus 164
exhaustive cuts at every fill bar and the bar after it in review, and 26
cuts on manufactured later-excursion days. Definitions as built where §2.3
was silent are in the printout's closing block; the ones that matter are
below.

| stage | NQ round 1 | NQ literal | NQ fresh-ifvg | **NQ decided** | ES round 1 | ES literal | ES fresh-ifvg | **ES decided** |
|---|---|---|---|---|---|---|---|---|
| sessions | 149 | 149 | 149 | 149 | 149 | 149 | 149 | 149 |
| sweeps | 93 | 78 | 78 | **116** | 97 | 95 | 95 | **128** |
| confirmations | 50 | 66 | 49 | **90** | 46 | 76 | 53 | **83** |
| zones | 34 | 66 | 49 | **90** | 24 | 76 | 53 | **83** |
| fills | 7 | 41 | 20 | **42** | 1 | 51 | 26 | **40** |

**The number asked for: 82 fills over 149 sessions on the two instruments,
0.55 a session-pair.** Before any `SKIP`, 100 filled trades is about 180
sessions — roughly eight and a half months of both instruments running. On
one instrument it is twice that. The 1-minute entry confirmation of §3.1 will
move this somewhat in either direction; nothing here measures it.

**What each decision did, measured.**

1. **`ifvg` fresh-only** does what it was meant to: `ifvg` is on 54 of 90 NQ
   and 42 of 83 ES confirmation bars, not 65 of 66 and 76 of 76, and `bos`
   (52 / 43) and `ote` (68 / 65) now carry confirmations of their own —
   `ifvg` is the sole first-to-fire on 22 / 22 days, `ote` on 8 / 9.
2. **The `eq` retrace rule, as written, is inert on these frames.** The
   excursion is met at the confirmation bar on 90 of 90 and 83 of 83
   confirmed days, because the dealing range at confirmation is never
   narrower than 1.29 ATR (NQ median 3.08, ES 2.89): the 09:30–09:45 bars
   dwarf a 14-bar ATR that is mostly overnight bars, so 0.5 ATR beyond the
   midpoint is always already true. The zone stage still equals the
   confirmation stage; `eq` is the zone used on 79 / 90 and 71 / 83 setups
   and on 39 of 42 and 33 of 40 fills. The later-bar path of the rule was
   exercised only on a scratch run at 2.5 ATR (invariants and causality
   held). The one reading in the log that *would* bite: on 14 of 39 NQ and
   11 of 33 ES `eq` fills the touching bar opened at or beyond the midpoint —
   price was already through it at the open, not retracing into it. A rule
   on the touching bar ("opens on the far side of EQ") would remove those;
   a threshold in ATR will not, at any value the range does not already
   clear. **Co-occurrence:** at `eq` fills another zone type was in
   discount on 25 / 22 (an `ob` on 24 / 17, a breaker on 7 / 10, a leg
   `fvg` never); 14 / 11 `eq` fills had no other zone at all.
3. **Ambiguity within a class recovers every round-1 sweep and adds a
   third more.** Disagreement between classes happens on 4 NQ / 6 ES bars
   all season; round 1's 93 / 97 sweep days are all kept but one (ES
   03-10, ASIA short against H4 long). Sweeps go *up* to 116 / 128 for two
   reasons: H1/H4 alone make 12 / 24 sweep days, and — this is a
   consequence of §2.3's wording, not of the H1/HVN case that motivated
   it — **round 1's own six levels are now three classes** (ASIA, LON, PD),
   so a wick through ASIA_L and LON_H that closes back inside both is no
   longer round 1's ambiguity (ASIA returns long, LON returns nothing).
   Abstention is almost entirely H1 (an H1 swing high and low bracketing
   the open: 40 / 34 sweep bars), and on 15 / 12 sweep bars a deeper
   same-side level sat inside an abstaining class and is recorded but is
   not the setup's. If the six session levels were one class and H1 / H4
   two more, round 1's ambiguity would be restored for the six; that is a
   one-line change and a decision for the user, not made here.
4. **POC / HVN as targets.** Logged only, by price: T1 is the nearest level
   beyond the entry of any name. Under that reading T1 is a POC / HVN on 12
   / 12 fills and sits a median 0.35 / 0.59 ATR from the entry — but on 16 /
   13 fills it is a *same-side-named* directional level (an `*_L` above a
   long's entry) and on 9 / 5 it is the swept level itself, because the
   entry is taken beyond the level the sweep took. For `tjr_human`, T1
   should be the nearest **opposing-named** level (a high for a long) or a
   POC / HVN beyond the entry; the CSV carries the names, so this is a
   re-cut, not a re-run. Price ties between two names (3 NQ / 1 ES fills)
   are one target, reported under both names.

**Three readings the reviewers recorded, none a change.** The sweep test
does not condition on the bar's open: 54 of 116 NQ and 42 of 128 ES decided
sweeps are bars that *opened* beyond the level and closed back through it
(round 1's own behaviour — 56 / 45 of its sweeps are the same), so the
sweep count is not a count of wick-and-reclaim bars; if T-7 wants that
reading it is a new primitive, not an edit. The first ~20 sessions of any
frame run on a shallower composite and a thinner H4 swing history
(`composite_depth` in the CSV), so a live detector must be warmed on at
least 20 completed sessions plus the 4-hour history before its levels
match these. And "levels the wick took" is round 1's range-cover test, so
an opposite-side level resting beyond the open is counted as taken on 18 /
29 of the 105 / 94 logged opposite-side breaches; that affects only the
logged count.

**One proof-helper gap, fixed in the commit that records this:** the
script's `as_of()` did not forget the fresh-inversion bookkeeping when it
forgot the sweep; the truncated run was the correct side and no printed
field or count depends on it.

### 2.5 Decisions on §2.4 (2026-09-18) — the final spec, and one last re-run

Three corrections by the user to §2.3, made on funnel counts only; still no
outcome measured. They are the spec `tjr_human` is built to. The re-run under
them is §2.6 (`--mode final`; `--mode decided` and `--mode literal` still
reproduce §2.4 and §2.1 byte for byte).

1. **`eq`: the open-side test, in addition to §2.3's excursion rule.** §2.3's
   0.5 ATR excursion is inert (§2.4) and `eq` still carried 39 of 42 and 33
   of 40 fills. A touch of the `eq` zone now counts only on a bar that **opens
   on the far side of the midpoint** — long: open > EQ; short: open < EQ. A
   bar that opens at or through the midpoint is not a retrace into it and
   does not fill; a later bar that opens on the far side and trades back to
   the midpoint does. While `eq` is the freshest zone it is the zone in play:
   a bar refused by the open-side test does not fall through to an older zone
   on that bar. On 1-minute entries the test is applied to the 1-minute bar
   that touches.
2. **Level classes: round 1's six session levels are ONE class.** `SESSION` =
   ASIA_H/L, LON_H/L, PDH/PDL; `H1` = H1_SH/SL; `H4` = H4_SH/SL. §2.3's
   wording split round 1's own rule by accident: a wick through a session low
   and a session high is round 1's ambiguity again (the `SESSION` class
   abstains). The combination rule of §2.3 is otherwise unchanged — among the
   classes that return a sweep, one side → sweep, opposite sides → ambiguous —
   and the re-run reports how many sweeps come from bars on which `SESSION`
   abstained and a swing class returned.
3. **Targets: the nearest opposing-NAMED level, or a POC / HVN, beyond the
   entry.** For a long: the nearest of ASIA_H, LON_H, PDH, H1_SH, H4_SH,
   POC_PREV, HVN_* **above** the entry is T1, the next one beyond it T2;
   shorts mirror with the lows. A same-side-named level is never a target —
   targeting the level the sweep just took is wrong by construction. Levels at
   the same price are one target carrying both names. No such level beyond
   the entry → no T1 for that trade, counted and reported.

### 2.6 The final re-run under §2.5 (2026-09-18) — 66 fills over 149 sessions

`tjr_wide_funnel.py --mode final` (now the default; `--mode decided` and
`--mode literal` reproduce §2.4 and §2.1 byte for byte). Printout
`results/tjr_intraday/part2_final_funnel.txt`, day rows
`part2_final_days_{NQ,ES}.csv`. Two adversarial reviews over two rounds,
neither refuted; causality clean on the script's ten cuts per instrument and
on roughly 1,500 reviewer cuts per instrument, including every bar from 09:35
to 10:15 on every day, random-walk garbling of everything after the cut, and
255 variants of each refused bar's high, low and close. No outcome read, no
trial counted.

| stage | NQ round 1 | NQ decided | **NQ final** | ES round 1 | ES decided | **ES final** |
|---|---|---|---|---|---|---|
| sessions | 149 | 149 | 149 | 149 | 149 | 149 |
| sweeps | 93 | 116 | **112** | 97 | 128 | **124** |
| confirmations | 50 | 90 | **88** | 46 | 83 | **79** |
| zones | 34 | 90 | **88** | 24 | 83 | **79** |
| fills | 7 | 42 | **32** | 1 | 40 | **34** |

**66 fills over 149 sessions, 0.44 a session-pair: 100 filled trades is about
226 sessions of both instruments — close to eleven months, not eight and a
half.** And §9.2's dead-setup rule is not in the funnel: on 15 of the 66
fills the wick stop had already traded before the fill (one NQ fill has its
entry beyond the stop and no R at all). With those gone the rate is about 51
per 149 sessions and the sample is nearer fourteen months. The 1-minute
trigger will move it again. The sample stays 100 (§5); this is what it costs.

**Where the fills went.** NQ: 42 decided − 4 lost and + 1 gained by the class
change = 39; − 9 left unfilled by the open-side test = 30; + 2 by the
zone-in-play reading below = 32. ES: 40 − 4 + 1 = 37; − 3 = 34; ± 0 = 34.

1. **The open-side test bites and `eq` is now a retrace.** It refused 38 bars
   on 17 NQ setups and 13 bars on 8 ES setups, every one a bar that opened at
   or through the midpoint; 12 and 5 of those setups never filled, 3 and 3
   filled later on `eq` from the far side, and 2 NQ setups filled on the `fvg`
   the refused bar itself completed. `eq` still carries 29 of 32 and 26
   of 34 fills, but each of them now opens strictly beyond the midpoint,
   trades back to it and enters exactly at it. Co-occurrence at `eq` fills:
   no other zone on 14 / 11, an `ob` on 13 / 12, a breaker on 2 / 6.
2. **One `SESSION` class restores round 1's ambiguity rule.** Round-1 sweep
   days lost: 0 of 93 on NQ, 1 of 97 on ES (03-10, `SESSION` short against H4
   long). Class disagreement all season: 0 NQ bars, 3 ES. The sweeps above
   round 1's are the swing classes': on 19 NQ and 28 ES days only H1 or H4
   made the sweep; on 6 / 5 sweep bars `SESSION` abstained and a swing class
   returned; on 26 / 42 `SESSION` was breached without closing back, or
   untouched, and a swing class returned. The setup's level is a session level
   on 68 / 71 sweeps, H1 on 15 / 26, H4 on 29 / 27.
3. **Targets are opposing-named, and that exposes a second problem.** No
   target is a same-side level or the swept level any more (under §2.3's
   by-price reading 10 / 8 of these fills would have had one). But the
   geometry at entry — no bar after the fill is read — says:

   | against the wick stop | NQ | ES |
   |---|---|---|
   | T1 reward-to-risk, median (quartiles) | 0.31 R (0.11 – 0.64) | 0.77 R (0.43 – 1.09) |
   | T2 reward-to-risk, median | 0.80 R | 1.48 R |
   | fills with T1 closer than 1 R | 24 of 31 | 23 of 34 |
   | fills where T1's price **had already traded** before the fill | 29 of 32 | 27 of 34 |

   The nearest opposing level beyond the entry is usually one the
   displacement leg already ran through on its way up: price swept, broke
   structure through that level, then retraced to the midpoint, and the
   "target" sits between the entry and a high the session has already made.
   It is not an untaken pool. TJR's target is the next liquidity *not yet
   taken*. As specified, the whole position would exit at a median third of
   an R on NQ, which would cap the very excursions §0 set out to measure.
   **The user's call before trade one** (§9.10): keep §2.5's nearest level,
   or require the target not to have traded since the 09:30 open. The
   detector journals both for every signal, so the choice costs nothing to
   make late — but it must be made before `arm`.

**One reading adopted here that §2.5 did not contain, flagged by review as the
user's to overrule.** On an entry-window bar *i*, the zone in play is the
freshest zone known **before** bar *i*; a gap or breaker that bar *i* itself
completes is in play from *i* + 1. Round 1's convention (§2.1, still used by
the decided and literal modes) makes the new zone the freshest *on* bar *i*
and so uses that bar's close to cancel a fill that a resting order on the
older zone takes inside it — a dependence on the close of the bar being
filled. No truncation test can see the difference; it is about the order of
events inside one 5-minute bar. It moves 2 NQ fills (04-06, 08-11) and the
bar of one ES fill (06-16): 66 with it, 64 under round 1's convention. The
1-minute detector (§9.2) uses the causal reading.

**Conventions recorded, not changed:** the open-side test applies to `eq`
only — a bar that opens through an `fvg` or a breaker fills at its own open;
EQ can fall off the quarter-tick grid and §2.5 gives no rounding rule; the
CSV column named `outcome` is the funnel stage reached, not a trade result.

## 3. Part 3 — `tjr_human` (T-7)

### 3.1 Signal layer

- Windows and instruments as round 2: sweep 09:30–09:50 ET, entry
  09:50–10:10 ET, flat by 15:55; NQ and ES; 5-minute structure with 1-minute
  entry confirmation (the two-timeframe model of `DESIGN-tjr-intraday.md`
  §11.3).
- Sweep: wick through a §2.3 sweep level (the directional set; ambiguity
  judged within a level class — the three classes of §2.5), close back inside.
- Confirmation: any of the three §2 confirmations, on 1-minute bars, after
  the sweep; `ifvg` on a fresh inversion only (§2.3).
- Zone: any of the four §2 zones, in discount / premium of the dealing range;
  `eq` only as a retrace into it (§2.3's excursion and §2.5's open-side test).
- Entry: price returns to the zone **and** a 1-minute candle closes out of
  the zone in the trade direction. Enter at the next bar's open.
- Initial stop: **the human's choice at fill, within [wick, 2.0 ATR]** (§3.3
  `STOP`, §5 amendment); the wick stop (round 1's definition) is the default
  when no choice arrives in time.
- Targets: the nearest opposing-named level, or POC / HVN, beyond the entry
  (T1), then the one beyond it (T2) — §2.5.
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
| `STOP <price>` | from signal until 60 s after the fill, once per trade | the initial stop; accepted only between the wick stop and the 2.0 ATR stop (both prices are in the signal alert, with the 1.0 ATR, 1.5 ATR and session stops for reference); outside that band → rejected and logged with the requested price; none by the deadline → the wick stop |
| `MOVE STOP` | while in the trade, after the `STOP` window | **only in the trade's favour**; a widen is rejected and logged as an attempt with the requested price |
| `EXIT NOW` | while in the trade | closes at market; logged |

No other input. No sizing, no target changes, no entry choice. The initial
stop is the one place the human may be wider than the rules; from the moment
it is set, the stop only tightens.

### 3.4 Journaling — the bot writes everything, the human writes nothing

Per action: timestamp (ms), price, seconds since entry, unrealised R at that
moment, the action, a reason code from `{noise, structure_changed, news,
gut}`. Per trade: level type, confirmation type, zone type, the other zone types in
discount at the fill (§2.3 co-occurrence), instrument, entry, the five
candidate stops at fill (wick, 1.0 / 1.5 / 2.0 ATR, session) and the one
chosen with seconds from fill to choice, every stop move, exit, exit reason,
realised R against the chosen stop, MAE, MFE, whether T1 / T2 printed and
when. And per trade, on the same entry: what each of the ten mechanical exits
of §5 (the wick fixed stop, the five §1 rules from it, the four round-2 fixed
stops) and a uniformly random stop in [0.5, 2.0] ATR would have done, each
in R against its own initial risk.

### 3.5 → §5. Pre-registration.

### 3.6 Weekly report (Sunday, from the journal, by the box)

- human R/trade vs each of the ten mechanical exits and the random stop
- behaviour: stop width chosen (distribution, R by width); stop moves within
  2 minutes of entry; **exits while underwater before any of the ten exits
  would have closed the trade** (count, R at exit, what the trade then did
  under the human's own initial stop); widen attempts; signals skipped and
  what they did
- branches: R by level type, confirmation type, zone type; `eq` fills by
  co-occurring zone type
- trades to go; current trajectory against the §5 bar

### 3.7 Interim reviews at 25 and 50 filled trades (added 2026-09-18)

At 100 trades a year is most of what it takes (§2.6), so the trajectory is
shown twice on the way, **without touching the pre-registration**. At the
25th and the 50th filled trade the box writes one review, from the journal,
against the same benchmarks as the final test:

- human R per trade, and each of the ten mechanical exits' R per trade on
  the same entries; which exit is best so far
- the paired difference against the best-of-ten so far: mean, standard
  error, and the count of trades the human beat it on
- the behaviour statistics of §3.6, including exits while underwater before
  any rule would have exited, and the stop width chosen
- branches (level, confirmation, zone), skipped signals and what they did

What an interim review is not: it computes no deflated statistic and
declares nothing; it cannot stop the experiment early, extend it, change the
sample, the benchmark set, the count of fourteen or the 0.95 bar; and the
best-of-ten named at 25 or 50 binds nothing — the scored benchmark is chosen
on all 100 entries. The one effect it can have is on the human, who will
have seen it; that is part of what is being measured, and each review is
journaled with its timestamp so the trades before and after it can be told
apart.

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

**Amendment, 2026-09-12, before trade one.** Part 1's table changed the
experiment: the four round-1 trades whose target printed later did so at
10:20–14:05 after going 1.2–2.1 R against the entry (MAE −1.78, −2.08, −1.78,
−1.20), and three of the eight stopped on their entry bar. A layer that
harvests that is holding through hours of drawdown, not tightening — and a
human who may only tighten from the wick cannot beat the fixed stop on exactly
the trades that point to an edge. So:

- **The initial stop is the human's, chosen at fill within [wick, 2.0 ATR]**
  (§3.3 `STOP`). The choice and the candidates are logged. It is one more
  adjustable choice: **`n_trials` = 10**, not 9. After the fill the
  widen-rejection stays; the stop only moves in the trade's favour.
- **Benchmark set: ten mechanical exits** on the same 100 entries — the wick
  fixed stop, the five §1 trailing rules from it, and the four round-2 fixed
  stops (1.0, 1.5, 2.0 ATR, session; `DESIGN-tjr-intraday.md` §11.4). The
  scored benchmark is the best of the ten on those entries, in-sample for the
  rules as before. Nothing in the set can widen after fill, so the set spans
  exactly the freedom the human has at fill and none of the freedom after it.
- **R is measured against each exit's own initial risk**, the human's against
  the stop the human chose — as the round-2 table did. That is what a
  position sized to its stop experiences; it is also why a wide stop is not
  free: the same excursion pays fewer R.
- **Pass, unchanged in form:** `d_i = R_human,i − R_best_of_ten,i` over the 100
  trades through `validate.deflated_sharpe` with `n_trials = 10`, ≥ 0.95,
  `mean(d) > 0`.
- **Behaviour statistic, reported, not scored:** trades exited by `EXIT NOW`
  while underwater (unrealised R < 0) before any of the ten mechanical exits
  would have closed them — the count, R at exit, and what each did afterwards
  under the human's own initial stop.
- Ten is the count as instructed (one more choice). A stricter count would
  also add the four new fixed-stop comparators, fourteen. If that is the
  count wanted, it is said before trade one; after, it is fixed.

**Second amendment, 2026-09-18, before trade one — supersedes the count
above.**

- **`n_trials` = 14.** The ten mechanical exits of the benchmark set (the
  wick fixed stop, the five §1 trailing rules, the four round-2 fixed
  widths) plus the four human controls (`SKIP`, `STOP`, `MOVE STOP`,
  `EXIT NOW`). Pass: `d_i = R_human,i − R_best_of_ten,i` over the 100 trades
  through `validate.deflated_sharpe` with `n_trials = 14`, ≥ 0.95,
  `mean(d) > 0`.
- **The sample stays 100.** About 180 sessions of both instruments at §2.6's
  rate; not shrunk because it is long.
- **Interim reviews at 25 and 50 filled trades** (§3.7): descriptive, against
  the same ten benchmarks, journaled; they decide nothing and change
  nothing here.

**Third amendment, 2026-09-18, before trade one — two definitions the test
needs and this section did not give.** Found by review of the build: the
pass bar depended on a choice that was not written here, and this section is
hashed into `ARMED`.

- **The 100** are the first 100 filled trades in fill order: taken while
  armed, confirmed by the closed entry bar, not skipped, not voided.
- **The call.** `validate.deflated_sharpe(returns, ppy, trial_sharpes,
  n_trials)` with `returns` = the 100 paired differences `d_i` against the
  best of ten — the exit with the highest mean net R on those 100 entries,
  ties to the first in §5's order; `ppy = 1` (one period is one trade;
  nothing is annualised and it cancels); **`trial_sharpes` = the ten
  per-trade Sharpe ratios, mean over standard deviation, of `R_human −
  R_exit_k`**, one for each mechanical exit, so the expected maximum is set by
  how much the human's edge varies across the comparators; `n_trials = 14`.
  Considered and not taken: the eleven raw R-series Sharpes; a theoretical
  `1/√(n−1)`.

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

**Approved 2026-09-12, as recommended.** The VPS detects the signal and pushes
to Telegram; `SKIP`, `STOP`, `MOVE STOP` and `EXIT NOW` are commands to the
same bot (T-4's); the TradingView MCP bridge draws the §3.2 chart and does
nothing else. Interim, for the 100-trade sample: the Windows PC runs the same
detector on the TradingView feed and pushes to the same bot. The CME feed is
now flagged by both T-5 and T-7; its monthly cost is put in front of the user
when either deploys, before anything is bought.

## 8. What is not in scope

Sizing, target changes, entry discretion, a live account, a second instrument
set, and any change to §5 after trade one.

## 9. Build contract for `tjr_human` (2026-09-18, before the code)

### 9.0 What this is, and the rule it is an exception to

`tjr_human` is a strategy that failed the gauntlet (`DESIGN-tjr-intraday.md`
§12) and is tagged `folklore`. Standing rules 2 and 4 say such a strategy is
never deployed, and that a paper run is a deployment. T-7 is the user's own,
explicit exception, made in the message that wrote those rules, and it is
bounded so that it stays an experiment on the human and not a deployment of
the strategy: **no order leaves the process** (a simulated ledger — Alpaca
serves no futures, and no broker is constructed anywhere in this package);
paper only; 100 filled trades; pre-registered in §5; nothing runs on the VPS;
nothing takes an entry until the user arms it (§9.7).

### 9.1 Layout

| where | what |
|---|---|
| `quantlab/tjr_human/detector.py` | pure and incremental: bars in, events out; no I/O, no clock of its own |
| `quantlab/tjr_human/exits.py` | the ten mechanical exits of §5 and the random stop, causal replay on the entry-timeframe bars |
| `quantlab/tjr_human/trade.py` | one trade's state machine and the rules of the four human controls |
| `quantlab/tjr_human/commands.py` | parse and validate `SKIP`, `STOP`, `MOVE STOP`, `EXIT NOW` and reason codes; the Telegram inbound transport |
| `quantlab/tjr_human/journal.py` | the JSONL journal, its schema, readers |
| `quantlab/tjr_human/report.py` | weekly report, interim reviews at 25 and 50, the final test at 100 |
| `quantlab/tjr_human/chart.py` | display layer through the `tv` CLI; may fail without consequence |
| `quantlab/tjr_human/runner.py` | the loop: feed → detector → alert → commands → trade → journal |
| `tjr_human.py` | CLI: `detect`, `replay`, `status`, `report`, `review`, `final`, `arm` |
| `tests/test_tjr_human.py` | standalone runner, as the other test files |

### 9.2 The signal, on two timeframes

Input is a 1-minute frame per instrument. Structure is read on completed
5-minute context bars built from it, and on 1-hour and 4-hour bins
(`primitives.resample_context`), all by clock arithmetic; a forming bar is
never read. Definitions are those of `tjr_wide_funnel.py --mode final`
(§2.1, §2.3, §2.5) wherever they apply, and:

| element | timeframe | definition |
|---|---|---|
| levels (ASIA, LON, PD, H1, H4 swings, POC_PREV, HVN) | frozen at the 09:25 close | as the funnel; the volume profile is built on the 1-minute bars |
| sweep | 5-minute context, read at the 1-minute bar that completes it, 09:30–09:49 | per-class test, three classes (§2.5) |
| confirmation | every 1-minute close after the sweep's completing minute, to 10:09 | `bos`: close beyond the frozen pre-sweep 5-minute swing; `ote`: close beyond extreme + 0.79 × (swing − extreme); `ifvg`: close through a **fresh** opposing 1-minute gap of day D |
| dealing range, EQ | at the confirmation minute | sweep extreme to the furthest 1-minute extreme since the sweep bar's first minute; EQ the midpoint |
| zones, in discount / premium | `fvg` on 1-minute bars (on the leg, or later in the window); `eq` with §2.3's excursion (ATR(14) of the 5-minute context at the newest completed context bar) and §2.5's open-side test on the touching minute; `ob` and `breaker` on the 5-minute context | freshest stamp wins, as the funnel |
| entry trigger | 1-minute | a minute touches the zone in play (for `eq`, having opened on the far side), then that minute or a later one **closes out of the zone in the trade direction**; the entry is the **next minute's open**, which must be stamped 09:50–10:09 |
| dead setup | 1-minute | price trades beyond the wick stop before the entry → the setup is journaled `invalidated` and the day is done |
| stops at fill | 5-minute ATR at the newest completed context bar | wick, 1.0 / 1.5 / 2.0 ATR, session — `tjr_intraday.stop_price` |
| targets | levels frozen at 09:25 | §2.5; **the bot exits the whole position at T1**; T2 is journaled (whether and when it printed), not traded |
| flat | 1-minute | the 15:55 close |

One setup and one position per instrument per day. Costs are the retail
futures costs of `DESIGN-tjr-intraday.md` §8 (ES 1 bp, NQ 0.5 bp round trip),
charged on every simulated fill, the benchmarks' included.

### 9.3 Events and the human's window

`signal` (confirmation and a zone in play: everything §3.2 draws, with
indicative stop prices) → `armed_entry` is not an event, the bot simply waits →
`fill` (exact candidate stops) → stop moves, `exit`. The signal is pushed
through `quantlab.alerts.notify` (T-4's transport and its 20 s bound) and the
chart is drawn; neither can delay or stop the detector. Four kinds are added
to the pushed set for this package — `signal`, `trade` (fill, stop set, exit),
`reply` (the bot's answer to a command) and, added by review, `status` (the
daily liveness lines and every failure, recovery, hole and clock notice) —
and T-4's three are untouched.

| control | accepted | rule |
|---|---|---|
| `SKIP [reason]` | from `signal` until the fill | the setup is journaled to the end as if traded, with what every exit would have done |
| `STOP <wick\|1.0\|1.5\|2.0\|session\|price> [reason]` | from `signal` until 60 s after the fill, once | before the fill a named width is resolved at the fill, a price is validated at the fill; inside [wick, 2.0 ATR] or rejected and logged; nothing valid by the fill → the wick stop is live from the first tick. **The 60 s after the fill are a grace for the message, not for the market: the wick stop is live during them and a trade it stops stays stopped.** To be wider than the wick on the entry bar, choose before the fill |
| `MOVE STOP <price> [reason]` | in the trade, after the `STOP` window | only in the trade's favour; a widen is rejected and journaled with the requested price |
| `EXIT NOW [reason]` | in the trade | simulated fill at the next observed price; executed first, reason attached after |

Reason codes are `noise`, `structure_changed`, `news`, `gut`. A command
without one is executed — latency matters more than bookkeeping — journaled
as `unspecified`, and the bot asks; a following message that is only a code
attaches to it. Commands are accepted only from `TELEGRAM_CHAT_ID`; every
inbound message is journaled, accepted or not. The token is read from the
environment by the transport and never printed, logged or journaled.

### 9.4 Journal

One JSONL file per instrument-month under `tjr_human_runs/`, append-only,
every record with a UTC millisecond stamp: `signal`, `command`, `fill`,
`stop_set`, `stop_move`, `reject`, `exit`, `skip`, `invalidated`,
`benchmarks` (written after the session closes: each of the ten exits and the
random stop on that entry, R against its own risk, exit reason, exit time;
MAE, MFE, whether and when T1 and T2 printed), `review`, `observe`. Fields as
§3.4. The human writes nothing.

### 9.5 Reports

`report` (weekly, §3.6), `review` (at the 25th and 50th filled trade, §3.7;
written once each, journaled), `final` (at the 100th: `d_i` against the best
of ten on all 100 entries through `validate.deflated_sharpe` with
`n_trials = 14`; pass ≥ 0.95 and `mean(d) > 0`; refuses to run before 100 and
says how many to go). `status` says where the run is and nothing about
whether it is winning.

### 9.6 Feed, interim path

The Windows PC, TradingView Desktop, a two-pane layout (NQ1! and ES1!, 1
minute). The runner reads bars through the repo's `tv` CLI, closed bars only
(a bar is closed once a newer one exists), merges them into a local 1-minute
store seeded from `data/{NQ,ES}_1min_tv.csv`, and refuses to start if the
store does not cover the 20 completed sessions and the 4-hour history the
levels need (§2.4). It polls every 5 s from 09:25 to 10:15 ET and while in a
trade, slower otherwise. The chart in replay mode is refused, as
`TradingViewFeed` refuses it. The same detector runs unchanged on a VPS feed
when one exists; the monthly cost of that feed goes to the user first.

On the PC there is no systemd to load an environment file. The runner reads
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from its own process environment
and from nowhere else; how they get there is a launcher the user writes by
hand and keeps outside the repo. No file in the repo, no test, no replay and
no chat session holds, prints or asks for the token.

### 9.7 Observe, then arm

The runner starts in **observe** mode: it detects, alerts, draws and journals
`observe` records, and takes no entry. Trade one is the start of the
pre-registered sample, so it needs an act: `tjr_human.py arm --reason "..."`
writes `ARMED` with the commit hash of the code and the hash of §5; from the
next signal the bot takes entries and counts. Disarming is refused once a
trade has filled (§5: nothing changes after trade one); stopping the process
is always possible and is journaled on restart as a gap.

### 9.8 The offline replay measures the machine, not the strategy

`tjr_human.py replay --csv … --human none|<script.json>` drives the same
runner over stored 1-minute bars with a simulated clock and a scripted human.
It exists to test the machinery. **On market data it prints machinery facts
only** — signals, fills, invalidations, commands accepted and rejected,
journal integrity, alert and draw calls — and no R, no win rate and no
per-exit totals: those would be a look at outcomes of a new TJR variant
across ten exits and two instruments, which is a third round by another
name, and the mechanical question is closed. Exit arithmetic is verified on
hand-built days in the tests. A `--show-outcomes` flag exists for synthetic
data and refuses market data files.

### 9.9 Acceptance

- The detector is causal: truncating the 1-minute frame at any minute —
  inside a forming 5-minute bar, at a sweep's completing minute, at a
  confirmation, at a touch, at the closing-out minute, at the fill —
  reproduces every earlier event and the cut day as of the cut.
- Fed the same bars in one call or minute by minute, the detector emits the
  same events.
- On hand-built days: every control's rule (a `STOP` outside the band, a
  widen, a second `STOP`, a `SKIP` after the fill, a command from another
  chat id) is rejected and journaled; a wick stop hit inside the 60 s grace
  stays hit; `EXIT NOW` fills at the next observed price; the ten exits and
  the random stop give the R the hand calculation gives.
- `final` refuses before 100; `review` writes once at 25 and once at 50;
  `n_trials` is 14 in code and in the printout.
- No broker, no order, no network except Telegram; with the two variables
  unset nothing tries the network and the runner still journals.
- The three existing test files and `run.py --strategy tjr --synthetic
  --quick` are unchanged; nothing in `quantlab/paper.py`, the registry or
  the gauntlet is touched.

### 9.10 Choices made where §3 was silent — the user can overrule any before trade one

1. The whole position exits at T1; T2 is journaled only.
2. `STOP` takes a named width before the fill, and the 60 s after the fill
   do not suspend the wick stop.
3. Observe mode until `arm`.
4. The replay prints no outcomes on market data.
5. A setup dies if price trades beyond the wick stop before the entry. (On
   the funnel's final fills that is 15 of 66 — §2.6.)
6. The zone in play on a bar is the freshest zone known **before** that bar
   (§2.6's causal reading; 2 of the funnel's 66 fills depend on it).
7. **`TARGET_RULE = "nearest"`**, §2.5 as the user wrote it. But §2.6 found
   that target already traded before the fill on 56 of 66 fills and a median
   0.31 R (NQ) / 0.77 R (ES) from the entry. The detector therefore computes
   and journals, for every signal, both the nearest target and the nearest
   **untaken** one — an opposing-named level or POC / HVN beyond the entry
   that price has not traded at since the 09:30 open — and the constant
   chooses which one the bot exits at. It is recorded in `ARMED` with the
   hashes and cannot change after trade one. `"untaken"` is recommended.

### 9.11 As built (2026-09-18)

Three build stages, then review by three lenses — causality of the detector,
conformity to this pre-registration, operations and secrets — through five
rounds with a fix after each of the first four; the last round found no
defect. About 11,000 lines with their tests; `tests/test_tjr_human.py` (104
tests) and the three older test files pass;
`run.py --strategy tjr --synthetic --quick` unchanged. **The live feed has
never run against a real TradingView session** — it is tested against a fake
`tv` CLI only, and TradingView was not running when that was checked on
2026-09-18 — and nothing is armed, scheduled or installed. **The first
observe session is where the pane read and the chart-style read get checked
against the real app.**

What the reviews established: fed in one call or minute by minute over 41,160
NQ and 41,154 ES bars the detector emits byte-identical events; 95 genuine
truncations reproduce every earlier event and the cut day as of the cut; a
fill does not move when its minute's high, low and close are replaced and
only the open kept; no forming 5-minute, 1-hour or 4-hour bin is read and no
swing is admitted by index. Every control's rule and refusal reproduces on
hand-built days, the 60 s grace included; the ten exits and the random stop
match hand arithmetic and `tjr_trailing_benchmark.replay` on 180 synthetic
replays; reviews are written once at 25 and once at 50 and compute no
deflated statistic; `final` refuses at 99 and runs at 100 with `n_trials =
14`; observe mode takes no entry; `arm` cannot be undone once an entry has
been taken, the entry minute of trade one included. No broker and no order
anywhere in the package; the only network user is Telegram; with the two
variables unset nothing tries the network; six kills mid-write left a clean
journal.

**Definitions made concrete where §9.2–§9.8 were silent.**

- *Freeze and windows.* Levels freeze on the minute stamped 09:29 (it
  completes the 09:25 bar). Sweep bins 09:30–09:45; closing-out minutes
  09:49–10:08, so entries are stamped 09:50–10:09.
- *Warm history* = 20 completed sessions **and** 100 completed 4-hour bins. A
  cold day is never routed. The 30-session seed files therefore give 10
  tradable sessions.
- *Dead setup* runs from the sweep's completing minute, not from the signal;
  "trades beyond" is low ≤ stop (long). An entry open at or beyond the wick
  stop is no trade.
- *Touch and close-out* are one pattern, consumed by its close-out: one closed
  out before 09:49 is spent and a new touch is required. The same minute may
  touch and close out.
- *ATR and session extreme at the fill* are read as of the minute before the
  entry minute, because the entry is that minute's open.
- *1-minute zones.* Leg gaps from the sweep-extreme minute + 2 through the
  confirmation minute, later gaps only when completed inside the entry
  window. `ob` and breaker stamps are the last minute of their 5-minute bin.
  The zone in play on a minute is the freshest known before it (§9.10 item 6).
- *Untaken* = the session's extreme over completed minutes from 09:30 has not
  reached the level; trading exactly at it counts as taken.
- *No target beyond the entry* → the trade still fills and ends by stop or
  the 15:55 flat.
- *Exits.* Names `wick_fixed`, `wick_be_1r`, `wick_atr1.0`, `wick_atr1.5`,
  `wick_swing`, `wick_hybrid`, `fixed_atr1.0`, `fixed_atr1.5`, `fixed_atr2.0`,
  `fixed_session`, `random_stop`. The target is not tested on the entry
  minute; later minutes test stop, then target, then the flat; both in one
  minute is the stop. Costs are charged once, on the entry notional. The
  random stop is seeded from a hash of the setup id.
- *The fill.* A closed-bars feed delivers the entry minute a minute late, so
  the runner opens the trade provisionally at the forming bar's open and the
  closed bar confirms it, re-bases it, or voids it. Only confirmed fills
  count. Closed bars decide stops and targets, with the conventions of the
  ten exits; the forming bar is used for the entry, for the price a `MOVE
  STOP` is tested against, and to fill an `EXIT NOW`.
- *Commands.* Telegram's own message date decides the `SKIP` boundary and,
  when it is not later than receipt, the `STOP` window. **A `SKIP` that
  Telegram dates before the fill voids the fill even when it arrives after
  it** — on Telegram's date only, never on the PC clock, and never when the
  message is dated before the signal existed; both clocks are journaled. A
  voided fill counts nowhere: not toward the 100, the reviews or `final`, and
  it returns the entry count to zero, so `arm` and `disarm` are possible
  again. A `SKIP` that names no instrument is routed by the message time, not
  by each setup's phase on arrival. A command older than 120 s on arrival is
  stale and not executed; a backlog at startup gets one reply. An optional `NQ` / `ES` word addresses a command;
  without it the one setup it can apply to is chosen, two is ambiguous. A
  `STOP` rejected outright does not consume the one choice. A post-fill
  `STOP` takes effect from the minute after the entry minute.
- *Underwater exit* = an `EXIT NOW` at gross R < 0 whose minute is strictly
  earlier than the earliest exit among the ten.
- *Journal.* Extra kinds `event`, `armed`, `gap`, `final`, `run`, `feed`; the
  sequence number is global; a record containing the token's value is
  refused by the writer; a torn last line is skipped with a warning.
- *Token.* It must look like a bot token exactly as it stands (digits, a
  colon, the secret; no space, quote or trailing CR). Anything else turns
  Telegram off for the process with one log line that never carries the
  value; masking covers the raw, stripped, percent-encoded and escaped
  forms. The same rule now guards T-4's path in `quantlab/alerts.py`.
- *Restart.* State is rebuilt by re-feeding the store; a trade left open from
  a past session is closed by replaying its stop, target and flat on the
  stored bars, with exit reason `restart_replay`, and says the human had no
  chance to act — such a trade still counts toward the 100 (§9.13 item 6). If
  the store lacks that session, or has even one missing trading minute
  between the entry and 15:55, the runner refuses to start, names the session
  and the back-fill command, and pushes the refusal; no flag unlocks it.
  Nothing is abandoned silently, at any age.
- *Feed.* `tv ohlcv` reads only the active chart, so by default one `tv ui
  eval` call reads both panes (`pane focus` then `ohlcv` is the fallback).
  Every batch is checked before it is merged — quarter-tick grid, whole
  minutes 60 s apart, high and low bracketing open and close, at least one
  bar with a wick — and the chart type must be Bars or Candles where the app
  exposes it, so Heikin Ashi, Renko or Line Break bars do not reach the
  append-only store. Every sliding window of both real 1-minute files passes
  (about 135,000 each, no refusal). Known limit: a Heikin Ashi series rounded
  to the tick, or a small-brick Renko with wicks, can pass the batch checks,
  and then only the chart-type read stands in the way. The store
  is never rewritten; upstream revisions are journaled. Refusals to start: a
  cold store, a hole of ten trading minutes in the two newest sessions
  unless `--accept-holes`, a chart in replay mode, a PC clock more than 180 s
  off the bars, a store whose first bar differs from the one `ARMED`
  recorded. A day whose levels froze on an un-accepted hole is not routed.
  The process asks Windows not to sleep; it cannot stop a closed lid.
- *Liveness.* At the first poll after 09:25 ET the phone gets one line — mode,
  filled n of 100, store freshness, warm or not — and after 10:10 one line per
  instrument that did not fill, with why. No morning line means the runner
  is down. A failing state is pushed again each trading day it lasts. None of
  these carries R.
- *`arm`* records the commit, the hash of §5, `TARGET_RULE`, `n_trials`, the
  late-fill switch and each store's first bar, refuses uncommitted code
  unless told otherwise, and refuses a directory a replay has written to.
- *Replay.* A file is synthetic only if a sidecar names its hash and it is
  not under `data/`. On market data the journal lives in a temporary
  directory and is deleted; `--show-outcomes` and `--journal-dir` are
  refused; the facts printed are counts, names and flags, not one float.

### 9.12 Dry-run (2026-09-18) — the machine, not the strategy

`tjr_human.py replay` over `data/{NQ,ES}_1min_tv.csv`, no human, 47 s, nothing
left on disk, no R, win rate or total anywhere in its output (grepped).
Thirty sessions seen per instrument; the first twenty are cold, **ten are
routed** (2026-08-31 to 2026-09-11).

| routed sessions, events | NQ | ES |
|---|---|---|
| no sweep | 2 | 3 |
| sweeps | 8 | 7 |
| confirmations = signals | 5 | 5 |
| entry triggers = fills | 2 | 2 |
| invalidated (wick stop traded first) | 4 | 3 |
| expired (no touch, or no close-out) | 2 | 2 |

Four fills in ten session-pairs: too few to say more than that it is in line
with the funnel's 0.34 – 0.44 a session-pair, so **100 trades is about a
year**. Every signal's zone was `eq`. 109 journal records, integrity ok; 24
notices recorded and none sent; 247 shapes queued and none drawn. A second
replay, on synthetic minutes with a scripted human, exercised every control
and every refusal; `status` shows counts and no result; `final` refused at 5
of 100.

### 9.13 Before `arm` — the user's decisions, most consequential first

1. **What "the H1 and H4 swing levels" means.** As specified they are the
   nearest admissible swings over *all history held*, so the store's first
   bar is part of the definition: review measured that a 30-session
   1-minute store does not reproduce the 149-session funnel's levels (sweeps
   differ on 4 of 10 warm NQ days) and that they converge only from about 90
   sessions — and TradingView serves about 30 sessions of 1-minute history.
   Either bound the lookback in the definition (recommended: H1 swings of the
   last 5 completed sessions, H4 of the last 20, both inside the warm rule,
   with the funnel re-cut once for the count), or keep all history and build
   the 1-hour and 4-hour bins from the 149-session 5-minute file. As built
   only guards exist: `ARMED` pins the store's first bar.
2. **The `STOP` band.** "[wick, 2.0 ATR]" assumed the wick stop is the tight
   one. With entries at the midpoint of a three-ATR range it is usually the
   wide one: wider than 2.0 ATR on 10 of the 16 fills of the 30-session files.
   As built the band is the two prices sorted, so on most fills the human can
   only be *tighter* than the wick — the opposite of the amendment's purpose —
   and the named widths 1.0 and 1.5 are rejected although they are benchmark
   members. §5 is hashed, so the band is restated there before `arm`: as
   built; or from 1.0 ATR out to the wick stop plus 1.0 ATR; or the tightest
   to the widest of the five candidates.
3. **`TARGET_RULE`**: `nearest` as written, or `untaken` (§2.6, recommended).
4. **The `trial_sharpes` definition** now in §5's third amendment: confirm it
   or replace it.
5. **Touch and close-out**: a pattern closed out before 09:49 is spent (as
   built), or the earlier touch stays armed. Changes the day's outcome on 2
   of 23 NQ and 1 of 21 ES swept days.
6. **A fill the human never had a chance to manage** (caught up from the
   store after a restart): counts toward the 100 as built; a switch exists
   to exclude fills delivered more than N seconds late.
7. **Dead setup from the sweep** (as built) or only from the signal — about a
   third of invalidations.

Conventions a person could also overrule, recorded in §9.11: the exit order
inside a minute; costs on the entry notional; tick-triggered stops filling
at the stop price; the 120 s stale cut-off and the 10-minute reason window;
whether review text is pushed; whether a changed §5 hash versus `ARMED`
should refuse rather than warn; that with a slow PC clock `STOP`, `MOVE STOP`
and `EXIT NOW` still read PC time when Telegram's date is later than receipt
(the `STOP` window stretches by the skew, inside the 90 s the clock check
allows); that a real exchange halt inside an abandoned trade's session would
leave a hole no back-fill can close and no override exists for it.

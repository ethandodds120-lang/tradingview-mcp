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

## 3. Part 3 — `tjr_human` (T-7)

### 3.1 Signal layer

- Windows and instruments as round 2: sweep 09:30–09:50 ET, entry
  09:50–10:10 ET, flat by 15:55; NQ and ES; 5-minute structure with 1-minute
  entry confirmation (the two-timeframe model of `DESIGN-tjr-intraday.md`
  §11.3).
- Sweep: wick through a §2.3 sweep level (the directional set; ambiguity
  judged within a level class), close back inside.
- Confirmation: any of the three §2 confirmations, on 1-minute bars, after
  the sweep; `ifvg` on a fresh inversion only (§2.3).
- Zone: any of the four §2 zones, in discount / premium of the dealing range;
  `eq` only as a retrace into it (§2.3).
- Entry: price returns to the zone **and** a 1-minute candle closes out of
  the zone in the trade direction. Enter at the next bar's open.
- Initial stop: **the human's choice at fill, within [wick, 2.0 ATR]** (§3.3
  `STOP`, §5 amendment); the wick stop (round 1's definition) is the default
  when no choice arrives in time.
- Targets: next opposing key level of any §2 type, POC and HVN included (T1),
  then the one beyond it (T2).
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

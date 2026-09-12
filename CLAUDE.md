# TradingView MCP — Claude Instructions

84 tools for reading and controlling a live TradingView Desktop chart via CDP (port 9222).

## Decision Tree — Which Tool When

### "What's on my chart right now?"
1. `chart_get_state` → symbol, timeframe, chart type, list of all indicators with entity IDs
2. `data_get_study_values` → current numeric values from all visible indicators (RSI, MACD, BBands, EMAs, etc.)
3. `quote_get` → real-time price, OHLC, volume for current symbol

### "What levels/lines/labels are showing?"
Custom Pine indicators draw with `line.new()`, `label.new()`, `table.new()`, `box.new()`. These are invisible to normal data tools. Use:

1. `data_get_pine_lines` → horizontal price levels drawn by indicators (deduplicated, sorted high→low)
2. `data_get_pine_labels` → text annotations with prices (e.g., "PDH 24550", "Bias Long ✓")
3. `data_get_pine_tables` → table data formatted as rows (e.g., session stats, analytics dashboards)
4. `data_get_pine_boxes` → price zones / ranges as {high, low} pairs

Use `study_filter` parameter to target a specific indicator by name substring (e.g., `study_filter: "Profiler"`).

### "Give me price data"
- `data_get_ohlcv` with `summary: true` → compact stats (high, low, range, change%, avg volume, last 5 bars)
- `data_get_ohlcv` without summary → all bars (use `count` to limit, default 100)
- `quote_get` → single latest price snapshot

### "Analyze my chart" (full report workflow)
1. `quote_get` → current price
2. `data_get_study_values` → all indicator readings
3. `data_get_pine_lines` → key price levels from custom indicators
4. `data_get_pine_labels` → labeled levels with context (e.g., "Settlement", "ASN O/U")
5. `data_get_pine_tables` → session stats, analytics tables
6. `data_get_ohlcv` with `summary: true` → price action summary
7. `capture_screenshot` → visual confirmation

### "Change the chart"
- `chart_set_symbol` → switch ticker (e.g., "AAPL", "ES1!", "NYMEX:CL1!")
- `chart_set_timeframe` → switch resolution (e.g., "1", "5", "15", "60", "D", "W")
- `chart_set_type` → switch chart style (Candles, HeikinAshi, Line, Area, Renko, etc.)
- `chart_manage_indicator` → add or remove studies (use full name: "Relative Strength Index", not "RSI")
- `chart_scroll_to_date` → jump to a date (ISO format: "2025-01-15")
- `chart_set_visible_range` → zoom to exact date range (unix timestamps)

### "Work on Pine Script"
1. `pine_set_source` → inject code into editor
2. `pine_smart_compile` → compile with auto-detection + error check
3. `pine_get_errors` → read compilation errors
4. `pine_get_console` → read log.info() output
5. `pine_get_source` → read current code back (WARNING: can be very large for complex scripts)
6. `pine_save` → save to TradingView cloud
7. `pine_new` → create blank indicator/strategy/library
8. `pine_open` → load a saved script by name

### "Practice trading with replay"
1. `replay_start` with `date: "2025-03-01"` → enter replay mode
2. `replay_step` → advance one bar
3. `replay_autoplay` → auto-advance (set speed with `speed` param in ms)
4. `replay_trade` with `action: "buy"/"sell"/"close"` → execute trades
5. `replay_status` → check position, P&L, current date
6. `replay_stop` → return to realtime

### "Screen multiple symbols"
- `batch_run` with `symbols: ["ES1!", "NQ1!", "YM1!"]` and `action: "screenshot"` or `"get_ohlcv"`

### "Draw on the chart"
- `draw_shape` → horizontal_line, trend_line, rectangle, text (pass point + optional point2)
- `draw_list` → see what's drawn
- `draw_remove_one` → remove by ID
- `draw_clear` → remove all

### "Manage alerts"
- `alert_create` → set price alert (condition: "crossing", "greater_than", "less_than")
- `alert_list` → view active alerts
- `alert_delete` → remove alerts

### "Navigate the UI"
- `ui_open_panel` → open/close pine-editor, strategy-tester, watchlist, alerts, trading
- `ui_click` → click buttons by aria-label, text, or data-name
- `layout_switch` → load a saved layout by name
- `ui_fullscreen` → toggle fullscreen
- `capture_screenshot` → take a screenshot (regions: "full", "chart", "strategy_tester")

### "TradingView isn't running"
- `tv_launch` → auto-detect and launch TradingView with CDP on Mac/Win/Linux
- `tv_health_check` → verify connection is working

## Context Management Rules

These tools can return large payloads. Follow these rules to avoid context bloat:

1. **Always use `summary: true` on `data_get_ohlcv`** unless you specifically need individual bars
2. **Always use `study_filter`** on pine tools when you know which indicator you want — don't scan all studies unnecessarily
3. **Never use `verbose: true`** on pine tools unless the user specifically asks for raw drawing data with IDs/colors
4. **Avoid calling `pine_get_source`** on complex scripts — it can return 200KB+. Only read if you need to edit the code.
5. **Avoid calling `data_get_indicator`** on protected/encrypted indicators — their inputs are encoded blobs. Use `data_get_study_values` instead for current values.
6. **Use `capture_screenshot`** for visual context instead of pulling large datasets — a screenshot is ~300KB but gives you the full visual picture
7. **Call `chart_get_state` once** at the start to get entity IDs, then reference them — don't re-call repeatedly
8. **Cap your OHLCV requests** — `count: 20` for quick analysis, `count: 100` for deeper work, `count: 500` only when specifically needed

### Output Size Estimates (compact mode)
| Tool | Typical Output |
|------|---------------|
| `quote_get` | ~200 bytes |
| `data_get_study_values` | ~500 bytes (all indicators) |
| `data_get_pine_lines` | ~1-3 KB per study (deduplicated levels) |
| `data_get_pine_labels` | ~2-5 KB per study (capped at 50) |
| `data_get_pine_tables` | ~1-4 KB per study (formatted rows) |
| `data_get_pine_boxes` | ~1-2 KB per study (deduplicated zones) |
| `data_get_ohlcv` (summary) | ~500 bytes |
| `data_get_ohlcv` (100 bars) | ~8 KB |
| `capture_screenshot` | ~300 bytes (returns file path, not image data) |

## Tool Conventions

- All tools return `{ success: true/false, ... }`
- Entity IDs (from `chart_get_state`) are session-specific — don't cache across sessions
- Pine indicators must be **visible** on chart for pine graphics tools to read their data
- `chart_manage_indicator` requires **full indicator names**: "Relative Strength Index" not "RSI", "Moving Average Exponential" not "EMA", "Bollinger Bands" not "BB"
- Screenshots save to `screenshots/` directory with timestamps
- OHLCV capped at 500 bars, trades at 20 per request
- Pine labels capped at 50 per study by default (pass `max_labels` to override)

## Architecture

```
Claude Code ←→ MCP Server (stdio) ←→ CDP (localhost:9222) ←→ TradingView Desktop (Electron)
```

Pine graphics path: `study._graphics._primitivesCollection.dwglines.get('lines').get(false)._primitivesDataById`

---

# quantlab — Strategy Research Harness

A separate concern from the MCP tools above: a backtest engine plus a validation
gauntlet, living in `quantlab/`. `run.py` is the gauntlet CLI, `paper.py` the
forward-test CLI. The MCP layer is one of its bar sources (`--tv-symbol`), nothing
more — orders never go out through TradingView.

## The two families

Strategies are split by **what kind of claim they make**. This is encoded, not
just documented: `Strategy.family` is validated at construction and a strategy
cannot reach the registry without declaring one.

**PREDICTIVE** — a chart pattern or market structure tells you what price does
next. A liquidity sweep implies a reversal; a fair value gap implies a
retracement. The claim is about a *specific setup* and rests on a story about
intent. Discretionary logic made mechanical.
→ `tjr`, `fvg` (in `quantlab/strategies/predictive/`)

**SYSTEMATIC** — no view on any individual setup. Harvests a statistical property
measured across the whole dataset: return autocorrelation, volatility clustering,
drift. No narrative about why price *should* move; the claim is about an average.
→ `tsmom`, `ma_cross`, `donchian`, `trend_filter`, `rsi_meanrev`
   (in `quantlab/strategies/systematic/`)

**BENCHMARK** — neither. What both families have to beat.
→ `buy_hold`, `random_entry` (in `quantlab/strategies/benchmarks.py`)

**Both families predict.** Do not describe systematic strategies as "not
predicting" — a momentum rule claims next period's return is related to the last
one, which is a prediction and a falsifiable one. The difference is what the claim
rests on: a pattern implying intent, versus a statistical property persisting.

## Metadata

Every `Strategy` carries `family`, `thesis`, `evidence`, `source`.

`evidence` is one of `published` | `folklore` | `untested`, and it describes what
is known about the *effect*, not how well the strategy backtests. Do not inflate
it. `folklore` is not an insult — it means untested here, which is the entire
reason the harness exists. `tjr` and `fvg` are `folklore` and should stay that way
unless someone produces a citation.

## Commands

```bash
python run.py --strategy tjr --synthetic --quick   # one strategy, full gauntlet
python run.py --compare --synthetic                # all, grouped by family
python run.py --family predictive --csv NQ_5min.csv
python run.py --head-to-head --csv NQ_5min.csv     # families against each other
```

`--head-to-head` surfaces **cost drag** as the interesting column: predictive
strategies trade setups rather than averages, so they trade far more often and
need a proportionally larger gross edge to finish level.

## Rules when working in quantlab/

1. **Never introduce lookahead.** `strategies/predictive/primitives.py:confirmed_swings`
   carries the guard: a pivot at bar `p` is only admissible at `p + swing_right`,
   which is what its `confirmed_at` tag records. Gate on `confirmed_at`, never on
   `index`. No `.shift(-n)`, no `rolling(center=True)`, no `argrelextrema`.
2. **Do not tune parameters or grids while refactoring.** Different numbers after a
   restructuring means a bug, not an improvement. Capture before/after and diff.
3. **Build predictive strategies from the primitives**, not by copying the pivot
   loop. If a primitive is missing, add it — but wiring a new primitive into an
   existing strategy changes its results and is a separate change.
4. **Do not delete strategies that perform badly.** `rsi_meanrev` and `fvg` are the
   calibration; they are supposed to fail.
5. **No live execution layer.** Forward tests may route to an Alpaca *paper*
   account. Live routing is gated behind `--live` + `allow_live` and is not
   something to enable on your own initiative.
6. `random_entry` fails the causality gate. That is a **false positive** — its RNG
   stream position depends on `len(df)`, not on any future bar. Do not "fix" it:
   `validate.random_benchmark` draws from it to score every other strategy's random
   gate, so changing it silently moves every verdict.
7. **The gauntlet prices in its own search.** Gates 5 and 6 (`deflated_sharpe`,
   `probability_of_backtest_overfitting`, `drawdown_distribution`) exist because
   everything above them reports the best of N grid combinations. When adding a
   parameter to a grid, remember it raises the bar the strategy has to clear —
   N goes up, so the Sharpe expected from noise goes up too.
8. **PBO is noisy.** Measured on pure noise over 25 seeds: mean 0.54, sd 0.19,
   range 0.21–0.90. Its gate sits at 0.6, not the nominal 0.5, because gating on
   the boundary fails honest strategies about half the time. Never report a single
   PBO figure as if it were precise.
9. **Bootstrap drawdowns in blocks, never IID.** Volatility clustering is where
   drawdowns come from. On a clustered series the IID bootstrap reported -20.5% at
   the 95th percentile where the block bootstrap reported -33.4%.

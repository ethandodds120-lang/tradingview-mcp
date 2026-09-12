# Strategy evidence — what the `evidence` tags in the registry rest on

Condensed from two research passes (September 2026). The registry's
`evidence` field cites this file. Peer-reviewed or working-paper sources are
listed as such; practitioner backtests are labelled and their numbers are
directional, not precise.

## The evidence hierarchy, strongest first

1. **Time-series (trend) momentum in diversified futures.** Moskowitz, Ooi &
   Pedersen, *Time Series Momentum*, JFE 104 (2012): 58 contracts, 1985–2009,
   every one positively predictable; Hurst, Ooi & Pedersen (2017) extend it a
   century back. Critique: a large share of the result is the volatility scaling
   rather than the signal (Huang, Li, Wang, Zhou). Survives costs at
   institutional scale. → `tsmom`, `trend_filter`, `ma_cross`, `donchian`:
   **published**.
2. **Volatility / variance risk premium.** Carr & Wu (2009); Fallon, Park & Yu
   (FAJ 2015): shorting vol earns Sharpe 0.5–1.5 across asset classes and is
   extremely negatively skewed. Not in the registry.
3. **Cross-sectional momentum.** Jegadeesh & Titman (1993); Asness, Moskowitz &
   Pedersen (2013). Daniel & Moskowitz, *Momentum Crashes*, JFE 122 (2016):
   rare, severe, forecastable crashes driven by the short leg. → `xs_momentum`:
   **published**, crash risk documented.
4. **Pairs / stat-arb.** Gatev, Goetzmann & Rouwenhorst, RFS 19 (2006): ~11%
   excess 1962–2002. Do & Faff (2010, 2012): decayed sharply after the 1990s,
   fragile to costs. Not in the registry.
5. **Short-horizon reversal.** Real, but Heston, Korajczyk & Sadka attribute it
   to liquidity imbalance and bid-ask bounce lasting under an hour — a taker
   pays it rather than earns it. → `rsi_meanrev`, `xs_reversal`: **folklore**,
   kept as the calibration that is supposed to fail on costs.
6. **Classical technical rules.** Brock, Lakonishok & LeBaron (1992) found 26
   rules beat cash on the DJIA 1897–1986; Sullivan, Timmermann & White (JF 1999)
   re-ran them under White's Reality Check: the best rule had no out-of-sample
   edge in the following decade. The canonical warning that in-sample rule
   success dies out of sample.
7. **ICT / Smart Money Concepts.** No reputable peer-reviewed test exists of
   fair value gaps, order blocks or liquidity sweeps (Google Scholar, arXiv, SSRN
   searched; the one journal hit is a pay-to-publish outlet using SMC as an ML
   feature, not testing it). Best independent mechanical backtest (StatOasis,
   practitioner): four ICT entries codified on daily SPY/QQQ/DIA/IWM, none
   significant; best t = +1.22; 0 of 648 backtests beat buy-and-hold. FXNX
   (practitioner): mechanical SMC win rates 38–48% with double-digit losing
   streaks; the 70–80% discretionary rates collapse to ~41% when coded.
   → `tjr`, `fvg`, `tjr_intraday`, `tjr_intraday_smt`: **folklore**.
   In-house (2026-09, `DESIGN-tjr-intraday.md` §9): TJR's session rules made
   mechanical on 149 sessions of ES and NQ 5-minute bars — 12 trades across
   both instruments and the SMT variant, 0 wins, walk-forward OOS Sharpe −1.4
   to −1.8, every one of 32 grid trials at or below zero. The direction call
   was right 5 of 8 times; the stop rule sits inside one bar's range and fires
   first.

## What ICT gestures at that *is* documented

- **Stop clustering at round numbers and price cascades through them.** Osler,
  *Currency Orders and Exchange Rate Dynamics*, JF (2003), and NY Fed Staff
  Report 150: from a complete bank order book, stop-loss orders cluster just
  beyond round numbers and take-profits at them, and price responds to stop
  clusters. This is the real mechanism behind the "liquidity sweep" intuition,
  documented rigorously twenty years before the framework that renamed it.
- **Intraday periodicity.** Heston, Korajczyk & Sadka, JF 65 (2010): return
  continuation at half-hour intervals that are multiples of a trading day,
  persisting ~40 days. The respectable version of "killzones"; the authors'
  own conclusion is that it is not a foundation for a strategy.
- **Order-flow imbalance predicts short-horizon moves** (Cont, Kukanov &
  Stoikov 2014) — but needs level-1/2 data at second horizons; not reachable
  from OHLCV bars.

Untested or unsupported as standalone edges: the specific three-candle FVG
heuristic, the market-maker "stop hunt" narrative, "Power of 3", Optimal Trade
Entry, session-window entries.

## Decay: assume any published edge is smaller now

- McLean & Pontiff, JF 71 (2016): 97 predictors, returns 26% lower out of
  sample and 58% lower after publication.
- Chordia, Subrahmanyam & Tong, JAE 58 (2014): prominent anomalies roughly
  halved after decimalization. (International robustness contested — Auer &
  Rottmann 2019.)

## Methodology the gauntlet implements, and its sources

- Deflated Sharpe: Bailey & López de Prado, JPM 40(5) (2014). → `validate.deflated_sharpe`
- Probability of backtest overfitting via CSCV: Bailey, Borwein, López de Prado
  & Zhu, J. Computational Finance (2017). → `validate.probability_of_backtest_overfitting`
- White's Reality Check / Hansen's SPA: the tools behind Sullivan et al.
- The t-stat bar of 3.0 rather than 2.0: Harvey, Liu & Zhu (2016). → `verdict`
- Purged / embargoed CV: López de Prado, *Advances in Financial Machine
  Learning* (2018). Not yet implemented; walk-forward carries lookback context
  without an embargo.
- Triple penance (expected time under water ~3× the drawdown's duration under
  serial correlation): Bailey & López de Prado, J. Risk (2014). → the block
  bootstrap in `validate.drawdown_distribution` is the empirical version.

## Base rates, for calibration

- Chague, De-Losso & Giovannetti (2020): of 1,551 Brazilian day traders who
  persisted more than 300 days, 97% lost money; 1.1% earned more than minimum
  wage.
- Barber, Lee, Liu & Odean (JFM 2013): under 1% of Taiwanese day traders earn
  reliable abnormal returns net of fees.
- Prop-firm challenge pass rates converge on 5–10% per attempt, and ~70% of
  failures are risk-limit breaches — sizing and discipline, not signal.

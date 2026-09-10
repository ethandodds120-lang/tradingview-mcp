# quantlab

A quantitative strategy research harness. Backtest engine, five strategies, and a
four-part validation gauntlet designed to kill strategies that only *look* good.

There is no live execution layer. That is deliberate — see "Why no bot yet" below.

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
python run.py --strategy fvg --csv NQ_5min.csv --cost-bps 4

# real daily data
python run.py --strategy donchian --yahoo SPY --start 2005-01-01
```

## What's in it

**Engine** (`engine.py`) — vectorized, deliberately pessimistic:
- Signals lagged one full bar. You cannot trade on a bar you haven't finished.
- Costs charged on every unit of position change, commission and slippage separately.
- Volatility targeting from trailing data only, with a no-trade band so you aren't
  charged for micro-rebalances.

**Strategies** (`strategies.py`):
| name | what it is |
|---|---|
| `tsmom` | time-series momentum, the Moskowitz/Ooi/Pedersen rule — your baseline |
| `ma_cross` | moving average crossover |
| `donchian` | turtle-style channel breakout |
| `rsi_meanrev` | short-term mean reversion — included to show how costs kill things |
| `fvg` | fair value gaps, same logic as the Pine indicator |
| `buy_hold` | the benchmark you actually have to beat |

**Validation** (`validate.py`) — the point of the whole thing:

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

## Adding your own strategy

```python
def my_signal(df, threshold=1.5):
    # return desired position at each bar: -1 short, 0 flat, +1 long
    # you may use df up to and including row t. never .shift(-n).
    z = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()
    return -np.sign(z) * (z.abs() > threshold)

REGISTRY["my_strat"] = Strategy("my_strat", my_signal,
                                {"threshold": 1.5},
                                {"threshold": [1.0, 1.5, 2.0, 2.5]})
```

The grid is what walk-forward optimizes over and what sensitivity sweeps.

## Why no bot yet

Order routing is a weekend of work. Knowing whether a signal is real is the entire
job, and it's the part that decides whether the bot makes or loses money.

The published research is blunt about this. McLean and Pontiff found that
peer-reviewed, statistically validated return predictors decline about 26%
out-of-sample and 58% post-publication. Those are the *good* ideas — the ones that
survived academic refereeing. Whatever survives your own search will be weaker than
that, not stronger.

So the sequence is: pass the gauntlet, then paper trade for months, then talk about
capital. If a strategy passes all four tests and holds up on forward data you
collected after building it, that's the point to add execution — and I'll write
that layer with you when you get there.

Note also that the TradingView MCP bridge explicitly excludes automated trading in
its terms; live execution would go through a broker API, not through the chart.

## Honest limitations

- Close-to-close fills. Real fills are worse, especially on gaps and at the open.
- Single instrument. Real trend following gets most of its Sharpe from diversifying
  across dozens of uncorrelated markets — a single-asset test understates it.
- No borrow costs, no margin interest, no overnight financing, no tax.
- Survivorship bias if you feed it a ticker that exists today because it did well.

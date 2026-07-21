# Post-Earnings Announcement Drift (PEAD) Analysis

I trade stocks and options, and I pay close attention to how prices move around earnings.
That's what pulled me into post-earnings announcement drift (PEAD): the idea that after a
company beats or misses earnings, its stock keeps drifting in that direction for days or
weeks instead of re-pricing instantly. It's a real, documented effect in finance research,
and the research also says it should be strongest in small, under-covered stocks and weakest
in mega-caps everyone already watches. I wanted to actually test that myself on real data
instead of just taking it on faith.

**Quick summary:** tested on 2,953 earnings events across 60 stocks, using six independent
statistical methods. No significant effect anywhere, and the null result held up (got stronger,
actually) as the sample grew from 807 to 2,953 events. Along the way I caught two real bugs:
one where my own test suite quietly deleted real production data, and one where an
unwinsorized regression produced a false-positive result that a placebo check exposed.

See [`analysis.ipynb`](analysis.ipynb) for a narrative version with charts rendered inline.

## The question

1. Does PEAD show up, and is it predictable, in real market data?
2. Does the effect get stronger as analyst coverage drops, the way the literature says it
   should? Tested against my own three-tier sample, not just cited.

## Data & methodology

Earnings surprises come from Alpha Vantage for 12 of 20 large-cap tickers, and yfinance for
everything else (the other 8 large-caps plus all mid/small-cap). I validated yfinance against
Alpha Vantage on overlapping data first, surprise percentages matched within about 0.2 points,
before trusting it as the primary source, after Alpha Vantage's free key stayed rate-limited
for over 24 hours across two calendar days, well past its advertised reset.

Daily prices come from Financial Modeling Prep for large-caps and yfinance for mid/small-caps
(FMP's free tier only whitelists a small set of large-cap symbols). SPY is the benchmark used
to compute abnormal drift: a stock's raw move minus the S&P 500's move over the same window,
which isolates the earnings-specific reaction from whatever the broad market was doing.

The universe is 60 stocks across three market-cap tiers, 20 large/20 mid/20 small, spread
across Tech, Financials, Healthcare, Consumer, Defense, and Industrials. Price history goes
back to 2006 (or IPO date) rather than a short recent window, since Alpha Vantage's earnings
history already went back to 1996 for large-caps, so extending price coverage unlocked
hundreds of already-available historical events for free.

"Day 0" is the reported earnings date if released pre-market, otherwise the next trading day.
Large-cap uses Alpha Vantage's explicit report-time field for this; mid/small-cap defaults to
post-market since that field isn't reliably available from yfinance, a disclosed
simplification that's reasonable since most companies report after close anyway.

Signals tested: surprise size, 5-day pre-earnings momentum, Day-0 volume vs. the trailing
20-day average, and volatility change. Drift windows: 5, 10, and 20 trading days after Day 0.
Everything is pulled into a normalized Postgres schema (Docker) and joined through a SQL view
built on layered window functions (`LEAD`/`LAG`, rolling `AVG`/`STDDEV_SAMP`) to compute
forward/trailing returns, volume, and volatility per ticker.

## Results

2,953 earnings events across all 60 tickers, up to 20 years of history where available.

### Quintile buckets

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -10.8% | +0.20% | 0.514 |
| Miss | +1.5% | +0.61% | 0.016 |
| Meet | +6.5% | +0.00% | 0.983 |
| Beat | +15.2% | +0.17% | 0.547 |
| Big beat | +47.3% | +0.19% | 0.579 |

If PEAD were real here, this should read like a staircase. It doesn't.

### Coverage hypothesis (Spearman correlation, by tier)

| Tier | Window | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|---|
| Large-cap | 10d | 1,237 | 20 | 0.006 | 0.835 |
| Large-cap | 20d | 1,237 | 20 | 0.018 | 0.518 |
| Mid-cap | 10d | 831 | 20 | 0.001 | 0.966 |
| Mid-cap | 20d | 831 | 20 | 0.046 | 0.186 |
| Small-cap | 10d | 863 | 20 | -0.021 | 0.531 |
| Small-cap | 20d | 863 | 20 | -0.004 | 0.903 |

### Cluster-robust regression (and a bug I caught mid-analysis)

Repeated events from the same company aren't fully independent, so standard errors should be
clustered by ticker rather than treated as one pile of i.i.d. observations. My first attempt
produced a suspiciously "significant" large-cap result that contradicted the Spearman test on
identical data. Turned out to be two compounding problems: a handful of extreme outlier
surprise values (up to +6,567%, from near-zero EPS estimates) dominating an unwinsorized
linear fit, and unreliable inference because large-cap only had 12 ticker-clusters at the time
(the rule of thumb wants 30-50+). Fixed by winsorizing at the 1st/99th percentile and flagging
any tier with too few clusters to trust.

| Tier | Window | n | clusters | Coef | Cluster-robust p | Corrected p |
|---|---|---|---|---|---|---|
| Large-cap | 10d | 1,237 | 20 | -0.0023 | 0.603 | 0.603 |
| Large-cap | 20d | 1,237 | 20 | 0.0103 | 0.076 | 0.151 |
| Mid-cap | 10d | 831 | 20 | 0.0086 | 0.054 | 0.151 |
| Mid-cap | 20d | 831 | 20 | 0.0172 | 0.020 | 0.122 |
| Small-cap | 10d | 863 | 20 | -0.0039 | 0.365 | 0.438 |
| Small-cap | 20d | 863 | 20 | 0.0057 | 0.291 | 0.436 |

Every tier now has a full 20 clusters (large-cap started at 12 since it began as
Alpha Vantage-only; sourcing the remaining 8 via yfinance fixed this). The one borderline
number, mid-cap at 20 days, doesn't survive Benjamini-Hochberg correction (0.122).

### Classifier: random split vs. walk-forward

A random 80/20 split scored 52.1% (logistic regression) and 53.8% (random forest) against a
50.6% baseline. But a random split on time-series data risks lookahead bias: a model partly
trained on later events predicting an earlier one, same principle as avoiding lookahead bias
in a trading backtest. 5-fold walk-forward validation (only training on chronologically
earlier events) tells a different story: 49.3% and 50.9% average accuracy against a 51.5%
baseline. Both sit at or below baseline in nearly every fold. The walk-forward result is the
one I trust.

### Pipeline validity check

Raw drift tested against SPY's return should come back strongly significant, since most
stocks move with the broad market. It does: r=0.434, p=1.00×10⁻¹³⁵. Good, the null result
elsewhere isn't because the pipeline is broken.

### Event study and placebo check

Average daily abnormal return, 10 days before to 20 after Day 0, cumulated. Abnormal return
spikes right on Day 0 (+0.61% mean vs. ~0.03-0.13% on other days), then the curve goes flat.
The market reprices instantly here, it doesn't drift.

A raw test of "any positive drift after Day 0" (ignoring surprise direction) does come back
significant on its own: mean +0.58%, p=0.0003. So I ran a placebo check, the identical test on
random non-earnings days, 100 times with different draws rather than trusting one lucky
comparison. The real result sits in the lower half of that distribution: placebo mean +0.78%
(range +0.26% to +1.45%) vs. the real +0.58%. Empirical p-value: 0.860. That "significant"
drift isn't earnings-specific. It's this sample's general upward tendency over the period, and
random days without any news show it just as much. Without this check I'd have reported +0.58%
as evidence for PEAD, and I'd have been wrong.

### Market model: proper beta-adjusted abnormal returns

Everywhere above, "abnormal drift" assumes every stock moves 1-for-1 with the market. The
actual academic standard (Brown & Warner 1985) estimates each stock's real beta from a clean
250-day window before the event, with a 30-day gap so the event can't leak into the estimate.
Average beta here is 1.13, meaning these are higher-than-market-sensitivity stocks, so the
simpler method was crediting some of that generic extra sensitivity to "abnormal" earnings
movement. Beta-adjusted, the post-Day-0 drift almost entirely disappears: mean CAR change
Day 0 to Day +20 is -0.065% (p=0.701). A cleaner confirmation of what the placebo check
already found.

### Multiple comparison correction

Applied separately to the 8 quintile/tier tests and the 6 cluster-robust regressions. Nothing
survives in either family. The same pattern (one test looks marginal alone, none survive
correction) reproduced at four different sample sizes as the dataset grew from 807 to 2,953.

## Interpretation

No significant relationship between surprise size and abnormal drift, in any tier, across six
independent methods. The coverage hypothesis didn't hold up either: every tier stayed
indistinguishable from zero, and quadrupling the sample size converged estimates closer to
zero, not further. That's the signature of a genuinely absent effect, not an underpowered test.

The placebo check is the strongest single piece of evidence here. It shows a result that looks
statistically significant on its own can be fully explained by general sample drift that has
nothing to do with earnings, and that testing for that directly, instead of assuming a small
p-value means what it looks like, is what separates a credible result from a false positive.

## Limitations

- Mid/small-cap Day-0 timing defaults to "post-market" rather than a confirmed report time
- A handful of originally-targeted small-cap tickers got dropped for lack of historical data
  density, itself a small sign that lower-coverage stocks have thinner historical records
- The market-model beta estimate needs a clean ~280-day window; 96 of 2,953 events don't have
  one and are excluded from just that analysis, though included everywhere else
- Each family of tests was corrected for multiple comparisons within itself; a maximally strict
  version would correct across all families run over the project's lifetime jointly. Since
  every family already came back null, that wouldn't change the conclusion, but it's worth
  naming as the more rigorous option

## What this demonstrates

**Data engineering**: two earnings APIs and two price APIs feeding a normalized Postgres
schema in Docker, idempotent ingestion with real error handling (a third-party API returned a
rate-limit notice with an HTTP 200 instead of an error code, so my original version silently
treated it as "zero results" instead of failing loudly), data quality checks, lineage
tracking, and a SQL view built on window functions instead of pulling everything into Python.

**Statistics**: quintile bucketing, cluster-robust regression with an outlier-driven false
positive diagnosed and fixed along the way, a market-beta validity check, walk-forward
cross-validation instead of a leaky random split, Benjamini-Hochberg correction across two
test families, and an event-study CAR with a 100-run placebo control that caught a result
that looked real but wasn't.

**Software practices**: a `pytest` suite that independently recomputes expected values from
synthetic fixtures and checks the SQL view against them exactly, `ruff` and the test suite
both wired into CI, a Streamlit dashboard with a static-snapshot fallback for when there's no
live database, and a narrative Jupyter notebook as a companion to the pipeline scripts.

**A bug in the project's own safety net**: the test suite had a fixture that assumed a date
range would always be free of real data. True when I wrote it, false once I extended real
price history further back, so running the tests quietly deleted real production data as a
side effect. Caught by comparing row counts before and after instead of just trusting "tests
passed," then fixed by making the fixture back up and restore whatever's really there.

## Running it

```bash
./setup.sh   # starts Postgres, creates the venv, installs deps, applies schema+view
```

Then, with `FMP_API_KEY` and `ALPHAVANTAGE_API_KEY` set in `.env`:

```bash
python ingest.py                          # large-cap tickers
python ingest_yfinance.py                 # mid/small-cap tickers (no keys needed)
python backfill_earnings_yfinance.py      # fallback earnings source for any AV-rate-limited tickers
python backfill_history.py                # extend price history back to 2006/IPO
python data_quality_checks.py             # validate the loaded data
python eda.py                             # quintile + significance analysis
python tier_analysis.py                   # coverage hypothesis test + cluster-robust regression
python model.py                           # classifier
python validity_checks.py                 # pipeline sanity check + multiple comparison correction
python event_study.py                     # cumulative abnormal return event study + placebo check
python market_model.py                    # beta-adjusted market-model event study
pytest tests/ -v                          # test suite
streamlit run dashboard.py                # interactive dashboard (live DB)
python export_snapshot.py                 # refresh the static snapshot for deployment
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb  # rebuild the notebook
```

The dashboard also runs with no database at all, falling back to the committed
`snapshot/earnings_drift.csv`. That means anyone can clone this repo and run
`streamlit run dashboard.py` with zero setup, and it's also what would power a public hosted
deployment, since a hosted instance has no access to the local Postgres container.

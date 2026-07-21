# Post-Earnings Announcement Drift (PEAD) Analysis

I built this to actually test a market anomaly instead of just reading about it. Post-earnings
announcement drift (PEAD) is the idea that after a company beats or misses earnings estimates,
its stock keeps drifting in that direction for days or weeks afterward instead of re-pricing
instantly. It's a real, well-documented effect in the finance literature, and the literature
also says it should be strongest in small, under-covered stocks and weakest in mega-caps
everyone already watches closely. So I wanted to know: does that actually hold up if I test it
myself, on real data, instead of taking it on faith?

Short answer: no, not in this sample, and I ended up testing that "no" from six different
angles before I trusted it. Full writeup below. See [`analysis.ipynb`](analysis.ipynb) for a
narrative version with charts and tables already rendered inline (no setup needed to view it).

## The question

1. Does PEAD show up, and is it predictable, in real market data?
2. Does the effect get stronger as analyst coverage drops, the way the literature says it
   should? Tested directly against my own three-tier sample, not just cited.

## Data & methodology

Earnings surprises (reported EPS vs. estimated EPS) come from Alpha Vantage for 12 of the 20
large-cap tickers, and yfinance for everything else — the other 8 large-caps plus all
mid/small-cap tickers. I validated yfinance against Alpha Vantage on overlapping data first
(surprise percentages matched within about 0.2 points) before trusting it as the primary
source for the rest, after Alpha Vantage's free-tier key stayed rate-limited for over 24
hours across two calendar days, well past its advertised daily reset.

Daily price history comes from Financial Modeling Prep for large-caps and yfinance for
mid/small-caps (FMP's free tier only allows a small whitelist of large-cap symbols, so
mid/small-caps needed a different source entirely). SPY is the market benchmark used to
compute abnormal drift: a stock's raw price move minus the S&P 500's move over the same
window, which isolates the earnings-specific reaction from whatever the broad market was
doing that week.

The universe is 60 stocks across three market-cap tiers, 20 large/20 mid/20 small, each tier
spread across Tech, Financials, Healthcare, Consumer, Defense, and Industrials so no single
sector dominates. Price history goes back to 2006 (or IPO date, whichever is later) rather
than just a recent window, since Alpha Vantage's earnings history already went back to 1996
for large-caps — extending price coverage unlocked hundreds of already-available historical
earnings events for free, no extra API calls needed.

"Day 0" is the reported earnings date if the release was pre-market, otherwise the next
trading day (a post-market release means nobody can trade on it until the following session
opens). Large-cap uses Alpha Vantage's explicit report-time field for this; mid/small-cap
falls back to assuming post-market, since that field isn't reliably available from yfinance —
a disclosed simplification, and reasonable since most companies do report after close anyway.

Signals tested: surprise size, 5-day pre-earnings momentum, Day-0 volume relative to the
trailing 20-day average, and volatility change (10 days after vs. 20 days before). Drift
windows: 5, 10, and 20 trading days after Day 0.

Everything's pulled via each API into a normalized Postgres schema (running in Docker), then
joined through a SQL view built on layered window functions (`LEAD`/`LAG`, rolling
`AVG`/`STDDEV_SAMP` with `ROWS BETWEEN` frames) to compute forward/trailing returns, volume,
and volatility per ticker. One source of truth per fact, no duplicated data.

## Results

2,953 earnings events across all 60 tickers, spanning up to 20 years of history where
available.

### Bucketed by surprise quintile

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -10.8% | +0.20% | 0.514 |
| Miss | +1.5% | +0.61% | 0.016 |
| Meet | +6.5% | +0.00% | 0.983 |
| Beat | +15.2% | +0.17% | 0.547 |
| Big beat | +47.3% | +0.19% | 0.579 |

If PEAD were real here, this table should read like a staircase from negative to positive.
It doesn't.

### Coverage hypothesis: Spearman correlation, by tier, at two drift horizons

| Tier | Window | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|---|
| Large-cap | 10d | 1,237 | 20 | 0.006 | 0.835 |
| Large-cap | 20d | 1,237 | 20 | 0.018 | 0.518 |
| Mid-cap | 10d | 831 | 20 | 0.001 | 0.966 |
| Mid-cap | 20d | 831 | 20 | 0.046 | 0.186 |
| Small-cap | 10d | 863 | 20 | -0.021 | 0.531 |
| Small-cap | 20d | 863 | 20 | -0.004 | 0.903 |

### Cluster-robust regression (and a real bug I caught mid-analysis)

Repeated earnings events from the same company aren't fully independent, so standard errors
should be clustered by ticker rather than treated as one big pile of i.i.d. observations, the
way the Spearman test above implicitly does. First attempt at this produced a suspiciously
"significant" large-cap result that flatly contradicted the Spearman test on the exact same
data. Turned out to be two compounding problems: a handful of extreme outlier surprise values
(up to +6,567%, from companies whose estimated EPS was near zero) dominating an unwinsorized
linear fit, and unreliable cluster-robust inference because large-cap only had 12
ticker-clusters at the time (the econometric rule of thumb wants 30-50+). Fixed by winsorizing
surprise_percentage at the 1st/99th percentile and flagging any tier with too few clusters to
trust.

| Tier | Window | n | clusters | Coef | Cluster-robust p | Reliable? | Corrected p |
|---|---|---|---|---|---|---|---|
| Large-cap | 10d | 1,237 | 20 | -0.0023 | 0.603 | Yes | 0.603 |
| Large-cap | 20d | 1,237 | 20 | 0.0103 | 0.076 | Yes | 0.151 |
| Mid-cap | 10d | 831 | 20 | 0.0086 | 0.054 | Yes | 0.151 |
| Mid-cap | 20d | 831 | 20 | 0.0172 | 0.020 | Yes | 0.122 |
| Small-cap | 10d | 863 | 20 | -0.0039 | 0.365 | Yes | 0.438 |
| Small-cap | 20d | 863 | 20 | 0.0057 | 0.291 | Yes | 0.436 |

Every tier now has a full 20 ticker-clusters (large-cap originally had only 12, since it
started out as Alpha Vantage-only; sourcing the remaining 8 tickers via yfinance fixed this
along the way), so all six results here are reliable. The one borderline number, mid-cap at
20 days with a raw p of 0.020, doesn't survive Benjamini-Hochberg correction across the 6
tests (corrected p=0.122).

### Classifier: random split vs. walk-forward

Logistic regression and random forest, predicting whether a stock beats the market in the 10
days after earnings. A random 80/20 split scored 52.1% and 53.8% against a 50.6% baseline, but
a random split on time-series data like this risks lookahead bias: a model partly trained on
later events predicting an earlier one. Same principle as avoiding lookahead bias in a trading
backtest, which came up early on in this project too.

5-fold walk-forward validation (only ever training on chronologically earlier events) tells a
different story: 49.3% average accuracy for logistic regression and 50.9% for random forest,
against a 51.5% average baseline. Both sit at or below baseline in nearly every fold. The
random-split numbers were probably a mild lookahead artifact; the walk-forward result is the
one I'd actually trust.

### Pipeline validity check

Before trusting a null result, it's worth checking the pipeline can detect a real relationship
at all. Raw (non-abnormal) drift tested against SPY's return over the same window should come
back strongly significant, since most stocks move with the broad market. It does: r=0.434,
p=1.00×10⁻¹³⁵ across all 2,953 events. Good — the null result elsewhere isn't because the
pipeline is broken.

### Event study: cumulative abnormal return, day by day

Instead of only checking fixed 10/20-day checkpoints, I computed the average daily abnormal
return for every trading day from 10 before to 20 after Day 0, then cumulated it. This is the
textbook way to visualize an effect like this. The result: abnormal return spikes sharply
right on Day 0 (+0.61% mean, versus roughly 0.03-0.13% on other days), then the curve goes
essentially flat for the following 20 days instead of climbing the way real PEAD would show.
The market reprices instantly here, it doesn't drift.

A raw test of "is there any positive drift in the 20 days after Day 0" (regardless of surprise
direction) does come back significant on its own: mean +0.58%, p=0.0003. Before trusting that,
I ran a placebo check — the identical test on random, non-earnings days for the same stocks —
100 times with different random draws, rather than trusting a single comparison that could
just be lucky. The real result sits in the lower half of that empirical distribution: placebo
mean +0.78% (range +0.26% to +1.45% across the 100 runs) versus the real +0.58%. The empirical
p-value — the share of random-day runs with an effect at least that large — is 0.860. In other
words, that "significant" drift isn't earnings-specific at all. It's just this stock sample's
general tendency to drift upward over the period, and random days without any earnings news
show it just as much, usually more. Without this check I'd have reported +0.58% as evidence
for PEAD, and I'd have been wrong.

### Market model: proper beta-adjusted abnormal returns

Everywhere above, "abnormal drift" assumes every stock moves 1-for-1 with the market. The
actual academic standard (the market model, Brown & Warner 1985) is more careful about this:
estimate each stock's real beta from a clean 250-day window before the event, with a 30-day
gap so the event itself can never leak into the estimate, then measure abnormal return against
that stock-specific expectation instead of a flat market assumption.

Average beta across this sample turns out to be 1.13 (median 1.09) — these are
higher-than-market-sensitivity stocks, which means the simpler method was quietly crediting
some of that generic extra sensitivity to "abnormal" earnings movement that wasn't really
about the earnings at all. Once properly beta-adjusted, the post-Day-0 continuation drift
almost entirely disappears: mean CAR change from Day 0 to Day +20 is -0.065% (p=0.701), nowhere
close to significant, and the curve actually declines slightly after the reaction instead of
climbing. This is a cleaner, more theoretically sound confirmation of what the placebo check
already found a different way.

### Multiple comparison correction

Applied separately to the 8 quintile/tier significance tests and the 6 cluster-robust
regression tests. Nothing survives correction in either family. The same pattern — one test
looks marginally significant on its own, nothing survives correction — reproduced
independently at four different sample sizes as the dataset grew from 807 to 1,635 to 2,564
to 2,953 events.

## Interpretation

No statistically significant relationship between earnings surprise size and abnormal
post-earnings drift, in any tier, across six different analytical lenses: bucketed
significance testing, cluster-robust regression, a walk-forward-validated classifier, a
market-beta validity check, an event-study CAR with a 100-run placebo comparison, and a
proper beta-adjusted market model. The coverage hypothesis (that less-covered stocks should
show a stronger effect) didn't hold up either — every tier stayed indistinguishable from
zero, and quadrupling the sample size made the estimates converge closer to zero, not further
from it. That's the signature of a genuinely absent effect, not an underpowered test.

The placebo check is the strongest single piece of evidence here. It shows that even a
result that looks statistically significant on its own can be fully explained by general
sample-level drift that has nothing to do with earnings, and that actually testing for that
directly — instead of assuming a small p-value means what it looks like it means — is what
separates a credible result from a false positive.

## Limitations

- Mid/small-cap Day-0 timing defaults to "post-market" rather than a confirmed report time
  (large-cap uses Alpha Vantage's actual report-time field where that source was used, but
  falls back to the same default for the 8 tickers sourced via yfinance instead)
- A handful of originally-targeted small-cap tickers got dropped from the final view for lack
  of historical data density — itself a small, consistent sign that lower-coverage stocks
  tend to have thinner historical records
- The market-model beta estimate needs a clean ~280-day window before each event; 96 of 2,953
  events (mostly early in a ticker's price history) don't have one and are excluded from just
  that analysis, though they're included everywhere else
- Every family of tests here was corrected for multiple comparisons within itself, but this
  project ran several such families over its lifetime. A maximally strict version would
  correct across all of them jointly. Since every family already came back null, that stricter
  correction wouldn't change the conclusion, but it's worth naming as the more rigorous option

## What this demonstrates

The short version: a real multi-source data pipeline, a genuinely rigorous statistical case
built from six independent angles, and honest reporting of a null result instead of fishing
until something looked significant.

**Data engineering** — two earnings APIs and two price APIs feeding a normalized Postgres
schema in Docker, idempotent ingestion with real error handling (including a bug I actually
hit: a third-party API returning a rate-limit notice with an HTTP 200 instead of an error
code, so the original version silently treated it as "zero results" instead of failing
loudly), data quality checks, `ingested_at` lineage tracking, and a SQL view built on layered
CTEs and window functions rather than pulling everything into Python.

**Statistics** — quintile bucketing with significance testing, cluster-robust regression
(with the outlier-driven false positive I diagnosed and fixed along the way), a
market-beta validity check, walk-forward cross-validation instead of a leaky random split,
Benjamini-Hochberg correction across two separate test families, and an event-study CAR
analysis with a 100-run placebo control that caught a result that looked real but wasn't.

**Software practices** — a `pytest` suite that independently recomputes expected values from
synthetic fixtures and checks the SQL view against them exactly, `ruff` linting and the test
suite both wired into GitHub Actions CI, an interactive Streamlit dashboard with a
static-snapshot fallback for when there's no live database (verified by deliberately breaking
the DB connection and confirming it actually falls back), and a narrative Jupyter notebook as
a readable companion to the pipeline scripts.

**A real bug in the project's own safety net** — the test suite had a fixture that assumed a
certain date range would always be free of real data. That was true when I wrote it, and
became false once I extended real price history further back, so running the test suite
quietly deleted real production data as a side effect. I caught it by comparing row counts
before and after rather than just trusting "tests passed," restored the data, and rewrote the
fixture to back up and restore whatever's really there instead of assuming.

## Running it

```bash
./setup.sh   # starts Postgres, creates the venv, installs deps, applies schema+view
```

Then, with `FMP_API_KEY` and `ALPHAVANTAGE_API_KEY` set in `.env`:

```bash
python ingest.py                          # large-cap tickers
python ingest_yfinance.py                 # mid/small-cap tickers (no keys needed)
python backfill_earnings_yfinance.py      # fallback earnings source for any AV-rate-limited tickers (no keys needed)
python backfill_history.py                # extend price history back to 2006/IPO for everything already loaded
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
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb  # rebuild the notebook with fresh outputs
```

The dashboard also runs with no database at all, falling back to the committed
`snapshot/earnings_drift.csv`. That means anyone can clone this repo and run `streamlit run
dashboard.py` immediately with zero setup, and it's also what would power a public hosted
deployment (Streamlit Community Cloud, say), since a hosted instance has no access to the
local Postgres container.

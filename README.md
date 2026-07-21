# Post-Earnings Announcement Drift (PEAD) Analysis

## Question

Post-earnings announcement drift (PEAD) is a well-documented market anomaly: after a company
reports earnings that beat or miss analyst estimates, its stock price historically keeps
drifting in that direction for days or weeks afterward, rather than instantly re-pricing.
Academic research has also found the effect is strongest in small-cap, low-analyst-coverage
stocks (which get re-priced slowly) and weakest in heavily-covered mega-caps.

This project asks two questions:

1. Does PEAD show up, and is it predictable, in real market data?
2. Does the effect actually get stronger as analyst coverage decreases, as the literature
   suggests — tested directly with our own three-tier sample, not just cited?

## Data & Methodology

- **Earnings surprises** (reported EPS vs. estimated EPS, by quarter): Alpha Vantage for the
  large-cap tier, yfinance for the mid- and small-cap tiers (validated against Alpha Vantage
  on overlapping data — surprise percentages matched within ~0.2 points)
- **Daily price history**: Financial Modeling Prep for large-caps, yfinance for mid/small-caps
  (FMP's free tier only allows a small whitelist of large-cap symbols)
- **Market benchmark**: SPY, used to compute *abnormal drift* — a stock's raw price move minus
  the S&P 500's move over the same window, isolating the earnings-specific reaction from
  broad market movement
- **Universe**: 52 stocks across three market-cap tiers (large/mid/small), each spanning Tech,
  Financials, Healthcare, Consumer, Defense, and Industrials
- **History depth**: price data extends back to 2006 (or IPO date, whichever is later) rather
  than a short recent window — since Alpha Vantage's earnings history already went back to
  1996 for large-caps, extending price coverage unlocked hundreds of already-available
  historical earnings events for free, without any additional API calls against rate-limited
  endpoints
- **"Day 0" definition**: the reported earnings date itself if released pre-market, otherwise
  the next trading day. For the large-cap tier this uses Alpha Vantage's explicit report-time
  field; for mid/small-caps, where that field isn't reliably available, we default to
  "post-market" (a disclosed simplification — most companies do report after close)
- **Signals tested**: earnings surprise size, pre-earnings 5-day price momentum, volume spike
  on Day 0 relative to the trailing 20-day average, and volatility change (10 days after vs.
  20 days before)
- **Drift window**: 5 and 10 trading days after Day 0

Data pulled via each API, stored in a normalized PostgreSQL schema (running in Docker),
joined via a SQL view using layered window functions (`LEAD`/`LAG`, rolling `AVG`/`STDDEV_SAMP`
with `ROWS BETWEEN` frames) to compute forward/trailing returns, volume, and volatility
per ticker — no data duplication, one source of truth per fact.

## Results

**2,564 earnings events** across 52 tickers (12 large-cap, 20 mid-cap, 20 small-cap), spanning
up to 20 years of history where available.

**Bucketed by surprise quintile (all tiers combined):**

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -11.8% | +0.33% | 0.316 |
| Miss | +1.5% | +0.64% | **0.028** |
| Meet | +7.4% | +0.00% | 0.994 |
| Beat | +16.9% | +0.19% | 0.566 |
| Big beat | +50.0% | +0.19% | 0.612 |

**Coverage hypothesis test** (Spearman correlation between surprise size and abnormal drift,
by tier, at two drift horizons):

| Tier | Window | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|---|
| Large-cap | 10d | 848 | 12 | -0.006 | 0.872 |
| Large-cap | 20d | 848 | 12 | 0.023 | 0.504 |
| Mid-cap | 10d | 831 | 20 | 0.001 | 0.966 |
| Mid-cap | 20d | 831 | 20 | 0.046 | 0.186 |
| Small-cap | 10d | 863 | 20 | -0.021 | 0.531 |
| Small-cap | 20d | 863 | 20 | -0.004 | 0.903 |

**Cluster-robust regression** (properly accounts for repeated earnings events from the same
company not being fully independent — a real limitation of the Spearman test above, which
implicitly treats every event as independent). surprise_percentage is winsorized at the
1st/99th percentile first: an initial unwinsorized pass showed a "significant" large-cap
result driven entirely by a handful of extreme outliers (surprise values up to +6,567%, from
near-zero EPS estimates) — a linear regression is highly sensitive to that kind of leverage
point in a way the rank-based Spearman test isn't, which is exactly why Spearman was the
primary test from the start.

| Tier | Window | n | n tickers (clusters) | Coef | Cluster-robust p | Reliable? | Corrected p |
|---|---|---|---|---|---|---|---|
| Large-cap | 10d | 848 | 12 | -0.0022 | 0.578 | No (<20 clusters) | 0.578 |
| Large-cap | 20d | 848 | 12 | 0.0099 | 0.034 | No (<20 clusters) | 0.103 |
| Mid-cap | 10d | 831 | 20 | 0.0086 | 0.054 | Yes | 0.109 |
| Mid-cap | 20d | 831 | 20 | 0.0172 | **0.020** | Yes | 0.103 |
| Small-cap | 10d | 863 | 20 | -0.0039 | 0.365 | Yes | 0.438 |
| Small-cap | 20d | 863 | 20 | 0.0057 | 0.291 | Yes | 0.436 |

The large-cap tier's cluster-robust p-values are flagged unreliable regardless of their value:
with only 12 ticker-clusters, cluster-robust standard errors are known to behave erratically
(the econometric rule of thumb wants 30-50+ clusters). The one borderline mid-cap result
(p=0.020, with a large-enough 20 clusters to trust) does not survive Benjamini-Hochberg
correction across all 6 tests (corrected p=0.103).

**Classifier** (logistic regression + random forest, 2,542 events, 509 held out for testing):
logistic regression scored 49.7% vs. a 50.3% baseline; random forest scored 53.0%. Neither is
a meaningful, consistent improvement.

**Pipeline validity check**: raw (non-abnormal) drift tested against SPY's return over the
same window should come back strongly significant if the pipeline measures things correctly
(basic market beta). It does, decisively: r = 0.431, p = 1.39 × 10⁻¹¹⁶ (n = 2,564).

**Event study (cumulative abnormal return)**: rather than only checking fixed 10/20-day
checkpoints, the average daily abnormal return was computed for every trading day from 10
days before to 20 days after Day 0, then cumulated. The result is the textbook signature of
*no drift*: abnormal return spikes sharply exactly on Day 0 (+0.61% mean, vs. ~0.03-0.13% on
other days) — the market reprices instantly — and the CAR curve is essentially flat for the
20 days afterward, rather than climbing steadily the way real PEAD would show.

A naive test of "is there ANY positive drift in the 20 days after Day 0" (not conditioned on
surprise direction) does come back statistically significant (mean +0.67%, p=0.0002) — but a
**placebo check** (the same test run on random, non-earnings days for the same stocks) shows
an even larger, more significant effect (mean +1.23%, p<0.0001). That means the apparent
post-Day-0 drift isn't earnings-specific at all — it's just this stock sample's general
tendency to drift upward over the study period, showing up equally (or more) on days with no
earnings event whatsoever. Without this placebo check, that +0.67% result would have been
easy to mistakenly report as evidence for PEAD.

**Multiple comparison correction**: applied separately to the 8 quintile/tier significance
tests and the 6 cluster-robust regression tests. Nothing survives correction in either family,
and this same pattern (one test looks marginally significant in isolation, none survive
correction) reproduced independently at three different sample sizes as the dataset grew from
807 to 1,635 to 2,564 events.

## Interpretation

**No statistically significant relationship was found between earnings surprise size and
abnormal post-earnings drift, in any tier, using any of five different analytical lenses**
(bucketed significance test, cluster-robust regression, supervised classifier, market-beta
validity check, and event-study CAR with placebo comparison). The coverage hypothesis
predicted the surprise-drift relationship should strengthen as coverage decreases; instead,
every tier stayed statistically indistinguishable from zero, and quadrupling the sample size
(807 → 2,564 events) made estimates converge closer to zero rather than revealing a hidden
effect — the signature of a genuinely absent relationship rather than an underpowered test.

The event-study placebo check is the single strongest piece of evidence here: it shows that
even a seemingly "significant" post-earnings pattern can be fully explained by general
sample-level drift unrelated to earnings at all, and that testing this directly (rather than
assuming a significant p-value means what it appears to mean) is what separates a credible
result from a false positive.

## Limitations

- Mid/small-cap Day-0 timing uses a "post-market" default rather than confirmed report timing
- Train/test split for the classifier is a simple random 80/20 split, not time-aware; a
  stricter walk-forward validation would be required before treating any effect as tradeable
- 8 of the original 20 target large-cap tickers are still missing earnings data (though they
  do have full price history), blocked by Alpha Vantage's free-tier daily quota
- A handful of originally-targeted small-cap tickers were dropped from the final view due to
  insufficient historical data density — itself a small, consistent data point about
  lower-coverage stocks having thinner historical records
- Large-cap tier has only 12 ticker-clusters, below the threshold generally considered
  reliable for cluster-robust inference — flagged explicitly in the results rather than
  silently trusted

## What this demonstrates

- End-to-end multi-source data pipeline (2 earnings APIs, 2 price APIs), normalized relational
  schema, Dockerized Postgres, idempotent ingestion with proper error handling (including a
  real bug caught and fixed: a third-party API silently returning a rate-limit notice with a
  200 status instead of an error)
- Advanced SQL: layered CTEs, window functions (`LEAD`, `LAG`, rolling `AVG`/`STDDEV_SAMP`
  with custom frames) for time-series feature engineering
- A market-adjusted (abnormal return) metric, not just raw price change
- A specific, falsifiable hypothesis (coverage moderates PEAD strength) tested directly with
  a purpose-built 3-tier sample, rather than assumed from literature
- Statistical significance testing and a supervised classifier, both used to cross-check the
  same conclusion
- `pytest` integration tests that independently recompute expected drift/volume/volatility
  values from synthetic fixtures and verify the SQL view matches exactly
- An interactive Streamlit dashboard for exploring results by tier/sector with a ticker
  drill-down
- Data quality checks (OHLC consistency, duplicate detection, referential integrity between
  tickers and their tier/sector mapping, outlier flagging) run standalone and in CI
- Continuous integration (GitHub Actions): every push spins up a fresh Postgres instance and
  runs the full test suite against it
- Data lineage tracking (`ingested_at` timestamps) on every ingested row
- A pipeline validity/sanity check (does the methodology detect a known, expected relationship
  before trusting it on an unknown one) and a multiple-comparison correction that caught what
  would otherwise have been a reported false positive
- A robustness check across two different drift horizons (10d and 20d), showing the
  conclusion isn't an artifact of picking one particular window
- Cluster-robust regression, with the underlying assumptions checked rather than assumed:
  diagnosed and fixed an outlier-driven false positive in an unwinsorized first pass, and
  explicitly flagged when a tier has too few clusters for the correction to be trustworthy
- A full event-study (cumulative abnormal return) analysis with a placebo control — caught
  a seemingly significant result that turned out to be a general sample artifact, not a real
  earnings effect, by testing random non-earnings days the same way and comparing
- A public, no-database-required dashboard deployment path (static snapshot fallback),
  verified by deliberately breaking the DB connection and confirming the fallback triggers
- Finding and fixing a real bug in the project's own test suite: a fixture that assumed a
  date range would always be free of real data quietly deleted real production data once
  that assumption stopped holding, after price history was extended further back — caught
  by comparing before/after row counts rather than trusting "tests passed" at face value,
  then fixed by backing up and restoring real data instead of relying on the assumption
- Honest reporting of a null result, with a literature-grounded explanation, rather than
  overfitting until something "worked"

## Running it

```bash
./setup.sh   # starts Postgres, creates the venv, installs deps, applies schema+view
```

Then, with `FMP_API_KEY` and `ALPHAVANTAGE_API_KEY` set in `.env`:

```bash
python ingest.py                          # large-cap tickers
python ingest_yfinance.py                 # mid/small-cap tickers (no keys needed)
python backfill_history.py                # extend price history back to 2006/IPO for everything already loaded
python data_quality_checks.py             # validate the loaded data
python eda.py                             # quintile + significance analysis
python tier_analysis.py                   # coverage hypothesis test + cluster-robust regression
python model.py                           # classifier
python validity_checks.py                 # pipeline sanity check + multiple comparison correction
python event_study.py                     # cumulative abnormal return event study + placebo check
pytest tests/ -v                          # test suite
streamlit run dashboard.py                # interactive dashboard (live DB)
python export_snapshot.py                 # refresh the static snapshot for deployment
```

The dashboard also runs without a database at all, using the committed `snapshot/earnings_drift.csv` — this is what powers the public deployment (see below), and lets anyone clone the repo and run `streamlit run dashboard.py` immediately with zero setup.

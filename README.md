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
- **Universe**: 40 stocks across three market-cap tiers (large/mid/small), each spanning Tech,
  Financials, Healthcare, Consumer, Defense, and Industrials
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

**Bucketed by surprise quintile (12 original large-cap tickers):**

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -5.0% | +1.32% | 0.161 |
| Miss | +2.8% | -0.04% | 0.937 |
| Meet | +7.3% | +0.09% | 0.907 |
| Beat | +13.3% | +1.34% | 0.152 |
| Big beat | +31.5% | -0.51% | 0.535 |

No bucket is statistically significant (all p > 0.05).

**Coverage hypothesis test, across all three tiers** (Spearman correlation between surprise
size and abnormal 10-day drift):

| Tier | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|
| Large-cap | 349 | 12 | -0.031 | 0.569 |
| Mid-cap | 273 | 10 | -0.056 | 0.360 |
| Small-cap | 185 | 7 | -0.084 | 0.255 |

**Classifier** (logistic regression + random forest, predicting whether a stock beats the
market in the 10 days after earnings, using surprise size, momentum, volume spike, volatility
change, tier, and sector as features, 807 events across all tiers): logistic regression scored
53.1% accuracy vs. a 51.9% baseline (always guess the majority class) — not a meaningful
improvement. Random forest scored 48.1%, below baseline.

## Interpretation

**No statistically significant relationship was found between earnings surprise size and
abnormal post-earnings drift, in any tier.** The coverage hypothesis predicted the
surprise-drift correlation should strengthen and turn positive as coverage decreases (small-cap
should show real PEAD; large-cap shouldn't). Instead, the correlation is negative in all three
tiers, and its magnitude does grow modestly from large-cap to small-cap — the opposite sign
from classic PEAD, hinting at mild reversal rather than drift, though this is not statistically
significant either and should not be over-claimed given the modest small-cap sample (185
events, 7 tickers).

Two independent methods (a bucketed significance test and a supervised classifier) reached the
same conclusion on the original large-cap sample, and expanding to a genuine 3-tier universe
with real low-coverage stocks did not change that conclusion. This is a stronger, more honestly
supported result than a single confirming test would have been — we set out to verify a
specific literature claim ourselves, rather than just citing it, and reported what the data
actually showed.

## Limitations

- Small-cap sample (185 events, 7 of 10 target tickers — 3 were dropped due to insufficient
  historical earnings/price data, itself a small data point consistent with lower coverage)
  is still modest; a larger sample could detect a smaller effect than this one can rule out
- Mid/small-cap Day-0 timing uses a "post-market" default rather than confirmed report timing
- Train/test split for the classifier is a simple random 80/20 split, not time-aware; a
  stricter walk-forward validation would be required before treating any effect as tradeable
- 8 of the original 20 large-cap tickers are still pending backfill, blocked by Alpha Vantage's
  free-tier daily quota

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
python data_quality_checks.py             # validate the loaded data
python eda.py                             # quintile + significance analysis
python tier_analysis.py                   # coverage hypothesis test
python model.py                           # classifier
pytest tests/ -v                          # test suite
streamlit run dashboard.py                # interactive dashboard
```

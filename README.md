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

**1,635 earnings events** across 32 tickers (12 large-cap, 10 mid-cap, 10 small-cap), spanning
up to 20 years of history where available.

**Bucketed by surprise quintile (all tiers combined):**

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -15.4% | +0.30% | 0.457 |
| Miss | +1.5% | +0.83% | **0.025** |
| Meet | +7.5% | -0.11% | 0.710 |
| Beat | +16.9% | +0.54% | 0.227 |
| Big beat | +50.0% | -0.08% | 0.852 |

**Coverage hypothesis test, across all three tiers** (Spearman correlation between surprise
size and abnormal 10-day drift):

| Tier | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|
| Large-cap | 848 | 12 | -0.006 | 0.872 |
| Mid-cap | 404 | 10 | -0.007 | 0.883 |
| Small-cap | 383 | 10 | -0.025 | 0.623 |

With ~2x the sample size of the original pass, the tier-level correlations converged even
closer to zero rather than revealing a hidden effect — exactly what you'd expect if the true
relationship is genuinely absent, since more data narrows the estimate around its real value.

**Classifier** (logistic regression + random forest, same feature set, 1,635 events, 327 held
out for testing): logistic regression scored 52.3% accuracy vs. a 50.8% baseline (always guess
the majority class); random forest scored 52.6%. Neither is a meaningful improvement — and
with a much larger test set than the original pass, this null result is now far less likely
to be a small-sample fluke.

**Pipeline validity check**: before trusting a null result, it's worth asking whether the
pipeline can detect a real relationship at all. As a sanity check, raw (non-abnormal) drift
was tested against SPY's return over the same window — stocks are expected to move with the
broad market (basic market beta), so this should come back strongly significant if the
pipeline is measuring things correctly. It does, decisively: r = 0.444, p = 1.05 × 10⁻⁸⁰
(n = 1,649). This gives real confidence that the PEAD null result reflects the data, not a
broken pipeline.

**Multiple comparison correction**: 8 significance tests were run in total (5 surprise
buckets + 3 tiers). Run in isolation, the "Miss" bucket's raw p-value (0.025) would have
looked significant at the standard 0.05 threshold — but after a Benjamini-Hochberg
false-discovery-rate correction across all 8 tests, its corrected p-value is 0.202, and
nothing survives correction. This pattern repeated (a different bucket, same outcome) when
the dataset was later expanded from 807 to 1,635 events, reinforcing that it's a real
multiple-testing artifact rather than a one-off coincidence.

## Interpretation

**No statistically significant relationship was found between earnings surprise size and
abnormal post-earnings drift, in any tier, at any sample size tested.** The coverage
hypothesis predicted the surprise-drift correlation should strengthen and turn positive as
coverage decreases (small-cap should show real PEAD; large-cap shouldn't). Instead, all three
tiers show a correlation indistinguishable from zero, and doubling the sample size made the
estimates converge closer to zero, not further from it — the signature of a genuinely absent
effect rather than an underpowered test.

Three independent methods (a bucketed significance test with multiple-comparison correction,
a supervised classifier, and a market-beta validity check) were used to cross-check this
conclusion, and it held at both the original (807-event) and expanded (1,635-event) sample
sizes. This is a substantially stronger and more honestly supported result than a single
confirming test would have been — we set out to verify a specific literature claim ourselves,
rather than just citing it, and reported what the data actually showed at every stage.

## Limitations

- Mid/small-cap Day-0 timing uses a "post-market" default rather than confirmed report timing
- Train/test split for the classifier is a simple random 80/20 split, not time-aware; a
  stricter walk-forward validation would be required before treating any effect as tradeable
- 8 of the original 20 large-cap tickers are still missing earnings data (though they do have
  full price history), blocked by Alpha Vantage's free-tier daily quota
- 3 originally-targeted small-cap tickers (NATH, UTMD, CATO) were dropped from the final view
  due to insufficient historical data density — itself a small, consistent data point about
  lower-coverage stocks having thinner historical records

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
python tier_analysis.py                   # coverage hypothesis test
python model.py                           # classifier
python validity_checks.py                 # pipeline sanity check + multiple comparison correction
pytest tests/ -v                          # test suite
streamlit run dashboard.py                # interactive dashboard (live DB)
python export_snapshot.py                 # refresh the static snapshot for deployment
```

The dashboard also runs without a database at all, using the committed `snapshot/earnings_drift.csv` — this is what powers the public deployment (see below), and lets anyone clone the repo and run `streamlit run dashboard.py` immediately with zero setup.

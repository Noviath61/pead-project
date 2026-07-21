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

- **Earnings surprises** (reported EPS vs. estimated EPS, by quarter): Alpha Vantage for 12 of
  20 large-cap tickers, yfinance for the rest (8 large-cap tickers plus all mid/small-cap) —
  yfinance was validated against Alpha Vantage on overlapping data first (surprise percentages
  matched within ~0.2 points) before being trusted as the source for the remaining tickers,
  after Alpha Vantage's free-tier key stayed rate-limited far longer than its advertised daily
  reset (over 24 hours, across two calendar days)
- **Daily price history**: Financial Modeling Prep for large-caps, yfinance for mid/small-caps
  (FMP's free tier only allows a small whitelist of large-cap symbols)
- **Market benchmark**: SPY, used to compute *abnormal drift* — a stock's raw price move minus
  the S&P 500's move over the same window, isolating the earnings-specific reaction from
  broad market movement
- **Universe**: 60 stocks across three market-cap tiers (20 large/20 mid/20 small), each
  spanning Tech, Financials, Healthcare, Consumer, Defense, and Industrials
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
- **Drift window**: 5, 10, and 20 trading days after Day 0

Data pulled via each API, stored in a normalized PostgreSQL schema (running in Docker),
joined via a SQL view using layered window functions (`LEAD`/`LAG`, rolling `AVG`/`STDDEV_SAMP`
with `ROWS BETWEEN` frames) to compute forward/trailing returns, volume, and volatility
per ticker — no data duplication, one source of truth per fact.

## Results

**2,953 earnings events** across 60 tickers (20 large-cap, 20 mid-cap, 20 small-cap), spanning
up to 20 years of history where available.

**Bucketed by surprise quintile (all tiers combined):**

| Surprise bucket | Median surprise | Avg. abnormal drift (10d) | p-value |
|---|---|---|---|
| Big miss | -10.8% | +0.20% | 0.514 |
| Miss | +1.5% | +0.61% | **0.016** |
| Meet | +6.5% | +0.00% | 0.983 |
| Beat | +15.2% | +0.17% | 0.547 |
| Big beat | +47.3% | +0.19% | 0.579 |

**Coverage hypothesis test** (Spearman correlation between surprise size and abnormal drift,
by tier, at two drift horizons):

| Tier | Window | n events | n tickers | Spearman r | p-value |
|---|---|---|---|---|---|
| Large-cap | 10d | 1,237 | 20 | 0.006 | 0.835 |
| Large-cap | 20d | 1,237 | 20 | 0.018 | 0.518 |
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
| Large-cap | 10d | 1,237 | 20 | -0.0023 | 0.603 | Yes | 0.603 |
| Large-cap | 20d | 1,237 | 20 | 0.0103 | 0.076 | Yes | 0.151 |
| Mid-cap | 10d | 831 | 20 | 0.0086 | 0.054 | Yes | 0.151 |
| Mid-cap | 20d | 831 | 20 | 0.0172 | **0.020** | Yes | 0.122 |
| Small-cap | 10d | 863 | 20 | -0.0039 | 0.365 | Yes | 0.438 |
| Small-cap | 20d | 863 | 20 | 0.0057 | 0.291 | Yes | 0.436 |

Every tier now has a full 20 ticker-clusters (large-cap originally had only 12 — resolved by
sourcing the remaining 8 tickers' earnings via yfinance once Alpha Vantage's key stayed
rate-limited past its advertised reset window), so all six results above are reliable. The one
borderline result (mid-cap 20d, p=0.020) does not survive Benjamini-Hochberg correction across
all 6 tests (corrected p=0.122).

**Classifier** (logistic regression + random forest, 2,931 events): a random 80/20 split
scored 52.1% (logistic regression) and 53.8% (random forest) vs. a 50.6% baseline — but a
random split for time-series data like this risks lookahead bias (a model partly trained on
*later* events predicting an *earlier* one), the same principle discussed early in this
project for avoiding it in backtests. **5-fold walk-forward validation** (only ever training
on chronologically earlier events) gives a materially different, more trustworthy picture:
49.3% average accuracy for logistic regression and 50.9% for random forest, against a 51.5%
average baseline — both models sit at or below baseline in nearly every fold. The random-split
numbers were likely a mild lookahead-bias artifact; the walk-forward result is the one that
should actually be trusted.

**Pipeline validity check**: raw (non-abnormal) drift tested against SPY's return over the
same window should come back strongly significant if the pipeline measures things correctly
(basic market beta). It does, decisively: r = 0.434, p = 1.00 × 10⁻¹³⁵ (n = 2,953).

**Event study (cumulative abnormal return)**: rather than only checking fixed 10/20-day
checkpoints, the average daily abnormal return was computed for every trading day from 10
days before to 20 days after Day 0, then cumulated. The result is the textbook signature of
*no drift*: abnormal return spikes sharply exactly on Day 0 (+0.61% mean, vs. ~0.03-0.13% on
other days) — the market reprices instantly — and the CAR curve is essentially flat for the
20 days afterward, rather than climbing steadily the way real PEAD would show.

A naive test of "is there ANY positive drift in the 20 days after Day 0" (not conditioned on
surprise direction) does come back statistically significant on its own (mean +0.58%,
p=0.0003) — but a **placebo check**, repeated 100 times with different random draws of
non-earnings days for the same stocks (rather than trusting a single draw, which could just
be lucky), shows the real result sitting in the *lower half* of the resulting empirical
distribution: placebo mean +0.78% (range +0.26% to +1.45% across the 100 runs), vs. the real
+0.58%. The empirical p-value — the fraction of random-day runs with a mean at least as large
as the real earnings-day result — is **0.860**. That means the apparent post-Day-0 drift isn't
earnings-specific at all; it's just this stock sample's general tendency to drift upward over
the study period, and random non-earnings days show it just as much (usually more). Without
this check, that +0.58% result would have been easy to mistakenly report as evidence for PEAD.

**Market model (proper beta-adjusted abnormal returns)**: the "abnormal drift" used everywhere
above assumes every stock moves 1-for-1 with the market (a simple market-adjusted return).
The academic standard (the market model, Brown & Warner 1985) is more precise: estimate each
stock's actual beta from a clean 250-day window *before* the event (with a 30-day buffer so
the event itself can never leak into the estimate — the same lookahead-bias discipline used
throughout this project), then measure abnormal return against that stock-specific
expectation instead of a flat market assumption. Average beta across this sample is **1.13**
(median 1.09) — these are higher-than-market-sensitivity stocks, which means the simpler
method was silently crediting some of that generic extra sensitivity to "abnormal" earnings
movement. Once properly beta-adjusted, the post-Day-0 continuation drift **almost entirely
disappears**: mean CAR change from Day 0 to Day +20 is -0.065% (t=-0.38, p=0.701) — not
remotely significant, and the curve is flat-to-slightly-declining after the reaction rather
than climbing. This is a cleaner, more theoretically correct confirmation of the same
conclusion the placebo check already reached from a different angle.

**Multiple comparison correction**: applied separately to the 8 quintile/tier significance
tests and the 6 cluster-robust regression tests. Nothing survives correction in either family,
and this same pattern (one test looks marginally significant in isolation, none survive
correction) reproduced independently at four different sample sizes as the dataset grew from
807 to 1,635 to 2,564 to 2,953 events.

## Interpretation

**No statistically significant relationship was found between earnings surprise size and
abnormal post-earnings drift, in any tier, using any of six different analytical lenses**
(bucketed significance test, cluster-robust regression, walk-forward-validated classifier,
market-beta validity check, event-study CAR with a 100-run placebo comparison, and a proper
beta-adjusted market-model event study). The coverage hypothesis predicted the surprise-drift
relationship should strengthen as coverage decreases;
instead, every tier stayed statistically indistinguishable from zero, and more than tripling
the sample size (807 → 2,953 events) made estimates converge closer to zero rather than
revealing a hidden effect — the signature of a genuinely absent relationship rather than an
underpowered test.

The event-study placebo check is the single strongest piece of evidence here: it shows that
even a seemingly "significant" post-earnings pattern can be fully explained by general
sample-level drift unrelated to earnings at all, and that testing this directly (rather than
assuming a significant p-value means what it appears to mean) is what separates a credible
result from a false positive.

## Limitations

- Mid/small-cap Day-0 timing uses a "post-market" default rather than confirmed report timing
  (large-cap uses Alpha Vantage's explicit report-time field where that source was used, but
  falls back to the same default for the 8 tickers sourced via yfinance instead)
- A handful of originally-targeted small-cap tickers were dropped from the final view due to
  insufficient historical data density — itself a small, consistent data point about
  lower-coverage stocks having thinner historical records
- The market-model beta estimate needs a clean ~280-day window before each event; 96 of 2,953
  events (mostly earlier in a ticker's price history) don't have one and are excluded from that
  specific analysis, though they're included everywhere else
- Every individual family of tests (quintile buckets, tier correlations, cluster-robust
  regressions) was corrected for multiple comparisons *within* that family, but this project
  ran many such families across its lifetime; a maximally strict analysis would correct across
  all of them jointly. Given every family already came back null, a stricter joint correction
  would not change the conclusion — but it's worth naming as the more rigorous alternative

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
- A full event-study (cumulative abnormal return) analysis with a placebo control repeated
  100 times to build an actual empirical null distribution (not trusting one lucky/unlucky
  random draw) — caught a seemingly significant result that turned out to be a general sample
  artifact, not a real earnings effect, once compared against that distribution
- Walk-forward (time-series) cross-validation for the classifier, catching that a naive random
  80/20 split was likely giving a mild lookahead-bias-inflated result — the same principle
  discussed early in this project for avoiding it in trading-strategy backtests
- A proper market-model (beta-adjusted) event study, not just a flat market-adjustment —
  betas estimated from a pre-event window with a deliberate gap before the event itself, so
  the event's own reaction can never leak into the estimate, and this more rigorous method
  produced an even cleaner null result than the simpler one
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
```

The dashboard also runs without a database at all, using the committed `snapshot/earnings_drift.csv` as a fallback — this lets anyone clone the repo and run `streamlit run dashboard.py` immediately with zero setup, and is also what would power a public hosted deployment (e.g. Streamlit Community Cloud), since a hosted instance has no access to the local Postgres container.

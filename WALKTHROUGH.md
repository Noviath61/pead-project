# Walkthrough: what every script in this repo does

This is a plain-English tour of every script, grouped by what it's for rather than
alphabetically. For the actual results and numbers, see `README.md` (full methodology) or
`FINDINGS.md` (1-page summary). This file answers a different question: if you're browsing
the code, what does each file actually do, and why does it exist?

## Getting data in

- **`db.py`** - one shared function (`get_engine()`) that opens a Postgres connection. Every
  other script imports this instead of repeating its own connection string.
- **`ingest.py`** - pulls the original 20 large-cap tickers: earnings history from Alpha
  Vantage, daily prices from Financial Modeling Prep (FMP). Both are free but rate-limited.
- **`ingest_yfinance.py`** - pulls the original 40 mid/small-cap tickers using yfinance
  instead, since FMP's free tier doesn't cover most of them and yfinance has no earnings-data
  rate limit to speak of.
- **`backfill_earnings_yfinance.py`** - a handful of the large-cap tickers still needed
  earnings data from yfinance as a fallback when Alpha Vantage was rate-limited; this fills
  those in specifically.
- **`backfill_history.py`** - extends every ticker's price history back to 2006 (or its IPO
  date if later), since the original ingestion only pulled a shorter window at first.
- **`ingest_expansion.py`** - adds 65 more tickers (all via yfinance) to widen the universe
  from 60 to 125, using the same tier/sector design. Every symbol in here was checked against
  yfinance for real price and earnings history before being added.
- **`data_quality_checks.py`** - a set of sanity checks against the loaded data: no negative
  prices, no duplicate rows, every ticker has a tier/sector mapping, no literal `NaN` values
  in numeric columns (a real bug this caught, see `README.md`). Run after any ingestion.
- **`export_full_dataset.py`** / **`load_full_dataset.py`** - dump the three ingested tables
  to compressed CSVs (`data_export/`), and restore them into a fresh database in seconds.
  This is what makes the whole project reproducible without needing your own API keys.
- **`export_snapshot.py`** - exports just the `earnings_drift` view to a CSV the dashboard can
  fall back to when there's no live database (e.g., a hosted deployment).
- **`export_charts.py`** - regenerates three of the README's chart images (quintile drift,
  event-study CAR, placebo distribution) directly from the database.

*(Not Python, but worth mentioning: `schema.sql`, `migrate_tiers.sql`, `migrate_lineage.sql`,
`schema_ff_factors.sql`, and `create_view.sql` define the database schema and the
`earnings_drift` SQL view that most analysis scripts read from. `queries.sql` is a standalone
set of business-question queries in pure SQL, no Python at all.)*

## Does earnings surprise size predict drift? (the core PEAD question)

- **`eda.py`** - the first, simplest test: bucket every event into quintiles by surprise size
  and check average drift per bucket. If PEAD were real, this should look like a staircase.
- **`tier_analysis.py`** - the "coverage hypothesis" test: does the correlation between
  surprise size and drift get stronger in less-covered (smaller) stocks? Also where the
  cluster-robust regression lives, correcting for the fact that repeated events from the same
  company aren't independent observations.
- **`sector_analysis.py`** - the same coverage-hypothesis test, sliced by sector instead of
  market-cap tier, since a tier can hide a sector-specific effect.
- **`signal_analysis.py`** - tests two alternative signals (Day-0 volume spike, volatility
  change) as predictors of drift, for symmetry with the surprise-size test.
- **`model.py`** - a classifier (logistic regression + random forest) predicting drift
  *direction*, tested both with a random 80/20 split and proper walk-forward
  (chronological) cross-validation to avoid lookahead bias.
- **`model_v2.py`** - checks whether adding `jump_ratio` (the volatility work's strongest
  signal) as a feature actually improves the classifier from `model.py`. It doesn't, and
  that's an honest, expected result explained in the README.
- **`validity_checks.py`** - two things: a sanity check that raw drift correlates with SPY's
  return (it should, confirming the pipeline works), and Benjamini-Hochberg multiple-
  comparison correction across every quintile/tier test run.
- **`event_study.py`** - the academic-standard cumulative abnormal return (CAR) event study,
  day by day from 10 days before to 20 after the event, plus a 100-run placebo check on random
  non-earnings days to make sure any apparent drift isn't just general sample drift.
- **`market_model.py`** - re-does the abnormal-return calculation properly, estimating each
  stock's actual market beta from a clean pre-event window instead of assuming beta=1.
- **`load_ff_factors.py`** - downloads Fama-French daily factor data (market, size, value)
  from Ken French's public data library.
- **`fama_french_model.py`** - the full 3-factor version of the market model, controlling for
  size and value as well as beta.
- **`economic_significance.py`** - prices the most obvious naive PEAD trade (long "big beat",
  short "big miss") against a realistic trading-cost assumption, since statistical
  significance and economic significance are different questions.
- **`survivorship_check.py`** - quantifies how much this ticker universe (companies still
  around and doing well today) inflates the general upward drift seen in the placebo check.
- **`power_analysis.py`** - a formal check that the tests above had enough statistical power
  to detect a real effect if one existed, so a null result means something.
- **`backtest_equity_curve.py`** - turns the naive strategy into an actual compounded equity
  curve (Sharpe ratio, max drawdown, win rate) instead of a single pooled average return.

## What actually matters for selling options around earnings

- **`backtest_math.py`** - a small library of pure, unit-tested math functions (straddle
  pricing, loss-capping, variance decomposition) shared by every script below, so the same
  formula isn't reimplemented five times with five chances to get it wrong differently.
- **`volatility_risk_premium.py`** - measures the "jump ratio": how much bigger is the
  earnings-day move than a normal day for that same stock? This is the volatility-side
  question, separate from PEAD's direction question.
- **`straddle_backtest.py`** - prices a historical-volatility-only at-the-money straddle
  (Brenner-Subrahmanyam approximation) and sells it into every event, to see if that would
  have made money.
- **`iron_condor_backtest.py`** - the same trade with the loss capped by protective wings
  (an iron condor), to see how much a defined-risk structure changes the picture.
- **`garch_volatility_forecast.py`** - fits a GARCH(1,1) model per ticker (a smarter
  volatility forecast than a flat rolling window) and checks whether it changes the
  jump-ratio conclusion.
- **`garch_straddle_backtest.py`** - reruns the full straddle and iron condor backtests with
  GARCH-priced volatility instead of the rolling window, on the identical events, for an
  apples-to-apples comparison.
- **`holding_period_sensitivity.py`** - the backtests above all assume a 1-day option. This
  reprices both backtests at 1, 2, 3, and 5 assumed trading days of holding period, since a
  real option's actual holding period isn't always 1 day.
- **`volatility_crush_check.py`** - checks whether the Day-0 volatility spike lingers into the
  following two weeks, or reverts back to normal almost immediately.
- **`bootstrap_confidence_intervals.py`** - naive vs. cluster bootstrap confidence intervals
  around the headline tier-level correlations, checking whether the same clustering concern
  from `tier_analysis.py` applies to a completely different statistical tool.
- **`vix_regime_analysis.py`** - conditions both the PEAD null result and the volatility-
  selling picture on the broader market's VIX level at the time (calm/normal/stressed), a new
  variable this project hadn't used elsewhere.

## Live tools (not backtests - these hit real market data right now)

- **`live_iv_check.py`** - pulls a real options chain and earnings calendar from yfinance for
  a given ticker, prices the at-the-money straddle, and compares the market's currently-priced
  expected move to that stock's own historical earnings-day pattern. This is the one part of
  the project meant to be rerun before an actual trade, not read as a fixed result.
- **`earnings_screener.py`** - runs the same comparison across the whole tracked universe
  instead of one ticker at a time, ranking results by how rich or cheap current pricing looks.

## Putting it together

- **`dashboard.py`** - a Streamlit dashboard covering both the PEAD and volatility/options
  tracks, with sidebar filters, a per-ticker drill-down, and a live options-chain section.
  Falls back to a static snapshot automatically when there's no database connection.
- **`build_notebook.py`** - generates `analysis.ipynb`, a narrative walkthrough of the same
  results with charts and tables rendered inline, for anyone who'd rather read than run code.

## Tests

- **`tests/test_backtest_math.py`** - unit tests for every function in `backtest_math.py`,
  each hand-calculated and asserted independently of the implementation.
- **`tests/test_live_iv_check.py`** - tests for the date-shifting logic that decides which
  option expiration actually captures an earnings reaction (pre-market vs. post-market
  reporters).
- **`tests/test_drift_view.py`** - independently recomputes expected values from synthetic
  fixtures and checks the `earnings_drift` SQL view against them exactly.
- **`tests/test_ingest.py`** - tests `to_float_or_none`, the small helper `ingest.py` uses to
  safely parse numeric fields from API responses that sometimes return `None` or garbage.

# Resume bullets

Drawn from this project. Trim to fit whatever format/length you need; these are written a bit
long so you have room to cut rather than pad.

1. Designed and built an end-to-end financial research pipeline (Postgres + Docker) ingesting
   20 years of earnings and price history for 125 stocks across 3 external APIs, with
   idempotent upserts, data-quality checks, and lineage tracking; caught a Postgres
   NUMERIC/NaN sorting bug that silently corrupted raw SQL query results.

2. Tested a classic market-efficiency hypothesis (post-earnings announcement drift) against
   6,044 real earnings events using 12+ independent statistical methods - cluster-robust
   regression, walk-forward-validated ML classifiers, an event-study with a 100-run placebo
   control, and a full Fama-French 3-factor model - reaching a null result that grew more
   confident, not less, as the sample size more than doubled.

3. Built and shipped a live decision-support tool that prices real-time options-chain data via
   yfinance and compares it to a stock's own historical earnings-day volatility; used it for
   an actual trade, which surfaced two production bugs invisible to unit testing (wrong option
   contract selected for after-hours reporters; wrong historical holding-period comparison),
   diagnosed and fixed both with regression tests same-day.

4. Fit a GARCH(1,1) volatility-forecasting model across 125 tickers and used it to reprice a
   6,069-event options backtest end-to-end; correctly identified a genuine change in
   statistical significance as the sample widened (rather than dismissing it as noise) and
   explained the underlying mechanism driving the discrepancy.

5. Shipped a Streamlit dashboard with an automatic live-database-to-static-snapshot fallback,
   verified end-to-end with `streamlit.testing.v1.AppTest` rather than manual browser checks,
   plus a scheduled GitHub Actions workflow that runs a 125-ticker screener daily and commits
   results back to the repo.

6. Diagnosed a false-positive regression coefficient caused by outlier leverage from extreme
   (>6,000%) earnings-surprise values, traced it to near-zero EPS estimates, and fixed it via
   winsorization and a minimum-cluster-count validity check - caught only because it
   contradicted an independent, outlier-robust correlation test on identical data.

7. Built a CI pipeline (pytest, mypy, ruff) that runs 20+ analysis scripts against real
   production-scale data on every push, backed by a reproducible data export/import path
   (600K+ rows, compressed CSVs) that replaces a multi-day rate-limited re-ingestion with a
   database restore in seconds.

8. Extended a 60-ticker research universe to 125 tickers mid-project without breaking any
   existing invariant (benchmark row counts, tier/sector balance, CI, dashboard); validated
   every new ticker against live data before ingestion and found/fixed two hardcoded-count
   bugs and one unhandled edge-case crash that the wider, more varied data surfaced.

9. Identified and fixed a compounding bug in a backtest engine that produced a mathematically
   impossible -494% max drawdown from naive cumulative summation; replaced it with proper
   geometric compounding and correctly distinguished a real position-sizing artifact from an
   actual strategy finding.

10. Used bootstrap resampling to test a non-obvious hypothesis about statistical clustering,
    found the opposite of the expected result, verified it wasn't sampling noise by rerunning
    with different seeds and resample counts, and reported the honest outcome instead of the
    expected one.

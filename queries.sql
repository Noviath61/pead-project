-- Standalone showcase queries against the earnings_drift view, answering real questions
-- directly in SQL rather than pandas. Each one runs independently: `psql ... -f queries.sql`
-- or copy any single query into a client. All numbers match what the Python scripts report,
-- since they read from the same view.


-- 1. Biggest earnings beats on record, and what actually happened to the stock afterward.
-- The NaN filter matters: Postgres NUMERIC allows a literal NaN value, and sorts it as
-- larger than every real number - without this filter, any NaN rows silently take over
-- the top of the list instead of the real biggest beats (a real bug this project hit and
-- fixed at the source; see ingest_yfinance.py and README's "What this demonstrates").
SELECT symbol, reported_date, surprise_percentage, abnormal_drift_10d_pct
FROM earnings_drift
WHERE surprise_percentage != 'NaN'
ORDER BY surprise_percentage DESC
LIMIT 10;


-- 2. Biggest misses, same question.
SELECT symbol, reported_date, surprise_percentage, abnormal_drift_10d_pct
FROM earnings_drift
WHERE surprise_percentage != 'NaN'
ORDER BY surprise_percentage ASC
LIMIT 10;


-- 3. Average abnormal drift by sector, ranked. Answers: "which sector reacts most to
--    earnings, and is any of it real?" (see the p-value column - none of it is, after
--    correction, but this is the query a stakeholder would actually ask for first.)
SELECT
    sector,
    count(*) AS n_events,
    round(avg(abnormal_drift_10d_pct), 3) AS avg_abnormal_drift_10d_pct,
    round(stddev(abnormal_drift_10d_pct), 3) AS stddev_abnormal_drift_10d_pct
FROM earnings_drift
GROUP BY sector
ORDER BY avg_abnormal_drift_10d_pct DESC;


-- 4. Tier x sector matrix: average abnormal drift for every combination, so you can see
--    at a glance whether any specific niche (e.g. small-cap Healthcare) stands out.
SELECT
    tier,
    sector,
    count(*) AS n_events,
    round(avg(abnormal_drift_10d_pct), 3) AS avg_abnormal_drift_10d_pct
FROM earnings_drift
GROUP BY tier, sector
ORDER BY tier, avg_abnormal_drift_10d_pct DESC;


-- 5. Quintile bucketing done natively in SQL with NTILE(), the SQL-native equivalent of
--    pandas' qcut() used everywhere else in this project. A common interview question is
--    "how would you bucket continuous data into quantiles in SQL" - this is the answer.
WITH bucketed AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY surprise_percentage) AS surprise_quintile
    FROM earnings_drift
    WHERE surprise_percentage != 'NaN'
)
SELECT
    surprise_quintile,
    count(*) AS n_events,
    round(min(surprise_percentage), 1) AS min_surprise_pct,
    round(max(surprise_percentage), 1) AS max_surprise_pct,
    round(avg(abnormal_drift_10d_pct), 3) AS avg_abnormal_drift_10d_pct
FROM bucketed
GROUP BY surprise_quintile
ORDER BY surprise_quintile;


-- 6. Native SQL correlation check (Postgres has a built-in CORR aggregate - Pearson, not
--    Spearman, so treat this as a cross-check rather than the primary test, which uses
--    Spearman in Python specifically because it's outlier-robust). Same null pattern.
SELECT
    tier,
    count(*) AS n_events,
    round(corr(surprise_percentage, abnormal_drift_10d_pct)::numeric, 4) AS pearson_r
FROM earnings_drift
GROUP BY tier
ORDER BY tier;


-- 7. Rolling 4-quarter average surprise per ticker - a genuine rolling-window use case,
--    answering "is this company's earnings performance trending up or down lately?"
SELECT
    symbol,
    reported_date,
    surprise_percentage,
    round(avg(surprise_percentage) OVER (
        PARTITION BY symbol ORDER BY reported_date
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 2) AS trailing_4q_avg_surprise_pct
FROM earnings_drift
ORDER BY symbol, reported_date;


-- 8. Which tickers most consistently beat estimates? Answers a real research-desk question:
--    "who almost never misses" vs. "who's a coin flip."
SELECT
    symbol,
    count(*) AS n_events,
    count(*) FILTER (WHERE surprise_percentage > 0) AS n_beats,
    round(100.0 * count(*) FILTER (WHERE surprise_percentage > 0) / count(*), 1) AS pct_beats
FROM earnings_drift
GROUP BY symbol
HAVING count(*) >= 20
ORDER BY pct_beats DESC
LIMIT 10;


-- 9. Each sector's single best- and worst-performing ticker by average abnormal drift,
--    using DENSE_RANK() to grab exactly one row per sector per side without a self-join.
WITH ranked AS (
    SELECT
        sector,
        symbol,
        avg(abnormal_drift_10d_pct) AS avg_drift,
        DENSE_RANK() OVER (PARTITION BY sector ORDER BY avg(abnormal_drift_10d_pct) DESC) AS best_rank,
        DENSE_RANK() OVER (PARTITION BY sector ORDER BY avg(abnormal_drift_10d_pct) ASC) AS worst_rank
    FROM earnings_drift
    GROUP BY sector, symbol
)
SELECT sector, symbol, round(avg_drift, 3) AS avg_abnormal_drift_10d_pct, 'best' AS side
FROM ranked WHERE best_rank = 1
UNION ALL
SELECT sector, symbol, round(avg_drift, 3), 'worst'
FROM ranked WHERE worst_rank = 1
ORDER BY sector, side;


-- 10. Year-over-year average abnormal drift, checking whether the (null) result is stable
--     across different market regimes rather than an artifact of any single period.
SELECT
    extract(YEAR FROM reported_date) AS year,
    count(*) AS n_events,
    round(avg(abnormal_drift_10d_pct), 3) AS avg_abnormal_drift_10d_pct
FROM earnings_drift
GROUP BY year
ORDER BY year;


-- 11. Volume spike leaders: the 10 events with the most unusual Day-0 trading volume
--     relative to their own trailing baseline, regardless of surprise direction.
SELECT symbol, reported_date, surprise_percentage, volume_spike_ratio, abnormal_drift_10d_pct
FROM earnings_drift
WHERE volume_spike_ratio IS NOT NULL
ORDER BY volume_spike_ratio DESC
LIMIT 10;

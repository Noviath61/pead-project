DROP VIEW IF EXISTS earnings_drift;
CREATE VIEW earnings_drift AS
WITH daily_returns AS (
    SELECT
        symbol,
        date,
        close,
        volume,
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
            / LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS daily_return
    FROM daily_prices
),
price_features AS (
    SELECT
        symbol,
        date,
        close,
        volume,
        LAG(close, 5)   OVER w AS close_minus_5d,
        LEAD(close, 5)  OVER w AS close_plus_5d,
        LEAD(close, 10) OVER w AS close_plus_10d,
        LEAD(close, 20) OVER w AS close_plus_20d,
        AVG(volume) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS avg_volume_20d_before,
        STDDEV_SAMP(daily_return) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS volatility_20d_before,
        STDDEV_SAMP(daily_return) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 1 FOLLOWING AND 10 FOLLOWING) AS volatility_10d_after
    FROM daily_returns
    WINDOW w AS (PARTITION BY symbol ORDER BY date)
),
reaction_day AS (
    SELECT
        e.symbol,
        e.reported_date,
        e.report_time,
        e.reported_eps,
        e.estimated_eps,
        e.surprise_percentage,
        e.source,
        CASE
            WHEN e.report_time = 'pre-market' THEN e.reported_date
            ELSE (
                SELECT MIN(dp.date)
                FROM daily_prices dp
                WHERE dp.symbol = e.symbol AND dp.date > e.reported_date
            )
        END AS day0_date
    FROM earnings_events e
)
SELECT
    r.symbol,
    tt.tier,
    tt.sector,
    r.reported_date,
    r.report_time,
    r.source AS earnings_source,
    r.reported_eps,
    r.estimated_eps,
    r.surprise_percentage,
    r.day0_date,
    p.close AS day0_close,
    ROUND((p.close - p.close_minus_5d) / p.close_minus_5d * 100, 2) AS pre_earnings_momentum_pct,
    ROUND((p.close_plus_5d - p.close) / p.close * 100, 2)  AS drift_5d_pct,
    ROUND((p.close_plus_10d - p.close) / p.close * 100, 2) AS drift_10d_pct,
    ROUND((spy.close_plus_5d - spy.close) / spy.close * 100, 2)  AS spy_drift_5d_pct,
    ROUND((spy.close_plus_10d - spy.close) / spy.close * 100, 2) AS spy_drift_10d_pct,
    ROUND((p.close_plus_5d - p.close) / p.close * 100, 2)
        - ROUND((spy.close_plus_5d - spy.close) / spy.close * 100, 2)  AS abnormal_drift_5d_pct,
    ROUND((p.close_plus_10d - p.close) / p.close * 100, 2)
        - ROUND((spy.close_plus_10d - spy.close) / spy.close * 100, 2) AS abnormal_drift_10d_pct,
    ROUND((p.close_plus_20d - p.close) / p.close * 100, 2) AS drift_20d_pct,
    ROUND((spy.close_plus_20d - spy.close) / spy.close * 100, 2) AS spy_drift_20d_pct,
    ROUND((p.close_plus_20d - p.close) / p.close * 100, 2)
        - ROUND((spy.close_plus_20d - spy.close) / spy.close * 100, 2) AS abnormal_drift_20d_pct,
    ROUND(p.volume / NULLIF(p.avg_volume_20d_before, 0), 2) AS volume_spike_ratio,
    ROUND(p.volatility_10d_after / NULLIF(p.volatility_20d_before, 0), 2) AS volatility_change_ratio
FROM reaction_day r
JOIN ticker_tiers tt
    ON tt.symbol = r.symbol
JOIN price_features p
    ON p.symbol = r.symbol AND p.date = r.day0_date
JOIN price_features spy
    ON spy.symbol = 'SPY' AND spy.date = r.day0_date
WHERE r.surprise_percentage IS NOT NULL
    AND (r.day0_date - r.reported_date) <= 5
    AND p.close_plus_10d IS NOT NULL
    AND p.close_plus_20d IS NOT NULL
    AND spy.close_plus_10d IS NOT NULL
    AND spy.close_plus_20d IS NOT NULL
    AND p.close_minus_5d IS NOT NULL
    AND p.avg_volume_20d_before IS NOT NULL
    AND p.volatility_20d_before IS NOT NULL;

CREATE TABLE IF NOT EXISTS earnings_events (
    symbol              TEXT NOT NULL,
    fiscal_date_ending  DATE NOT NULL,
    reported_date       DATE NOT NULL,
    report_time         TEXT,
    reported_eps        NUMERIC,
    estimated_eps       NUMERIC,
    surprise            NUMERIC,
    surprise_percentage NUMERIC,
    PRIMARY KEY (symbol, reported_date)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    symbol  TEXT NOT NULL,
    date    DATE NOT NULL,
    open    NUMERIC,
    high    NUMERIC,
    low     NUMERIC,
    close   NUMERIC,
    volume  BIGINT,
    PRIMARY KEY (symbol, date)
);

ALTER TABLE earnings_events ADD COLUMN IF NOT EXISTS source TEXT;
UPDATE earnings_events SET source = 'alpha_vantage' WHERE source IS NULL;
ALTER TABLE earnings_events ALTER COLUMN fiscal_date_ending DROP NOT NULL;

CREATE TABLE IF NOT EXISTS ticker_tiers (
    symbol TEXT PRIMARY KEY,
    tier   TEXT NOT NULL,
    sector TEXT NOT NULL
);

INSERT INTO ticker_tiers (symbol, tier, sector) VALUES
    ('AAPL', 'large', 'Tech'), ('MSFT', 'large', 'Tech'), ('GOOGL', 'large', 'Tech'),
    ('AMZN', 'large', 'Tech'), ('META', 'large', 'Tech'), ('NVDA', 'large', 'Tech'),
    ('TSLA', 'large', 'Tech'), ('AMD', 'large', 'Tech'), ('HOOD', 'large', 'Tech'),
    ('TSM', 'large', 'Tech'),
    ('JPM', 'large', 'Financials'), ('BAC', 'large', 'Financials'), ('V', 'large', 'Financials'),
    ('UNH', 'large', 'Healthcare'), ('JNJ', 'large', 'Healthcare'),
    ('WMT', 'large', 'Consumer'), ('COST', 'large', 'Consumer'),
    ('LMT', 'large', 'Defense'), ('BA', 'large', 'Defense'),
    ('GE', 'large', 'Industrials'),
    ('ASAN', 'mid', 'Tech'), ('PD', 'mid', 'Tech'),
    ('WAL', 'mid', 'Financials'), ('EWBC', 'mid', 'Financials'),
    ('ICUI', 'mid', 'Healthcare'), ('PODD', 'mid', 'Healthcare'),
    ('ANF', 'mid', 'Consumer'), ('FIVE', 'mid', 'Consumer'),
    ('CR', 'mid', 'Industrials'), ('WTS', 'mid', 'Industrials'),
    ('FLWS', 'small', 'Consumer'), ('BBW', 'small', 'Consumer'),
    ('NATH', 'small', 'Consumer'), ('CATO', 'small', 'Consumer'),
    ('UTMD', 'small', 'Healthcare'), ('JBSS', 'small', 'Consumer'),
    ('ASUR', 'small', 'Tech'), ('MCRI', 'small', 'Consumer'),
    ('HAFC', 'small', 'Financials'), ('SCHL', 'small', 'Consumer')
ON CONFLICT (symbol) DO UPDATE SET tier = EXCLUDED.tier, sector = EXCLUDED.sector;

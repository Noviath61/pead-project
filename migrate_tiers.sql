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
    ('HAFC', 'small', 'Financials'), ('SCHL', 'small', 'Consumer'),
    ('BILL', 'mid', 'Tech'), ('TWLO', 'mid', 'Tech'),
    ('PNFP', 'mid', 'Financials'), ('GBCI', 'mid', 'Financials'),
    ('TDOC', 'mid', 'Healthcare'), ('HAE', 'mid', 'Healthcare'),
    ('CHWY', 'mid', 'Consumer'),
    ('HII', 'mid', 'Defense'), ('TXT', 'mid', 'Defense'),
    ('AOS', 'mid', 'Industrials'),
    ('PRGS', 'small', 'Tech'), ('SPSC', 'small', 'Tech'),
    ('FFIN', 'small', 'Financials'), ('CASH', 'small', 'Financials'),
    ('ANIP', 'small', 'Healthcare'), ('HSTM', 'small', 'Healthcare'),
    ('SHOO', 'small', 'Consumer'), ('BOOT', 'small', 'Consumer'),
    ('AAON', 'small', 'Industrials'), ('SXI', 'small', 'Industrials'),
    -- Universe expansion (ingest_expansion.py): roughly doubles the original 60 tickers,
    -- same 3-tier x 6-sector design, all validated against yfinance before being added here.
    ('ORCL', 'large', 'Tech'), ('CSCO', 'large', 'Tech'), ('IBM', 'large', 'Tech'), ('INTC', 'large', 'Tech'),
    ('WFC', 'large', 'Financials'), ('GS', 'large', 'Financials'), ('MS', 'large', 'Financials'), ('C', 'large', 'Financials'),
    ('PFE', 'large', 'Healthcare'), ('MRK', 'large', 'Healthcare'), ('ABT', 'large', 'Healthcare'), ('LLY', 'large', 'Healthcare'),
    ('PG', 'large', 'Consumer'), ('KO', 'large', 'Consumer'), ('PEP', 'large', 'Consumer'),
    ('RTX', 'large', 'Defense'), ('NOC', 'large', 'Defense'), ('GD', 'large', 'Defense'),
    ('HON', 'large', 'Industrials'), ('MMM', 'large', 'Industrials'), ('CAT', 'large', 'Industrials'),
    ('FFIV', 'mid', 'Tech'), ('JKHY', 'mid', 'Tech'), ('ZBRA', 'mid', 'Tech'), ('TYL', 'mid', 'Tech'),
    ('SEIC', 'mid', 'Financials'), ('WBS', 'mid', 'Financials'), ('CBSH', 'mid', 'Financials'), ('UMBF', 'mid', 'Financials'),
    ('MMSI', 'mid', 'Healthcare'), ('OMCL', 'mid', 'Healthcare'), ('CHE', 'mid', 'Healthcare'), ('ENSG', 'mid', 'Healthcare'),
    ('CAKE', 'mid', 'Consumer'), ('TXRH', 'mid', 'Consumer'), ('CROX', 'mid', 'Consumer'), ('DECK', 'mid', 'Consumer'),
    ('CW', 'mid', 'Defense'), ('HEI', 'mid', 'Defense'), ('TDY', 'mid', 'Defense'),
    ('GGG', 'mid', 'Industrials'), ('NDSN', 'mid', 'Industrials'), ('WSO', 'mid', 'Industrials'),
    ('PLXS', 'small', 'Tech'), ('DGII', 'small', 'Tech'), ('NVEC', 'small', 'Tech'), ('ROG', 'small', 'Tech'),
    ('TRMK', 'small', 'Financials'), ('FMBH', 'small', 'Financials'), ('LKFN', 'small', 'Financials'), ('NBTB', 'small', 'Financials'),
    ('USPH', 'small', 'Healthcare'), ('UFPT', 'small', 'Healthcare'), ('CRVL', 'small', 'Healthcare'), ('ANIK', 'small', 'Healthcare'),
    ('CAL', 'small', 'Consumer'), ('WEYS', 'small', 'Consumer'), ('LAKE', 'small', 'Consumer'), ('CULP', 'small', 'Consumer'),
    ('AVAV', 'small', 'Defense'), ('ATRO', 'small', 'Defense'),
    ('LNN', 'small', 'Industrials'), ('ROCK', 'small', 'Industrials'), ('TRS', 'small', 'Industrials'), ('PATK', 'small', 'Industrials')
ON CONFLICT (symbol) DO UPDATE SET tier = EXCLUDED.tier, sector = EXCLUDED.sector;

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import engine, UPSERT_EARNINGS, UPSERT_PRICE

TEST_SYMBOL = "ZZZTEST"
BENCHMARK_SYMBOL = "SPY"

# A date range safely before PRICE_START (2019-01-01) used everywhere else in this
# project, so these synthetic rows can never collide with real ingested data —
# in this dev DB or in a fresh CI database that has no data at all.
START_DATE = "2010-01-04"

# Fixed, hand-chosen daily returns (not all identical, so volatility is nonzero
# and testable). Index 0..24 = 25 trading days before Day 0; 25 = Day 0 itself;
# 26..35 = 10 trading days after Day 0.
DAILY_RETURNS = [
    0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.015, 0.01, 0.0, -0.005,
    0.012, -0.008, 0.02, -0.01, 0.005, 0.01, -0.02, 0.015, -0.005, 0.008,
    -0.01, 0.02, -0.015, 0.01, 0.0,
    0.03,   # Day 0 "reaction" jump
    0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.01, -0.012, 0.005, 0.02,
]

# A different fixed series for the "market benchmark," so the abnormal-drift
# check is genuinely subtracting a different series, not comparing a symbol to itself.
BENCHMARK_DAILY_RETURNS = [
    0.002, 0.001, -0.003, 0.004, 0.0, 0.002, -0.001, 0.003, 0.001, -0.002,
    0.002, 0.001, -0.001, 0.002, 0.0, 0.001, -0.002, 0.003, 0.001, -0.001,
    0.002, 0.0, 0.001, -0.002, 0.002,
    0.005,
    0.001, 0.0, 0.002, -0.001, 0.001, 0.002, -0.001, 0.001, 0.0, 0.002,
]


def _closes_from_returns(start_price, returns):
    closes = [start_price]
    for r in returns[1:]:
        closes.append(closes[-1] * (1 + r))
    return closes


@pytest.fixture
def synthetic_data():
    trading_days = pd.bdate_range(start=START_DATE, periods=len(DAILY_RETURNS))
    reported_date = trading_days[24].date()   # last day "before" Day 0
    day0_date = trading_days[25].date()

    closes = _closes_from_returns(100.0, DAILY_RETURNS)
    benchmark_closes = _closes_from_returns(400.0, BENCHMARK_DAILY_RETURNS)
    volumes = [1_000_000 + i * 1000 for i in range(len(DAILY_RETURNS))]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM earnings_events WHERE symbol = :s"), {"s": TEST_SYMBOL})
        conn.execute(text("DELETE FROM daily_prices WHERE symbol IN (:s, :b) AND date < '2019-01-01'"),
                     {"s": TEST_SYMBOL, "b": BENCHMARK_SYMBOL})
        conn.execute(text("""
            INSERT INTO ticker_tiers (symbol, tier, sector) VALUES (:s, 'test', 'Test')
            ON CONFLICT (symbol) DO NOTHING
        """), {"s": TEST_SYMBOL})

        for d, c, v in zip(trading_days, closes, volumes):
            conn.execute(UPSERT_PRICE, {
                "symbol": TEST_SYMBOL, "date": d.date(),
                "open": c, "high": c, "low": c, "close": c, "volume": v,
            })

        for d, c in zip(trading_days, benchmark_closes):
            conn.execute(UPSERT_PRICE, {
                "symbol": BENCHMARK_SYMBOL, "date": d.date(),
                "open": c, "high": c, "low": c, "close": c, "volume": 1_000_000,
            })

        conn.execute(UPSERT_EARNINGS, {
            "symbol": TEST_SYMBOL, "fiscal_date_ending": None,
            "reported_date": reported_date, "report_time": "post-market",
            "reported_eps": 1.10, "estimated_eps": 1.00,
            "surprise": 0.10, "surprise_percentage": 10.0,
            "source": "test",
        })

    yield {
        "trading_days": trading_days, "closes": closes, "benchmark_closes": benchmark_closes,
        "volumes": volumes, "reported_date": reported_date, "day0_date": day0_date,
    }

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM earnings_events WHERE symbol = :s"), {"s": TEST_SYMBOL})
        conn.execute(text("DELETE FROM daily_prices WHERE symbol IN (:s, :b) AND date < '2019-01-01'"),
                     {"s": TEST_SYMBOL, "b": BENCHMARK_SYMBOL})
        conn.execute(text("DELETE FROM ticker_tiers WHERE symbol = :s"), {"s": TEST_SYMBOL})


def test_day0_date_is_next_trading_day_after_post_market_report(synthetic_data):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT day0_date FROM earnings_drift WHERE symbol = :s"), {"s": TEST_SYMBOL}
        ).fetchone()
    assert row is not None, "expected a row in earnings_drift for the synthetic symbol"
    assert row.day0_date == synthetic_data["day0_date"]


def test_drift_and_momentum_match_independent_calculation(synthetic_data):
    closes = synthetic_data["closes"]
    day0_close = closes[25]

    expected_momentum = (day0_close - closes[20]) / closes[20] * 100
    expected_drift_5d = (closes[30] - day0_close) / day0_close * 100
    expected_drift_10d = (closes[35] - day0_close) / day0_close * 100

    with engine.connect() as conn:
        row = conn.execute(
            text("""SELECT pre_earnings_momentum_pct, drift_5d_pct, drift_10d_pct
                     FROM earnings_drift WHERE symbol = :s"""), {"s": TEST_SYMBOL}
        ).fetchone()

    assert float(row.pre_earnings_momentum_pct) == pytest.approx(expected_momentum, abs=0.01)
    assert float(row.drift_5d_pct) == pytest.approx(expected_drift_5d, abs=0.01)
    assert float(row.drift_10d_pct) == pytest.approx(expected_drift_10d, abs=0.01)


def test_volume_spike_and_volatility_match_independent_calculation(synthetic_data):
    volumes = synthetic_data["volumes"]
    returns_before_window = DAILY_RETURNS[5:25]
    returns_after_window = DAILY_RETURNS[26:36]

    expected_avg_volume_before = np.mean(volumes[5:25])
    expected_volume_spike = volumes[25] / expected_avg_volume_before
    expected_vol_before = np.std(returns_before_window, ddof=1)
    expected_vol_after = np.std(returns_after_window, ddof=1)
    expected_vol_change = expected_vol_after / expected_vol_before

    with engine.connect() as conn:
        row = conn.execute(
            text("""SELECT volume_spike_ratio, volatility_change_ratio
                     FROM earnings_drift WHERE symbol = :s"""), {"s": TEST_SYMBOL}
        ).fetchone()

    assert float(row.volume_spike_ratio) == pytest.approx(expected_volume_spike, rel=0.01)
    assert float(row.volatility_change_ratio) == pytest.approx(expected_vol_change, rel=0.05)


def test_abnormal_drift_subtracts_benchmark_return(synthetic_data):
    closes = synthetic_data["closes"]
    benchmark_closes = synthetic_data["benchmark_closes"]
    day0_close = closes[25]

    expected_drift_10d = round((closes[35] - day0_close) / day0_close * 100, 2)
    expected_benchmark_drift_10d = round(
        (benchmark_closes[35] - benchmark_closes[25]) / benchmark_closes[25] * 100, 2
    )
    expected_abnormal = expected_drift_10d - expected_benchmark_drift_10d

    with engine.connect() as conn:
        row = conn.execute(
            text("""SELECT abnormal_drift_10d_pct
                     FROM earnings_drift WHERE symbol = :s"""), {"s": TEST_SYMBOL}
        ).fetchone()

    assert float(row.abnormal_drift_10d_pct) == pytest.approx(expected_abnormal, abs=0.05)

import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_iv_check import shift_to_reaction_date


def test_pre_market_reacts_same_day():
    earnings_date = datetime.date(2026, 7, 22)  # a Wednesday
    assert shift_to_reaction_date(earnings_date, "pre-market") == earnings_date


def test_post_market_reacts_next_trading_day():
    # Wednesday report, post-market -> reaction is Thursday
    earnings_date = datetime.date(2026, 7, 22)
    assert shift_to_reaction_date(earnings_date, "post-market") == datetime.date(2026, 7, 23)


def test_unknown_report_time_defaults_to_post_market():
    # This is the exact case that mattered live: yfinance's calendar never says pre/post-market,
    # so the default has to assume post-market rather than silently treating it as same-day.
    earnings_date = datetime.date(2026, 7, 22)
    assert shift_to_reaction_date(earnings_date, None) == datetime.date(2026, 7, 23)


def test_post_market_report_on_friday_reacts_next_monday():
    # A Friday report should roll over the weekend to Monday, not Saturday.
    earnings_date = datetime.date(2026, 7, 24)  # a Friday
    assert shift_to_reaction_date(earnings_date, "post-market") == datetime.date(2026, 7, 27)

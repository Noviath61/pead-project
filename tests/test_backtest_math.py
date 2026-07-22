import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_math import (
    brenner_subrahmanyam_premium_pct,
    chain_has_no_contracts,
    compound_wealth_index,
    max_drawdown_pct,
    cap_losses,
    isolate_earnings_move_pct,
    would_clip_to_zero,
)


def test_brenner_subrahmanyam_premium_pct_matches_hand_calculation():
    daily_vol = pd.Series([0.01, 0.02, 0.0])
    result = brenner_subrahmanyam_premium_pct(daily_vol)
    assert result.tolist() == pytest.approx([0.8, 1.6, 0.0])


def test_brenner_subrahmanyam_premium_pct_custom_const():
    daily_vol = pd.Series([0.01, 0.02])
    result = brenner_subrahmanyam_premium_pct(daily_vol, const=0.4)
    assert result.tolist() == pytest.approx([0.4, 0.8])


def test_brenner_subrahmanyam_premium_pct_scales_by_sqrt_of_trading_days():
    daily_vol = pd.Series([0.01])
    result_1day = brenner_subrahmanyam_premium_pct(daily_vol, trading_days=1)
    result_4day = brenner_subrahmanyam_premium_pct(daily_vol, trading_days=4)
    # sqrt(4) = 2, so a 4-day holding period should be exactly double the 1-day price
    assert result_4day.iloc[0] == pytest.approx(result_1day.iloc[0] * 2)
    assert result_4day.iloc[0] == pytest.approx(1.6)


def test_compound_wealth_index_full_size_matches_hand_calculation():
    # +10%, -10%, +20% compounded at full size: 1.00 -> 1.10 -> 0.99 -> 1.188
    returns = pd.Series([10.0, -10.0, 20.0])
    result = compound_wealth_index(returns, position_size_fraction=1.0)
    assert result.tolist() == pytest.approx([1.10, 0.99, 1.188])


def test_compound_wealth_index_partial_size_matches_hand_calculation():
    # Same returns, but each trade only risks half of capital:
    # 1.00 -> 1.05 -> 0.9975 -> 1.09725
    returns = pd.Series([10.0, -10.0, 20.0])
    result = compound_wealth_index(returns, position_size_fraction=0.5)
    assert result.tolist() == pytest.approx([1.05, 0.9975, 1.09725])


def test_compound_wealth_index_never_negative_regardless_of_losing_streak():
    # A naive cumsum of these same returns would go negative; proper compounding
    # can approach zero but never cross it - this is the exact bug this project
    # caught and fixed in backtest_equity_curve.py.
    returns = pd.Series([-90.0] * 5)
    result = compound_wealth_index(returns, position_size_fraction=1.0)
    assert (result > 0).all()


def test_max_drawdown_pct_matches_hand_calculation():
    # Peak of 1.2 at index 1, trough of 0.8 at index 4: (0.8 / 1.2 - 1) * 100
    wealth_index = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
    result = max_drawdown_pct(wealth_index)
    assert result == pytest.approx((0.8 / 1.2 - 1) * 100)


def test_max_drawdown_pct_is_zero_for_a_monotonically_rising_curve():
    wealth_index = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown_pct(wealth_index) == pytest.approx(0.0)


def test_cap_losses_binds_when_uncapped_loss_exceeds_the_wing():
    pnl_pct = pd.Series([-10.0])
    credit_pct = pd.Series([2.0])
    result = cap_losses(pnl_pct, credit_pct, wing_multiplier=3)
    assert result.tolist() == pytest.approx([-6.0])


def test_cap_losses_does_not_bind_when_loss_is_within_the_wing():
    pnl_pct = pd.Series([-3.0])
    credit_pct = pd.Series([2.0])
    result = cap_losses(pnl_pct, credit_pct, wing_multiplier=3)
    assert result.tolist() == pytest.approx([-3.0])


def test_cap_losses_leaves_a_winning_trade_untouched():
    pnl_pct = pd.Series([5.0])
    credit_pct = pd.Series([2.0])
    result = cap_losses(pnl_pct, credit_pct, wing_multiplier=3)
    assert result.tolist() == pytest.approx([5.0])


def test_isolate_earnings_move_pct_with_zero_non_event_days_returns_raw_move():
    # No non-event days between now and expiration means the whole straddle price IS the
    # earnings move, nothing to net out.
    result = isolate_earnings_move_pct(
        raw_expected_move_pct=5.26, normal_daily_vol_pct=2.33, non_event_trading_days=0
    )
    assert result == pytest.approx(5.26)


def test_isolate_earnings_move_pct_matches_hand_calculation():
    # total variance (10/100)^2 = 0.01, normal variance (2/100)^2 * 5 = 0.002,
    # earnings variance 0.008, sqrt(0.008) * 100 = 8.944...
    result = isolate_earnings_move_pct(
        raw_expected_move_pct=10.0, normal_daily_vol_pct=2.0, non_event_trading_days=5
    )
    assert result == pytest.approx(8.94427, abs=1e-4)


def test_isolate_earnings_move_pct_clips_at_zero_instead_of_going_negative():
    # This is the exact case a far-dated expiration produced during development: normal
    # volatility over many non-event days can explain MORE variance than the whole straddle
    # priced in, which would make the earnings-specific variance negative and its square root
    # undefined. Clipped at zero rather than raising or returning NaN.
    result = isolate_earnings_move_pct(
        raw_expected_move_pct=11.45, normal_daily_vol_pct=2.36, non_event_trading_days=27
    )
    assert result == pytest.approx(0.0)


def test_would_clip_to_zero_true_for_the_far_dated_regression_case():
    assert would_clip_to_zero(
        raw_expected_move_pct=11.45, normal_daily_vol_pct=2.36, non_event_trading_days=27
    ) is True


def test_would_clip_to_zero_true_even_within_a_short_horizon():
    # The bug this caught live: AAPL was only 8 trading days out, well within a "short and
    # therefore trustworthy" horizon, but its own recent realized vol was high enough that
    # the netting assumption still broke down. Being close to the event doesn't guarantee
    # this can't happen.
    assert would_clip_to_zero(
        raw_expected_move_pct=4.9506, normal_daily_vol_pct=2.3679, non_event_trading_days=7
    ) is True


def test_would_clip_to_zero_false_when_the_straddle_price_comfortably_covers_normal_vol():
    assert would_clip_to_zero(
        raw_expected_move_pct=6.6406, normal_daily_vol_pct=1.8260, non_event_trading_days=7
    ) is False


def test_chain_has_no_contracts_true_when_either_side_is_empty():
    # The exact bug this caught live: a handful of thin, illiquid tickers (added when the
    # ticker universe was widened) list an expiration with zero call or put contracts, which
    # crashed with an unhandled IndexError instead of a clean skip.
    has_calls = pd.DataFrame({"strike": [100.0], "bid": [1.0], "ask": [1.2]})
    no_contracts = pd.DataFrame({"strike": [], "bid": [], "ask": []})
    assert chain_has_no_contracts(no_contracts, has_calls) is True
    assert chain_has_no_contracts(has_calls, no_contracts) is True
    assert chain_has_no_contracts(no_contracts, no_contracts) is True


def test_chain_has_no_contracts_false_when_both_sides_have_rows():
    has_calls = pd.DataFrame({"strike": [100.0], "bid": [1.0], "ask": [1.2]})
    has_puts = pd.DataFrame({"strike": [100.0], "bid": [0.9], "ask": [1.1]})
    assert chain_has_no_contracts(has_calls, has_puts) is False

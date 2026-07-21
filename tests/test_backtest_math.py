import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_math import (
    brenner_subrahmanyam_premium_pct,
    compound_wealth_index,
    max_drawdown_pct,
    cap_losses,
)


def test_brenner_subrahmanyam_premium_pct_matches_hand_calculation():
    daily_vol = pd.Series([0.01, 0.02, 0.0])
    result = brenner_subrahmanyam_premium_pct(daily_vol)
    assert result.tolist() == pytest.approx([0.8, 1.6, 0.0])


def test_brenner_subrahmanyam_premium_pct_custom_const():
    daily_vol = pd.Series([0.01, 0.02])
    result = brenner_subrahmanyam_premium_pct(daily_vol, const=0.4)
    assert result.tolist() == pytest.approx([0.4, 0.8])


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

import pandas as pd

# Brenner & Subrahmanyam (1988): ATM straddle price ~= 0.8 * S * daily_sigma * sqrt(T).
BRENNER_SUBRAHMANYAM_CONST = 0.8


def brenner_subrahmanyam_premium_pct(
    daily_vol: pd.Series, const: float = BRENNER_SUBRAHMANYAM_CONST, trading_days: int = 1
) -> pd.Series:
    return const * daily_vol * 100 * (trading_days ** 0.5)


def compound_wealth_index(trade_return_pct: pd.Series, position_size_fraction: float = 1.0) -> pd.Series:
    return (1 + trade_return_pct / 100 * position_size_fraction).cumprod()


def max_drawdown_pct(wealth_index: pd.Series) -> float:
    running_max = wealth_index.cummax()
    return float(((wealth_index / running_max - 1) * 100).min())


def cap_losses(pnl_pct: pd.Series, credit_pct: pd.Series, wing_multiplier: float) -> pd.Series:
    max_loss_pct = -wing_multiplier * credit_pct
    return pnl_pct.clip(lower=max_loss_pct)


def isolate_earnings_move_pct(
    raw_expected_move_pct: float, normal_daily_vol_pct: float, non_event_trading_days: int
) -> float:
    # Variance additivity: subtract out what normal daily vol explains, clipped at zero.
    total_variance = (raw_expected_move_pct / 100) ** 2
    normal_variance = (normal_daily_vol_pct / 100) ** 2 * non_event_trading_days
    earnings_variance = max(total_variance - normal_variance, 0.0)
    return (earnings_variance ** 0.5) * 100


def chain_has_no_contracts(calls: pd.DataFrame, puts: pd.DataFrame) -> bool:
    return calls.empty or puts.empty


def would_clip_to_zero(
    raw_expected_move_pct: float, normal_daily_vol_pct: float, non_event_trading_days: int
) -> bool:
    # True when isolate_earnings_move_pct's clipping actually bound - not a real 0%, a sign
    # the vol-stays-constant assumption doesn't hold here.
    total_variance = (raw_expected_move_pct / 100) ** 2
    normal_variance = (normal_daily_vol_pct / 100) ** 2 * non_event_trading_days
    return normal_variance >= total_variance

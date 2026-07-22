import pandas as pd

# Brenner & Subrahmanyam (1988), "A Simple Formula to Compute the Implied Standard
# Deviation": an at-the-money call or put is worth about 0.4 * S * sigma * sqrt(T), so an
# ATM straddle (call + put) is about 0.8 * S * sigma * sqrt(T). With T = 1 trading day and
# sigma expressed as a DAILY standard deviation already (no annualizing needed, the sqrt(T)
# term collapses to 1 when sigma and T are in the same units), this reduces to:
#     straddle price, as a % of the stock price  =  0.8 * daily_sigma
BRENNER_SUBRAHMANYAM_CONST = 0.8


def brenner_subrahmanyam_premium_pct(
    daily_vol: pd.Series, const: float = BRENNER_SUBRAHMANYAM_CONST, trading_days: int = 1
) -> pd.Series:
    # trading_days=1 (the default) is the original single-day option every earlier caller in
    # this project assumed. Straddle price scales with sqrt(T), so a multi-day holding period
    # just needs that same sqrt(T) term applied on top of the daily-vol base case.
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
    # An option's price reflects the WHOLE period until expiration, not just an embedded
    # event day. Variance is additive across independent days, so the event-specific variance
    # is whatever's left after subtracting what this stock's own ordinary daily vol would
    # explain over the non-event days - clipped at zero, since a bad volatility estimate or a
    # too-long non-event window can otherwise subtract out more variance than was ever there.
    total_variance = (raw_expected_move_pct / 100) ** 2
    normal_variance = (normal_daily_vol_pct / 100) ** 2 * non_event_trading_days
    earnings_variance = max(total_variance - normal_variance, 0.0)
    return (earnings_variance ** 0.5) * 100


def chain_has_no_contracts(calls: pd.DataFrame, puts: pd.DataFrame) -> bool:
    # A handful of thin, illiquid names list an expiration with no call or put contracts at
    # all - not a calculation issue, there's simply nothing to price an ATM straddle from.
    # Indexing into either DataFrame's first row without checking this first raises an
    # unhandled IndexError instead of a clean "no usable comparison" skip.
    return calls.empty or puts.empty


def would_clip_to_zero(
    raw_expected_move_pct: float, normal_daily_vol_pct: float, non_event_trading_days: int
) -> bool:
    # True whenever isolate_earnings_move_pct's clipping actually bound, meaning the "normal"
    # variance assumption alone explains as much or more than the whole option price - not a
    # real 0% answer, a sign the underlying assumption (vol stays constant into the event)
    # doesn't hold well enough here to trust the result at all.
    total_variance = (raw_expected_move_pct / 100) ** 2
    normal_variance = (normal_daily_vol_pct / 100) ** 2 * non_event_trading_days
    return normal_variance >= total_variance

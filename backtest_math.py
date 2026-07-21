import pandas as pd

# Brenner & Subrahmanyam (1988), "A Simple Formula to Compute the Implied Standard
# Deviation": an at-the-money call or put is worth about 0.4 * S * sigma * sqrt(T), so an
# ATM straddle (call + put) is about 0.8 * S * sigma * sqrt(T). With T = 1 trading day and
# sigma expressed as a DAILY standard deviation already (no annualizing needed, the sqrt(T)
# term collapses to 1 when sigma and T are in the same units), this reduces to:
#     straddle price, as a % of the stock price  =  0.8 * daily_sigma
BRENNER_SUBRAHMANYAM_CONST = 0.8


def brenner_subrahmanyam_premium_pct(
    daily_vol: pd.Series, const: float = BRENNER_SUBRAHMANYAM_CONST
) -> pd.Series:
    return const * daily_vol * 100


def compound_wealth_index(trade_return_pct: pd.Series, position_size_fraction: float = 1.0) -> pd.Series:
    return (1 + trade_return_pct / 100 * position_size_fraction).cumprod()


def max_drawdown_pct(wealth_index: pd.Series) -> float:
    running_max = wealth_index.cummax()
    return float(((wealth_index / running_max - 1) * 100).min())


def cap_losses(pnl_pct: pd.Series, credit_pct: pd.Series, wing_multiplier: float) -> pd.Series:
    max_loss_pct = -wing_multiplier * credit_pct
    return pnl_pct.clip(lower=max_loss_pct)

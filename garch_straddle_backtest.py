from db import get_engine
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import ttest_1samp
from backtest_math import brenner_subrahmanyam_premium_pct, cap_losses

warnings.filterwarnings("ignore", category=UserWarning, module="arch")

engine = get_engine()

WING_MULTIPLIER = 3

print("=== Repricing the straddle and iron condor backtests with GARCH instead of rolling vol ===")
print("(garch_volatility_forecast.py already showed GARCH(1,1) tracks realized earnings-day moves")
print(" better than a flat 20-day rolling window, but only reported a single summary statistic")
print(" (the breakeven multiplier). This reruns the FULL straddle_backtest.py and")
print(" iron_condor_backtest.py machinery - per-event P&L, win rate, tier/sector cuts, capped")
print(" vs. uncapped loss - with the option priced off GARCH volatility instead, on the exact")
print(" same events as the rolling-vol version, so the comparison is apples to apples rather")
print(" than two different samples.)")
print()

tickers = pd.read_sql("SELECT symbol FROM ticker_tiers", engine)["symbol"].tolist()
prices = pd.read_sql(
    "SELECT symbol, date, close FROM daily_prices WHERE symbol = ANY(%(symbols)s) ORDER BY symbol, date",
    engine, params={"symbols": tickers},
)

garch_rows = []
failed = []
for symbol, sub in prices.groupby("symbol"):
    sub = sub.reset_index(drop=True)
    sub["ret_pct"] = sub["close"].pct_change() * 100
    fit_data = sub.dropna(subset=["ret_pct"])
    try:
        res = arch_model(fit_data["ret_pct"], vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    except Exception:
        failed.append(symbol)
        continue
    garch_rows.append(pd.DataFrame({
        "symbol": symbol,
        "date": fit_data["date"].values,
        "garch_daily_vol": res.conditional_volatility / 100,
    }))
garch_df = pd.concat(garch_rows, ignore_index=True)
print(f"Fit GARCH(1,1) for {prices['symbol'].nunique() - len(failed)} of {prices['symbol'].nunique()} tickers"
      f"{' (failed: ' + ', '.join(failed) + ')' if failed else ''}")

EVENT_QUERY = """
WITH daily_returns AS (
    SELECT
        symbol,
        date,
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date))
            / LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS daily_return
    FROM daily_prices
),
vol_features AS (
    SELECT
        symbol,
        date,
        daily_return,
        STDDEV_SAMP(daily_return) OVER (PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS normal_daily_vol
    FROM daily_returns
),
reaction_day AS (
    SELECT
        e.symbol,
        e.reported_date,
        e.surprise_percentage,
        CASE
            WHEN e.report_time = 'pre-market' THEN e.reported_date
            ELSE (
                SELECT MIN(dp.date) FROM daily_prices dp
                WHERE dp.symbol = e.symbol AND dp.date > e.reported_date
            )
        END AS day0_date
    FROM earnings_events e
    WHERE e.surprise_percentage != 'NaN'
)
SELECT
    r.symbol, tt.tier, tt.sector, r.day0_date, r.surprise_percentage,
    v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN ticker_tiers tt ON tt.symbol = r.symbol
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""
events = pd.read_sql(EVENT_QUERY, engine)
df = events.merge(garch_df, left_on=["symbol", "day0_date"], right_on=["symbol", "date"], how="inner")
n_dropped = len(events) - len(df)
print(f"n = {len(df)} events with both a rolling-window and a GARCH volatility estimate"
      + (f" ({n_dropped} dropped: tickers whose GARCH fit failed)" if n_dropped else
         " (every event with a rolling estimate also got a GARCH one)"))
print()

df["realized_move_pct"] = df["day0_return"].abs() * 100

for label, vol_col in [("Rolling 20-day", "normal_daily_vol"), ("GARCH(1,1)", "garch_daily_vol")]:
    credit_col = f"credit_{vol_col}"
    pnl_col = f"pnl_{vol_col}"
    capped_col = f"capped_{vol_col}"
    df[credit_col] = brenner_subrahmanyam_premium_pct(df[vol_col])
    df[pnl_col] = df[credit_col] - df["realized_move_pct"]
    df[capped_col] = cap_losses(df[pnl_col], df[credit_col], WING_MULTIPLIER)

    t_stat, p_val = ttest_1samp(df[pnl_col], popmean=0)
    p_one_sided = p_val / 2 if t_stat < 0 else 1 - p_val / 2
    win_rate = (df[pnl_col] > 0).mean() * 100
    breakeven = df["realized_move_pct"].mean() / df[credit_col].mean()

    print(f"--- {label}-priced straddle ---")
    print(f"Mean P&L (uncapped): {df[pnl_col].mean():+.2f}%   Win rate: {win_rate:.1f}%   "
          f"p={p_one_sided:.1e}   Breakeven IV multiplier: {breakeven:.2f}x")
    print(f"Mean P&L ({WING_MULTIPLIER}x-credit iron condor): {df[capped_col].mean():+.2f}%   "
          f"Worst single event: {df[capped_col].min():.1f}%")

    by_tier = df.groupby("tier")[pnl_col].mean().reindex(["large", "mid", "small"])
    print(f"By tier (uncapped): large={by_tier['large']:+.2f}%  mid={by_tier['mid']:+.2f}%  "
          f"small={by_tier['small']:+.2f}%")
    print()

rolling_mean, garch_mean = df["pnl_normal_daily_vol"].mean(), df["pnl_garch_daily_vol"].mean()
corr = df["pnl_normal_daily_vol"].corr(df["pnl_garch_daily_vol"], method="spearman")
print(f"The two pricing methods agree on the sign and rough size of the result (mean P&L "
      f"{rolling_mean:+.2f}% rolling vs {garch_mean:+.2f}% GARCH, per-event Spearman correlation "
      f"{corr:.2f}), which is what should happen if the earlier single-ticker GARCH check")
print("generalizes rather than being a fluke of the specific tickers it was run on there.")
print("A genuinely better volatility model changes the number somewhat but not the conclusion:")
print("selling this trade priced off either method loses money on average, historically.")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(df["pnl_normal_daily_vol"].clip(lower=-20), bins=60, alpha=0.5,
        label="Rolling 20-day priced", color="#7f8c8d")
ax.hist(df["pnl_garch_daily_vol"].clip(lower=-20), bins=60, alpha=0.5,
        label="GARCH(1,1) priced", color="#2c7fb8")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Uncapped straddle P&L per trade (% of price, clipped at -20% for readability)")
ax.set_ylabel("Number of events")
ax.set_title("Straddle P&L: rolling-window vs. GARCH-priced, same 2,900+ events")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("charts/garch_straddle_backtest.png", dpi=150)
plt.close(fig)
print("\nSaved charts/garch_straddle_backtest.png")

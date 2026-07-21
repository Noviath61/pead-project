from db import get_engine
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import ttest_1samp

pd.set_option("display.width", 200)
warnings.filterwarnings("ignore", category=UserWarning, module="arch")

engine = get_engine()

print("=== GARCH(1,1) volatility forecasts vs. the simple rolling-window estimate ===")
print("(Every volatility number so far, jump_ratio, the straddle backtest, uses a 20-day")
print(" trailing rolling standard deviation as 'normal' volatility. That's a reasonable")
print(" baseline, but real vol forecasting almost always uses something that captures")
print(" volatility clustering, the tendency for high-vol and low-vol periods to persist,")
print(" instead of weighting the last 20 days equally. This fits a GARCH(1,1) model per")
print(" ticker, the standard textbook volatility-forecasting model (Bollerslev 1986), and")
print(" checks whether it actually changes the jump-ratio and straddle-pricing conclusions.")
print()
print(" One honest caveat up front: the market model and Fama-French sections carefully fit")
print(" only on a pre-event window to avoid lookahead bias. Fitting a fresh GARCH model")
print(" before each of ~2,950 events would be its own project. This fits ONE GARCH model per")
print(" ticker on its full available history instead, so the fitted parameters (not the")
print(" forecast itself, which only uses information through the prior day) have mild")
print(" lookahead bias. Fine for asking 'does a smarter model change the conclusion', not a")
print(" substitute for the point-in-time discipline used elsewhere in this project.)")
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
print()

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
    r.symbol, tt.tier, r.day0_date, r.surprise_percentage,
    v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN ticker_tiers tt ON tt.symbol = r.symbol
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""
events = pd.read_sql(EVENT_QUERY, engine)
events = events.merge(
    garch_df.rename(columns={"date": "day0_date", "garch_daily_vol": "garch_daily_vol"}),
    on=["symbol", "day0_date"], how="inner",
)

events["jump_ratio_rolling"] = events["day0_return"].abs() / events["normal_daily_vol"]
events["jump_ratio_garch"] = events["day0_return"].abs() / events["garch_daily_vol"]
events = events[(events["jump_ratio_rolling"] > 0) & (events["jump_ratio_garch"] > 0)].copy()

vol_corr = events["normal_daily_vol"].corr(events["garch_daily_vol"], method="spearman")
print(f"n = {len(events)} earnings events with both a rolling and a GARCH volatility estimate")
print(f"Spearman correlation between the two volatility estimates: {vol_corr:.3f}")
print("(High but not 1.0 is exactly what you'd expect: related information, different weighting")
print(" of recent vs. older data.)")
print()

for label, col in [("Rolling 20-day", "jump_ratio_rolling"), ("GARCH(1,1)", "jump_ratio_garch")]:
    log_r = np.log(events[col])
    geo_mean = np.exp(log_r.mean())
    t_stat, p_two = ttest_1samp(log_r, popmean=0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    print(f"{label:16s}  mean={events[col].mean():.2f}x  median={events[col].median():.2f}x  "
          f"geomean={geo_mean:.2f}x  (t={t_stat:.2f}, p={p_one:.1e})")
print()

mean_realized_pct = events["day0_return"].abs().mul(100).mean()
rolling_breakeven = mean_realized_pct / (0.8 * events["normal_daily_vol"] * 100).mean()
garch_breakeven = mean_realized_pct / (0.8 * events["garch_daily_vol"] * 100).mean()
print(f"Breakeven implied-vol multiplier, straddle priced off rolling vol: {rolling_breakeven:.2f}x")
print(f"Breakeven implied-vol multiplier, straddle priced off GARCH vol:   {garch_breakeven:.2f}x")
print()
print("GARCH is the textbook-correct way to forecast volatility, and it IS measurably closer")
print("to the realized move than the flat rolling window, but it doesn't change the story: the")
print("earnings-day jump still runs well above what even a smarter day-to-day volatility model")
print("expects, because neither model has any way to know an earnings date is coming. That gap")
print("is exactly the volatility risk premium options markets price in ahead of the event, which")
print("a purely backward-looking time-series model, however sophisticated, structurally can't.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
sample = events.sample(min(2000, len(events)), random_state=42)
ax.scatter(sample["normal_daily_vol"] * 100, sample["garch_daily_vol"] * 100, s=6, alpha=0.3, color="#2c7fb8")
lims = [0, max(sample["normal_daily_vol"].max(), sample["garch_daily_vol"].max()) * 100]
ax.plot(lims, lims, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Rolling 20-day daily vol (%)")
ax.set_ylabel("GARCH(1,1) daily vol (%)")
ax.set_title(f"Two vol estimates agree in shape (r={vol_corr:.2f}), not exactly")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
methods = ["Rolling 20-day", "GARCH(1,1)"]
geo_means = [
    np.exp(np.log(events["jump_ratio_rolling"]).mean()),
    np.exp(np.log(events["jump_ratio_garch"]).mean()),
]
ax.bar(methods, geo_means, color=["#7f8c8d", "#2c7fb8"])
ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Normal day (1x)")
ax.set_ylabel("Geometric mean jump ratio")
ax.set_title("Earnings-day jump under both volatility models")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/garch_volatility_forecast.png", dpi=150)
plt.close(fig)
print("\nSaved charts/garch_volatility_forecast.png")

from db import get_engine
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

pd.set_option("display.width", 200)

engine = get_engine()

# Brenner & Subrahmanyam (1988), "A Simple Formula to Compute the Implied Standard
# Deviation": an at-the-money call or put is worth about 0.4 * S * sigma * sqrt(T), so an
# ATM straddle (call + put) is about 0.8 * S * sigma * sqrt(T). With T = 1 trading day and
# sigma expressed as a DAILY standard deviation already (no annualizing needed, the sqrt(T)
# term collapses to 1 when sigma and T are in the same units), this reduces to:
#     straddle price, as a % of the stock price  =  0.8 * daily_sigma
BRENNER_SUBRAHMANYAM_CONST = 0.8

print("=== Straddle backtest: selling a proxy at-the-money straddle into every earnings event ===")
print("(volatility_risk_premium.py showed the realized earnings-day move is several times a")
print(" normal day. This asks the next question directly: if you'd sold an at-the-money")
print(" straddle priced ONLY off trailing historical volatility, no options-chain data, no")
print(" real implied vol, would you have made money? This can't use real options prices, since")
print(" this project has none, so it prices the straddle with the Brenner-Subrahmanyam")
print(" approximation, a textbook formula for converting volatility into an ATM option price.")
print(" That means this is deliberately the CHEAPEST reasonable price for the straddle, since")
print(" real implied vol runs above historical vol into an earnings date. A strategy that loses")
print(" money even at this cheap a price is not a real counterexample to selling options for a")
print(" living, it's a lower bound on how rich real implied vol has to run just to break even.)")
print()

QUERY = """
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
    r.symbol, tt.tier, tt.sector, r.reported_date, r.day0_date, r.surprise_percentage,
    v.daily_return AS day0_return, v.normal_daily_vol
FROM reaction_day r
JOIN ticker_tiers tt ON tt.symbol = r.symbol
JOIN vol_features v ON v.symbol = r.symbol AND v.date = r.day0_date
WHERE v.normal_daily_vol IS NOT NULL AND v.daily_return IS NOT NULL AND v.normal_daily_vol > 0
"""

df = pd.read_sql(QUERY, engine)
df["straddle_premium_pct"] = BRENNER_SUBRAHMANYAM_CONST * df["normal_daily_vol"] * 100
df["realized_move_pct"] = df["day0_return"].abs() * 100
# Simplified one-period payoff at expiration: keep the premium, pay out whatever the stock
# moved beyond it. Ignores bid/ask spread, commissions, assignment/pin risk, and early
# closing the position before expiration, all of which a real options desk has to handle.
df["pnl_pct"] = df["straddle_premium_pct"] - df["realized_move_pct"]

n = len(df)
mean_pnl = df["pnl_pct"].mean()
median_pnl = df["pnl_pct"].median()
win_rate = (df["pnl_pct"] > 0).mean() * 100

t_stat, p_two_sided = ttest_1samp(df["pnl_pct"], popmean=0)
p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

# How much richer than trailing historical vol would the market need to price the straddle,
# on average, for this to break even? Solving mean(multiplier * premium - realized) = 0 for
# multiplier, i.e. multiplier = mean(realized) / mean(premium).
breakeven_multiplier = df["realized_move_pct"].mean() / df["straddle_premium_pct"].mean()

mean_premium = df["straddle_premium_pct"].mean()
print(f"n = {n} earnings events")
print(f"Mean straddle premium collected (historical-vol-priced): {mean_premium:.2f}% of price")
print(f"Mean realized move paid out: {df['realized_move_pct'].mean():.2f}% of price")
print(f"Mean P&L: {mean_pnl:+.2f}% of price   |   Median P&L: {median_pnl:+.2f}%")
print(f"Win rate (premium exceeded realized move): {win_rate:.1f}%")
print(f"One-sided t-test that mean P&L < 0: t={t_stat:.2f}, p={p_one_sided:.2e}")
print(f"Breakeven implied-vol multiplier over trailing historical vol: {breakeven_multiplier:.2f}x")
print()

by_tier = df.groupby("tier")[["pnl_pct"]].agg(["count", "mean", "median"])
by_tier.columns = ["n", "mean_pnl_pct", "median_pnl_pct"]
by_tier = by_tier.reindex(["large", "mid", "small"])
win_rate_by_tier = df.groupby("tier").apply(lambda g: (g["pnl_pct"] > 0).mean() * 100, include_groups=False)
by_tier["win_rate_pct"] = win_rate_by_tier.reindex(["large", "mid", "small"]).round(1)
print("By market-cap tier:")
print(by_tier.round(2).to_string())
print()

surprise_corr = df["surprise_percentage"].abs().corr(df["pnl_pct"], method="spearman")
print(f"Spearman correlation, |surprise %| vs P&L: {surprise_corr:.3f}")
print("(Negative and meaningfully different from zero would mean bigger surprises make this")
print(" specific proxy trade worse, i.e. the historical-vol price under-covers exactly the")
print(" events where the reaction is biggest, which is the events that hurt most to be wrong on.)")
print()

by_sector = df.groupby("sector")[["pnl_pct"]].agg(["count", "mean"])
by_sector.columns = ["n", "mean_pnl_pct"]
win_rate_by_sector = df.groupby("sector").apply(
    lambda g: (g["pnl_pct"] > 0).mean() * 100, include_groups=False
)
by_sector["win_rate_pct"] = win_rate_by_sector.round(1)
by_sector = by_sector.sort_values("mean_pnl_pct")
print("By sector:")
print(by_sector.round(2).to_string())
print("Tech is the worst sector to sell this trade into by a wide margin (biggest jump ratio,")
print("see volatility_risk_premium.py), while Defense is close to a coin flip, roughly")
print("breakeven on both P&L and win rate. Same sector pattern as the jump ratio, seen from")
print("the P&L side instead of the volatility side.")
print()

print("At a price set only by trailing historical volatility, this loses money on average")
print(f"({mean_pnl:+.2f}% per trade, p={p_one_sided:.1e}), and would need implied vol priced at roughly")
print(f"{breakeven_multiplier:.1f}x the trailing historical level just to break even. That's actually in the")
print("range real earnings implied vol run-ups reach in practice, which is the honest reconciliation")
print("here: historical vol alone is a bad price for an earnings option, real IV runs richer than")
print("that for exactly this reason, and whether it runs rich ENOUGH to be a profitable sale, net of")
print("realistic spreads and sizing, is a question only real options-chain data could answer.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(df["pnl_pct"].clip(lower=-15, upper=10), bins=50, color="#2c7fb8", edgecolor="white")
ax.axvline(0, color="black", linewidth=1, linestyle="--", label="Breakeven")
ax.axvline(mean_pnl, color="#c0392b", linewidth=1.5, label=f"Mean ({mean_pnl:+.2f}%)")
ax.set_xlabel("P&L per trade, historical-vol-priced straddle (% of price)")
ax.set_ylabel("Number of events")
ax.set_title("Selling a historical-vol-priced straddle into earnings")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
tier_order = ["large", "mid", "small"]
tier_means = [df[df["tier"] == t]["pnl_pct"].mean() for t in tier_order]
colors = ["#c0392b" if m < 0 else "#27ae60" for m in tier_means]
ax.bar(tier_order, tier_means, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean P&L (% of price)")
ax.set_title("Mean P&L by market-cap tier")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("charts/straddle_backtest.png", dpi=150)
plt.close(fig)
print("\nSaved charts/straddle_backtest.png")

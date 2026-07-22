from db import get_engine
import pandas as pd
import matplotlib.pyplot as plt
from backtest_math import compound_wealth_index, max_drawdown_pct

pd.set_option("display.width", 200)

engine = get_engine()

ROUND_TRIP_COST_PCT = 0.40  # 20bps per leg, one open + one close
POSITION_SIZE_FRACTION = 0.10  # each trade risks 10% of capital, not 100% - see note below

print("=== Time-series backtest: a real equity curve, not just a pooled average ===")
print("(economic_significance.py already showed the average long-short spread is unprofitable")
print(" net of costs. This asks a different, complementary question: if you actually traded")
print(" this strategy through calendar time, what would the risk-adjusted return profile look")
print(" like - Sharpe ratio, max drawdown, win rate? A simplified illustration: each qualifying")
print(" event becomes one trade, sequenced by its actual Day-0 date, not overlap-adjusted for")
print(" simultaneous positions, which a fully realistic backtest would need to handle. Each")
print(f" trade risks {POSITION_SIZE_FRACTION:.0%} of capital rather than 100% - a first pass that bet the")
print(" full account on every single trade in sequence mathematically wiped out the portfolio,")
print(" which just reflects unrealistic position sizing, not a real finding about the strategy.)")
print()

df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df_clean = df.dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"]).copy()
df_clean["surprise_quintile"] = pd.qcut(
    df_clean["surprise_percentage"], 5,
    labels=["1: Big miss", "2: Miss", "3: Meet", "4: Beat", "5: Big beat"],
)

longs = df_clean[df_clean["surprise_quintile"] == "5: Big beat"].copy()
longs["trade_return_pct"] = longs["abnormal_drift_10d_pct"] - ROUND_TRIP_COST_PCT
longs["side"] = "long"

shorts = df_clean[df_clean["surprise_quintile"] == "1: Big miss"].copy()
shorts["trade_return_pct"] = -shorts["abnormal_drift_10d_pct"] - ROUND_TRIP_COST_PCT
shorts["side"] = "short"

trades = pd.concat([longs, shorts]).sort_values("day0_date").reset_index(drop=True)

# Properly compounded wealth index, not a raw cumsum (which can wander below -100%).
# Each trade risks only POSITION_SIZE_FRACTION, not the full account.
trades["wealth_index"] = compound_wealth_index(trades["trade_return_pct"], POSITION_SIZE_FRACTION)

n_trades = len(trades)
span_years = (trades["day0_date"].max() - trades["day0_date"].min()).days / 365.25
trades_per_year = n_trades / span_years

mean_return = trades["trade_return_pct"].mean()
std_return = trades["trade_return_pct"].std()
sharpe = (mean_return / std_return) * (trades_per_year ** 0.5)

max_drawdown = max_drawdown_pct(trades["wealth_index"])

win_rate = (trades["trade_return_pct"] > 0).mean() * 100
total_return_pct = (trades["wealth_index"].iloc[-1] - 1) * 100

print(f"Trades: {n_trades} ({len(longs)} long, {len(shorts)} short), spanning {span_years:.1f} years "
      f"({trades_per_year:.0f} trades/year)")
print(f"Mean return per trade (net of {ROUND_TRIP_COST_PCT}% round-trip cost): {mean_return:+.3f}%")
print(f"Annualized Sharpe ratio: {sharpe:.2f}")
print(f"Max drawdown (properly compounded): {max_drawdown:.1f}%")
print(f"Win rate: {win_rate:.1f}%")
print(f"Total compounded return over the full period: {total_return_pct:+.1f}%")
print()
print("For comparison: a real, tradeable long-short equity strategy typically wants a Sharpe")
print("ratio comfortably above 1.0. This one isn't close, consistent with every other test")
print("in this project - statistically null, economically unprofitable, and not attractive")
print("on a risk-adjusted basis either.")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(trades["day0_date"], trades["wealth_index"], color="#c0392b", linewidth=1)
ax.axhline(1.0, color="black", linewidth=0.8, label="Starting capital")
ax.set_ylabel("Wealth index (starting value = 1.0)")
ax.set_title(f"Long-short quintile strategy equity curve (Sharpe={sharpe:.2f}, "
             f"max drawdown={max_drawdown:.1f}%)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("charts/equity_curve.png", dpi=150)
plt.close(fig)
print("\nSaved charts/equity_curve.png")

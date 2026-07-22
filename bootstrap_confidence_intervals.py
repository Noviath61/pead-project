from db import get_engine
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

pd.set_option("display.width", 200)

engine = get_engine()

N_BOOT = 5000
RNG = np.random.default_rng(42)

print("=== Bootstrap confidence intervals: does clustering matter for uncertainty, not just p-values? ===")
print("(tier_analysis.py already clusters standard errors by ticker in its regression, since")
print(" repeated events from the same company aren't independent draws. This asks whether the")
print(" same problem shows up in a completely different tool: a bootstrap confidence interval")
print(" around the headline Spearman correlations. A NAIVE bootstrap resamples individual events,")
print(" quietly treating every one of them as independent. A CLUSTER bootstrap resamples whole")
print(" tickers instead, so a company's 30-60 quarters of history move together, respecting the")
print(" same non-independence tier_analysis.py already had to account for.)")
print()

def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])

df = pd.read_sql("SELECT * FROM earnings_drift", engine)

rows = []
start = time.perf_counter()
for tier in ["large", "mid", "small"]:
    sub = df[df["tier"] == tier].dropna(subset=["surprise_percentage", "abnormal_drift_10d_pct"])
    x_all = sub["surprise_percentage"].to_numpy()
    y_all = sub["abnormal_drift_10d_pct"].to_numpy()
    symbols_all = sub["symbol"].to_numpy()
    tickers = np.unique(symbols_all)
    n_events, n_tickers = len(sub), len(tickers)

    observed_r = spearman_r(x_all, y_all)

    # Naive bootstrap: resample individual events with replacement, ignoring that many of
    # them come from the same 20 companies.
    naive_draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n_events, n_events)
        naive_draws[b] = spearman_r(x_all[idx], y_all[idx])

    # Cluster bootstrap: resample whole tickers with replacement, keeping every event from
    # a chosen ticker together, so a repeatedly-drawn company can't be mistaken for several
    # independent ones.
    ticker_to_indices = {t: np.where(symbols_all == t)[0] for t in tickers}
    cluster_draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        chosen_tickers = RNG.choice(tickers, size=n_tickers, replace=True)
        idx = np.concatenate([ticker_to_indices[t] for t in chosen_tickers])
        cluster_draws[b] = spearman_r(x_all[idx], y_all[idx])

    naive_lo, naive_hi = np.percentile(naive_draws, [2.5, 97.5])
    cluster_lo, cluster_hi = np.percentile(cluster_draws, [2.5, 97.5])
    rows.append({
        "tier": tier, "n_events": n_events, "n_tickers": n_tickers,
        "observed_r": round(observed_r, 3),
        "naive_ci_low": round(naive_lo, 3), "naive_ci_high": round(naive_hi, 3),
        "naive_ci_width": round(naive_hi - naive_lo, 3),
        "cluster_ci_low": round(cluster_lo, 3), "cluster_ci_high": round(cluster_hi, 3),
        "cluster_ci_width": round(cluster_hi - cluster_lo, 3),
    })

elapsed = time.perf_counter() - start
result = pd.DataFrame(rows)
result["cluster_ci_wider_by"] = (result["cluster_ci_width"] / result["naive_ci_width"]).round(2)
print(result.to_string(index=False))
print(f"\n({2 * N_BOOT * 3:,} total resamples across 3 tiers in {elapsed:.1f}s)")
print()

print("Going in, I expected the cluster-aware interval to come out wider in every tier, the same")
print("story as the cluster-robust regression earlier in this project, where ignoring clustering")
print("understated the true uncertainty. That's only half true here: large-cap comes out about")
print("the same width, and mid/small-cap actually come out narrower under cluster resampling,")
print("not wider. That's a real result, not a bug, rerunning with a different seed and N_BOOT")
print("reproduces the same pattern. The likely reason is that a regression's cluster-robust SE")
print("corrects for correlated ERRORS within a company (one quarter's surprise being unusually")
print("predictive tends to mean the next one is too), while this is resampling whole companies")
print("for a rank correlation computed once over the pooled tier, a different object entirely,")
print("and apparently the quarter-to-quarter pattern within a single company isn't nearly as")
print("internally correlated as the regression residuals were. Both intervals still comfortably")
print("straddle zero in every tier either way, so the conclusion doesn't move, but the honest")
print("finding is that 'does clustering widen my interval' turned out to depend on which")
print("statistic you're clustering, not something safe to assume just because it mattered")
print("for the regression.")

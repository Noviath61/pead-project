from db import get_engine
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

pd.set_option("display.width", 200)

engine = get_engine()

print("=== Does the engineered jump_ratio feature add any predictive power? ===")
print("(model.py already tests surprise size, pre-earnings momentum, volume spike, and")
print(" volatility change as features for predicting drift DIRECTION. This project has since")
print(" engineered a new feature, volatility_risk_premium.py's jump_ratio, the size of the")
print(" Day-0 move relative to a normal day, which turned out to be one of the single")
print(" strongest, most statistically significant numbers in this whole project. It would be")
print(" sloppy not to test whether adding it actually helps the classifier, instead of just")
print(" assuming a strong standalone signal must also be a useful predictive feature.)")
print()

BASE_FEATURES = [
    "surprise_percentage", "pre_earnings_momentum_pct", "volume_spike_ratio", "volatility_change_ratio",
]
CATEGORICAL_FEATURES = ["tier", "sector"]

df = pd.read_sql("SELECT * FROM earnings_drift", engine)

JUMP_QUERY = """
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
)
SELECT symbol, date AS day0_date, ABS(daily_return) / normal_daily_vol AS jump_ratio
FROM vol_features
WHERE normal_daily_vol IS NOT NULL AND normal_daily_vol > 0 AND daily_return IS NOT NULL
"""
jump_df = pd.read_sql(JUMP_QUERY, engine)

df = df.merge(jump_df, on=["symbol", "day0_date"], how="left")

feature_sets = {
    "Baseline (model.py's 4 features)": BASE_FEATURES,
    "Baseline + jump_ratio": BASE_FEATURES + ["jump_ratio"],
}
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
for set_name, numeric_features in feature_sets.items():
    all_features = numeric_features + CATEGORICAL_FEATURES
    sub = df.dropna(subset=all_features + ["abnormal_drift_10d_pct"]).copy()
    sub["target_up"] = (sub["abnormal_drift_10d_pct"] > 0).astype(int)
    sub = sub.sort_values("reported_date").reset_index(drop=True)

    X = sub[all_features]
    y = sub["target_up"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    for model_name, model in models.items():
        fold_accuracies, fold_baselines = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
            y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline([("prep", preprocessor), ("model", model)])
            pipeline.fit(X_train_f, y_train_f)
            preds_f = pipeline.predict(X_test_f)

            fold_accuracies.append(accuracy_score(y_test_f, preds_f))
            fold_baselines.append(y_test_f.value_counts(normalize=True).max())

        results.append({
            "feature_set": set_name, "model": model_name, "n": len(sub),
            "avg_accuracy": round(sum(fold_accuracies) / len(fold_accuracies), 4),
            "avg_baseline": round(sum(fold_baselines) / len(fold_baselines), 4),
        })

result_df = pd.DataFrame(results)
result_df["beats_baseline"] = result_df["avg_accuracy"] > result_df["avg_baseline"]
print(result_df.to_string(index=False))
print()

pivot = result_df.pivot(index="model", columns="feature_set", values="avg_accuracy")
lift = (pivot["Baseline + jump_ratio"] - pivot["Baseline (model.py's 4 features)"]) * 100
print("Accuracy change from adding jump_ratio (percentage points, walk-forward average):")
for model_name, delta in lift.items():
    print(f"  {model_name}: {delta:+.2f} pp")
print()

if (lift.abs() < 1.0).all():
    print("Adding the strongest standalone signal in this whole project barely moves the needle")
    print("on directional accuracy, both changes stay under 1 percentage point either way. That's")
    print("not a contradiction: jump_ratio measures the SIZE of the reaction, not which way it")
    print("goes, so there's no real reason it should help predict direction, and it doesn't.")
    print("Consistent with everything else here, a strong, statistically real feature by one")
    print("measure isn't automatically a useful predictive feature by another.")
else:
    print("Adding jump_ratio moved walk-forward accuracy by more than 1 percentage point for at")
    print("least one model, worth a closer look before drawing any conclusion from it.")

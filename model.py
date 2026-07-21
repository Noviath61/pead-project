from db import get_engine
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option("display.width", 200)

engine = get_engine()

NUMERIC_FEATURES = [
    "surprise_percentage", "pre_earnings_momentum_pct", "volume_spike_ratio", "volatility_change_ratio",
]
CATEGORICAL_FEATURES = ["tier", "sector"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df = df.dropna(subset=FEATURES + ["abnormal_drift_10d_pct"])
df["target_up"] = (df["abnormal_drift_10d_pct"] > 0).astype(int)
df = df.sort_values("reported_date").reset_index(drop=True)

X = df[FEATURES]
y = df["target_up"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

baseline_accuracy = y_test.value_counts(normalize=True).max()
print(f"Rows used: {len(df)}  (after dropping rows with missing features)")
print(f"Baseline (always predict majority class): {baseline_accuracy:.3f}")
print(f"Test set size: {len(y_test)}  |  Train set size: {len(y_train)}")
print()

for name, model in models.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(f"--- {name} (random 80/20 split) ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["down/flat", "up"]))
    print()

print("=== Walk-forward (time-series) cross-validation ===")
print("(A random split can leak information: predicting an older event using a model")
print(" trained partly on LATER events is a form of lookahead bias. Walk-forward validation")
print(" only ever trains on events that happened chronologically before the ones being")
print(" predicted - the same principle discussed early on for backtesting trading strategies.)")
print()

tscv = TimeSeriesSplit(n_splits=5)
for name, model in models.items():
    fold_accuracies = []
    fold_baselines = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
        y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_train_f, y_train_f)
        preds_f = pipeline.predict(X_test_f)

        fold_accuracies.append(accuracy_score(y_test_f, preds_f))
        fold_baselines.append(y_test_f.value_counts(normalize=True).max())

    print(f"--- {name} (5-fold walk-forward) ---")
    for i, (acc, base) in enumerate(zip(fold_accuracies, fold_baselines), start=1):
        print(f"  fold {i}: accuracy={acc:.3f}  baseline={base:.3f}")
    print(f"  average accuracy={sum(fold_accuracies)/len(fold_accuracies):.3f}  "
          f"average baseline={sum(fold_baselines)/len(fold_baselines):.3f}")
    print()

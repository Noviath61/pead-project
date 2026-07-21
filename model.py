import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pd.set_option("display.width", 200)
load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
engine = create_engine(DB_URL)

NUMERIC_FEATURES = ["surprise_percentage", "pre_earnings_momentum_pct", "volume_spike_ratio", "volatility_change_ratio"]
CATEGORICAL_FEATURES = ["tier", "sector"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

df = pd.read_sql("SELECT * FROM earnings_drift", engine)
df = df.dropna(subset=FEATURES + ["abnormal_drift_10d_pct"])
df["target_up"] = (df["abnormal_drift_10d_pct"] > 0).astype(int)

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

    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["down/flat", "up"]))
    print()

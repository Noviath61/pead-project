import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from scipy import stats

load_dotenv()

st.set_page_config(page_title="PEAD Analysis", layout="wide")


SNAPSHOT_PATH = "snapshot/earnings_drift.csv"


@st.cache_data
def load_data():
    try:
        db_url = (
            f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
            f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
        )
        engine = create_engine(db_url)
        return pd.read_sql("SELECT * FROM earnings_drift", engine), "live database"
    except Exception:
        df = pd.read_csv(SNAPSHOT_PATH, parse_dates=["reported_date", "day0_date"])
        return df, "static snapshot"


df, data_source = load_data()
if data_source == "static snapshot":
    st.info(
        "Showing a static data snapshot (no live database connection available in this "
        "environment). Run this locally with Postgres for live, re-runnable analysis.",
        icon="ℹ️",
    )

st.title("Post-Earnings Announcement Drift (PEAD) Analysis")
st.caption(
    "Does an earnings surprise predict abnormal (market-adjusted) stock drift in the "
    "days after? Tested across large/mid/small-cap tiers to check whether the effect "
    "is stronger in less-covered stocks, as academic literature suggests."
)

st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("Tier", options=sorted(df["tier"].unique()), default=sorted(df["tier"].unique()))
sectors = st.sidebar.multiselect("Sector", options=sorted(df["sector"].unique()), default=sorted(df["sector"].unique()))

filtered = df[df["tier"].isin(tiers) & df["sector"].isin(sectors)].dropna(
    subset=["surprise_percentage", "abnormal_drift_10d_pct"]
)

col1, col2, col3 = st.columns(3)
col1.metric("Earnings events", len(filtered))
col2.metric("Tickers", filtered["symbol"].nunique())
col3.metric("Tiers included", filtered["tier"].nunique())

st.divider()

st.subheader("Does surprise size correlate with abnormal drift, by tier?")
rows = []
for tier in sorted(filtered["tier"].unique()):
    sub = filtered[filtered["tier"] == tier]
    if len(sub) < 5:
        continue
    r, p = stats.spearmanr(sub["surprise_percentage"], sub["abnormal_drift_10d_pct"])
    rows.append({"tier": tier, "n": len(sub), "spearman_r": round(r, 3), "p_value": round(p, 4),
                 "significant (p<0.05)": "yes" if p < 0.05 else "no"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Avg. abnormal drift by surprise quintile")
    quintiled = filtered.copy()
    quintiled["surprise_quintile"] = pd.qcut(
        quintiled["surprise_percentage"], 5,
        labels=["1: Big miss", "2: Miss", "3: Meet", "4: Beat", "5: Big beat"],
        duplicates="drop",
    )
    bucket_summary = quintiled.groupby("surprise_quintile", observed=True)["abnormal_drift_10d_pct"].mean().reset_index()
    fig = px.bar(bucket_summary, x="surprise_quintile", y="abnormal_drift_10d_pct",
                 labels={"abnormal_drift_10d_pct": "Avg abnormal drift (10d, %)", "surprise_quintile": "Surprise bucket"})
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Surprise size vs. abnormal drift")
    fig2 = px.scatter(filtered, x="surprise_percentage", y="abnormal_drift_10d_pct", color="tier",
                       hover_data=["symbol", "reported_date"],
                       labels={"surprise_percentage": "Earnings surprise (%)", "abnormal_drift_10d_pct": "Abnormal drift (10d, %)"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Event study: cumulative abnormal return around earnings (Day 0)")
st.caption(
    "The academic-standard way to visualize this: average abnormal return per day, "
    "relative to the earnings reaction date, cumulated over time. A real PEAD effect "
    "would show the line continuing to climb (or fall) steadily after Day 0. A flat "
    "line after Day 0 means the market's reaction was essentially instant."
)
try:
    car_overall = pd.read_csv("snapshot/event_study_overall.csv")
    fig3 = px.line(car_overall, x="offset", y="car_pct",
                    labels={"offset": "Trading days relative to Day 0", "car_pct": "Cumulative abnormal return (%)"})
    fig3.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Day 0 (earnings reaction)")
    st.plotly_chart(fig3, use_container_width=True)
except FileNotFoundError:
    st.caption("Run `python event_study.py` to generate this chart.")

st.divider()

st.subheader("Ticker drill-down")
symbol = st.selectbox("Symbol", options=sorted(filtered["symbol"].unique()))
ticker_df = filtered[filtered["symbol"] == symbol].sort_values("reported_date")
st.dataframe(
    ticker_df[["reported_date", "report_time", "surprise_percentage", "day0_date",
               "drift_10d_pct", "abnormal_drift_10d_pct", "volume_spike_ratio", "volatility_change_ratio"]],
    use_container_width=True, hide_index=True,
)

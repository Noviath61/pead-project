import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from db import get_engine

st.set_page_config(page_title="PEAD Analysis", layout="wide")


SNAPSHOT_PATH = "snapshot/earnings_drift.csv"


@st.cache_data
def load_data():
    try:
        engine = get_engine()
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
tier_options = sorted(df["tier"].unique())
sector_options = sorted(df["sector"].unique())
tiers = st.sidebar.multiselect("Tier", options=tier_options, default=tier_options)
sectors = st.sidebar.multiselect("Sector", options=sector_options, default=sector_options)

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
    bucket_summary = (
        quintiled.groupby("surprise_quintile", observed=True)["abnormal_drift_10d_pct"]
        .mean().reset_index()
    )
    bucket_labels = {
        "abnormal_drift_10d_pct": "Avg abnormal drift (10d, %)",
        "surprise_quintile": "Surprise bucket",
    }
    fig = px.bar(bucket_summary, x="surprise_quintile", y="abnormal_drift_10d_pct", labels=bucket_labels)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Surprise size vs. abnormal drift")
    scatter_labels = {
        "surprise_percentage": "Earnings surprise (%)",
        "abnormal_drift_10d_pct": "Abnormal drift (10d, %)",
    }
    fig2 = px.scatter(
        filtered, x="surprise_percentage", y="abnormal_drift_10d_pct", color="tier",
        hover_data=["symbol", "reported_date"], labels=scatter_labels,
    )
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
    fig3 = px.line(
        car_overall, x="offset", y="car_pct",
        labels={"offset": "Trading days relative to Day 0", "car_pct": "Cumulative abnormal return (%)"},
    )
    fig3.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Day 0 (earnings reaction)")
    st.plotly_chart(fig3, use_container_width=True)
except FileNotFoundError:
    st.caption("Run `python event_study.py` to generate this chart.")

st.divider()

st.subheader("Volatility around earnings: what actually matters for selling options")
st.caption(
    "A different question from PEAD: not which way the stock moves after earnings, but how "
    "much it moves on the day itself, relative to a normal trading day for that same stock. "
    "This is the part of the project closest to actually trading options around earnings."
)
try:
    jump_df = pd.read_csv("snapshot/volatility_jump.csv", parse_dates=["day0_date"])
    straddle_df = pd.read_csv("snapshot/straddle_pnl.csv", parse_dates=["day0_date"])
    jump_filtered = jump_df[jump_df["tier"].isin(tiers) & jump_df["sector"].isin(sectors)]
    straddle_filtered = straddle_df[straddle_df["tier"].isin(tiers) & straddle_df["sector"].isin(sectors)]

    vcol1, vcol2, vcol3 = st.columns(3)
    geo_mean = np.exp(np.log(jump_filtered.loc[jump_filtered["jump_ratio"] > 0, "jump_ratio"]).mean())
    vcol1.metric("Geometric mean jump ratio", f"{geo_mean:.2f}x",
                 help="Earnings-day move divided by a normal day's move for that stock")
    mean_straddle_pnl = straddle_filtered["pnl_pct"].mean()
    vcol2.metric(
        "Historical-vol-priced straddle, mean P&L", f"{mean_straddle_pnl:+.2f}%",
        help="Selling an at-the-money straddle priced off trailing volatility (Brenner-Subrahmanyam)",
    )
    vcol3.metric("Straddle win rate", f"{(straddle_filtered['pnl_pct'] > 0).mean() * 100:.1f}%")

    vleft, vright = st.columns(2)
    with vleft:
        fig4 = px.histogram(
            jump_filtered[jump_filtered["jump_ratio"] <= 8], x="jump_ratio", nbins=40,
            labels={"jump_ratio": "Earnings-day move / normal day's move"},
            title="Distribution of the earnings-day volatility jump",
        )
        fig4.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="Normal day")
        st.plotly_chart(fig4, use_container_width=True)
    with vright:
        sector_pnl = straddle_filtered.groupby("sector")["pnl_pct"].mean().reset_index()
        fig5 = px.bar(
            sector_pnl.sort_values("pnl_pct"), x="sector", y="pnl_pct",
            labels={"sector": "Sector", "pnl_pct": "Mean straddle P&L (%)"},
            title="Historical-vol-priced straddle P&L by sector",
        )
        fig5.add_hline(y=0, line_color="black")
        st.plotly_chart(fig5, use_container_width=True)
    st.caption(
        "None of this measures real implied volatility, there's no options-chain data in this "
        "project. It shows the realized side only: how big the reaction actually was, and what "
        "it would have cost to sell into it at a price set by historical volatility alone."
    )
except FileNotFoundError:
    st.caption(
        "Run `python volatility_risk_premium.py` and `python straddle_backtest.py` to "
        "generate this section."
    )

st.divider()

st.subheader("Live check: is the market pricing this correctly right now?")
st.caption(
    "Pulls real options-chain data and today's earnings calendar from yfinance, and compares "
    "the market's current expected move to this specific ticker's own historical earnings "
    "reaction. Unlike everything else on this page, this is not reproducible: it reflects "
    "live prices and today's earnings calendar, not the fixed historical dataset."
)
if data_source != "live database":
    st.caption(
        "Needs a live database connection for the historical baseline query, not available "
        "in this environment. Run this locally with Postgres to use this section."
    )
else:
    @st.cache_data(ttl=900)
    def run_live_check(symbols: tuple[str, ...]):
        from live_iv_check import build_richness_table
        return build_richness_table(list(symbols), get_engine())

    symbols_input = st.text_input("Tickers (comma-separated)", value="HOOD, NVDA, GOOGL")
    if st.button("Run live check"):
        symbols = tuple(s.strip().upper() for s in symbols_input.split(",") if s.strip())
        try:
            with st.spinner("Pulling live options data from yfinance..."):
                live_result, live_messages = run_live_check(symbols)
        except Exception as exc:
            st.error(f"Live check failed: {exc}")
        else:
            for message in live_messages:
                st.caption(message)
            if live_result.empty:
                st.caption("No tickers produced a usable comparison.")
            else:
                st.dataframe(live_result, use_container_width=True, hide_index=True)
                st.caption(
                    "Descriptive context from this project's own historical data, not a trading "
                    "signal or a recommendation. Results cached for 15 minutes to avoid "
                    "hammering yfinance on repeated clicks."
                )

st.divider()

st.subheader("Ticker drill-down")
symbol = st.selectbox("Symbol", options=sorted(filtered["symbol"].unique()))
ticker_df = filtered[filtered["symbol"] == symbol].sort_values("reported_date")
st.dataframe(
    ticker_df[["reported_date", "report_time", "surprise_percentage", "day0_date",
               "drift_10d_pct", "abnormal_drift_10d_pct", "volume_spike_ratio", "volatility_change_ratio"]],
    use_container_width=True, hide_index=True,
)

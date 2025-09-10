import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Options Flow Summarizer — Weighted Bias + Gauges", layout="wide")

# ============================
# Normalization
# ============================
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("~", "price") for c in df.columns]
    ren = {"priceprice":"underlying_price","price":"underlying_price","price_":"underlying_price",
           "open_int":"open_interest","oi":"open_interest","trade":"trade_price",
           "exp":"expires","expiration":"expires"}
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # IV parsing
    if "iv" in df.columns:
        df["iv"] = df["iv"].astype(str).str.replace("%","",regex=False).str.strip()
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        if df["iv"].max(skipna=True) and df["iv"].max(skipna=True) > 3:
            df["iv"] = df["iv"] / 100.0

    # Expiration date parsing
    if "expires" in df.columns:
        exp = pd.to_datetime(df["expires"], errors="coerce", utc=True)
        df["exp_date"] = exp.dt.tz_localize(None).dt.date
    else:
        df["exp_date"] = pd.NaT

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.capitalize()

    # Ensure minimum columns exist
    required = ["symbol","type","strike","exp_date","dte","trade_price",
                "size","premium","open_interest","iv","underlying_price","volume","side"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Numeric casts
    num_cols = ["strike","dte","trade_price","size","premium",
                "open_interest","iv","underlying_price","volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strike distance percentage
    df["strike_distance_pct"] = (df["strike"] - df["underlying_price"]) / df["underlying_price"] * 100.0
    return df


# ============================
# Aggregation
# ============================
def _aggregate(df: pd.DataFrame):
    grp_cols = ["type","strike","exp_date"]

    grouped = df.groupby(grp_cols, as_index=False).agg(
        premium_sum=("premium","sum"),
        oi=("open_interest","max"),
        vol=("volume","sum"),
        largest_trade=("size","max"),
        under_px=("underlying_price","last"),
        iv_mean=("iv","mean")
    )

    totals_df  = df.groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"total_size"})
    inst500_df = df[df["size"]>=500].groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"inst_size_ge500"})

    by_strike = grouped.merge(totals_df, on=grp_cols, how="left").merge(inst500_df, on=grp_cols, how="left")

    for col in ["total_size","inst_size_ge500"]:
        if col not in by_strike.columns:
            by_strike[col] = 0.0
    by_strike = by_strike.fillna(0.0)

    by_strike["vol_oi_ratio"] = np.where(by_strike["oi"]>0, by_strike["vol"]/(by_strike["oi"]+1.0), np.nan)
    return by_strike


# ============================
# Weighted Bias Calculation
# ============================
def _weighted_bias_row(row):
    sign = 1.0 if str(row.get("type","")).lower() == "call" else -1.0
    score = sign * (np.log1p(row.get("premium_sum", 0.0)) / 15.0)
    if row.get("vol_oi_ratio", np.nan) > 1.0:
        score *= 1.2
    if row.get("inst_size_ge500", 0.0) > 0:
        score *= 1.3
    return score


def add_weighted_bias(df):
    df["weighted_bias_score"] = df.apply(_weighted_bias_row, axis=1)
    df["weighted_bias_label"] = df["weighted_bias_score"].apply(
        lambda x: "Bullish" if x > 0.8 else ("Bearish" if x < -0.8 else "Neutral")
    )
    return df


# ============================
# Gauge Visuals
# ============================
def _gauge(value, title, min_val=-100, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_val, 0], 'color': "#ffcccc"},
                {'range': [0, max_val], 'color': "#ccffcc"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
    return fig


def overall_gauge_score(df):
    if df.empty:
        return 0.0
    return float(np.clip(df["weighted_bias_score"].sum(), -3.0, 3.0) / 3.0 * 100.0)


# ============================
# Streamlit UI
# ============================
st.title("Options Flow Summarizer — Weighted Bias + Gauges")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    df = _norm_cols(raw)
    st.success("CSV normalized successfully!")

    # Aggregation
    agg = _aggregate(df)
    agg = add_weighted_bias(agg)

    # Overall gauge
    score = overall_gauge_score(agg)
    st.subheader("Overall Options Bias Gauge")
    st.plotly_chart(_gauge(score, "Net Bias"))

    # Strongest bullish and bearish
    most_bullish = agg.sort_values("weighted_bias_score", ascending=False).head(1)
    most_bearish = agg.sort_values("weighted_bias_score", ascending=True).head(1)

    col1, col2 = st.columns(2)
    if not most_bullish.empty:
        row = most_bullish.iloc[0]
        col1.subheader("Most Bullish Level")
        col1.plotly_chart(_gauge(row["weighted_bias_score"]*30, "Bullish Level"), use_container_width=True)
        col1.write(row)

    if not most_bearish.empty:
        row = most_bearish.iloc[0]
        col2.subheader("Most Bearish Level")
        col2.plotly_chart(_gauge(row["weighted_bias_score"]*30, "Bearish Level"), use_container_width=True)
        col2.write(row)

    # Data table
    st.subheader("Detailed View")
    st.dataframe(agg.sort_values("premium_sum", ascending=False))

else:
    st.info("Upload a CSV to get started.")

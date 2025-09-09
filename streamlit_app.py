import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Flow Summarizer", layout="wide")

# --------------------------- Helpers ---------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("~", "price") for c in df.columns]

    ren = {
        "priceprice": "underlying_price",
        "price": "underlying_price",
        "price_": "underlying_price",
        "open_int": "open_interest",
        "oi": "open_interest",
        "trade": "trade_price",
        "exp": "expires",
        "expiration": "expires",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # IV parsing
    if "iv" in df.columns:
        df["iv"] = df["iv"].astype(str).str.replace("%", "", regex=False).str.strip()
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        if df["iv"].max(skipna=True) and df["iv"].max(skipna=True) > 3:
            df["iv"] = df["iv"] / 100.0

    # --- FIXED Expires parsing ---
    if "expires" in df.columns:
        exp = pd.to_datetime(df["expires"], errors="coerce", utc=True)
        if not pd.api.types.is_datetime64_any_dtype(exp):
            exp = pd.to_datetime(df["expires"].astype(str), errors="coerce", utc=True)
        df["exp_date"] = exp.dt.tz_localize(None).dt.date
    else:
        df["exp_date"] = pd.NaT

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.capitalize()

    # Ensure required minimal set
    required = [
        "symbol", "type", "strike", "exp_date", "dte", "trade_price",
        "size", "premium", "open_interest", "iv", "underlying_price",
        "volume", "side"
    ]
    for col in required:
        if col not in df.columns:
            if col in ("volume", "open_interest", "iv"):
                df[col] = np.nan
            elif col == "side":
                df[col] = "unknown"
            else:
                df[col] = np.nan

    num_cols = ["strike","dte","trade_price","size","premium",
                "open_interest","iv","underlying_price","volume"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _money(x):
    try:
        return f"${float(x):,.0f}"
    except:
        return str(x)

# --------------------------- Summary Logic ---------------------------

def summarize_data(df: pd.DataFrame):
    total_premium = df["premium"].sum(skipna=True)
    calls_premium = df.loc[df["type"] == "Call", "premium"].sum(skipna=True)
    puts_premium = df.loc[df["type"] == "Put", "premium"].sum(skipna=True)

    # IV
    iv_mean = df["iv"].mean(skipna=True)
    iv_max = df["iv"].max(skipna=True)

    # Top expirations
    top_expirations = df.groupby("exp_date")["premium"].sum().sort_values(ascending=False).head(5)

    # Top strikes
    top_strikes = (
        df.groupby(["type", "strike"])["premium"]
        .sum()
        .reset_index()
        .sort_values("premium", ascending=False)
        .head(10)
    )

    # Volume / OI
    df["vol_oi_ratio"] = df.apply(
        lambda r: r["volume"] / (r["open_interest"] + 1.0) if pd.notnull(r["open_interest"]) else np.nan,
        axis=1
    )

    # Narrative
    narrative = []

    # Premium balance
    if puts_premium > calls_premium * 1.5:
        narrative.append("**Flow heavily bearish:** Puts dominate premium flow.")
    elif calls_premium > puts_premium * 1.5:
        narrative.append("**Flow heavily bullish:** Calls dominate premium flow.")
    else:
        narrative.append("**Flow balanced:** Calls and puts are roughly equal.")

    # IV level
    if iv_mean > 0.8:
        narrative.append("**IV is elevated** → Good environment for premium selling (CSPs, covered calls, credit spreads).")
    elif iv_mean < 0.3:
        narrative.append("**IV is low** → Good environment for long debit strategies (long calls/puts, debit spreads).")

    # Top levels analysis
    narrative.append("\n**Key Strikes and Levels:**")
    for _, row in top_strikes.iterrows():
        t = row["type"]
        narrative.append(f"- {t} {row['strike']}: {_money(row['premium'])} total premium")

    return {
        "total_premium": total_premium,
        "calls_premium": calls_premium,
        "puts_premium": puts_premium,
        "iv_mean": iv_mean,
        "iv_max": iv_max,
        "top_expirations": top_expirations,
        "top_strikes": top_strikes,
        "narrative": narrative
    }

# --------------------------- UI ---------------------------

st.title("Options Flow Summarizer — Key Levels, OI/Volume/Size, Strategy Hints")
st.caption("Upload an options-flow CSV (e.g., Barchart export). No manual inputs required.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    with st.expander("Raw Preview", expanded=False):
        st.dataframe(raw.head(25), use_container_width=True)

    df = _norm_cols(raw)

    st.success("CSV normalized successfully! Here's a preview:")
    st.dataframe(df.head(10))

    # Generate summary
    results = summarize_data(df)

    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Premium", _money(results["total_premium"]))
    col2.metric("Calls Premium", _money(results["calls_premium"]))
    col3.metric("Puts Premium", _money(results["puts_premium"]))
    col4.metric("Mean IV", f"{results['iv_mean']:.2f}")

    st.subheader("Top Expirations by Premium")
    st.dataframe(results["top_expirations"].rename("Premium ($)").to_frame())

    st.subheader("Top Strikes by Premium")
    st.dataframe(results["top_strikes"])

    st.subheader("Narrative Summary")
    for line in results["narrative"]:
        st.markdown(line)

    # Download summary as Markdown
    summary_md = "# Options Flow Summary\n\n" + "\n".join(results["narrative"])
    st.download_button(
        label="Download Summary (Markdown)",
        data=summary_md.encode("utf-8"),
        file_name="options_flow_summary.md",
        mime="text/markdown"
    )

else:
    st.info("Upload a CSV to get started. Barchart-style exports work best.")
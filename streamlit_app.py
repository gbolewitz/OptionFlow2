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
    st.success("CSV parsed successfully! Here's a preview of normalized data:")
    st.dataframe(df.head(10))
else:
    st.info("Upload a CSV to get started. Barchart-style exports work best.")
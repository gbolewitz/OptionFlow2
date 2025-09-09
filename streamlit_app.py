# =============================
# streamlit_app.py
# =============================
# Cloud web app that lets users upload an options-flow CSV (like Barchart export)
# and returns a structured summary: key levels, trend/flow, OI vs Volume vs Size,
# strike-distance zones, and strategy considerations (long call/put/CSP/spreads).
#
# Deploy on Streamlit Cloud:
# 1) Put this file and requirements.txt in a public GitHub repo
# 2) Go to https://share.streamlit.io, connect the repo, select streamlit_app.py
# 3) Open the app URL, upload your CSV, and read/download the summary.
#
# No manual inputs required in this version.

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Flow Summarizer", layout="wide")

# --------------------------- Helpers ---------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lowercase snake_case
    df.columns = [c.strip().lower().replace(" ", "_").replace("~", "price") for c in df.columns]

    # Soft renames for common variants
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

    # Parse IV (e.g., "126.39%" -> 1.2639)
    if "iv" in df.columns:
        df["iv"] = (df["iv"].astype(str).str.replace("%", "", regex=False).str.strip())
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
        if df["iv"].max(skipna=True) and df["iv"].max(skipna=True) > 3:
            df["iv"] = df["iv"] / 100.0

    # Split bid_x_size / ask_x_size -> bid/ask price + size
    def split_px_sz(x):
        if isinstance(x, str) and "x" in x:
            parts = [p.strip() for p in x.split("x")]
            if len(parts) >= 2:
                try: px = float(parts[0])
                except: px = np.nan
                try: sz = float(parts[1])
                except: sz = np.nan
                return pd.Series([px, sz])
        return pd.Series([np.nan, np.nan])

    if "bid_x_size" in df.columns:
        df[["bid_price", "bid_size"]] = df["bid_x_size"].apply(split_px_sz)
    if "ask_x_size" in df.columns:
        df[["ask_price", "ask_size"]] = df["ask_x_size"].apply(split_px_sz)

    # Expires -> date only
    if "expires" in df.columns:
        df["exp_date"] = pd.to_datetime(df["expires"], errors="coerce").dt.date
    else:
        df["exp_date"] = pd.NaT

    # Normalize types
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.capitalize()

    # Ensure required minimal set exists
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

    # Numeric casts
    num_cols = ["strike","dte","trade_price","size","premium",
                "open_interest","iv","underlying_price","volume",
                "bid_price","bid_size","ask_price","ask_size"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _strike_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strike_distance_pct"] = (df["strike"] - df["underlying_price"]) / df["underlying_price"] * 100.0
    bins = [-np.inf, -10, -5, -2, 2, 5, 10, np.inf]
    labels = [
        "Deep Below (<=-10%)", "Below (-10%..-5%)", "Slightly Below (-5%..-2%)",
        "ATM (-2%..+2%)", "Slightly Above (+2%..+5%)", "Above (+5%..+10%)",
        "Deep Above (>=+10%)"
    ]
    df["strike_zone"] = pd.cut(df["strike_distance_pct"], bins=bins, labels=labels)
    return df


def _aggregate_key_tables(df: pd.DataFrame):
    # Totals
    tot_prem = df["premium"].sum(skipna=True)
    call_prem = df.loc[df["type"] == "Call", "premium"].sum(skipna=True)
    put_prem = df.loc[df["type"] == "Put", "premium"].sum(skipna=True)

    # IV stats
    iv_mean = df["iv"].mean(skipna=True)
    iv_p75 = df["iv"].quantile(0.75)
    iv_max = df["iv"].max(skipna=True)

    # Expiration concentration (top 5 by premium)
    exp_prem = df.groupby("exp_date")["premium"].sum().sort_values(ascending=False).head(5)

    # Top 10 strikes by premium (group by type+strike)
    top_strikes = (
        df.groupby(["type","strike"])["premium"]
          .sum().reset_index()
          .sort_values("premium", ascending=False).head(10)
    )

    # DTE buckets (premium)
    dte_bins = [-np.inf, 2, 7, 30, 60, 120, np.inf]
    dte_lbls = ["<=2", "3-7", "8-30", "31-60", "61-120", ">120"]
    dte_bucket = pd.cut(df["dte"], bins=dte_bins, labels=dte_lbls)
    dte_prem = df.groupby(dte_bucket)["premium"].sum().reindex(dte_lbls)

    # OI/Volume ratios & institutional size by (type,strike,exp_date)
    grp_cols = ["type","strike","exp_date"]
    by_strike = df.groupby(grp_cols).agg(
        premium_sum=("premium","sum"),
        oi=("open_interest","max"),
        vol=("volume","sum"),
        largest_trade=("size","max"),
        total_size=("size","sum"),
        inst_size_100=("size", lambda s: float(s[s>=100].sum()) if (s>=100).any() else 0.0),
        inst_size_500=("size", lambda s: float(s[s>=500].sum()) if (s>=500).any() else 0.0),
        under_px=("underlying_price","last"),
    ).reset_index()

    by_strike["vol_oi_ratio"] = by_strike.apply(
        lambda r: (r["vol"] / (r["oi"] + 1.0)) if pd.notnull(r["oi"]) else np.nan, axis=1
    )
    by_strike["inst_share_ge100"] = by_strike.apply(
        lambda r: (r["inst_size_100"] / r["total_size"]) if r["total_size"] and r["total_size"]>0 else np.nan, axis=1
    )
    by_strike["inst_share_ge500"] = by_strike.apply(
        lambda r: (r["inst_size_500"] / r["total_size"]) if r["total_size"] and r["total_size"]>0 else np.nan, axis=1
    )
    by_strike["strike_distance_pct"] = (by_strike["strike"] - by_strike["under_px"]) / by_strike["under_px"] * 100.0

    return {
        "tot_prem": tot_prem,
        "call_prem": call_prem,
        "put_prem": put_prem,
        "iv_mean": iv_mean,
        "iv_p75": iv_p75,
        "iv_max": iv_max,
        "exp_prem": exp_prem,
        "top_strikes": top_strikes,
        "dte_prem": dte_prem,
        "by_strike": by_strike,
    }


def _money(x):
    try: return f"${float(x):,.0f}"
    except: return str(x)


def _insights_text(tables):
    lines = []
    tot = tables["tot_prem"]; c = tables["call_prem"]; p = tables["put_prem"]
    call_share = (c / tot * 100.0) if tot else 0.0
    put_share  = (p / tot * 100.0) if tot else 0.0

    lines.append("## Heuristic Signals")
    if put_share > 60:
        lines.append("- Near-term **bearish/hedging** tone (puts dominant by premium).")
    elif call_share > 60:
        lines.append("- Flow **bullish-leaning** (calls dominant by premium).")
    else:
        lines.append("- Flow **mixed/neutral** (puts and calls balanced).")

    iv_mean = tables["iv_mean"] or 0.0
    if iv_mean >= 0.8:
        lines.append("- **IV is elevated** → premium-selling setups (CSPs, credit spreads) attractive.")
    elif iv_mean <= 0.3:
        lines.append("- **IV is low** → favor long debit strategies if directional.")

    # DTE concentration
    dte_prem = tables["dte_prem"].fillna(0)
    near = float(dte_prem.get("3-7",0) + dte_prem.get("<=2",0))
    mid  = float(dte_prem.get("8-30",0) + dte_prem.get("31-60",0))
    long = float(dte_prem.get("61-120",0) + dte_prem.get(">120",0))
    if near > (mid + long):
        lines.append("- Premium concentrated **in the next 7 days** → event/earnings hedging likely.")
    elif long > (near + mid):
        lines.append("- Premium concentrated **far-dated** → longer-term conviction positioning.")

    # OI/Volume/Size narratives for top groups
    bys = tables["by_strike"].copy()
    top_levels = bys.sort_values("premium_sum", ascending=False).head(8)

    lines.append("\n## Key Levels & Institutional Activity")
    for _, r in top_levels.iterrows():
        t, strike, exp = r["type"], r["strike"], r["exp_date"]
        prem = _money(r["premium_sum"]); voi = r["vol_oi_ratio"]
        inst100 = r["inst_share_ge100"]; inst500 = r["inst_share_ge500"]; dist = r["strike_distance_pct"]

        story_bits = []
        if not np.isnan(voi):
            if voi > 1.0: story_bits.append("new positioning (Volume > OI)")
            elif voi < 0.5: story_bits.append("likely closing/rolling (Volume < OI)")
            else: story_bits.append("balanced open/close")
        if not np.isnan(inst500) and inst500 > 0.2:
            story_bits.append("**mega-block** prints present (>=500 contracts)")
        elif not np.isnan(inst100) and inst100 > 0.3:
            story_bits.append("institutional-sized activity (>=100 contracts)")

        loc = ""
        if not np.isnan(dist):
            if dist <= -2: loc = "(below spot: potential support)"
            elif -2 < dist < 2: loc = "(near ATM: control zone)"
            else: loc = "(above spot: potential resistance)"

        lines.append(f"- {t} {strike} exp {exp}: {prem} — {'; '.join(story_bits) if story_bits else 'flow noted'} {loc}")

    # Strategy suggestions (non-advice)
    lines.append("\n## Potential Plays (Not Advice)")
    if iv_mean >= 0.8:
        lines.append("- **Cash-Secured Puts (CSPs):** target strikes 2–5% below spot where puts show high premium and OI; 30–45 DTE.")
        lines.append("- **Put Credit Spreads:** define risk near support clusters with strong put OI.")
        lines.append("- **Covered Calls:** if long shares, monetize high IV at resistance strikes flagged by call clusters.")
    else:
        lines.append("- **Debit Call/Put Spreads:** when IV is moderate/low and a directional bias is present.")

    # Bias by comparing leading put/call cluster premia near spot
    bys_near = bys[abs(bys["strike_distance_pct"]) <= 5].copy()
    near_put  = bys_near.loc[bys_near["type"]=="Put","premium_sum"].sum()
    near_call = bys_near.loc[bys_near["type"]=="Call","premium_sum"].sum()
    if near_put > near_call * 1.5:
        lines.append("- **Short-term bias:** Puts concentrated near spot → consider **bearish** structures (put spreads) or **defensive CSPs**.")
    elif near_call > near_put * 1.5:
        lines.append("- **Short-term bias:** Calls concentrated near spot → consider **bullish** structures (call spreads).")
    else:
        lines.append("- **Short-term bias:** Mixed near spot → consider neutral income (iron condors) or wait for confirmation.")

    return "\n".join(lines)


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

    # Normalize & enrich
    df = _norm_cols(raw)
    df = _strike_distance(df)

    # Aggregate tables
    tables = _aggregate_key_tables(df)

    # Top metrics header
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Premium", _money(tables["tot_prem"]))
    col2.metric("Calls Premium", _money(tables["call_prem"]))
    col3.metric("Puts Premium", _money(tables["put_prem"]))
    ivm = tables["iv_mean"] if not pd.isna(tables["iv_mean"]) else 0.0
    col4.metric("Mean IV (dec)", f"{ivm:.2f}")

    st.subheader("Expiration Concentration (Top 5 by Premium)")
    st.dataframe(tables["exp_prem"].rename("premium").to_frame(), use_container_width=True)

    st.subheader("Largest Strikes by Premium (Top 10, all expiries)")
    st.dataframe(tables["top_strikes"], use_container_width=True)

    st.subheader("DTE Buckets (Premium Sum)")
    st.dataframe(tables["dte_prem"].rename("premium").to_frame(), use_container_width=True)

    st.subheader("Per-Strike Detail (Type/Strike/Expiry)")
    show_cols = [
        "type","strike","exp_date","premium_sum","oi","vol","vol_oi_ratio",
        "largest_trade","total_size","inst_share_ge100","inst_share_ge500",
        "under_px","strike_distance_pct"
    ]
    st.dataframe(tables["by_strike"][show_cols].sort_values("premium_sum", ascending=False), use_container_width=True)

    # Compose narrative summary
    summary_md = []
    summary_md.append("# Options Flow Summary\n")
    summary_md.append(f"**Total premium:** {_money(tables['tot_prem'])}")
    summary_md.append(f"- Calls: {_money(tables['call_prem'])}")
    summary_md.append(f"- Puts: {_money(tables['put_prem'])}\n")

    summary_md.append("## Implied Volatility\n")
    summary_md.append(f"- Mean IV: {tables['iv_mean']:.2f}")
    summary_md.append(f"- 75th pct IV: {tables['iv_p75']:.2f}")
    summary_md.append(f"- Max IV: {tables['iv_max']:.2f}\n")

    summary_md.append("## Expiration Concentration (Top 5)\n")
    for d, val in tables["exp_prem"].items():
        summary_md.append(f"- {d}: {_money(val)}")

    summary_md.append("\n## Largest Strikes by Premium (Top 10)\n")
    for _, row in tables["top_strikes"].iterrows():
        summary_md.append(f"- {row['type']} {row['strike']}: {_money(row['premium'])}")

    summary_md.append("\n## DTE Buckets (Premium)\n")
    for idx, val in tables["dte_prem"].items():
        summary_md.append(f"- {idx}: {_money(val)}")

    summary_md.append("\n" + _insights_text(tables))

    summary_text = "\n".join(summary_md)

    st.subheader("Narrative Summary")
    st.markdown(summary_text)

    st.download_button(
        label="Download Summary (Markdown)",
        data=summary_text.encode("utf-8"),
        file_name="options_flow_summary.md",
        mime="text/markdown",
    )

else:
    st.info("Upload a CSV to get started. Barchart-style exports work best.")
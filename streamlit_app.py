import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Options Flow Summarizer (Pro Signals) — v1.1", layout="wide")

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

    # Robust expires parsing
    if "expires" in df.columns:
        exp = pd.to_datetime(df["expires"], errors="coerce", utc=True)
        if not pd.api.types.is_datetime64_any_dtype(exp):
            exp = pd.to_datetime(df["expires"].astype(str), errors="coerce", utc=True)
        df["exp_date"] = exp.dt.tz_localize(None).dt.date
    else:
        df["exp_date"] = pd.NaT

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.capitalize()

    # Ensure minimal set
    required = ["symbol","type","strike","exp_date","dte","trade_price",
                "size","premium","open_interest","iv","underlying_price","volume","side"]
    for col in required:
        if col not in df.columns:
            if col in ("volume","open_interest","iv"):
                df[col] = np.nan
            elif col == "side":
                df[col] = "unknown"
            else:
                df[col] = np.nan

    # Numeric casts
    num_cols = ["strike","dte","trade_price","size","premium",
                "open_interest","iv","underlying_price","volume"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strike distance & zone
    df["strike_distance_pct"] = (df["strike"] - df["underlying_price"]) / df["underlying_price"] * 100.0
    bins = [-np.inf, -10, -5, -2, 2, 5, 10, np.inf]
    labels = ["Deep Below (<=-10%)","Below (-10%..-5%)","Slightly Below (-5%..-2%)",
              "ATM (-2%..+2%)","Slightly Above (+2%..+5%)","Above (+5%..+10%)","Deep Above (>=+10%)"]
    df["strike_zone"] = pd.cut(df["strike_distance_pct"], bins=bins, labels=labels)

    # Row-level Vol/OI for quick checks
    df["vol_oi_ratio_row"] = df.apply(lambda r: r["volume"]/(r["open_interest"]+1.0) if pd.notnull(r["open_interest"]) else np.nan, axis=1)
    return df

def _money(x):
    try: return f"${float(x):,.0f}"
    except: return str(x)

# ============================
# Aggregations (with robust merges to avoid KeyError)
# ============================
def _aggregate(df: pd.DataFrame):
    tot_prem = df["premium"].sum(skipna=True)
    call_prem = df.loc[df["type"]=="Call","premium"].sum(skipna=True)
    put_prem  = df.loc[df["type"]=="Put","premium"].sum(skipna=True)

    iv_mean = df["iv"].mean(skipna=True); iv_p75 = df["iv"].quantile(0.75); iv_max = df["iv"].max(skipna=True)
    exp_prem = df.groupby("exp_date")["premium"].sum().sort_values(ascending=False).head(5)
    top_strikes = (df.groupby(["type","strike"])["premium"].sum().reset_index().sort_values("premium", ascending=False).head(10))

    dte_bins = [-np.inf, 2, 7, 30, 60, 120, np.inf]; dte_lbls = ["<=2","3-7","8-30","31-60","61-120",">120"]
    dte_bucket = pd.cut(df["dte"], bins=dte_bins, labels=dte_lbls); dte_prem = df.groupby(dte_bucket)["premium"].sum().reindex(dte_lbls)

    grp_cols = ["type","strike","exp_date"]
    grouped = df.groupby(grp_cols, as_index=False).agg(
        premium_sum=("premium","sum"),
        oi=("open_interest","max"),
        vol=("volume","sum"),
        largest_trade=("size","max"),
        under_px=("underlying_price","last"),
        iv_mean=("iv","mean"),
    )

    # Totals and institutional sizes as DataFrames
    totals_df  = df.groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"total_size"})
    inst100_df = df[df["size"]>=100].groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"inst_size_ge100"})
    inst500_df = df[df["size"]>=500].groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"inst_size_ge500"})

    by_strike = grouped.merge(totals_df, on=grp_cols, how="left")
    by_strike = by_strike.merge(inst100_df, on=grp_cols, how="left")
    by_strike = by_strike.merge(inst500_df, on=grp_cols, how="left")

    # Ensure presence and fill
    for col in ["total_size","inst_size_ge100","inst_size_ge500"]:
        if col not in by_strike.columns:
            by_strike[col] = 0.0
    by_strike[["total_size","inst_size_ge100","inst_size_ge500"]] = by_strike[["total_size","inst_size_ge100","inst_size_ge500"]].fillna(0.0)

    # Derived metrics
    by_strike["vol_oi_ratio"] = by_strike.apply(lambda r: (r["vol"]/(r["oi"]+1.0)) if pd.notnull(r["oi"]) else np.nan, axis=1)
    by_strike["strike_distance_pct"] = (by_strike["strike"] - by_strike["under_px"]) / by_strike["under_px"] * 100.0
    by_strike["inst_share_ge100"] = np.where(by_strike["total_size"]>0, by_strike["inst_size_ge100"]/by_strike["total_size"], np.nan)
    by_strike["inst_share_ge500"] = np.where(by_strike["total_size"]>0, by_strike["inst_size_ge500"]/by_strike["total_size"], np.nan)

    return {"tot_prem":tot_prem,"call_prem":call_prem,"put_prem":put_prem,"iv_mean":iv_mean,"iv_p75":iv_p75,"iv_max":iv_max,
            "exp_prem":exp_prem,"top_strikes":top_strikes,"dte_prem":dte_prem,"by_strike":by_strike}

# ============================
# Hidden Signals
# ============================
def hidden_signals(df: pd.DataFrame, by_strike: pd.DataFrame):
    lines_bull, lines_bear = [], []

    # Closing (<0.5) vs Opening (>1.0)
    closing_clusters = by_strike[(by_strike["vol_oi_ratio"].notna()) & (by_strike["vol_oi_ratio"] < 0.5)]
    opening_clusters = by_strike[(by_strike["vol_oi_ratio"].notna()) & (by_strike["vol_oi_ratio"] > 1.0)]
    if not closing_clusters.empty:
        puts_close = closing_clusters[closing_clusters["type"]=="Put"]["premium_sum"].sum()
        calls_close = closing_clusters[closing_clusters["type"]=="Call"]["premium_sum"].sum()
        if puts_close > calls_close * 1.2: lines_bull.append("Heavy **put volume with low Vol/OI (<0.5)** → likely **closing hedges**, a *bullish* tell.")
        if calls_close > puts_close * 1.2: lines_bear.append("Heavy **call volume with low Vol/OI (<0.5)** → likely **closing calls**, a *bearish* tell.")
    if not opening_clusters.empty:
        puts_open = opening_clusters[opening_clusters["type"]=="Put"]["premium_sum"].sum()
        calls_open = opening_clusters[opening_clusters["type"]=="Call"]["premium_sum"].sum()
        if puts_open > calls_open * 1.2: lines_bear.append("**New put positioning** (Vol/OI > 1.0) dominates → *bearish* conviction.")
        if calls_open > puts_open * 1.2: lines_bull.append("**New call positioning** (Vol/OI > 1.0) dominates → *bullish* conviction.")

    # Put/Call IV skew near ATM (±5%)
    near = df[abs(df["strike_distance_pct"])<=5].copy()
    put_iv = near.loc[near["type"]=="Put","iv"].mean(); call_iv = near.loc[near["type"]=="Call","iv"].mean()
    if pd.notnull(put_iv) and pd.notnull(call_iv):
        if (put_iv - call_iv) >= 0.10: lines_bull.append("**Put IV >> Call IV near ATM** → market **over-hedged** with puts; upside surprise risk.")
        elif (call_iv - put_iv) >= 0.10: lines_bear.append("**Call IV >> Put IV near ATM** → **greedy upside pricing**; downside risk if disappointment.")

    # Near-term puts ABOVE spot (<=7 DTE) → hedging tell
    dtes = df[["exp_date","dte"]].drop_duplicates()
    near_term = by_strike[(by_strike["type"]=="Put") & (by_strike["strike_distance_pct"]>0)].merge(dtes, on="exp_date", how="left")
    nt_above = near_term[near_term["dte"]<=7]
    if not nt_above.empty and nt_above["premium_sum"].sum()>0:
        lines_bull.append("**Near-term puts ABOVE spot** with sizable OI/premium → consistent with **hedging longs**, not outright bearish bets.")

    # Institutional mega-blocks (>=500)
    blocks = by_strike[(by_strike["inst_share_ge500"].notna()) & (by_strike["inst_share_ge500"]>0.2)]
    if not blocks.empty:
        bull_blocks = blocks[blocks["type"]=="Call"]["premium_sum"].sum()
        bear_blocks = blocks[blocks["type"]=="Put"]["premium_sum"].sum()
        if bull_blocks > bear_blocks * 1.1: lines_bull.append("**Mega-block call prints** present → institutional *bullish* interest.")
        elif bear_blocks > bull_blocks * 1.1: lines_bear.append("**Mega-block put prints** present → institutional *bearish* interest.")

    # Far-dated call accumulation (>120 DTE)
    by_dte = df.groupby(["type","exp_date"])["dte"].mean().reset_index()
    long_dtes = by_dte[by_dte["dte"]>120]["exp_date"].tolist()
    leaps = by_strike[(by_strike["type"]=="Call") & (by_strike["exp_date"].isin(long_dtes))]
    if not leaps.empty and leaps["premium_sum"].sum()>0:
        lines_bull.append("**Far-dated call accumulation** (>120 DTE) detected → quiet *long-term bullish* setup.")

    return lines_bull, lines_bear

# ============================
# Narrative
# ============================
def build_narrative(tables, df):
    lines = []
    tot = tables["tot_prem"]; c = tables["call_prem"]; p = tables["put_prem"]
    call_share = (c / tot * 100.0) if tot else 0.0; put_share  = (p / tot * 100.0) if tot else 0.0
    lines.append("## Heuristic Signals")
    if put_share > 60: lines.append("- Near-term **bearish/hedging** tone (puts dominant by premium).")
    elif call_share > 60: lines.append("- Flow **bullish-leaning** (calls dominant by premium).")
    else: lines.append("- Flow **mixed/neutral** (puts and calls balanced).")
    iv_mean = tables["iv_mean"] or 0.0
    if iv_mean >= 0.8: lines.append("- **IV is elevated** → premium-selling setups (CSPs, credit spreads) attractive.")
    elif iv_mean <= 0.3: lines.append("- **IV is low** → favor long debit strategies if directional.")
    dte_prem = tables["dte_prem"].fillna(0)
    near = float(dte_prem.get("<=2",0) + dte_prem.get("3-7",0)); mid  = float(dte_prem.get("8-30",0) + dte_prem.get("31-60",0))
    long = float(dte_prem.get("61-120",0) + dte_prem.get(">120",0))
    if near > (mid + long): lines.append("- Premium concentrated **in the next 7 days** → event/earnings hedging likely.")
    elif long > (near + mid): lines.append("- Premium concentrated **far-dated** → longer-term conviction positioning.")
    bull, bear = hidden_signals(df, tables["by_strike"])
    if bull:
        lines.append("\n## Hidden Bullish Flags"); [lines.append(f"- {s}") for s in bull]
    if bear:
        lines.append("\n## Hidden Bearish Flags"); [lines.append(f"- {s}") for s in bear]
    bys = tables["by_strike"].copy().sort_values("premium_sum", ascending=False).head(8)
    lines.append("\n## Key Levels & Institutional Activity")
    for _, r in bys.iterrows():
        t, strike, exp = r["type"], r["strike"], r["exp_date"]; prem = _money(r["premium_sum"]); voi = r["vol_oi_ratio"]
        inst500 = r["inst_share_ge500"]; dist = r["strike_distance_pct"]
        story = []
        if not np.isnan(voi):
            if voi > 1.0: story.append("new positioning")
            elif voi < 0.5: story.append("closing/rolling")
            else: story.append("balanced")
        if not np.isnan(inst500) and inst500 > 0.2: story.append("mega-blocks")
        loc = "(near ATM)" if -2 < dist < 2 else ("(support zone)" if dist <= -2 else "(resistance zone)")
        lines.append(f"- {t} {strike} exp {exp}: {prem} — {', '.join(story)} {loc}")
    lines.append("\n## Potential Plays (Not Advice)")
    if iv_mean >= 0.8:
        lines.append("- **CSPs / Put Credit Spreads** at support clusters (puts 2–5% below spot with strong OI).")
        lines.append("- **Covered Calls** at resistance clusters if long shares.")
    else:
        lines.append("- **Debit Call/Put Spreads** when directional bias appears from hidden flags + key levels.")
    return "\n".join(lines)

# ============================
# UI
# ============================
st.title("Options Flow Summarizer — Key Levels, OI/Volume/Size + Hidden Signals")
st.caption("Upload an options-flow CSV (Barchart-style). No manual inputs required.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}"); st.stop()

    with st.expander("Raw Preview", expanded=False):
        st.dataframe(raw.head(25), use_container_width=True)

    df = _norm_cols(raw)
    st.success("CSV normalized successfully!")
    with st.expander("Normalized Preview", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    tables = _aggregate(df)

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

    st.subheader("Per-Strike Detail (Institutional/Skew Signals)")
    show_cols = ["type","strike","exp_date","premium_sum","oi","vol","vol_oi_ratio",
                 "largest_trade","total_size","inst_share_ge100","inst_share_ge500",
                 "under_px","strike_distance_pct","iv_mean"]
    st.dataframe(tables["by_strike"][show_cols].sort_values("premium_sum", ascending=False), use_container_width=True)

    narrative = build_narrative(tables, df)
    st.subheader("Narrative Summary")
    st.markdown(narrative)

    st.download_button("Download Summary (Markdown)",
                       data=("# Options Flow Summary (Pro)\n\n"+narrative).encode("utf-8"),
                       file_name="options_flow_summary_pro.md", mime="text/markdown")
else:
    st.info("Upload a CSV to get started.")
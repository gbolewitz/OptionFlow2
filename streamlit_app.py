import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Options Flow Summarizer — PRO (Hidden Signals + Weighted Gauges + Entry Plays)", layout="wide")

# ============================
# Normalization
# ============================
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
        # Handle raw % like 45 -> 0.45 if needed
        if df["iv"].max(skipna=True) and df["iv"].max(skipna=True) > 3:
            df["iv"] = df["iv"] / 100.0

    # Robust expiration parsing
    if "expires" in df.columns:
        exp = pd.to_datetime(df["expires"], errors="coerce", utc=True)
        if not pd.api.types.is_datetime64_any_dtype(exp):
            exp = pd.to_datetime(df["expires"].astype(str), errors="coerce", utc=True)
        df["exp_date"] = exp.dt.tz_localize(None).dt.date
    else:
        df["exp_date"] = pd.NaT

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.capitalize()

    # Ensure minimum columns exist
    required = [
        "symbol","type","strike","exp_date","dte","trade_price",
        "size","premium","open_interest","iv","underlying_price","volume","side"
    ]
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

    # Strike distance % (vs underlying)
    df["strike_distance_pct"] = (df["strike"] - df["underlying_price"]) / df["underlying_price"] * 100.0

    # Quick per-row Vol/OI ratio (rough)
    df["vol_oi_ratio_row"] = df.apply(
        lambda r: r["volume"] / (r["open_interest"] + 1.0) if pd.notnull(r["open_interest"]) else np.nan,
        axis=1
    )
    return df

def _money(x):
    try:
        return f"${float(x):,.0f}"
    except:
        return str(x)

# ============================
# Aggregation (robust merges)
# ============================
def _aggregate(df: pd.DataFrame):
    tot_prem = df["premium"].sum(skipna=True)
    call_prem = df.loc[df["type"]=="Call","premium"].sum(skipna=True)
    put_prem  = df.loc[df["type"]=="Put","premium"].sum(skipna=True)

    iv_mean = df["iv"].mean(skipna=True)
    iv_p75  = df["iv"].quantile(0.75)
    iv_max  = df["iv"].max(skipna=True)

    # Expiration / strike summaries for display
    exp_prem = df.groupby("exp_date")["premium"].sum().sort_values(ascending=False).head(5)
    top_strikes = (df.groupby(["type","strike"])["premium"].sum()
                     .reset_index()
                     .sort_values("premium", ascending=False)
                     .head(10))

    # DTE buckets
    dte_bins = [-np.inf, 2, 7, 30, 60, 120, np.inf]
    dte_lbls = ["<=2", "3-7", "8-30", "31-60", "61-120", ">120"]
    dte_bucket = pd.cut(df["dte"], bins=dte_bins, labels=dte_lbls)
    dte_prem = df.groupby(dte_bucket)["premium"].sum().reindex(dte_lbls)

    # By (type, strike, exp_date) with institutional + Vol/OI
    grp_cols = ["type","strike","exp_date"]
    grouped = df.groupby(grp_cols, as_index=False).agg(
        premium_sum=("premium","sum"),
        oi=("open_interest","max"),
        vol=("volume","sum"),
        largest_trade=("size","max"),
        under_px=("underlying_price","last"),
        iv_mean=("iv","mean"),
    )

    totals_df  = df.groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"total_size"})
    inst100_df = df[df["size"]>=100].groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"inst_size_ge100"})
    inst500_df = df[df["size"]>=500].groupby(grp_cols, as_index=False)["size"].sum().rename(columns={"size":"inst_size_ge500"})

    by_strike = (grouped
                 .merge(totals_df, on=grp_cols, how="left")
                 .merge(inst100_df, on=grp_cols, how="left")
                 .merge(inst500_df, on=grp_cols, how="left"))

    for col in ["total_size","inst_size_ge100","inst_size_ge500"]:
        if col not in by_strike.columns:
            by_strike[col] = 0.0
    by_strike[["total_size","inst_size_ge100","inst_size_ge500"]] = by_strike[["total_size","inst_size_ge100","inst_size_ge500"]].fillna(0.0)

    by_strike["vol_oi_ratio"] = by_strike.apply(
        lambda r: (r["vol"] / (r["oi"] + 1.0)) if pd.notnull(r["oi"]) else np.nan,
        axis=1
    )
    by_strike["strike_distance_pct"] = (by_strike["strike"] - by_strike["under_px"]) / by_strike["under_px"] * 100.0
    by_strike["inst_share_ge100"] = np.where(by_strike["total_size"]>0, by_strike["inst_size_ge100"]/by_strike["total_size"], np.nan)
    by_strike["inst_share_ge500"] = np.where(by_strike["total_size"]>0, by_strike["inst_size_ge500"]/by_strike["total_size"], np.nan)

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
        "by_strike": by_strike
    }

# ============================
# Weighted Bias (per key level) + Strength
# ============================
def _positioning_tag(voi):
    if pd.isna(voi): return "unknown"
    if voi > 1.0: return "new positioning"
    if voi < 0.5: return "closing/rolling"
    return "balanced"

def _zone_tag(dist):
    if pd.isna(dist): return "unknown"
    if dist <= -2: return "support zone"
    if -2 < dist < 2: return "near ATM"
    return "resistance zone"

def _weighted_bias_row(row):
    # Base polarity: Call=+1, Put=-1
    sign = 1.0 if str(row.get("type","")).lower() == "call" else -1.0
    score = 0.0

    # Positioning (Vol/OI)
    voi = row.get("vol_oi_ratio", np.nan)
    if not pd.isna(voi):
        if voi > 1.0:
            score += 1.0 * sign
            score += min(0.3 * (voi - 1.0), 0.6) * sign
        elif voi < 0.5:
            score += -0.7 * sign
            score += -min(0.3 * (0.5 - voi), 0.6) * sign

    # Institutional mega-block emphasis (>=500 share > 20%)
    inst_share500 = row.get("inst_share_ge500", np.nan)
    if not pd.isna(inst_share500) and inst_share500 > 0.2:
        score += 0.6 * sign

    # Zone adjustments
    dist = row.get("strike_distance_pct", np.nan)
    if not pd.isna(dist):
        if -2 < dist < 2:  # ATM
            score *= 1.10
        elif dist <= -2:   # below spot (support)
            score += 0.10 * sign
        else:              # above spot (resistance)
            score += -0.10 * sign

    # Premium scaling to reward significant flow
    prem = float(row.get("premium_sum", 0.0) or 0.0)
    scale = np.log1p(max(prem, 1.0)) / 15.0  # soft normalization
    score *= scale
    return score

def add_weighted_bias_and_strength(tables):
    bys = tables["by_strike"].copy()
    bys["weighted_bias_score"] = bys.apply(_weighted_bias_row, axis=1)

    def label(s):
        if s >= 0.8: return "Bullish"
        if s <= -0.8: return "Bearish"
        return "Neutral"
    bys["weighted_bias_label"] = bys["weighted_bias_score"].apply(label)

    # Level strength for ranking strongest key levels
    novelty = (bys["vol_oi_ratio"] - 1.0).abs().fillna(0.0)
    inst_boost = bys["inst_share_ge500"].fillna(0.0)
    prem_boost = np.log1p(bys["premium_sum"].fillna(0.0))
    bys["level_strength"] = (bys["weighted_bias_score"].abs() * (1 + 0.5*novelty) * (1 + inst_boost) * prem_boost)

    tables["by_strike"] = bys
    return tables

# ============================
# New Activity Scoring + Play Suggestions
# ============================
def _is_opening_flow(row) -> bool:
    """
    Proxy for opening interest when we don't have prior-day OI:
    - Vol/OI > 1.1 indicates more traded volume than existing OI (opening bias).
    - Large single prints also count as opening bias.
    """
    voi = row.get("vol_oi_ratio", np.nan)
    largest = row.get("largest_trade", 0.0) or 0.0
    return (pd.notna(voi) and voi > 1.1) or (largest >= 250)  # tweak threshold as you like

def _new_activity_score(row) -> float:
    """
    Higher = fresher, more actionable level.
    Combines: opening signal strength, premium size (log), institutional share, and ATM proximity.
    """
    voi = row.get("vol_oi_ratio", np.nan)
    prem = float(row.get("premium_sum", 0.0) or 0.0)
    inst = float(row.get("inst_share_ge500", 0.0) or 0.0)
    dist = row.get("strike_distance_pct", np.nan)

    # Base from opening strength
    base = 0.0
    if pd.notna(voi):
        if voi > 1.1:
            base = min((voi - 1.0) * 1.5, 2.0)  # cap extreme prints
        elif voi < 0.7:
            base = -0.5  # closing bias de-prioritized

    # Institutional boost
    base += min(inst * 2.0, 1.0)  # up to +1 for strong mega-block share

    # Premium scaling (favor material flow)
    base *= (np.log1p(prem) / 12.0)

    # ATM proximity nudge (slight boost to actionable levels)
    if pd.notna(dist):
        if -2 < dist < 2:
            base *= 1.10
        elif dist > 8 or dist < -8:
            base *= 0.85  # far OTM slightly discounted

    return float(base)

def _zone_tag_for_play(dist):
    if pd.isna(dist): return "unknown"
    if dist <= -2: return "support"
    if -2 < dist < 2: return "ATM"
    return "resistance"

def _suggest_play(row, iv_mean: float) -> tuple[str, str]:
    """
    Heuristic suggestions (NOT ADVICE).
    - Bullish + high IV → credit put structures at support (CSP / put credit spread)
    - Bullish + low/med IV → debit call spread slightly OTM
    - Bearish + high IV → call credit spread near resistance
    - Bearish + low/med IV → debit put spread slightly OTM
    """
    t = str(row.get("type","")).lower()
    dist = row.get("strike_distance_pct", np.nan)
    zone = _zone_tag_for_play(dist)
    strike = row.get("strike")
    exp = row.get("exp_date")
    label = row.get("weighted_bias_label","Neutral")
    score = float(row.get("weighted_bias_score", 0.0))

    # Direction from bias score sign
    bullish = score > 0

    # Choose structure based on IV regime
    high_iv = iv_mean >= 0.7
    width = 2.0  # default width; adjust for your ticker

    if bullish:
        if high_iv:
            # Bullish + high IV → get paid: CSP or put credit spread at/under support
            if zone in ("support","ATM"):
                return ("Cash-Secured Put / Put Credit Spread",
                        f"Sell {strike}P (or {strike}P/{strike-width}P credit spread) exp {exp} at {zone}.")
            else:
                # if resistance but bullish and high IV, safer: put credit spread below spot
                return ("Put Credit Spread",
                        f"Sell {strike-2}P / Buy {strike-2-width}P exp {exp} below spot to keep distance from resistance.")
        else:
            # Bullish + low/med IV → buy premium: debit call spread
            target = strike + width
            return ("Debit Call Spread",
                    f"Buy {strike}C / Sell {target}C exp {exp} ({zone}).")
    else:  # bearish
        if high_iv:
            # Bearish + high IV → get paid: call credit spread at/near resistance
            if zone in ("resistance","ATM"):
                return ("Call Credit Spread",
                        f"Sell {strike}C / Buy {strike+width}C exp {exp} at {zone}.")
            else:
                # at support but bearish + high IV → safer to position above
                return ("Call Credit Spread",
                        f"Sell {strike+2}C / Buy {strike+2+width}C exp {exp} nearer resistance.")
        else:
            # Bearish + low/med IV → buy premium: debit put spread
            target = strike - width
            return ("Debit Put Spread",
                    f"Buy {strike}P / Sell {target}P exp {exp} ({zone}).")

def add_new_activity_and_plays(tables):
    """
    Adds columns:
      - opening_flag
      - new_activity_score
      - entry_play / entry_note (suggestion)
    """
    bys = tables["by_strike"].copy()
    bys["opening_flag"] = bys.apply(_is_opening_flow, axis=1)
    bys["new_activity_score"] = bys.apply(_new_activity_score, axis=1)

    # Suggest plays only for rows flagged as opening
    iv_mean = float(tables["iv_mean"] or 0.0)
    plays = bys.apply(lambda r: _suggest_play(r, iv_mean) if r["opening_flag"] else ("—", "Filtered: not opening"), axis=1)
    bys["entry_play"] = [p[0] for p in plays]
    bys["entry_note"] = [p[1] for p in plays]

    tables["by_strike"] = bys
    return tables

# ============================
# Hidden Signals
# ============================
def hidden_signals(df: pd.DataFrame, by_strike: pd.DataFrame):
    lines_bull, lines_bear = [], []

    # Closing vs Opening using Vol/OI
    closing_clusters = by_strike[(by_strike["vol_oi_ratio"].notna()) & (by_strike["vol_oi_ratio"] < 0.5)]
    opening_clusters = by_strike[(by_strike["vol_oi_ratio"].notna()) & (by_strike["vol_oi_ratio"] > 1.0)]

    if not closing_clusters.empty:
        puts_close = closing_clusters[closing_clusters["type"]=="Put"]["premium_sum"].sum()
        calls_close = closing_clusters[closing_clusters["type"]=="Call"]["premium_sum"].sum()
        if puts_close > calls_close * 1.2:
            lines_bull.append("Heavy **put volume with low Vol/OI (<0.5)** → likely **closing hedges**, a *bullish* tell.")
        if calls_close > puts_close * 1.2:
            lines_bear.append("Heavy **call volume with low Vol/OI (<0.5)** → likely **closing calls**, a *bearish* tell.")

    if not opening_clusters.empty:
        puts_open = opening_clusters[opening_clusters["type"]=="Put"]["premium_sum"].sum()
        calls_open = opening_clusters[opening_clusters["type"]=="Call"]["premium_sum"].sum()
        if puts_open > calls_open * 1.2:
            lines_bear.append("**New put positioning** (Vol/OI > 1.0) dominates → *bearish* conviction.")
        if calls_open > puts_open * 1.2:
            lines_bull.append("**New call positioning** (Vol/OI > 1.0) dominates → *bullish* conviction.")

    # Put/Call IV skew near ATM (±5%)
    near = df[abs(df["strike_distance_pct"])<=5].copy()
    put_iv = near.loc[near["type"]=="Put","iv"].mean()
    call_iv = near.loc[near["type"]=="Call","iv"].mean()
    if pd.notnull(put_iv) and pd.notnull(call_iv):
        if (put_iv - call_iv) >= 0.10:
            lines_bull.append("**Put IV >> Call IV near ATM** → market **over-hedged** with puts; upside surprise risk.")
        elif (call_iv - put_iv) >= 0.10:
            lines_bear.append("**Call IV >> Put IV near ATM** → **greedy upside pricing**; downside risk if disappointment.")

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
        if bull_blocks > bear_blocks * 1.1:
            lines_bull.append("**Mega-block call prints** present → institutional *bullish* interest.")
        elif bear_blocks > bull_blocks * 1.1:
            lines_bear.append("**Mega-block put prints** present → institutional *bearish* interest.")

    # Far-dated call accumulation (>120 DTE)
    by_dte = df.groupby(["type","exp_date"])["dte"].mean().reset_index()
    long_dtes = by_dte[by_dte["dte"]>120]["exp_date"].tolist()
    leaps = by_strike[(by_strike["type"]=="Call") & (by_strike["exp_date"].isin(long_dtes))]
    if not leaps.empty and leaps["premium_sum"].sum()>0:
        lines_bull.append("**Far-dated call accumulation** (>120 DTE) detected → quiet *long-term bullish* setup.")

    return lines_bull, lines_bear

# ============================
# Narrative Builder
# ============================
def build_narrative(tables, df):
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
    near = float(dte_prem.get("<=2",0) + dte_prem.get("3-7",0))
    mid  = float(dte_prem.get("8-30",0) + dte_prem.get("31-60",0))
    long = float(dte_prem.get("61-120",0) + dte_prem.get(">120",0))
    if near > (mid + long):
        lines.append("- Premium concentrated **in the next 7 days** → event/earnings hedging likely.")
    elif long > (near + mid):
        lines.append("- Premium concentrated **far-dated** → longer-term conviction positioning.")

    # Hidden signals section
    bull, bear = hidden_signals(df, tables["by_strike"])
    if bull:
        lines.append("\n## Hidden Bullish Flags")
        for s in bull: lines.append(f"- {s}")
    if bear:
        lines.append("\n## Hidden B

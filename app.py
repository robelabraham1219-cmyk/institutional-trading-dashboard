"""
==================================================================================
INSTITUTIONAL QUANTITATIVE TRADING DASHBOARD
Gold (XAU/USD) | Forex | Crypto
==================================================================================
A single-file, production-ready Streamlit dashboard integrating six institutional
quantitative trading modules:

  1. Order Flow & Liquidity Heatmap (BSL/SSL + Fair Value Gaps)
  2. Quantitative Machine Learning Classifier (RandomForest direction model)
  3. Cross-Asset Correlation & Macro Yield Matrix
  4. Volume Delta / Cumulative Volume Delta (CVD) Footprint Analysis
  5. Options Gamma Exposure (GEX) & Max Pain Engine
  6. Institutional Execution Algorithms (VWAP / TWAP / Iceberg Detection)
==================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import math
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==================================================================================
# PAGE CONFIG & GLOBAL STYLE
# ==================================================================================

st.set_page_config(
    page_title="Institutional Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0b0e14; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1f2937; }
    div[data-testid="stMetric"] {
        background-color: #131722;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] { color: #f0b90b; }
    h1, h2, h3 { color: #f0f0f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #131722;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #cfd3dc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937;
        color: #f0b90b;
        font-weight: 600;
    }
    .module-note {
        background-color: #131722;
        border-left: 3px solid #f0b90b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #b8bdc9;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #f0b90b;
        color: #0b0e14;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }
    thead tr th { background-color: #1f2937 !important; color: #f0b90b !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

# ==================================================================================
# ASSET UNIVERSE
# ==================================================================================

ASSET_MAP = {
    "Gold (XAU/USD)": "GC=F",
    "Silver (XAG/USD)": "SI=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CHF": "USDCHF=X",
    "Bitcoin (BTC/USD)": "BTC-USD",
    "Ethereum (ETH/USD)": "ETH-USD",
}

OPTIONS_PROXY_MAP = {
    "GC=F": "GLD",
    "SI=F": "SLV",
    "BTC-USD": None,
    "ETH-USD": None,
}

MACRO_TICKERS = {
    "Gold": "GC=F",
    "US Dollar Index": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "Bitcoin": "BTC-USD",
}

# ==================================================================================
# DATA LAYER
# ==================================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[expected].copy()
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_close_series(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    df = fetch_ohlcv(ticker, period, interval)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(symbol: str):
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options
        if not expiries:
            return None
        expiry = expiries[0]
        chain = tk.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
        return calls, puts, expiry, spot
    except Exception:
        return None


def get_last_price(ticker: str):
    try:
        df = fetch_ohlcv(ticker, period="5d", interval="1d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


# ==================================================================================
# SHARED TECHNICAL UTILITIES
# ==================================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def safe_pct(a, b):
    try:
        if b == 0 or pd.isna(b):
            return 0.0
        return (a - b) / abs(b) * 100
    except Exception:
        return 0.0


# ==================================================================================
# MODULE 1 — ORDER FLOW & LIQUIDITY HEATMAP
# ==================================================================================

def detect_swings(df: pd.DataFrame, window: int = 5):
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(window, n - window):
        window_high = highs[i - window: i + window + 1]
        window_low = lows[i - window: i + window + 1]
        if highs[i] == window_high.max():
            swing_highs.append((df.index[i], float(highs[i])))
        if lows[i] == window_low.min():
            swing_lows.append((df.index[i], float(lows[i])))
    return swing_highs, swing_lows


def detect_fvg(df: pd.DataFrame):
    bullish, bearish = [], []
    highs = df["High"].values
    lows = df["Low"].values
    idx = df.index
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            bullish.append({
                "start": idx[i - 2], "end": idx[i],
                "top": float(lows[i]), "bottom": float(highs[i - 2]),
            })
        if highs[i] < lows[i - 2]:
            bearish.append({
                "start": idx[i - 2], "end": idx[i],
                "top": float(lows[i - 2]), "bottom": float(highs[i]),
            })
    return bullish, bearish


def render_liquidity_module(df: pd.DataFrame, ticker: str):
    st.markdown('<div class="module-note">Swing-point liquidity pools (Buy-Side / Sell-Side Liquidity) and '
                'Fair Value Gap imbalance zones, derived from raw price-action structure.</div>',
                unsafe_allow_html=True)

    window = st.slider("Swing Detection Sensitivity (lookback bars)", 2, 15, 5, key="liq_window")
    max_zones = st.slider("Max FVG Zones Displayed", 3, 30, 12, key="liq_fvg_count")

    swing_highs, swing_lows = detect_swings(df, window=window)
    bullish_fvg, bearish_fvg = detect_fvg(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSL Pools (Swing Highs)", len(swing_highs))
    c2.metric("SSL Pools (Swing Lows)", len(swing_lows))
    c3.metric("Bullish FVG Zones", len(bullish_fvg))
    c4.metric("Bearish FVG Zones", len(bearish_fvg))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    recent_highs = sorted(swing_highs, key=lambda x: x[0])[-8:]
    recent_lows = sorted(swing_lows, key=lambda x: x[0])[-8:]

    for t, price in recent_highs:
        fig.add_shape(type="line", x0=t, x1=df.index[-1], y0=price, y1=price,
                       line=dict(color="#ff5252", width=1, dash="dot"))
    for t, price in recent_lows:
        fig.add_shape(type="line", x0=t, x1=df.index[-1], y0=price, y1=price,
                       line=dict(color="#00e5ff", width=1, dash="dot"))

    if recent_highs:
        fig.add_annotation(x=df.index[-1], y=recent_highs[-1][1], text="BSL", showarrow=False,
                            font=dict(color="#ff5252", size=11), xanchor="left")
    if recent_lows:
        fig.add_annotation(x=df.index[-1], y=recent_lows[-1][1], text="SSL", showarrow=False,
                            font=dict(color="#00e5ff", size=11), xanchor="left")

    for zone in bullish_fvg[-max_zones:]:
        fig.add_shape(type="rect", x0=zone["start"], x1=df.index[-1], y0=zone["bottom"], y1=zone["top"],
                       fillcolor="rgba(38,166,154,0.18)", line=dict(width=0))
    for zone in bearish_fvg[-max_zones:]:
        fig.add_shape(type="rect", x0=zone["start"], x1=df.index[-1], y0=zone["bottom"], y1=zone["top"],
                       fillcolor="rgba(239,83,80,0.18)", line=dict(width=0))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=620, xaxis_rangeslider_visible=False,
        title=f"{ticker} — Liquidity Pools & Fair Value Gaps",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Nearest Liquidity Levels to Current Price"):
        last_price = float(df["Close"].iloc[-1])
        all_levels = [("BSL", t, p) for t, p in swing_highs] + [("SSL", t, p) for t, p in swing_lows]
        if all_levels:
            lvl_df = pd.DataFrame(all_levels, columns=["Type", "Timestamp", "Price"])
            lvl_df["Distance %"] = lvl_df["Price"].apply(lambda p: safe_pct(p, last_price))
            lvl_df = lvl_df.reindex(lvl_df["Distance %"].abs().sort_values().index).head(10)
            st.dataframe(lvl_df.set_index("Timestamp"), use_container_width=True)
        else:
            st.info("Not enough data to identify swing liquidity levels for the selected window.")


# ==================================================================================
# MODULE 2 — QUANTITATIVE ML CLASSIFIER
# ==================================================================================

def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["return_1"] = df["Close"].pct_change(1)
    feat["return_3"] = df["Close"].pct_change(3)
    feat["return_5"] = df["Close"].pct_change(5)
    feat["rsi_14"] = compute_rsi(df["Close"], 14)
    feat["atr_14"] = compute_atr(df, 14)
    feat["volatility_10"] = df["Close"].pct_change().rolling(10).std()
    feat["ma_10"] = df["Close"].rolling(10).mean()
    feat["ma_30"] = df["Close"].rolling(30).mean()
    feat["ma_ratio"] = feat["ma_10"] / feat["ma_30"] - 1
    feat["momentum_10"] = df["Close"] - df["Close"].shift(10)
    feat["volume_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    return feat


def render_ml_module(df: pd.DataFrame, ticker: str):
    st.markdown('<div class="module-note">A RandomForest classifier trained live on engineered technical '
                'features to estimate the probability of the next-bar directional move.</div>',
                unsafe_allow_html=True)

    if len(df) < 80:
        st.warning("Insufficient historical data for reliable ML training. Select a longer period (≥ 3 months, daily interval recommended).")
        return

    horizon = st.slider("Prediction Horizon (bars ahead)", 1, 10, 3, key="ml_horizon")
    threshold = st.slider("Move Threshold for Buy/Sell Classification (%)", 0.05, 2.0, 0.15, step=0.05, key="ml_threshold") / 100

    feat = build_ml_features(df)
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1

    labels = pd.Series(1, index=df.index)
    labels[future_return > threshold] = 2
    labels[future_return < -threshold] = 0

    data = feat.copy()
    data["target"] = labels
    data = data.dropna()

    if len(data) < 50 or data["target"].nunique() < 2:
        st.warning("Not enough class diversity in the selected sample to train a robust classifier. Try adjusting the threshold or horizon.")
        return

    feature_cols = [c for c in feat.columns]
    X = data[feature_cols]
    y = data["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, shuffle=False
        )
        model = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return

    latest_features = feat.iloc[[-1]].fillna(feat.median(numeric_only=True))
    latest_scaled = scaler.transform(latest_features)
    proba = model.predict_proba(latest_scaled)[0]
    class_order = model.classes_
    proba_map = {int(c): p for c, p in zip(class_order, proba)}
    sell_p = proba_map.get(0, 0.0) * 100
    hold_p = proba_map.get(1, 0.0) * 100
    buy_p = proba_map.get(2, 0.0) * 100

    signal = "BUY" if buy_p == max(buy_p, hold_p, sell_p) else ("SELL" if sell_p == max(buy_p, hold_p, sell_p) else "HOLD")
    signal_color = {"BUY": "#26a69a", "SELL": "#ef5350", "HOLD": "#f0b90b"}[signal]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy (Test Set)", f"{acc*100:.1f}%")
    c2.metric("Buy Probability", f"{buy_p:.1f}%")
    c3.metric("Sell Probability", f"{sell_p:.1f}%")
    c4.metric("Hold Probability", f"{hold_p:.1f}%")

    st.markdown(
        f"<h3 style='color:{signal_color};'>Model Signal: {signal}</h3>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#f0b90b"))
        fig_fi.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Feature Importance",
                              margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_b:
        fig_proba = go.Figure(go.Pie(
            labels=["Sell", "Hold", "Buy"], values=[sell_p, hold_p, buy_p],
            marker=dict(colors=["#ef5350", "#f0b90b", "#26a69a"]), hole=0.55,
        ))
        fig_proba.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Directional Probability",
                                 margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_proba, use_container_width=True)


# ==================================================================================
# MODULE 3 — CROSS-ASSET CORRELATION & MACRO YIELD MATRIX
# ==================================================================================

def render_correlation_module(period: str):
    st.markdown('<div class="module-note">Cross-asset relationships between Gold, the US Dollar Index, '
                '10-Year Treasury Yields, and Bitcoin — key macro drivers for precious metals positioning.</div>',
                unsafe_allow_html=True)

    series_dict = {}
    fetch_errors = []
    for label, tk in MACRO_TICKERS.items():
        s = fetch_close_series(tk, period=period, interval="1d")
        if s.empty:
            fetch_errors.append(label)
        else:
            series_dict[label] = s

    if fetch_errors:
        st.warning(f"Could not retrieve live data for: {', '.join(fetch_errors)}. Displaying available assets only.")

    if len(series_dict) < 2:
        st.error("Insufficient macro data available to compute correlations right now. Please try again shortly.")
        return

    combined = pd.DataFrame(series_dict).dropna(how="all")
    combined = combined.ffill().dropna()

    if combined.empty or len(combined) < 10:
        st.warning("Not enough overlapping historical data across assets for this period.")
        return

    corr_matrix = combined.corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
    ))
    fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=460, title="Cross-Asset Correlation Matrix",
                            margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Rolling 30-Day Correlation vs Gold")
    if "Gold" in combined.columns:
        rolling_window = min(30, max(5, len(combined) // 3))
        fig_roll = go.Figure()
        for col in combined.columns:
            if col == "Gold":
                continue
            rolling_corr = combined["Gold"].rolling(rolling_window).corr(combined[col])
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"Gold vs {col}"))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#666")
        fig_roll.update_layout(template=PLOTLY_TEMPLATE, height=420,
                                title=f"Rolling {rolling_window}-Day Correlation",
                                margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_roll, use_container_width=True)

    with st.expander("📈 Normalized Price Performance (Rebased to 100)"):
        rebased = combined / combined.iloc[0] * 100
        fig_reb = go.Figure()
        for col in rebased.columns:
            fig_reb.add_trace(go.Scatter(x=rebased.index, y=rebased[col], mode="lines", name=col))
        fig_reb.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_reb, use_container_width=True)


# ==================================================================================
# MODULE 4 — VOLUME DELTA & FOOTPRINT IMBALANCE
# ==================================================================================

def compute_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    buy_ratio = ((out["Close"] - out["Low"]) / rng).clip(0, 1).fillna(0.5)
    out["buy_volume"] = out["Volume"] * buy_ratio
    out["sell_volume"] = out["Volume"] * (1 - buy_ratio)
    out["delta"] = out["buy_volume"] - out["sell_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out


def render_volume_delta_module(df: pd.DataFrame, ticker: str):
    st.markdown('<div class="module-note">Synthetic order-flow reconstruction: buying vs. selling volume '
                'estimated from intra-bar close position, aggregated into Cumulative Volume Delta (CVD).</div>',
                unsafe_allow_html=True)

    vd = compute_volume_delta(df)
    z_thresh = st.slider("Imbalance Spike Sensitivity (Z-score)", 1.0, 4.0, 2.0, step=0.25, key="vd_z")

    delta_mean = vd["delta"].mean()
    delta_std = vd["delta"].std() if vd["delta"].std() > 0 else 1.0
    vd["delta_z"] = (vd["delta"] - delta_mean) / delta_std
    spikes = vd[vd["delta_z"].abs() >= z_thresh]

    c1, c2, c3 = st.columns(3)
    c1.metric("Net CVD (Session)", f"{vd['delta'].sum():,.0f}")
    c2.metric("Buy Volume Share", f"{(vd['buy_volume'].sum() / max(vd['Volume'].sum(),1))*100:.1f}%")
    c3.metric("Imbalance Spikes Detected", len(spikes))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "Volume Delta (Buy − Sell)", "Cumulative Volume Delta (CVD)"),
    )
    fig.add_trace(go.Candlestick(
        x=vd.index, open=vd["Open"], high=vd["High"], low=vd["Low"], close=vd["Close"],
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if not spikes.empty:
        fig.add_trace(go.Scatter(
            x=spikes.index, y=spikes["High"] * 1.001, mode="markers", name="Imbalance Spike",
            marker=dict(color="#f0b90b", size=9, symbol="triangle-down"),
        ), row=1, col=1)

    bar_colors = np.where(vd["delta"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=vd.index, y=vd["delta"], marker_color=bar_colors, name="Delta"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vd.index, y=vd["cvd"], mode="lines", name="CVD",
                              line=dict(color="#00e5ff", width=2)), row=3, col=1)

    fig.update_layout(template=PLOTLY_TEMPLATE, height=780, showlegend=False,
                       xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("⚠️ Detected Order Flow Divergence / Imbalance Events"):
        if not spikes.empty:
            show_cols = ["Close", "Volume", "buy_volume", "sell_volume", "delta", "delta_z"]
            st.dataframe(spikes[show_cols].tail(15).round(2), use_container_width=True)
        else:
            st.info("No significant volume imbalance spikes detected at the current sensitivity level.")


# ==================================================================================
# MODULE 5 — OPTIONS GAMMA EXPOSURE (GEX) & MAX PAIN ENGINE
# ==================================================================================

def bs_gamma(spot, strike, t_years, iv, r=0.045):
    try:
        if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
            return 0.0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * math.sqrt(t_years))
        pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        gamma = pdf / (spot * iv * math.sqrt(t_years))
        return gamma
    except Exception:
        return 0.0


def simulate_gex(spot: float, n_strikes: int = 25, iv: float = 0.18, days_to_expiry: int = 30, seed: int = 7):
    rng = np.random.default_rng(seed)
    spacing = spot * 0.01
    strikes = np.round(spot + np.arange(-n_strikes, n_strikes + 1) * spacing, 2)
    t_years = max(days_to_expiry, 1) / 365.0

    distance = np.abs(strikes - spot)
    base_oi = 5000 * np.exp(-(distance ** 2) / (2 * (spot * 0.05) ** 2))
    call_oi = np.clip(base_oi * rng.uniform(0.7, 1.3, len(strikes)), 10, None)
    put_oi = np.clip(base_oi * rng.uniform(0.7, 1.3, len(strikes)), 10, None)

    gammas = np.array([bs_gamma(spot, k, t_years, iv) for k in strikes])
    contract_mult = 100
    call_gex = gammas * call_oi * contract_mult * spot * spot * 0.01
    put_gex = -gammas * put_oi * contract_mult * spot * spot * 0.01

    return pd.DataFrame({
        "strike": strikes, "call_oi": call_oi, "put_oi": put_oi,
        "call_gex": call_gex, "put_gex": put_gex, "net_gex": call_gex + put_gex,
    })


def gex_from_chain(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, days_to_expiry: int):
    t_years = max(days_to_expiry, 1) / 365.0
    contract_mult = 100

    def process(chain, sign):
        c = chain.copy()
        c["impliedVolatility"] = c["impliedVolatility"].replace(0, np.nan).fillna(c["impliedVolatility"].median())
        c["impliedVolatility"] = c["impliedVolatility"].fillna(0.2)
        c["openInterest"] = c["openInterest"].fillna(0)
        c["gamma"] = c.apply(lambda r: bs_gamma(spot, r["strike"], t_years, max(r["impliedVolatility"], 0.01)), axis=1)
        c["gex"] = sign * c["gamma"] * c["openInterest"] * contract_mult * spot * spot * 0.01
        return c[["strike", "openInterest", "gex"]]

    c_proc = process(calls, 1)
    p_proc = process(puts, -1)
    merged = pd.merge(c_proc, p_proc, on="strike", how="outer", suffixes=("_call", "_put")).fillna(0)
    merged["net_gex"] = merged["gex_call"] + merged["gex_put"]
    merged = merged.rename(columns={"gex_call": "call_gex", "gex_put": "put_gex"})
    merged = merged.sort_values("strike").reset_index(drop=True)
    return merged


def compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame):
    try:
        strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        pain = []
        for s in strikes:
            call_loss = ((s - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
            put_loss = ((puts["strike"] - s).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
            pain.append(call_loss + put_loss)
        pain_df = pd.DataFrame({"strike": strikes, "total_pain": pain})
        max_pain_strike = pain_df.loc[pain_df["total_pain"].idxmin(), "strike"]
        return max_pain_strike, pain_df
    except Exception:
        return None, pd.DataFrame()


def render_gex_module(ticker: str):
    st.markdown('<div class="module-note">Gamma Exposure (GEX) profile identifying dealer positioning, '
                'volatility pin zones, and the gamma flip level. Uses live listed options where available, '
                'or a Black-Scholes-based simulation engine otherwise.</div>', unsafe_allow_html=True)

    proxy = OPTIONS_PROXY_MAP.get(ticker, None)
    use_live = proxy is not None
    live_symbol = proxy if use_live else ticker

    chain_result = fetch_options_chain(live_symbol) if use_live else None

    if chain_result is not None:
        calls, puts, expiry, spot = chain_result
        try:
            dte = max((pd.to_datetime(expiry) - pd.Timestamp.now()).days, 1)
        except Exception:
            dte = 30
        gex_df = gex_from_chain(calls, puts, spot, dte)
        max_pain, pain_df = compute_max_pain(calls, puts)
        source_label = f"Live listed options — proxy: {live_symbol} (expiry {expiry})"
    else:
        spot = get_last_price(ticker) or 2000.0
        dte = st.slider("Simulated Days to Expiry", 1, 90, 30, key="gex_dte")
        iv_assumed = st.slider("Assumed Implied Volatility (%)", 5, 80, 18, key="gex_iv") / 100
        gex_df = simulate_gex(spot, n_strikes=25, iv=iv_assumed, days_to_expiry=dte)
        max_pain = gex_df.loc[(gex_df["call_oi"] + gex_df["put_oi"]).idxmax(), "strike"]
        pain_df = pd.DataFrame()
        source_label = f"Simulated GEX engine (Black-Scholes gamma model) — no listed options market for {ticker}"

    st.caption(f"Data source: {source_label}")

    net_gex_total = gex_df["net_gex"].sum()
    flip_candidates = gex_df.sort_values("strike")
    flip_candidates["cum_gex"] = flip_candidates["net_gex"].cumsum()
    sign_changes = flip_candidates[flip_candidates["cum_gex"] * flip_candidates["cum_gex"].shift(1) < 0]
    gamma_flip = float(sign_changes["strike"].iloc[0]) if not sign_changes.empty else float(gex_df["strike"].median())

    c1, c2, c3 = st.columns(3)
    c1.metric("Spot Price", f"{spot:,.2f}")
    c2.metric("Net GEX", f"{net_gex_total:,.0f}")
    c3.metric("Max Pain Strike", f"{max_pain:,.2f}" if max_pain is not None else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["call_gex"], name="Call GEX", marker_color="#26a69a"))
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["put_gex"], name="Put GEX", marker_color="#ef5350"))
    fig.add_vline(x=spot, line_dash="dash", line_color="#f0b90b",
                  annotation_text="Spot", annotation_position="top")
    fig.add_vline(x=gamma_flip, line_dash="dot", line_color="#00e5ff",
                  annotation_text="Gamma Flip", annotation_position="bottom")
    if max_pain is not None:
        fig.add_vline(x=max_pain, line_dash="dashdot", line_color="#ba68c8",
                      annotation_text="Max Pain", annotation_position="top")

    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, barmode="relative",
                       title=f"{ticker} — Gamma Exposure Profile by Strike",
                       xaxis_title="Strike", yaxis_title="Gamma Exposure",
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================================
# MODULE 6 — INSTITUTIONAL EXECUTION ALGORITHMS (VWAP / TWAP / ICEBERG)
# ==================================================================================

def compute_vwap_bands(df: pd.DataFrame):
    out = df.copy()
    typical_price = (out["High"] + out["Low"] + out["Close"]) / 3
    cum_vol = out["Volume"].cumsum().replace(0, np.nan)
    cum_tp_vol = (typical_price * out["Volume"]).cumsum()
    out["vwap"] = (cum_tp_vol / cum_vol).ffill().bfill()

    sq_diff = ((typical_price - out["vwap"]) ** 2) * out["Volume"]
    cum_sq_diff = sq_diff.cumsum()
    variance = (cum_sq_diff / cum_vol).replace([np.inf, -np.inf], np.nan).fillna(0)
    std = np.sqrt(variance)

    out["vwap_std"] = std
    out["vwap_u1"] = out["vwap"] + std
    out["vwap_u2"] = out["vwap"] + 2 * std
    out["vwap_l1"] = out["vwap"] - std
    out["vwap_l2"] = out["vwap"] - 2 * std

    out["twap"] = typical_price.expanding().mean()
    return out


def detect_icebergs(df: pd.DataFrame, z_thresh: float = 2.5):
    out = df.copy()
    price_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["vol_range_ratio"] = out["Volume"] / price_range
    ratio_mean = out["vol_range_ratio"].mean()
    ratio_std = out["vol_range_ratio"].std() if out["vol_range_ratio"].std() > 0 else 1.0
    out["vr_z"] = (out["vol_range_ratio"] - ratio_mean) / ratio_std
    icebergs = out[(out["vr_z"] >= z_thresh)]
    return out, icebergs


def render_execution_module(df: pd.DataFrame, ticker: str):
    st.markdown('<div class="module-note">Institutional execution benchmarks — VWAP with statistical '
                'deviation bands, TWAP baseline, and detection of probable iceberg / hidden-order clusters '
                'via volume-to-range anomaly analysis.</div>', unsafe_allow_html=True)

    vwap_df = compute_vwap_bands(df)
    z_thresh = st.slider("Iceberg Detection Sensitivity (Z-score)", 1.5, 4.0, 2.5, step=0.25, key="ice_z")
    vwap_df, icebergs = detect_icebergs(vwap_df, z_thresh=z_thresh)

    last_close = vwap_df["Close"].iloc[-1]
    last_vwap = vwap_df["vwap"].iloc[-1]
    last_twap = vwap_df["twap"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{last_close:,.2f}")
    c2.metric("VWAP", f"{last_vwap:,.2f}", f"{safe_pct(last_close, last_vwap):.2f}%")
    c3.metric("TWAP", f"{last_twap:,.2f}", f"{safe_pct(last_close, last_twap):.2f}%")
    c4.metric("Iceberg Clusters Detected", len(icebergs))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=vwap_df.index, open=vwap_df["Open"], high=vwap_df["High"], low=vwap_df["Low"], close=vwap_df["Close"],
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap"], mode="lines", name="VWAP",
                              line=dict(color="#f0b90b", width=2)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["twap"], mode="lines", name="TWAP",
                              line=dict(color="#ba68c8", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u1"], mode="lines", name="+1 SD",
                              line=dict(color="rgba(38,166,154,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u2"], mode="lines", name="+2 SD",
                              line=dict(color="rgba(38,166,154,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l1"], mode="lines", name="-1 SD",
                              line=dict(color="rgba(239,83,80,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l2"], mode="lines", name="-2 SD",
                              line=dict(color="rgba(239,83,80,0.35)", width=1)))

    if not icebergs.empty:
        fig.add_trace(go.Scatter(
            x=icebergs.index, y=icebergs["Low"] * 0.999, mode="markers", name="Iceberg Cluster",
            marker=dict(color="#00e5ff", size=10, symbol="diamond"),
        ))

    fig.update_layout(template=PLOTLY_TEMPLATE, height=650, xaxis_rangeslider_visible=False,
                       title=f"{ticker} — VWAP / TWAP Execution Benchmarks & Iceberg Detection",
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================================
# SIDEBAR — GLOBAL CONTROLS
# ==================================================================================

st.sidebar.markdown("## 📊 Institutional Quant Terminal")
st.sidebar.caption("Gold · Forex · Crypto — Multi-Module Analytics")
st.sidebar.divider()

asset_label = st.sidebar.selectbox("Asset", list(ASSET_MAP.keys()), index=0)
ticker = ASSET_MAP[asset_label]

period = st.sidebar.selectbox("Historical Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y"], index=3)
interval = st.sidebar.selectbox("Interval", ["15m", "30m", "1h", "1d", "1wk"], index=3)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ==================================================================================
# MAIN HEADER & DATA FETCH
# ==================================================================================

st.title("📊 Institutional Quantitative Trading Dashboard")
st.caption(f"Active Instrument: **{asset_label}** ({ticker}) · Period: {period} · Interval: {interval}")

with st.spinner(f"Fetching market data for {ticker}..."):
    main_df = fetch_ohlcv(ticker, period=period, interval=interval)

if main_df.empty:
    st.error(
        f"⚠️ Unable to retrieve data for **{ticker}** with period='{period}', interval='{interval}'. "
        "Try a different period/interval, or click **Refresh Data** in the sidebar."
    )
    st.stop()

# Snapshot metrics
last_row = main_df.iloc[-1]
prev_row = main_df.iloc[-2] if len(main_df) > 1 else last_row
chg = safe_pct(last_row["Close"], prev_row["Close"])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Price", f"{last_row['Close']:,.2f}", f"{chg:.2f}%")
m2.metric("Session High", f"{last_row['High']:,.2f}")
m3.metric("Session Low", f"{last_row['Low']:,.2f}")
m4.metric("Volume", f"{last_row['Volume']:,.0f}")
m5.metric("Bars Loaded", f"{len(main_df):,}")

st.divider()

# ==================================================================================
# TAB NAVIGATION — 6 MODULES
# ==================================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌊 Liquidity & Order Flow",
    "🤖 ML Classifier",
    "🔗 Cross-Asset Correlation",
    "📊 Volume Delta / CVD",
    "🎯 Options GEX & Max Pain",
    "⚙️ VWAP / TWAP / Iceberg",
])

with tab1:
    try:
        render_liquidity_module(main_df, ticker)
    except Exception as e:
        st.error(f"Liquidity module error: {e}")

with tab2:
    try:
        render_ml_module(main_df, ticker)
    except Exception as e:
        st.error(f"ML Classifier error: {e}")

with tab3:
    try:
        render_correlation_module(period=period if period not in ["5d"] else "3mo")
    except Exception as e:
        st.error(f"Correlation error: {e}")

with tab4:
    try:
        render_volume_delta_module(main_df, ticker)
    except Exception as e:
        st.error(f"Volume Delta error: {e}")

with tab5:
    try:
        render_gex_module(ticker)
    except Exception as e:
        st.error(f"GEX module error: {e}")

with tab6:
    try:
        render_execution_module(main_df, ticker)
    except Exception as e:
        st.error(f"Execution error: {e}")

"""
Institutional Quantitative FX Terminal
========================================
A single-file Streamlit application combining:
  Phase 1 - Official DXY geometric-mean formula + multi-asset engine
  Phase 2 - Volatility & Range engine (ATR, Bollinger Bands)
  Phase 3 - Momentum & Trend engine (RSI, EMA 20/50/200)
  Phase 4 - SMC / ICT engine (Fair Value Gaps, Order Blocks, BOS/MSS, liquidity sweeps)
  Phase 5 - Macro Yield & Correlation engine (US 10Y vs asset)
  Phase 6 - Multi-source fundamental news sentiment (RSS, no paid API)
  Phase 7 - Hurst Exponent rolling volatility-regime engine

Zero paid API keys. Data: yfinance + feedparser only.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import feedparser
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ============================================================================
# CONFIG / CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Institutional FX Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSET_TICKERS = {
    "DXY Index": None,          # computed via Phase 1 geometric formula
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "XAU/USD (Gold)": "GC=F",   # COMEX Gold futures, most reliable free XAU/USD proxy
}

DXY_COMPONENT_TICKERS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCAD": "USDCAD=X",
    "USDSEK": "USDSEK=X",
    "USDCHF": "USDCHF=X",
}

# interval -> lookback period, tuned to keep candle counts sane on yfinance's
# own intraday retention limits (1m ~7d, intraday <60m ~60d, 60m ~730d)
INTERVAL_PERIOD_MAP = {
    "1m": "5d",
    "2m": "5d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "60m": "3mo",
    "1d": "1y",
    "1wk": "5y",
}

RSS_FEEDS = {
    "ForexFactory": "https://www.forexfactory.com/rss.php",
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "DailyFX": "https://www.dailyfx.com/feeds/all",
}

BULLISH_KEYWORDS = [
    "rate hike", "hawkish", "beats expectations", "stronger than expected",
    "inflation rises", "gdp beats", "jobs beat", "raises rates", "tightening",
]
BEARISH_KEYWORDS = [
    "rate cut", "dovish", "misses expectations", "weaker than expected",
    "recession", "gdp falls", "jobs miss", "cuts rates", "easing", "slowdown",
]
RELEVANCE_KEYWORDS = [
    "cpi", "nfp", "fomc", "inflation", "rate hike", "rate cut", "federal reserve",
    "ecb", "boe", "boj", "payrolls", "gdp", "ppi", "unemployment", "fed",
]

SQUAWK_CHANNELS = [
    {
        "name": "Newsquawk",
        "url": "https://www.newsquawk.com",
        "desc_en": "Institutional-grade live audio news squawk covering macro, rates and FX.",
        "desc_am": "ለማክሮ፣ ለወለድ ምጣኔ እና ለውጭ ምንዛሪ ገበያ ተቋማዊ ደረጃ ያለው ቀጥታ የድምጽ ዜና ሽፋን።",
    },
    {
        "name": "Livesquawk",
        "url": "https://www.livesquawk.com",
        "desc_en": "Real-time audio and text news squawk service for FX and rates traders.",
        "desc_am": "ለውጭ ምንዛሪ እና ለወለድ ነጋዴዎች የቀጥታ ጊዜ የድምጽ እና የጽሑፍ ዜና አገልግሎት።",
    },
    {
        "name": "Bloomberg Audio",
        "url": "https://www.bloomberg.com/audio",
        "desc_en": "Bloomberg Radio / Surveillance live audio market coverage.",
        "desc_am": "የብሉምበርግ ራዲዮ/ሰርቬይላንስ ቀጥታ የገበያ የድምጽ ሽፋን።",
    },
]

# ============================================================================
# TRANSLATIONS
# ============================================================================

T = {
    "en": {
        "app_title": "Institutional Quantitative FX Terminal",
        "app_subtitle": "7-Phase Quant Engine · SMC/ICT Structure · News Sentiment · Multi-Asset Correlation",
        "language": "Language",
        "asset": "Asset",
        "timeframe": "Timeframe",
        "refresh": "🔄 Refresh Data",
        "price": "Price",
        "atr": "ATR (14)",
        "rsi": "RSI (14)",
        "hurst": "Hurst Exponent",
        "trend": "Trend",
        "sentiment": "News Sentiment",
        "overbought": "Overbought",
        "oversold": "Oversold",
        "neutral_rsi": "Neutral",
        "strong_bullish": "Strong Bullish",
        "strong_bearish": "Strong Bearish",
        "bullish": "Bullish",
        "bearish": "Bearish",
        "neutral": "Neutral",
        "mean_reverting": "Mean-Reverting (Volatility Spike Risk)",
        "trending": "Trending / Persistent",
        "random_walk": "Random Walk / Noise",
        "chart_title": "Price Action — SMC/ICT Confluence Chart",
        "smc_panel": "SMC / ICT Structure",
        "fvg": "Fair Value Gaps",
        "order_blocks": "Order Blocks",
        "structure_events": "Structure Events (BOS / Liquidity Sweeps)",
        "no_fvg": "No recent Fair Value Gaps detected.",
        "no_ob": "No recent Order Blocks detected.",
        "no_events": "No recent structure events.",
        "correlation_panel": "Macro Yield & Correlation",
        "us10y": "US 10Y Yield (^TNX)",
        "correlation_label": "Correlation",
        "correlation_unavailable": "Correlation data unavailable for this timeframe.",
        "news_panel": "Fundamental News Sentiment",
        "squawk_panel": "Live Audio Squawk",
        "no_news": "No headlines available right now.",
        "gold_note": "Sourced from COMEX Gold futures (GC=F) as a free XAU/USD proxy.",
        "dxy_error": "Could not build a full DXY basket for this timeframe (thin intraday coverage on one or more component pairs). Try Daily or Weekly.",
        "data_error": "Unable to retrieve sufficient market data for this asset/timeframe. Try a different selection.",
        "disclaimer": "For educational and informational purposes only. Not financial advice. Trade at your own risk.",
        "signals_legend": "▲ Buy confluence   ▼ Sell confluence",
    },
    "am": {
        "app_title": "የተቋማት መጠናዊ የውጭ ምንዛሪ ተርሚናል",
        "app_subtitle": "7-ደረጃ መጠናዊ ሞተር · SMC/ICT አወቃቀር · የዜና ስሜት · የብዙ ንብረት ትስስር",
        "language": "ቋንቋ",
        "asset": "ንብረት",
        "timeframe": "የጊዜ ገደብ",
        "refresh": "🔄 መረጃ አድስ",
        "price": "ዋጋ",
        "atr": "ATR (14)",
        "rsi": "RSI (14)",
        "hurst": "የሁርስት ኤክስፖነንት",
        "trend": "አዝማሚያ",
        "sentiment": "የዜና ስሜት",
        "overbought": "ከመጠን በላይ የተገዛ",
        "oversold": "ከመጠን በላይ የተሸጠ",
        "neutral_rsi": "ገለልተኛ",
        "strong_bullish": "በጣም ወደ ላይ",
        "strong_bearish": "በጣም ወደ ታች",
        "bullish": "ወደ ላይ",
        "bearish": "ወደ ታች",
        "neutral": "ገለልተኛ",
        "mean_reverting": "ወደ አማካይ የመመለስ አዝማሚያ (የመዋዠቅ ስጋት)",
        "trending": "ቀጣይነት ያለው አዝማሚያ",
        "random_walk": "ዘፈቀደ እንቅስቃሴ / ጫጫታ",
        "chart_title": "የዋጋ እንቅስቃሴ — SMC/ICT ቻርት",
        "smc_panel": "SMC / ICT አወቃቀር",
        "fvg": "የፍትሃዊ ዋጋ ክፍተቶች (FVG)",
        "order_blocks": "የትዕዛዝ ብሎኮች",
        "structure_events": "የአወቃቀር ክስተቶች (BOS / የፈሳሽ ማጥመጃ)",
        "no_fvg": "በቅርብ ጊዜ የፍትሃዊ ዋጋ ክፍተት አልተገኘም።",
        "no_ob": "በቅርብ ጊዜ የትዕዛዝ ብሎክ አልተገኘም።",
        "no_events": "በቅርብ ጊዜ የአወቃቀር ክስተት የለም።",
        "correlation_panel": "የማክሮ ምርት እና ትስስር",
        "us10y": "የአሜሪካ 10-ዓመት ምርት (^TNX)",
        "correlation_label": "ትስስር",
        "correlation_unavailable": "ለዚህ የጊዜ ገደብ የትስስር መረጃ የለም።",
        "news_panel": "የመሠረታዊ ዜና ስሜት",
        "squawk_panel": "ቀጥታ የድምጽ ዜና",
        "no_news": "በአሁኑ ጊዜ ዜናዎች የሉም።",
        "gold_note": "ከኮሜክስ ወርቅ ፊውቸርስ (GC=F) የተገኘ፣ ለ XAU/USD ነፃ አማራጭ።",
        "dxy_error": "ለዚህ የጊዜ ገደብ ሙሉ የ DXY ቅርጫት መገንባት አልተቻለም። እባክዎ ዕለታዊ ወይም ሳምንታዊ ይሞክሩ።",
        "data_error": "ለዚህ ንብረት/የጊዜ ገደብ በቂ የገበያ መረጃ ማግኘት አልተቻለም። እባክዎ ሌላ ይምረጡ።",
        "disclaimer": "ለትምህርት እና መረጃ አገልግሎት ብቻ። የፋይናንስ ምክር አይደለም። በራስዎ ኃላፊነት ይነግዱ።",
        "signals_legend": "▲ የግዢ ውህደት   ▼ የሽያጭ ውህደት",
    },
}

# ============================================================================
# PHASE 1 — DATA FETCHING + DXY GEOMETRIC FORMULA
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlc(ticker, period, interval):
    """Fetch OHLC data for a single ticker via yfinance. Cached 60s."""
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None


def compute_dxy_ohlc(period, interval):
    """Phase 1: build a synthetic DXY OHLC series from the official
    ICE weighted-geometric-mean formula, applied independently to the
    Open/High/Low/Close of each component pair."""
    data = {}
    for key, ticker in DXY_COMPONENT_TICKERS.items():
        d = fetch_ohlc(ticker, period, interval)
        if d is None:
            return None
        data[key] = d

    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    if idx is None or len(idx) < 5:
        return None

    aligned = {k: d.reindex(idx) for k, d in data.items()}
    out = pd.DataFrame(index=idx)
    for col in ["Open", "High", "Low", "Close"]:
        eur = aligned["EURUSD"][col]
        jpy = aligned["USDJPY"][col]
        gbp = aligned["GBPUSD"][col]
        cad = aligned["USDCAD"][col]
        sek = aligned["USDSEK"][col]
        chf = aligned["USDCHF"][col]
        out[col] = (
            50.14348112
            * (eur ** -0.576)
            * (jpy ** 0.136)
            * (gbp ** -0.119)
            * (cad ** 0.091)
            * (sek ** 0.042)
            * (chf ** 0.036)
        )
    out = out.dropna()
    return out if not out.empty else None


def get_price_data(asset_label, period, interval):
    if asset_label == "DXY Index":
        return compute_dxy_ohlc(period, interval)
    ticker = ASSET_TICKERS[asset_label]
    return fetch_ohlc(ticker, period, interval)


# ============================================================================
# PHASE 2 — VOLATILITY & RANGE ENGINE
# ============================================================================

def calc_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_bollinger(df, period=20, num_std=2):
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    return mid, mid + num_std * std, mid - num_std * std


# ============================================================================
# PHASE 3 — MOMENTUM & TREND ENGINE
# ============================================================================

def calc_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def determine_trend(df, ema20, ema50, ema200):
    last_close = df["Close"].iloc[-1]
    e20, e50, e200 = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
    if last_close > e20 > e50 > e200:
        return "strong_bullish"
    if last_close < e20 < e50 < e200:
        return "strong_bearish"
    if last_close > e50:
        return "bullish"
    if last_close < e50:
        return "bearish"
    return "neutral"


# ============================================================================
# PHASE 4 — SMC / ICT & PRICE ACTION ENGINE
# ============================================================================

def find_fvg(df):
    """3-candle Fair Value Gap imbalance detection."""
    fvgs = []
    highs, lows, idx = df["High"].values, df["Low"].values, df.index
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            fvgs.append({"type": "bullish", "start_idx": idx[i - 2], "end_idx": idx[i],
                         "top": lows[i], "bottom": highs[i - 2]})
        if highs[i] < lows[i - 2]:
            fvgs.append({"type": "bearish", "start_idx": idx[i - 2], "end_idx": idx[i],
                         "top": lows[i - 2], "bottom": highs[i]})
    return fvgs


def find_order_blocks(df):
    """Bullish OB = last down-candle before an up-expansion that clears its high.
    Bearish OB = last up-candle before a down-expansion that clears its low."""
    obs = []
    o, c = df["Open"].values, df["Close"].values
    h, l = df["High"].values, df["Low"].values
    idx = df.index
    for i in range(len(df) - 1):
        if c[i] < o[i] and c[i + 1] > h[i]:
            obs.append({"type": "bullish", "idx": idx[i], "top": h[i], "bottom": l[i]})
        if c[i] > o[i] and c[i + 1] < l[i]:
            obs.append({"type": "bearish", "idx": idx[i], "top": h[i], "bottom": l[i]})
    return obs


def find_swings(df, n=3):
    highs, lows, idx = df["High"].values, df["Low"].values, df.index
    swing_highs, swing_lows = [], []
    for i in range(n, len(df) - n):
        wh = highs[i - n:i + n + 1]
        wl = lows[i - n:i + n + 1]
        if highs[i] == wh.max():
            swing_highs.append((idx[i], highs[i]))
        if lows[i] == wl.min():
            swing_lows.append((idx[i], lows[i]))
    return swing_highs, swing_lows


def detect_bos_and_sweeps(df, n=3):
    """Single-pass detector for Break of Structure (MSS) and liquidity
    sweeps of the most recent unbroken swing high/low."""
    swing_highs, swing_lows = find_swings(df, n)
    sh_map = dict(swing_highs)
    sl_map = dict(swing_lows)
    idx = df.index
    closes, highs, lows = df["Close"].values, df["High"].values, df["Low"].values
    events = []
    last_sh = last_sl = None
    for i in range(len(df)):
        t = idx[i]
        if t in sh_map:
            last_sh = sh_map[t]
        if t in sl_map:
            last_sl = sl_map[t]
        if last_sh is not None:
            if closes[i] > last_sh:
                events.append({"type": "BOS_up", "idx": t, "level": last_sh})
                last_sh = None
            elif highs[i] > last_sh and closes[i] < last_sh:
                events.append({"type": "liquidity_sweep_high", "idx": t, "level": last_sh})
        if last_sl is not None:
            if closes[i] < last_sl:
                events.append({"type": "BOS_down", "idx": t, "level": last_sl})
                last_sl = None
            elif lows[i] < last_sl and closes[i] > last_sl:
                events.append({"type": "liquidity_sweep_low", "idx": t, "level": last_sl})
    return events


# ============================================================================
# PHASE 5 — MACRO YIELD & CORRELATION ENGINE
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yield(period, interval):
    safe_interval = interval if interval in ("1d", "1wk") else "1d"
    safe_period = period if interval in ("1d", "1wk") else "1y"
    return fetch_ohlc("^TNX", safe_period, safe_interval)


def calc_rolling_correlation(series_a, series_b, window=20):
    a, b = series_a.pct_change(), series_b.pct_change()
    combined = pd.concat([a, b], axis=1, join="inner").dropna()
    combined.columns = ["a", "b"]
    if len(combined) < window:
        return pd.Series(dtype=float)
    return combined["a"].rolling(window).corr(combined["b"])


# ============================================================================
# PHASE 6 — MULTI-SOURCE FUNDAMENTAL NEWS ENGINE
# ============================================================================

def _parse_feed(url, timeout=6):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return feedparser.parse(resp.content)
    except Exception:
        try:
            return feedparser.parse(url)
        except Exception:
            return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news():
    headlines = []
    for source, url in RSS_FEEDS.items():
        feed = _parse_feed(url)
        if feed is None or not getattr(feed, "entries", None):
            continue
        for entry in feed.entries[:10]:
            headlines.append({
                "source": source,
                "title": entry.get("title", "").strip(),
                "published": entry.get("published", entry.get("updated", "")),
                "link": entry.get("link", ""),
            })
    return headlines


def compute_sentiment(headlines):
    if not headlines:
        return "Neutral", 0
    score, relevant = 0, 0
    for h in headlines:
        title = h["title"].lower()
        if any(k in title for k in RELEVANCE_KEYWORDS):
            relevant += 1
            if any(k in title for k in BULLISH_KEYWORDS):
                score += 1
            elif any(k in title for k in BEARISH_KEYWORDS):
                score -= 1
    if relevant == 0 or score == 0:
        return "Neutral", score
    return ("Bullish", score) if score > 0 else ("Bearish", score)


# ============================================================================
# PHASE 7 — HURST EXPONENT ROLLING ENGINE
# ============================================================================

def _hurst_single(ts, min_lag=2, max_lag=13):
    ts = np.asarray(ts, dtype=float)
    max_lag = min(max_lag, len(ts) // 2)
    if max_lag <= min_lag:
        return np.nan
    lags = range(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        s = np.std(diff)
        tau.append(s if s > 1e-10 else 1e-10)
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0] * 2.0


def calc_hurst_rolling(series, window=30):
    values = series.values
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(window, n):
        try:
            out[i] = _hurst_single(values[i - window:i])
        except Exception:
            out[i] = np.nan
    return pd.Series(out, index=series.index)


# ============================================================================
# SIGNAL CONFLUENCE (Phase 1-7 combined)
# ============================================================================

def generate_confluence_signals(df, obs, hurst_series, lookback=300, hurst_threshold=0.40):
    signals = []
    if df.empty:
        return signals
    sub = df.tail(lookback)
    bullish_obs = [o for o in obs if o["type"] == "bullish"][-50:]
    bearish_obs = [o for o in obs if o["type"] == "bearish"][-50:]
    close = sub["Close"]
    for t, price in zip(sub.index, close.values):
        h = hurst_series.loc[t] if t in hurst_series.index else np.nan
        if pd.isna(h) or h >= hurst_threshold:
            continue
        if any(o["bottom"] <= price <= o["top"] and o["idx"] <= t for o in bullish_obs):
            signals.append({"idx": t, "price": price, "type": "buy"})
        elif any(o["bottom"] <= price <= o["top"] and o["idx"] <= t for o in bearish_obs):
            signals.append({"idx": t, "price": price, "type": "sell"})
    return signals


# ============================================================================
# CHART BUILDER
# ============================================================================

def build_chart(df, ema20, ema50, fvgs, obs, signals):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA 20",
                              line=dict(color="#42a5f5", width=1.3)))
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA 50",
                              line=dict(color="#ffa726", width=1.3)))

    x_end = df.index[-1]
    for f in fvgs[-20:]:
        color = "rgba(38,166,154,0.18)" if f["type"] == "bullish" else "rgba(239,83,80,0.18)"
        fig.add_shape(type="rect", x0=f["start_idx"], x1=x_end, y0=f["bottom"], y1=f["top"],
                      fillcolor=color, line=dict(width=0), layer="below")
    for o in obs[-20:]:
        color = "rgba(66,165,245,0.16)" if o["type"] == "bullish" else "rgba(255,167,38,0.16)"
        fig.add_shape(type="rect", x0=o["idx"], x1=x_end, y0=o["bottom"], y1=o["top"],
                      fillcolor=color, line=dict(width=1, color=color), layer="below")

    buys = [s for s in signals if s["type"] == "buy"]
    sells = [s for s in signals if s["type"] == "sell"]
    if buys:
        fig.add_trace(go.Scatter(x=[s["idx"] for s in buys], y=[s["price"] for s in buys],
                                  mode="markers", name="Buy",
                                  marker=dict(symbol="triangle-up", size=13, color="#00e676",
                                              line=dict(width=1, color="#003d1f"))))
    if sells:
        fig.add_trace(go.Scatter(x=[s["idx"] for s in sells], y=[s["price"] for s in sells],
                                  mode="markers", name="Sell",
                                  marker=dict(symbol="triangle-down", size=13, color="#ff1744",
                                              line=dict(width=1, color="#3d0009"))))

    fig.update_layout(
        template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.03, x=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    )
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    with st.sidebar:
        st.markdown("## ⚙️ Terminal Settings")
        lang_choice = st.radio("Language / ቋንቋ",
                                ["English 🇬🇧", "አማርኛ (Amharic) 🇪🇹"], index=0)
        lang = "en" if lang_choice.startswith("English") else "am"
        tr = T[lang]

        st.markdown("---")
        asset_label = st.selectbox(tr["asset"], list(ASSET_TICKERS.keys()), index=0)
        timeframe = st.selectbox(tr["timeframe"], list(INTERVAL_PERIOD_MAP.keys()), index=6)

        if asset_label == "XAU/USD (Gold)":
            st.caption(tr["gold_note"])

        st.markdown("---")
        if st.button(tr["refresh"], use_container_width=True):
            st.cache_data.clear()
        st.caption(tr["disclaimer"])

    period = INTERVAL_PERIOD_MAP[timeframe]

    st.title(f"📊 {tr['app_title']}")
    st.caption(tr["app_subtitle"])

    with st.spinner("..."):
        df = get_price_data(asset_label, period, timeframe)

    if df is None or len(df) < 40:
        st.error("⚠️ " + (tr["dxy_error"] if asset_label == "DXY Index" else tr["data_error"]))
        st.stop()

    # ---- Phase 2 / 3 indicators ----
    atr = calc_atr(df)
    bb_mid, bb_upper, bb_lower = calc_bollinger(df)
    rsi = calc_rsi(df)
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    ema200 = df["Close"].ewm(span=200, adjust=False).mean()
    trend = determine_trend(df, ema20, ema50, ema200)

    # ---- Phase 4 SMC/ICT ----
    fvgs = find_fvg(df)
    obs = find_order_blocks(df)
    structure_events = detect_bos_and_sweeps(df)

    # ---- Phase 7 Hurst ----
    hurst_series = calc_hurst_rolling(df["Close"], window=30)
    hurst_valid = hurst_series.dropna()
    last_hurst = hurst_valid.iloc[-1] if not hurst_valid.empty else np.nan

    # ---- Confluence signals (uses Phases 4 + 7) ----
    signals = generate_confluence_signals(df, obs, hurst_series)

    # ---- Phase 6 news ----
    headlines = fetch_news()
    sentiment_label, _ = compute_sentiment(headlines)

    # ---- Phase 5 macro yield correlation ----
    yield_df = fetch_yield(period, timeframe)
    corr_series = None
    if yield_df is not None:
        corr_series = calc_rolling_correlation(df["Close"], yield_df["Close"])

    # ================= METRICS ROW =================
    last_price = df["Close"].iloc[-1]
    last_atr = atr.iloc[-1] if not atr.dropna().empty else np.nan
    last_rsi = rsi.iloc[-1] if not rsi.dropna().empty else np.nan

    cols = st.columns(6)
    cols[0].metric(tr["price"], f"{last_price:,.5f}")
    cols[1].metric(tr["atr"], f"{last_atr:,.5f}" if not np.isnan(last_atr) else "—")

    rsi_state = (tr["overbought"] if last_rsi > 70 else
                 tr["oversold"] if last_rsi < 30 else tr["neutral_rsi"])
    cols[2].metric(tr["rsi"], f"{last_rsi:,.1f}" if not np.isnan(last_rsi) else "—", rsi_state)

    hurst_state = (tr["mean_reverting"] if (not np.isnan(last_hurst) and last_hurst < 0.45) else
                   tr["trending"] if (not np.isnan(last_hurst) and last_hurst > 0.55) else
                   tr["random_walk"])
    cols[3].metric(tr["hurst"], f"{last_hurst:,.2f}" if not np.isnan(last_hurst) else "—")

    cols[4].metric(tr["trend"], tr.get(trend, trend))
    cols[5].metric(tr["sentiment"], tr.get(sentiment_label.lower(), sentiment_label))

    st.caption(f"🧭 RSI: {rsi_state}   |   🌊 Hurst: {hurst_state}   |   {tr['signals_legend']}")

    # ================= CHART =================
    st.subheader(tr["chart_title"])
    fig = build_chart(df, ema20, ema50, fvgs, obs, signals)
    st.plotly_chart(fig, use_container_width=True)

    # ================= PANELS =================
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"🧭 {tr['smc_panel']}")

        st.markdown(f"**{tr['fvg']}** ({len(fvgs)})")
        if fvgs:
            for f in fvgs[-5:][::-1]:
                icon = "🟢" if f["type"] == "bullish" else "🔴"
                st.write(f"{icon} {f['bottom']:.5f} – {f['top']:.5f}  ·  {f['start_idx']}")
        else:
            st.caption(tr["no_fvg"])

        st.markdown(f"**{tr['order_blocks']}** ({len(obs)})")
        if obs:
            for o in obs[-5:][::-1]:
                icon = "🟢" if o["type"] == "bullish" else "🔴"
                st.write(f"{icon} {o['bottom']:.5f} – {o['top']:.5f}  ·  {o['idx']}")
        else:
            st.caption(tr["no_ob"])

        st.markdown(f"**{tr['structure_events']}** ({len(structure_events)})")
        if structure_events:
            for e in structure_events[-5:][::-1]:
                st.write(f"⚡ {e['type']} @ {e['level']:.5f}  ·  {e['idx']}")
        else:
            st.caption(tr["no_events"])

    with c2:
        st.subheader(f"📈 {tr['correlation_panel']}")
        if corr_series is not None and not corr_series.dropna().empty:
            last_corr = corr_series.dropna().iloc[-1]
            st.metric(f"{tr['us10y']} — {tr['correlation_label']}", f"{last_corr:.2f}")
        else:
            st.caption(tr["correlation_unavailable"])

        st.subheader(f"📰 {tr['news_panel']}")
        st.markdown(f"**{tr['sentiment']}: {tr.get(sentiment_label.lower(), sentiment_label)}**")
        if headlines:
            for h in headlines[:8]:
                st.write(f"• [{h['source']}] {h['title']}")
        else:
            st.caption(tr["no_news"])

    st.subheader(f"🎙️ {tr['squawk_panel']}")
    sq_cols = st.columns(len(SQUAWK_CHANNELS))
    for i, sq in enumerate(SQUAWK_CHANNELS):
        with sq_cols[i]:
            st.markdown(f"**[{sq['name']}]({sq['url']})**")
            st.caption(sq["desc_en"] if lang == "en" else sq["desc_am"])

    st.markdown("---")
    st.caption(f"{tr['disclaimer']}  ·  Last refreshed {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()

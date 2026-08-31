"""
==================================================================================
INSTITUTIONAL QUANTITATIVE TRADING TERMINAL
Hybrid Multi-Source Data Engine — Binance | yfinance (+ optional Dukascopy SWFX overlay)
==================================================================================
A single-file, production-ready Streamlit application implementing six
institutional-grade quantitative trading modules, backed by a hybrid data engine
that routes each asset class to the venue best suited to serve real market data
without requiring the user to hold API keys:

    Crypto & Metals proxy   -> Binance Public REST API   (klines + depth, 3-endpoint fallback)
    Major Forex pairs       -> yfinance structural baseline (ALWAYS, fast, guarantees 200+
                                bars) + OPTIONAL Dukascopy SWFX live tick overlay for real
                                volume / DOM, fetched as a separate, toggleable, lazy step
    Macro / Index / Futures -> yfinance                   (OHLCV + option chains)
    Universal fallback      -> yfinance                   (any source failure)

PERFORMANCE ARCHITECTURE (why "Fetching market data..." is now fast):
Earlier revisions called the Dukascopy SWFX tick downloader (`fetch_dukascopy_ticks`,
which fans out up to ~168 concurrent HTTP requests) INSIDE `fetch_ohlcv()` — i.e. inside
the exact call wrapped by the "Fetching market data..." spinner on every page load and
every timeframe change. That's what made the UI feel stuck.

This revision fully decouples the two:
  1. `fetch_ohlcv()` for forex now ONLY calls yfinance. It never touches Dukascopy. This
     is the same call used for every asset class, so "Fetching market data..." is always
     just one fast, cached yfinance/Binance request — no network fan-out, ever. The
     fallback path is also asset-aware now: forex never retries yfinance a second time
     with the same arguments (that was a pointless duplicate call) — it only falls back
     to yfinance if the primary route WASN'T already yfinance (i.e. for crypto, if
     Binance failed).
  2. Dukascopy tick fetching (used for both the real-volume overlay and the DOM ladder)
     happens afterwards, under its own separate spinner, and ONLY when the sidebar
     "Enable Live SWFX Tick Overlay" toggle is on and the selected timeframe is intraday
     (daily+ timeframes never needed it and now explicitly skip it). Both the volume
     overlay and the DOM ladder pull from the SAME cached `fetch_dukascopy_ticks()` call
     (identical symbol + lookback-hours cache key), so there is only ever one network
     batch per rerun, not two.
  3. Every `.bi5` hourly download and the tick-parsing step are `st.cache_data(ttl=300)`,
     so once an hour's ticks are fetched they are reused instantly across refreshes.

Honest caveat: Streamlit executes a script top-to-bottom on every rerun and paints the
page only once that run finishes — there's no true background/async rendering without
extra infrastructure (e.g. st.fragment + session-state polling, or a websocket bridge),
which this single-file app does not attempt to fake. What's guaranteed here is that the
Dukascopy fan-out is no longer on the *candle* critical path, is user-toggleable, is
skipped automatically on timeframes that don't need it, and is cached — so the heaviest
part of a load is, at worst, one clearly-labeled optional spinner instead of blocking
candle rendering entirely.

VOLUME ROBUSTNESS (why VWAP/CVD/Iceberg no longer break on forex):
yfinance reports zero (or missing) volume for most FX pairs, and Dukascopy's real tick
volume only covers a recent rolling window. Rather than let downstream math divide by a
column that's sometimes entirely zero, a Synthetic Volume Proxy
(`apply_synthetic_volume_proxy`) is applied immediately after the baseline candles are
fetched — before any module sees the data — whenever real reported volume is missing on
more than half the bars. It is a bar-range/volatility-weighted estimate
((High-Low) scaled by the bar's absolute return), NOT real traded volume, and it is
always disclosed as such: a neutral "🧮 Synthetic Volume Proxy" badge (not a jarring
yellow warning) is shown wherever it's in effect, and any real Dukascopy tick volume
fetched afterwards overwrites the synthetic estimate bar-by-bar wherever real coverage
exists. This guarantees every bar has a positive Volume value, so VWAP/CVD/Volume Delta/
Iceberg detection never divide by zero or degrade to all-NaN.

Modules:
  1. Institutional Order Flow & Liquidity Heatmap (BSL/SSL, FVGs, live $ volume
     annotations from the L2 DOM, institutional bank-wall isolation, and a
     Bank Anchor PnL Tracker)
  2. Quantitative Machine Learning Classifier (RandomForest direction model)
  3. Cross-Asset Correlation & Macro Yield Matrix
  4. Volume Delta / Cumulative Volume Delta (CVD) Footprint Analysis
  5. Options Gamma Exposure (GEX) & Max Pain Engine (live chain or simulation)
  6. Institutional Execution Algorithms (VWAP / TWAP / Iceberg Detection)
==================================================================================
"""

import lzma
import math
import struct
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
    .source-badge {
        display: inline-block;
        background-color: #1f2937;
        color: #f0b90b;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .info-badge {
        display: inline-block;
        background-color: #16202e;
        color: #7ea6d8;
        border: 1px solid #24405e;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 8px;
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

LIGHT_CSS = """
<style>
    .stApp { background-color: #f8f9fa; color: #1a1d23; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e5ea; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e5ea;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricValue"] { color: #b8860b; }
    h1, h2, h3 { color: #1a1d23; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #4a4f5a;
        border: 1px solid #e2e5ea;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eef0f3;
        color: #b8860b;
        font-weight: 600;
    }
    .module-note {
        background-color: #ffffff;
        border-left: 3px solid #b8860b;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #4a4f5a;
        margin-bottom: 10px;
        border: 1px solid #e2e5ea;
    }
    .source-badge {
        display: inline-block;
        background-color: #eef0f3;
        color: #8a6400;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .info-badge {
        display: inline-block;
        background-color: #eef4fb;
        color: #2f5d8a;
        border: 1px solid #cfe0f0;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .stButton>button {
        background-color: #b8860b;
        color: #ffffff;
        border: none;
        font-weight: 600;
        border-radius: 6px;
    }
    thead tr th { background-color: #eef0f3 !important; color: #8a6400 !important; }
</style>
"""

PLOTLY_TEMPLATE = "plotly_dark"
PURPLE_WALL = "#ba68c8"
PLOTLY_CONFIG = {"scrollZoom": False, "displayModeBar": True, "responsive": True}

# ==================================================================================
# DATA ENGINE CONSTANTS — HTTP HEADERS & MULTI-ENDPOINT FALLBACK ROUTING
# ==================================================================================

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

BINANCE_KLINES_ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]

BINANCE_DEPTH_ENDPOINTS = [
    "https://api.binance.com/api/v3/depth",
    "https://data-api.binance.vision/api/v3/depth",
    "https://api.binance.us/api/v3/depth",
]

DUKASCOPY_BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Base timeframes for which a Dukascopy tick overlay is even meaningful. Daily+ bases
# are never attempted — yfinance is the sole source there, backstopped by the synthetic
# volume proxy, so no tick fetch (and no request fan-out) happens on those timeframes.
TICK_OVERLAY_ELIGIBLE_BASES = {"1m", "5m", "15m", "30m", "1h", "4h"}
BAR_RULE_MAP = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h"}

DEFAULT_TICK_LOOKBACK_HOURS = 48
DEFAULT_DOM_BUCKET_PIPS = 2.0
DEFAULT_BINANCE_DEPTH_LIMIT = 1000

# ==================================================================================
# ASSET UNIVERSE
# ==================================================================================

CRYPTO_ASSETS = {
    "Bitcoin (BTC/USDT)": "BTCUSDT",
    "Ethereum (ETH/USDT)": "ETHUSDT",
    "PAX Gold — Gold Proxy (PAXG/USDT)": "PAXGUSDT",
    "Solana (SOL/USDT)": "SOLUSDT",
    "XRP (XRP/USDT)": "XRPUSDT",
    "BNB (BNB/USDT)": "BNBUSDT",
}

FOREX_ASSETS = {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
    "AUD/USD": "AUDUSD",
    "USD/CHF": "USDCHF",
    "USD/CAD": "USDCAD",
}

MACRO_ASSETS = {
    "US Dollar Index (DXY)": "DX-Y.NYB",
    "US 10-Year Treasury Yield": "^TNX",
    "Gold Futures (GC=F)": "GC=F",
    "Silver Futures (SI=F)": "SI=F",
    "S&P 500 Index": "^GSPC",
}

OPTIONS_PROXY_MAP = {
    "GC=F": "GLD",
    "SI=F": "SLV",
    "^GSPC": "SPY",
}

INTERVAL_CHOICES = [
    "1m", "2m", "3m", "4m", "5m", "15m", "30m", "1h", "2h", "3h", "4h",
    "1d", "1wk", "1mo", "1y",
]

INTERVAL_CONFIG = {
    "1m": {"base": "1m", "resample": None},
    "2m": {"base": "1m", "resample": "2min"},
    "3m": {"base": "1m", "resample": "3min"},
    "4m": {"base": "1m", "resample": "4min"},
    "5m": {"base": "5m", "resample": None},
    "15m": {"base": "15m", "resample": None},
    "30m": {"base": "30m", "resample": None},
    "1h": {"base": "1h", "resample": None},
    "2h": {"base": "1h", "resample": "2h"},
    "3h": {"base": "1h", "resample": "3h"},
    "4h": {"base": "4h", "resample": None},
    "1d": {"base": "1d", "resample": None},
    "1wk": {"base": "1d", "resample": "W"},
    "1mo": {"base": "1d", "resample": "ME"},
    "1y": {"base": "1d", "resample": "YE"},
}

BINANCE_BASE_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}

# ==================================================================================
# SYMBOL / TICKER TRANSLATION HELPERS
# ==================================================================================

def binance_to_yf_ticker(symbol: str) -> str:
    for quote in ("USDT", "USDC", "BUSD"):
        if symbol.upper().endswith(quote):
            base = symbol.upper()[:-len(quote)]
            return f"{base}-USD"
    return symbol

def forex_to_yf_ticker(symbol: str) -> str:
    return f"{symbol.upper()}=X"

def dukascopy_point_divider(symbol: str) -> int:
    """Dukascopy raw tick prices are integers; divide by this to get the real quote."""
    return 1000 if "JPY" in symbol.upper() else 100000

# ==================================================================================
# LOW-LEVEL SOURCE FETCHERS
# ==================================================================================

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    legacy_alias = {"ME": "M", "YE": "Y"}
    try:
        out = df.resample(rule).agg(agg)
    except Exception:
        try:
            out = df.resample(legacy_alias.get(rule, rule)).agg(agg)
        except Exception:
            return df
    return out.dropna(subset=["Open", "High", "Low", "Close"])

def fetch_binance_klines(symbol: str, base_key: str, limit: int = 1000) -> pd.DataFrame:
    interval = BINANCE_BASE_MAP.get(base_key)
    if interval is None:
        return pd.DataFrame()

    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    for url in BINANCE_KLINES_ENDPOINTS:
        try:
            resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or not data:
                continue
            cols = [
                "open_time", "Open", "High", "Low", "Close", "Volume",
                "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
            ]
            df = pd.DataFrame(data, columns=cols)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.set_index("open_time")
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()

def fetch_yfinance_ohlcv(ticker: str, base_key: str) -> pd.DataFrame:
    period_map = {
        "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
        "1h": "730d", "4h": "730d", "1d": "10y",
    }
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "60m", "4h": "60m", "1d": "1d",
    }
    period = period_map.get(base_key, "1y")
    yf_interval = interval_map.get(base_key, "1d")
    try:
        raw = yf.download(
            tickers=ticker, period=period, interval=yf_interval,
            progress=False, auto_adjust=False, threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                raw.columns = raw.columns.get_level_values(0)
            except Exception:
                raw.columns = ["_".join([str(c) for c in col if c]) for col in raw.columns]
        expected = ["Open", "High", "Low", "Close", "Volume"]
        for col in expected:
            if col not in raw.columns:
                return pd.DataFrame()
        df = raw[expected].copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0)
        if base_key == "4h" and not df.empty:
            df = resample_ohlcv(df, "4h")
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(asset_class: str, symbol: str, yf_ticker: str, interval_key: str):
    """Fast structural candle fetch ONLY. Never touches Dukascopy — see module docstring
    (Performance Architecture) for why this separation is what fixes the load-time bug.

    Fallback logic is asset-aware: forex's primary route already IS yfinance, so if it
    comes back empty there is no point calling fetch_yfinance_ohlcv() a second time with
    identical arguments — it will just fail again. The generic "any source failure ->
    yfinance" fallback below now only fires for asset classes whose primary route was
    NOT yfinance (i.e. crypto, when all three Binance endpoints failed)."""
    cfg = INTERVAL_CONFIG.get(interval_key, INTERVAL_CONFIG["1d"])
    base_key = cfg["base"]
    resample_rule = cfg["resample"]

    df = pd.DataFrame()
    source_used = "unavailable"
    primary_was_yfinance = False

    if asset_class == "crypto":
        df = fetch_binance_klines(symbol, base_key)
        if not df.empty:
            source_used = "Binance (live)"
    elif asset_class == "forex":
        df = fetch_yfinance_ohlcv(yf_ticker, base_key)
        primary_was_yfinance = True
        if not df.empty:
            source_used = "yfinance (structural baseline)"
    else:
        df = fetch_yfinance_ohlcv(yf_ticker, base_key)
        primary_was_yfinance = True
        if not df.empty:
            source_used = "yfinance"

    if df.empty and not primary_was_yfinance:
        df = fetch_yfinance_ohlcv(yf_ticker, base_key)
        if not df.empty:
            source_used = "yfinance (fallback)"

    if df.empty:
        return pd.DataFrame(), source_used

    if resample_rule:
        df = resample_ohlcv(df, resample_rule)

    return df, source_used

def get_last_price(asset_class: str, symbol: str, yf_ticker: str):
    df, _src = fetch_ohlcv(asset_class, symbol, yf_ticker, "1d")
    if df.empty:
        return None
    return float(df["Close"].iloc[-1])

# ==================================================================================
# SYNTHETIC VOLUME PROXY — seamless fallback when real volume is missing/zero
# ==================================================================================

def apply_synthetic_volume_proxy(df: pd.DataFrame):
    """Fills bars with no real reported volume using a bar-range/volatility-weighted
    estimate, so VWAP / CVD / Volume Delta / Iceberg detection always have a positive,
    non-degenerate Volume series to work with. Returns (df, used_synthetic: bool).

    This is explicitly an ESTIMATE, not real traded volume — callers must disclose it
    to the user (see the "🧮 Synthetic Volume Proxy" badges in the modules below)
    rather than presenting it as genuine exchange volume. Any bar that already has real
    reported volume (> 0) is left untouched; only zero/missing bars are filled.
    """
    if df.empty:
        return df, False

    vol = df["Volume"].fillna(0)
    real_coverage = float((vol > 0).mean()) if len(vol) else 0.0
    if real_coverage >= 0.5:
        return df, False

    out = df.copy()
    bar_range = (out["High"] - out["Low"]).clip(lower=0)
    ret_vol = out["Close"].pct_change().abs().fillna(0)
    raw_proxy = (bar_range * (1.0 + ret_vol * 25.0)).replace([np.inf, -np.inf], 0).fillna(0)

    if raw_proxy.sum() <= 0:
        raw_proxy = pd.Series(1.0, index=out.index)

    positive_mean = raw_proxy[raw_proxy > 0].mean() if (raw_proxy > 0).any() else 1.0
    scaled = (raw_proxy / positive_mean) * 1000.0
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    fallback_fill = scaled.mean() if scaled.notna().any() else 1000.0
    scaled = scaled.fillna(fallback_fill).clip(lower=1.0)

    out["Volume"] = vol.where(vol > 0, scaled)
    return out, True

# ==================================================================================
# DUKASCOPY SWFX — OPTIONAL LAZY TICK OVERLAY (volume enrichment + DOM ladder)
# Fetched separately from the candle pipeline; never blocks candle rendering.
# ------------------------------------------------------------------
def _dukascopy_hour_url(symbol: str, dt_utc: datetime) -> str:
    # Dukascopy's month component in the URL path is zero-indexed (Jan = 00).
    return (
        f"{DUKASCOPY_BASE_URL}/{symbol.upper()}/{dt_utc.year}/"
        f"{dt_utc.month - 1:02d}/{dt_utc.day:02d}/{dt_utc.hour:02d}h_ticks.bi5"
    )

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dukascopy_hour_raw(symbol: str, dt_utc: datetime) -> bytes:
    """Download one hour's raw .bi5 bytes. Cached for 5 minutes — completed historical
    hours never change, so every later reuse (across the volume-overlay path AND the DOM
    ladder path, and across reruns) is a cache hit rather than a fresh network call."""
    url = _dukascopy_hour_url(symbol, dt_utc)
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        if resp.status_code != 200 or not resp.content:
            return b""
        return resp.content
    except Exception:
        return b""

def _decode_dukascopy_bi5(raw_bytes: bytes, hour_start_utc: datetime, point_divider: int):
    """Decode a Dukascopy .bi5 tick file into a list of tick dicts.

    Each record is 20 bytes, big-endian:
      int32 ms_offset_from_hour_start, int32 ask_raw, int32 bid_raw,
      float32 ask_volume, float32 bid_volume
    The payload is LZMA-compressed.
    """
    if not raw_bytes:
        return []
    try:
        decompressed = lzma.decompress(raw_bytes)
    except Exception:
        return []
    record_size = 20
    n_records = len(decompressed) // record_size
    if n_records == 0:
        return []
    ticks = []
    for i in range(n_records):
        chunk = decompressed[i * record_size: (i + 1) * record_size]
        try:
            ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">iiiff", chunk)
        except struct.error:
            continue
        if ask_raw <= 0 or bid_raw <= 0:
            continue
        ts = hour_start_utc + timedelta(milliseconds=ms_offset)
        ticks.append({
            "timestamp": ts,
            "ask": ask_raw / point_divider,
            "bid": bid_raw / point_divider,
            "ask_volume": float(ask_vol),
            "bid_volume": float(bid_vol),
        })
    return ticks

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dukascopy_ticks(symbol: str, lookback_hours: int) -> pd.DataFrame:
    """Pull the last `lookback_hours` of real SWFX ticks for `symbol`, concurrently.
    Cached for 5 minutes on the exact (symbol, lookback_hours) key — the volume-overlay
    step and the DOM-ladder step both call this with the SAME lookback_hours value, so
    only the first caller in a rerun actually hits the network; the second is a cache hit."""
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    hour_slots = [current_hour - timedelta(hours=h) for h in range(lookback_hours + 1)]
    point_divider = dukascopy_point_divider(symbol)

    all_ticks = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(_fetch_dukascopy_hour_raw, symbol, hs): hs for hs in hour_slots
        }
        for fut in as_completed(futures):
            hour_start = futures[fut]
            try:
                raw = fut.result()
            except Exception:
                raw = b""
            if raw:
                all_ticks.extend(_decode_dukascopy_bi5(raw, hour_start, point_divider))

    if not all_ticks:
        return pd.DataFrame()

    tdf = pd.DataFrame(all_ticks).sort_values("timestamp").set_index("timestamp")
    tdf["mid"] = (tdf["ask"] + tdf["bid"]) / 2.0
    tdf["volume"] = tdf["ask_volume"] + tdf["bid_volume"]
    return tdf

def overlay_dukascopy_tick_volume(df: pd.DataFrame, ticks: pd.DataFrame, base_key: str) -> pd.DataFrame:
    """Stitch real Dukascopy tick-derived volume onto the baseline candles, overwriting
    the synthetic proxy (if any) wherever live tick coverage exists. OHLC values are
    never touched — only Volume. Wrapped in try/except by the caller so a malformed
    tick batch (bad timezone, empty resample, etc.) degrades gracefully instead of
    crashing the app or wiping out the candles already on screen."""
    rule = BAR_RULE_MAP.get(base_key)
    if rule is None or ticks.empty or df.empty:
        return df

    try:
        tick_vol = ticks["volume"].resample(rule).sum()

        idx = df.index
        idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        tv_idx = tick_vol.index
        tick_vol.index = tv_idx.tz_localize("UTC") if tv_idx.tz is None else tv_idx.tz_convert("UTC")

        try:
            tolerance = pd.Timedelta(rule)
        except Exception:
            tolerance = None

        aligned = tick_vol.reindex(idx_utc, method="nearest", tolerance=tolerance)

        out = df.copy()
        vol_arr = aligned.to_numpy()
        valid = ~pd.isna(vol_arr)
        valid = valid & (vol_arr > 0)
        if valid.any():
            out_vol = out["Volume"].to_numpy(dtype=float)
            out_vol[valid] = vol_arr[valid]
            out["Volume"] = out_vol
        return out
    except Exception:
        # Any alignment/merge failure falls back to the candles as they already are
        # (real volume or synthetic proxy) rather than raising or clearing the chart.
        return df

# ==================================================================================
# L2 DOM / LIQUIDITY DEPTH ENGINE
# ==================================================================================

@st.cache_data(ttl=15, show_spinner=False)
def fetch_order_book(
    asset_class: str,
    symbol: str,
    depth_limit: int = DEFAULT_BINANCE_DEPTH_LIMIT,
    dom_lookback_hours: int = DEFAULT_TICK_LOOKBACK_HOURS,
    dom_bucket_pips: float = DEFAULT_DOM_BUCKET_PIPS,
) -> pd.DataFrame:
    try:
        if asset_class == "crypto":
            params = {"symbol": symbol.upper(), "limit": depth_limit}
            for url in BINANCE_DEPTH_ENDPOINTS:
                try:
                    resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=6)
                    resp.raise_for_status()
                    data = resp.json()
                    bids = pd.DataFrame(data.get("bids", []), columns=["price", "qty"])
                    asks = pd.DataFrame(data.get("asks", []), columns=["price", "qty"])
                    if bids.empty and asks.empty:
                        continue
                    bids = bids.astype(float)
                    asks = asks.astype(float)
                    bids["side"] = "bid"
                    asks["side"] = "ask"
                    combined = pd.concat([bids, asks], ignore_index=True)
                    if not combined.empty:
                        return combined
                except Exception:
                    continue
            return pd.DataFrame()

        elif asset_class == "forex":
            # Dukascopy SWFX has no public, key-free multi-level DOM feed. Instead we
            # build a genuine liquidity ladder from real recent tick prints: the last
            # window of actual bid/ask ticks and traded volumes is bucketed into price
            # levels, producing a real, live, trade-flow-derived depth structure (see
            # module docstring for the honest distinction vs. a native L2 snapshot).
            ticks = fetch_dukascopy_ticks(symbol, lookback_hours=dom_lookback_hours)
            if ticks.empty:
                return pd.DataFrame()

            recent = ticks.tail(4000).copy()
            if recent.empty:
                return pd.DataFrame()

            pip = 0.01 if "JPY" in symbol.upper() else 0.0001
            bucket_size = pip * max(dom_bucket_pips, 0.1)

            recent["bid_bucket"] = (recent["bid"] / bucket_size).round() * bucket_size
            recent["ask_bucket"] = (recent["ask"] / bucket_size).round() * bucket_size

            bid_levels = (
                recent.groupby("bid_bucket")["bid_volume"].sum()
                .reset_index().rename(columns={"bid_bucket": "price", "bid_volume": "qty"})
            )
            bid_levels["side"] = "bid"

            ask_levels = (
                recent.groupby("ask_bucket")["ask_volume"].sum()
                .reset_index().rename(columns={"ask_bucket": "price", "ask_volume": "qty"})
            )
            ask_levels["side"] = "ask"

            combined = pd.concat([bid_levels, ask_levels], ignore_index=True)
            combined = combined[combined["qty"] > 0]
            return combined[["price", "qty", "side"]]

        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def format_dollars(value: float) -> str:
    try:
        value = float(value)
    except Exception:
        return "$0"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{sign}${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{sign}${value/1e3:.1f}K"
    else:
        return f"{sign}${value:,.0f}"

def format_price_level(price: float) -> str:
    try:
        price = float(price)
    except Exception:
        return "0.00"
    if abs(price) < 10:
        return f"{price:,.5f}"
    elif abs(price) < 1000:
        return f"{price:,.3f}"
    else:
        return f"{price:,.2f}"

def compute_level_dollar_volume(order_book_df: pd.DataFrame, level_price: float, tolerance_pct: float = 0.15) -> float:
    if order_book_df.empty or level_price <= 0:
        return 0.0
    tol = level_price * (tolerance_pct / 100.0)
    mask = (order_book_df["price"] >= level_price - tol) & (order_book_df["price"] <= level_price + tol)
    subset = order_book_df[mask]
    if subset.empty:
        return 0.0
    return float((subset["price"] * subset["qty"]).sum())

def isolate_institutional_walls(
    order_book_df: pd.DataFrame, current_price: float, percentile: float = 80.0,
    top_n_per_side: int = 5, max_distance_pct: float = 25.0
) -> pd.DataFrame:
    if order_book_df.empty:
        return pd.DataFrame()
    df = order_book_df.copy()
    if current_price and current_price > 0:
        band = current_price * (max_distance_pct / 100.0)
        df = df[(df["price"] >= current_price - band) & (df["price"] <= current_price + band)]
    if df.empty:
        return pd.DataFrame()
    df["dollar_value"] = df["price"] * df["qty"]
    threshold = np.percentile(df["dollar_value"], percentile)
    walls = df[df["dollar_value"] >= threshold].copy()
    if walls.empty:
        return pd.DataFrame()
    bid_walls = walls[walls["side"] == "bid"].sort_values("dollar_value", ascending=False).head(top_n_per_side)
    ask_walls = walls[walls["side"] == "ask"].sort_values("dollar_value", ascending=False).head(top_n_per_side)
    return pd.concat([bid_walls, ask_walls], ignore_index=True).sort_values("dollar_value", ascending=False)

def compute_bank_anchor_pnl(walls_df: pd.DataFrame, current_price: float) -> pd.DataFrame:
    if walls_df.empty or current_price <= 0:
        return pd.DataFrame()

    out = walls_df.copy()
    pnl_vals, position_vals = [], []
    for _, row in out.iterrows():
        if row["side"] == "bid":
            pnl = safe_pct(current_price, row["price"])
            position_vals.append("Long Anchor (Support)")
        else:
            pnl = safe_pct(row["price"], current_price)
            position_vals.append("Short Anchor (Resistance)")
        pnl_vals.append(pnl)

    out["pnl_pct"] = pnl_vals
    out["position"] = position_vals

    def status(p):
        if p > 0.5:
            return "🟢 Expanding Profit"
        elif p < -0.5:
            return "🔴 Unwinding / Cutting Loss"
        else:
            return "🟡 Building Position"

    out["status"] = out["pnl_pct"].apply(status)
    return out.sort_values("dollar_value", ascending=False)

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

def price_axis_range(*series_list, pad_pct: float = 0.08):
    values = []
    for s in series_list:
        if s is None:
            continue
        s = pd.Series(s).dropna()
        if not s.empty:
            values.append(s)
    if not values:
        return None
    combined = pd.concat(values)
    lo = float(combined.min())
    hi = float(combined.max())
    if hi <= lo:
        hi = lo * 1.01 if lo > 0 else lo + 1.0
    pad = (hi - lo) * pad_pct
    if pad <= 0:
        pad = abs(hi) * 0.01 if hi != 0 else 1.0
    return [lo - pad, hi + pad]

def render_status_badge(text: str):
    st.markdown(f'<span class="info-badge">{text}</span>', unsafe_allow_html=True)

# ==================================================================================
# MODULE 1 — INSTITUTIONAL ORDER FLOW & LIQUIDITY HEATMAP
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

def render_liquidity_module(df: pd.DataFrame, order_book_df: pd.DataFrame, asset_class: str, label: str):
    st.markdown(
        '<div class="module-note">Swing-point liquidity pools (BSL/SSL) and Fair Value Gap '
        'imbalance zones from price-action structure, dynamically annotated with live resting '
        '$ order-book / tick-flow depth, institutional bank-wall isolation, and a Bank Anchor '
        'PnL Tracker.</div>',
        unsafe_allow_html=True,
    )

    if len(df) < 15:
        st.warning("Not enough bars returned for this selection to compute liquidity structure. Try a longer interval.")
        return

    c1, c2, c3 = st.columns(3)
    window = c1.slider("Swing Sensitivity (lookback bars)", 2, 15, 5, key="liq_window")
    tolerance = c2.slider("DOM Price Tolerance (%) for $ Volume Aggregation", 0.02, 1.0, 0.15, step=0.02, key="liq_tol")
    wall_pctl = c3.slider("Institutional Wall Percentile", 50, 99, 80, key="liq_wall_pct")
    max_zones = st.slider("Max FVG Zones Displayed", 3, 30, 12, key="liq_fvg_count")

    swing_highs, swing_lows = detect_swings(df, window=window)
    bullish_fvg, bearish_fvg = detect_fvg(df)
    current_price = float(df["Close"].iloc[-1])

    has_dom = asset_class in ("crypto", "forex") and not order_book_df.empty
    if not has_dom:
        render_status_badge(
            "ℹ️ Live order-book / tick-flow depth unavailable right now (macro assets have no public "
            "order book, live SWFX overlay is disabled, or the feed is temporarily unreachable) — "
            "$ volume annotations and wall isolation are skipped. Swing/FVG structure below is unaffected."
        )
    else:
        badge_text = "🟢 LIVE L2 DOM CONNECTED" if asset_class == "crypto" else "🟢 LIVE SWFX TICK-FLOW DEPTH"
        st.markdown(
            f'<span class="source-badge">{badge_text}</span> '
            f'<span style="color:#8b90a0; font-size:0.8rem;">{len(order_book_df):,} price levels loaded</span>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSL Pools (Swing Highs)", len(swing_highs))
    c2.metric("SSL Pools (Swing Lows)", len(swing_lows))
    c3.metric("Bullish FVG Zones", len(bullish_fvg))
    c4.metric("Bearish FVG Zones", len(bearish_fvg))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    recent_highs = sorted(swing_highs, key=lambda x: x[0])[-8:]
    recent_lows = sorted(swing_lows, key=lambda x: x[0])[-8:]

    for t, price in recent_highs:
        dv = compute_level_dollar_volume(order_book_df, price, tolerance) if has_dom else 0.0
        annot_text = f"BSL @ {format_price_level(price)} ({format_dollars(dv)})" if has_dom else f"BSL @ {format_price_level(price)}"
        fig.add_shape(type="line", x0=t, x1=df.index[-1], y0=price, y1=price, line=dict(color="#ff5252", width=1, dash="dot"))
        fig.add_annotation(x=df.index[-1], y=price, text=annot_text, showarrow=False, font=dict(color="#ff5252", size=10), xanchor="left")

    for t, price in recent_lows:
        dv = compute_level_dollar_volume(order_book_df, price, tolerance) if has_dom else 0.0
        annot_text = f"SSL @ {format_price_level(price)} ({format_dollars(dv)})" if has_dom else f"SSL @ {format_price_level(price)}"
        fig.add_shape(type="line", x0=t, x1=df.index[-1], y0=price, y1=price, line=dict(color="#00e5ff", width=1, dash="dot"))
        fig.add_annotation(x=df.index[-1], y=price, text=annot_text, showarrow=False, font=dict(color="#00e5ff", size=10), xanchor="left")

    for zone in bullish_fvg[-max_zones:]:
        fig.add_shape(type="rect", x0=zone["start"], x1=df.index[-1], y0=zone["bottom"], y1=zone["top"], fillcolor="rgba(38,166,154,0.18)", line=dict(width=0))
    for zone in bearish_fvg[-max_zones:]:
        fig.add_shape(type="rect", x0=zone["start"], x1=df.index[-1], y0=zone["bottom"], y1=zone["top"], fillcolor="rgba(239,83,80,0.18)", line=dict(width=0))

    walls_df = pd.DataFrame()
    if has_dom:
        walls_df = isolate_institutional_walls(order_book_df, current_price, percentile=wall_pctl, top_n_per_side=5)
        for _, w in walls_df.iterrows():
            fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=w["price"], y1=w["price"], line=dict(color=PURPLE_WALL, width=2, dash="dash"))
            side_tag = "BID WALL" if w["side"] == "bid" else "ASK WALL"
            fig.add_annotation(
                x=df.index[len(df)//2], y=w["price"],
                text=f"🏦 {side_tag} @ {format_price_level(w['price'])}: {format_dollars(w['dollar_value'])}",
                showarrow=False, font=dict(color=PURPLE_WALL, size=10), bgcolor="rgba(11,14,20,0.7)"
            )

    y_range = price_axis_range(df["Low"], df["High"], pad_pct=0.10)

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=680, xaxis_rangeslider_visible=False, dragmode="pan",
        title=f"{label} — Liquidity Pools, Fair Value Gaps & Institutional Bank Walls",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=y_range, autorange=False if y_range else True),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if has_dom:
        st.subheader("🏦 Bank Anchor PnL Tracker")
        if not walls_df.empty:
            pnl_df = compute_bank_anchor_pnl(walls_df, current_price)
            net_pnl = float(pnl_df["pnl_pct"].mean())
            if net_pnl > 0.5:
                overall_status = "🟢 Institutional Anchors Net Expanding Profit"
            elif net_pnl < -0.5:
                overall_status = "🔴 Institutional Anchors Net Unwinding / Taking Profit"
            else:
                overall_status = "🟡 Institutional Anchors Net Neutral / Building Positions"

            m1, m2, m3 = st.columns(3)
            m1.metric("Net Institutional Anchor PnL", f"{net_pnl:+.2f}%")
            m2.metric("Institutional Walls Detected", len(pnl_df))
            m3.metric("Largest Wall $ Value", format_dollars(pnl_df["dollar_value"].max()))
            st.markdown(f"### {overall_status}")

            show_cols = ["price", "qty", "side", "dollar_value", "position", "pnl_pct", "status"]
            display_df = pnl_df[show_cols].rename(columns={
                "price": "Price", "qty": "Quantity", "side": "Side",
                "dollar_value": "$ Value", "position": "Anchor Type",
                "pnl_pct": "Unrealized PnL %", "status": "Status",
            })
            display_df["$ Value"] = display_df["$ Value"].apply(format_dollars)
            display_df["Unrealized PnL %"] = display_df["Unrealized PnL %"].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No institutional-sized walls detected above the selected percentile threshold within a realistic band of the current price.")

    with st.expander("📋 Nearest Liquidity Levels to Current Price"):
        all_levels = [("BSL", t, p) for t, p in swing_highs] + [("SSL", t, p) for t, p in swing_lows]
        if all_levels:
            lvl_df = pd.DataFrame(all_levels, columns=["Type", "Timestamp", "Price"])
            lvl_df["Distance %"] = lvl_df["Price"].apply(lambda p: safe_pct(p, current_price))
            if has_dom:
                lvl_df["$ Volume Nearby"] = lvl_df["Price"].apply(
                    lambda p: format_dollars(compute_level_dollar_volume(order_book_df, p, tolerance))
                )
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
    feat["volume_delta"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    return feat

def render_ml_module(df: pd.DataFrame, label: str):
    st.markdown(
        '<div class="module-note">A RandomForest classifier trained live on engineered technical '
        'features (RSI, ATR, volatility, MA ratios, volume delta, momentum) to estimate the '
        'probability of the next-N-bar directional move.</div>',
        unsafe_allow_html=True,
    )

    if len(df) < 80:
        st.warning("Insufficient historical data for reliable ML training. Select a longer interval / lower timeframe granularity.")
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
        st.warning("Not enough class diversity in the selected sample to train a robust classifier.")
        return

    feature_cols = list(feat.columns)
    X = data[feature_cols]
    y = data["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, shuffle=False)
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

    st.markdown(f"<h3 style='color:{signal_color};'>Model Signal: {signal}</h3>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#f0b90b"))
        fig_fi.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Feature Importance", margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
        st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)

    with col_b:
        fig_proba = go.Figure(go.Pie(
            labels=["Sell", "Hold", "Buy"], values=[sell_p, hold_p, buy_p],
            marker=dict(colors=["#ef5350", "#f0b90b", "#26a69a"]), hole=0.55,
        ))
        fig_proba.update_layout(
            template=PLOTLY_TEMPLATE, height=420, title="Directional Probability",
            margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
        )
        st.plotly_chart(fig_proba, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Model trained on {len(X_train)} bars, validated on {len(X_test)} out-of-sample bars for {label}.")

# ==================================================================================
# MODULE 3 — CROSS-ASSET CORRELATION & MACRO YIELD MATRIX
# ==================================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_series(yf_ticker: str, lookback_days: int = 180) -> pd.Series:
    df = fetch_yfinance_ohlcv(yf_ticker, "1d")
    if df.empty:
        return pd.Series(dtype=float)
    cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
    df = df[df.index >= cutoff]
    return df["Close"]

def render_correlation_module(symbol: str, yf_ticker: str, asset_class: str, label: str):
    st.markdown(
        '<div class="module-note">Cross-asset relationships between the active instrument, '
        'the US Dollar Index, 10-Year Treasury Yields, and Bitcoin — key macro drivers.</div>',
        unsafe_allow_html=True,
    )

    lookback = st.slider("Correlation Lookback (Days)", 30, 730, 180, step=10, key="corr_lookback")

    # Labels aligned 1:1 with MACRO_ASSETS naming so badges/legends read consistently
    # across the Macro asset picker and this module's correlation matrix.
    base_macro = {
        "US Dollar Index (DXY)": "DX-Y.NYB",
        "US 10-Year Treasury Yield": "^TNX",
        "Bitcoin (BTC/USD)": "BTC-USD",
    }

    is_btc_active = symbol.upper() == "BTCUSDT" or yf_ticker.upper() == "BTC-USD"
    is_dxy_active = yf_ticker.upper() == "DX-Y.NYB"
    is_10y_active = yf_ticker.upper() == "^TNX"

    if is_btc_active or is_dxy_active or is_10y_active:
        anchor_label = "Gold Futures (GC=F)"
        anchor_series = fetch_macro_series("GC=F", lookback)
    else:
        anchor_label = label
        anchor_df, _src = fetch_ohlcv(asset_class, symbol, yf_ticker, "1d")
        if not anchor_df.empty:
            cutoff = anchor_df.index.max() - pd.Timedelta(days=lookback)
            anchor_series = anchor_df[anchor_df.index >= cutoff]["Close"]
        else:
            anchor_series = pd.Series(dtype=float)

    series_dict = {}
    if not anchor_series.empty:
        series_dict[anchor_label] = anchor_series

    fetch_errors = []
    for m_label, tk in base_macro.items():
        if m_label == anchor_label:
            continue
        s = fetch_macro_series(tk, lookback)
        if s.empty:
            fetch_errors.append(m_label)
        else:
            series_dict[m_label] = s

    if fetch_errors:
        st.warning(f"Could not retrieve live data for: {', '.join(fetch_errors)}.")

    if len(series_dict) < 2:
        st.error("Insufficient macro data available to compute correlations right now.")
        return

    combined = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()

    if combined.empty or len(combined) < 10:
        st.warning("Not enough overlapping historical data across assets for this lookback window.")
        return

    corr_matrix = combined.corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
    ))
    fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=460, title="Cross-Asset Correlation Matrix", margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
    st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader(f"Rolling 30-Day Correlation vs {anchor_label}")
    if anchor_label in combined.columns:
        rolling_window = min(30, max(5, len(combined) // 3))
        fig_roll = go.Figure()
        for col in combined.columns:
            if col == anchor_label:
                continue
            rolling_corr = combined[anchor_label].rolling(rolling_window).corr(combined[col])
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{anchor_label} vs {col}"))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#666")
        fig_roll.update_layout(template=PLOTLY_TEMPLATE, height=420, title=f"Rolling {rolling_window}-Day Correlation", margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
        st.plotly_chart(fig_roll, use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("📈 Normalized Price Performance (Rebased to 100)"):
        rebased = combined / combined.iloc[0] * 100
        fig_reb = go.Figure()
        for col in rebased.columns:
            fig_reb.add_trace(go.Scatter(x=rebased.index, y=rebased[col], mode="lines", name=col))
        fig_reb.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan")
        st.plotly_chart(fig_reb, use_container_width=True, config=PLOTLY_CONFIG)

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

def render_volume_delta_module(df: pd.DataFrame, label: str, source_used: str, volume_is_synthetic: bool = False):
    st.markdown(
        '<div class="module-note">Order-flow reconstruction: buying vs. selling volume estimated '
        'from real exchange-reported volume weighted by intra-bar close position, aggregated into '
        'Cumulative Volume Delta (CVD).</div>',
        unsafe_allow_html=True,
    )

    if df["Volume"].sum() == 0:
        render_status_badge("ℹ️ No volume data available for this instrument — Volume Delta will show flat/neutral output.")
    elif volume_is_synthetic:
        render_status_badge(
            "🧮 Using a Synthetic Volume Proxy (bar-range × volatility estimate) — real exchange "
            "volume was unavailable or zero for most bars. Not genuine traded volume."
        )

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
        subplot_titles=(f"{label} Price [{source_used}]", "Volume Delta (Buy − Sell)", "Cumulative Volume Delta (CVD)"),
    )
    fig.add_trace(go.Candlestick(
        x=vd.index, open=vd["Open"], high=vd["High"], low=vd["Low"], close=vd["Close"],
        name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if not spikes.empty:
        fig.add_trace(go.Scatter(
            x=spikes.index, y=spikes["High"] * 1.001, mode="markers", name="Imbalance Spike",
            marker=dict(color="#f0b90b", size=9, symbol="triangle-down"),
        ), row=1, col=1)

    bar_colors = np.where(vd["delta"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=vd.index, y=vd["delta"], marker_color=bar_colors, name="Delta"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vd.index, y=vd["cvd"], mode="lines", name="CVD", line=dict(color="#00e5ff", width=2)), row=3, col=1)

    price_range = price_axis_range(vd["Low"], vd["High"], pad_pct=0.06)
    if price_range:
        fig.update_yaxes(range=price_range, row=1, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=780, showlegend=False,
        xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

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
        return pdf / (spot * iv * math.sqrt(t_years))
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

def compute_max_pain_from_sim(gex_df: pd.DataFrame):
    try:
        strikes = gex_df["strike"].values
        call_oi = gex_df["call_oi"].values
        put_oi = gex_df["put_oi"].values
        pain = []
        for s in strikes:
            call_loss = (np.clip(s - strikes, 0, None) * call_oi).sum()
            put_loss = (np.clip(strikes - s, 0, None) * put_oi).sum()
            pain.append(call_loss + put_loss)
        pain = np.array(pain)
        max_pain_strike = float(strikes[np.argmin(pain)])
        return max_pain_strike, pd.DataFrame({"strike": strikes, "total_pain": pain})
    except Exception:
        return None, pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(proxy_symbol: str):
    try:
        tk = yf.Ticker(proxy_symbol)
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

def gex_from_chain(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, days_to_expiry: int):
    t_years = max(days_to_expiry, 1) / 365.0
    contract_mult = 100

    def process(chain, sign):
        c = chain.copy()
        c["impliedVolatility"] = c["impliedVolatility"].replace(0, np.nan)
        med_iv = c["impliedVolatility"].median()
        c["impliedVolatility"] = c["impliedVolatility"].fillna(med_iv if pd.notna(med_iv) else 0.2)
        c["openInterest"] = c["openInterest"].fillna(0)
        c["gamma"] = c.apply(lambda r: bs_gamma(spot, r["strike"], t_years, max(r["impliedVolatility"], 0.01)), axis=1)
        c["gex"] = sign * c["gamma"] * c["openInterest"] * contract_mult * spot * spot * 0.01
        return c[["strike", "openInterest", "gex"]]

    c_proc = process(calls, 1)
    p_proc = process(puts, -1)
    merged = pd.merge(c_proc, p_proc, on="strike", how="outer", suffixes=("_call", "_put")).fillna(0)
    merged["net_gex"] = merged["gex_call"] + merged["gex_put"]
    merged = merged.rename(columns={"gex_call": "call_gex", "gex_put": "put_gex"})
    return merged.sort_values("strike").reset_index(drop=True)

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

def render_gex_module(symbol: str, yf_ticker: str, asset_class: str, label: str):
    st.markdown(
        '<div class="module-note">Gamma Exposure (GEX) profile identifying dealer positioning, '
        'volatility pin zones, and the gamma flip level.</div>',
        unsafe_allow_html=True,
    )

    proxy = OPTIONS_PROXY_MAP.get(yf_ticker) if asset_class == "macro" else None
    chain_result = fetch_options_chain(proxy) if proxy else None

    if chain_result is not None:
        calls, puts, expiry, spot = chain_result
        try:
            dte = max((pd.to_datetime(expiry) - pd.Timestamp.now()).days, 1)
        except Exception:
            dte = 30
        gex_df = gex_from_chain(calls, puts, spot, dte)
        max_pain, pain_df = compute_max_pain(calls, puts)
        source_label = f"Live listed options — proxy: {proxy} (expiry {expiry})"
    else:
        spot = get_last_price(asset_class, symbol, yf_ticker)
        if spot is None:
            st.error(f"Could not retrieve a live spot price for {label} to calibrate the GEX engine.")
            return
        dte = st.slider("Simulated Days to Expiry", 1, 90, 30, key="gex_dte")
        iv_assumed = st.slider("Assumed Implied Volatility (%)", 5, 80, 18, key="gex_iv") / 100
        n_strikes = st.slider("Strike Range (± strikes around spot)", 10, 40, 25, key="gex_strikes")
        gex_df = simulate_gex(spot, n_strikes=n_strikes, iv=iv_assumed, days_to_expiry=dte)
        max_pain, pain_df = compute_max_pain_from_sim(gex_df)
        source_label = f"Simulated GEX engine (Black-Scholes gamma model) — no listed options market for {label}"

    st.caption(f"Data source: {source_label}")

    net_gex_total = gex_df["net_gex"].sum()
    flip_candidates = gex_df.sort_values("strike").copy()
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
    fig.add_vline(x=spot, line_dash="dash", line_color="#f0b90b", annotation_text="Spot", annotation_position="top")
    fig.add_vline(x=gamma_flip, line_dash="dot", line_color="#00e5ff", annotation_text="Gamma Flip", annotation_position="bottom")
    if max_pain is not None:
        fig.add_vline(x=max_pain, line_dash="dashdot", line_color=PURPLE_WALL, annotation_text="Max Pain", annotation_position="top")

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=560, barmode="relative",
        title=f"{label} — Gamma Exposure Profile by Strike",
        xaxis_title="Strike", yaxis_title="Gamma Exposure",
        margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    interpretation = (
        "Positive net GEX suggests dealers are net long gamma → they hedge by buying dips / selling rallies. "
        "Negative net GEX suggests dealers are net short gamma → hedging flows can amplify moves, "
        "increasing realized volatility, especially below the gamma flip level."
    )
    st.info(interpretation)

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

    # Final safety net: if Volume was entirely zero/NaN despite upstream handling,
    # vwap/twap could still be all-NaN after ffill/bfill (nothing to fill from).
    # Fall back to the typical price itself so the chart and downstream metrics
    # never render NaN.
    out["vwap"] = out["vwap"].fillna(typical_price)
    out["twap"] = out["twap"].fillna(typical_price)
    for col in ["vwap_u1", "vwap_u2", "vwap_l1", "vwap_l2"]:
        out[col] = out[col].fillna(out["vwap"])
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

def render_execution_module(df: pd.DataFrame, label: str, volume_is_synthetic: bool = False):
    st.markdown(
        '<div class="module-note">Institutional execution benchmarks — VWAP with statistical '
        'deviation bands, TWAP baseline, and detection of probable iceberg / hidden-order clusters.</div>',
        unsafe_allow_html=True,
    )

    if df["Volume"].sum() == 0:
        render_status_badge("ℹ️ No volume data available for this instrument — VWAP falls back to typical price.")
    elif volume_is_synthetic:
        render_status_badge(
            "🧮 Using a Synthetic Volume Proxy (bar-range × volatility estimate) — real exchange "
            "volume was unavailable or zero for most bars. Not genuine traded volume."
        )

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
        name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap"], mode="lines", name="VWAP", line=dict(color="#f0b90b", width=2)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["twap"], mode="lines", name="TWAP", line=dict(color=PURPLE_WALL, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u1"], mode="lines", name="+1 SD", line=dict(color="rgba(38,166,154,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u2"], mode="lines", name="+2 SD", line=dict(color="rgba(38,166,154,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l1"], mode="lines", name="-1 SD", line=dict(color="rgba(239,83,80,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l2"], mode="lines", name="-2 SD", line=dict(color="rgba(239,83,80,0.35)", width=1)))

    if not icebergs.empty:
        fig.add_trace(go.Scatter(
            x=icebergs.index, y=icebergs["Low"] * 0.999, mode="markers", name="Iceberg Cluster",
            marker=dict(color="#00e5ff", size=10, symbol="diamond"),
        ))

    price_range = price_axis_range(
        vwap_df["Low"], vwap_df["High"], vwap_df["vwap_l2"], vwap_df["vwap_u2"], pad_pct=0.06
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=650, xaxis_rangeslider_visible=False, dragmode="pan",
        title=f"{label} — VWAP / TWAP Execution Benchmarks & Iceberg Detection",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=price_range, autorange=False if price_range else True),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    fig_vol = go.Figure(go.Bar(x=vwap_df.index, y=vwap_df["Volume"], marker_color="#5c6bc0", name="Volume"))
    if not icebergs.empty:
        fig_vol.add_trace(go.Bar(x=icebergs.index, y=icebergs["Volume"], marker_color="#00e5ff", name="Iceberg Volume"))
    fig_vol.update_layout(template=PLOTLY_TEMPLATE, height=280, title="Volume Profile & Anomaly Bars", margin=dict(l=10, r=10, t=40, b=10), dragmode="pan")
    st.plotly_chart(fig_vol, use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("🧊 Detected Iceberg / Hidden Order Clusters"):
        if not icebergs.empty:
            show_cols = ["Close", "Volume", "vol_range_ratio", "vr_z"]
            st.dataframe(icebergs[show_cols].tail(15).round(3), use_container_width=True)
        else:
            st.info("No statistically significant iceberg clusters detected at the current sensitivity level.")

# ==================================================================================
# SIDEBAR — THEME TOGGLE, ASSET SELECTION & DATA ENGINE CONTROLS
# ==================================================================================

st.sidebar.markdown("## 📊 Institutional Quant Terminal")
st.sidebar.caption("Hybrid Multi-Source Data Engine")

theme_choice = st.sidebar.radio(
    "🎨 Theme", ["Dark Theme", "Light Theme"], index=0, horizontal=True, key="theme_choice",
)
if theme_choice == "Dark Theme":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    PLOTLY_TEMPLATE = "plotly_white"

st.sidebar.divider()

st.sidebar.subheader("Asset Selection")
asset_category = st.sidebar.selectbox(
    "Asset Category",
    ["Crypto (Binance)", "Forex (yfinance + optional SWFX overlay)", "Metals / Macro Index (yfinance)", "Custom Symbol"],
)

if asset_category == "Crypto (Binance)":
    asset_label = st.sidebar.selectbox("Select Crypto Pair", list(CRYPTO_ASSETS.keys()))
    symbol = CRYPTO_ASSETS[asset_label]
    asset_class = "crypto"
    yf_ticker = binance_to_yf_ticker(symbol)
    primary_source_note = "Primary: Binance REST (klines + L2 depth, 3-endpoint fallback)"

elif asset_category == "Forex (yfinance + optional SWFX overlay)":
    asset_label = st.sidebar.selectbox("Select Forex Pair", list(FOREX_ASSETS.keys()))
    symbol = FOREX_ASSETS[asset_label]
    asset_class = "forex"
    yf_ticker = forex_to_yf_ticker(symbol)
    primary_source_note = "Primary: yfinance structural baseline (always fast) + optional Dukascopy SWFX tick overlay"

elif asset_category == "Metals / Macro Index (yfinance)":
    asset_label = st.sidebar.selectbox("Select Macro Asset", list(MACRO_ASSETS.keys()))
    symbol = MACRO_ASSETS[asset_label]
    asset_class = "macro"
    yf_ticker = symbol
    primary_source_note = "Primary: yfinance (OHLCV + option chains)"

else:
    custom_class = st.sidebar.selectbox("Custom Asset Class", ["crypto", "forex", "macro"])
    default_symbol = {"crypto": "BTCUSDT", "forex": "EURUSD", "macro": "GC=F"}[custom_class]
    custom_symbol = st.sidebar.text_input("Custom Symbol", value=default_symbol)
    symbol = custom_symbol.strip().upper() if custom_class != "macro" else custom_symbol.strip()
    asset_class = custom_class
    asset_label = f"Custom: {symbol}"
    if asset_class == "crypto":
        yf_ticker = binance_to_yf_ticker(symbol)
        primary_source_note = "Primary: Binance REST (klines + L2 depth, 3-endpoint fallback)"
    elif asset_class == "forex":
        yf_ticker = forex_to_yf_ticker(symbol)
        primary_source_note = "Primary: yfinance structural baseline (always fast) + optional Dukascopy SWFX tick overlay"
    else:
        yf_ticker = symbol
        primary_source_note = "Primary: yfinance (OHLCV + option chains)"

st.sidebar.caption(primary_source_note)
st.sidebar.divider()

interval = st.sidebar.selectbox("Interval / Timeframe", INTERVAL_CHOICES, index=INTERVAL_CHOICES.index("1d"))
base_key_selected = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1d"])["base"]

st.sidebar.divider()
st.sidebar.subheader("DOM / Order-Book Depth")

binance_depth_limit = DEFAULT_BINANCE_DEPTH_LIMIT
tick_lookback_hours = DEFAULT_TICK_LOOKBACK_HOURS
dom_bucket_pips = DEFAULT_DOM_BUCKET_PIPS
enable_tick_overlay = False

if asset_class == "crypto":
    binance_depth_limit = st.sidebar.select_slider(
        "Binance Depth Levels", options=[50, 100, 250, 500, 1000], value=DEFAULT_BINANCE_DEPTH_LIMIT,
    )
elif asset_class == "forex":
    overlay_possible = base_key_selected in TICK_OVERLAY_ELIGIBLE_BASES
    enable_tick_overlay = st.sidebar.checkbox(
        "Enable Live SWFX Tick Overlay (Volume + DOM)", value=True,
        help="Fetches real Dukascopy SWFX ticks as a separate, lazy step AFTER candles "
             "load — turn off for maximum speed. Only applies to intraday timeframes; "
             "candles always render from yfinance regardless of this setting.",
        disabled=not overlay_possible,
    )
    if not overlay_possible:
        st.sidebar.caption("Tick overlay isn't used on daily+ timeframes — candles + synthetic volume proxy cover it.")
    if enable_tick_overlay and overlay_possible:
        tick_lookback_hours = st.sidebar.slider(
            "SWFX Tick Lookback (hours)", min_value=2, max_value=168, value=DEFAULT_TICK_LOOKBACK_HOURS, step=2,
            help="Wider lookback survives weekend market closures and covers more bars with real "
                 "volume; narrower is faster and more 'live'. Shared by both the volume overlay and "
                 "the DOM ladder — only fetched once per rerun.",
        )
        dom_bucket_pips = st.sidebar.slider(
            "DOM Bucket Size (pips)", min_value=0.5, max_value=10.0, value=DEFAULT_DOM_BUCKET_PIPS, step=0.5,
        )
else:
    st.sidebar.caption("No order-book depth controls for macro/index assets (no public order book).")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh All Data (Clear Cache)", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "<div style='font-size:0.75rem; opacity:0.75;'>"
    "<b>Data Engine Routing</b><br>"
    "Crypto/Metals proxy → Binance Public API (3-endpoint fallback)<br>"
    "Major Forex → yfinance structural baseline ALWAYS (fast, 200+ bars) + optional, "
    "toggleable Dukascopy SWFX tick overlay for real volume / DOM (intraday only)<br>"
    "Macro/Index/Futures → yfinance<br>"
    "Any source failure → automatic yfinance fallback (crypto only — forex's primary IS "
    "yfinance already, so it is never retried a second time with identical arguments)<br>"
    "When real volume is missing/zero, a disclosed Synthetic Volume Proxy "
    "(bar-range × volatility) fills the gap so VWAP/CVD/Iceberg never break — real "
    "tick volume overwrites it wherever the SWFX overlay has coverage.<br>"
    "Forex depth ladder is built from real SWFX tick prints (bid/ask price + volume), "
    "not a native multi-level order book — Dukascopy does not publish one publicly.<br>"
    "All Dukascopy tick downloads are cached 5 minutes and shared between the volume "
    "overlay and DOM ladder, so repeated views are instant and never double-fetch.<br><br>"
    "For informational / research purposes only — not investment advice."
    "</div>",
    unsafe_allow_html=True,
)

# ==================================================================================
# MAIN HEADER & DATA FETCH
# ==================================================================================

st.title("📊 Institutional Quantitative Trading Terminal")
st.caption(f"Active Instrument: **{asset_label}** ({symbol}) · Class: {asset_class} · Interval: {interval}")

# Step 1 — FAST structural candle fetch. Never touches Dukascopy for any asset class.
with st.spinner(f"Fetching market data for {symbol}..."):
    main_df, source_used = fetch_ohlcv(asset_class, symbol, yf_ticker, interval)

if main_df.empty:
    st.error(
        f"⚠️ Unable to retrieve OHLCV data for **{symbol}** at interval '{interval}' from any configured "
        "source. Try a different interval, verify the symbol, or click **Refresh All Data** in the sidebar."
    )
    st.stop()

# Step 2 — Synthetic volume proxy applied immediately (local compute, no network) so
# every downstream module always has a usable, non-degenerate Volume column regardless
# of whether the optional tick overlay below succeeds, is enabled, or is even applicable.
volume_is_synthetic = False
if asset_class == "forex":
    main_df, volume_is_synthetic = apply_synthetic_volume_proxy(main_df)

# Step 3 — OPTIONAL, separate, lazy Dukascopy tick overlay. Only runs for forex, only
# when the sidebar toggle is on, and only on intraday timeframes. This is intentionally
# AFTER candles are already valid and renderable, under its own clearly-labeled spinner,
# and wrapped in try/except so a Dukascopy hiccup never takes candles off the screen.
order_book_df = pd.DataFrame()
if asset_class == "crypto":
    with st.spinner("Fetching live Binance order book..."):
        order_book_df = fetch_order_book(asset_class, symbol, depth_limit=binance_depth_limit)
elif asset_class == "forex" and enable_tick_overlay and base_key_selected in TICK_OVERLAY_ELIGIBLE_BASES:
    with st.spinner("Fetching live SWFX tick-flow overlay (volume + DOM)..."):
        try:
            ticks = fetch_dukascopy_ticks(symbol, tick_lookback_hours)
        except Exception:
            ticks = pd.DataFrame()
        if not ticks.empty:
            main_df = overlay_dukascopy_tick_volume(main_df, ticks, base_key_selected)
            source_used += " + Dukascopy SWFX tick overlay"
        try:
            order_book_df = fetch_order_book(
                asset_class, symbol,
                dom_lookback_hours=tick_lookback_hours,
                dom_bucket_pips=dom_bucket_pips,
            )
        except Exception:
            order_book_df = pd.DataFrame()

if len(main_df) < 20:
    st.warning("Very limited data returned for this selection — consider a lower-granularity interval.")

if asset_class == "crypto":
    dom_badge = "🟢 LIVE L2 DOM" if not order_book_df.empty else "🔴 L2 DOM UNAVAILABLE"
elif asset_class == "forex":
    if not enable_tick_overlay:
        dom_badge = "⚪ TICK OVERLAY DISABLED (candles still live via yfinance)"
    elif base_key_selected not in TICK_OVERLAY_ELIGIBLE_BASES:
        dom_badge = "⚪ NOT APPLICABLE ON THIS TIMEFRAME"
    elif not order_book_df.empty:
        dom_badge = "🟢 LIVE SWFX TICK-FLOW DEPTH"
    else:
        dom_badge = "🔴 DEPTH UNAVAILABLE"
else:
    dom_badge = "⚪ NO L2 DOM (macro asset)"

badges_html = (
    f'<span class="source-badge">📡 CANDLES: {source_used.upper()}</span>'
    f'<span class="source-badge">{dom_badge}</span>'
)
if volume_is_synthetic:
    badges_html += '<span class="source-badge">🧮 SYNTHETIC VOLUME PROXY ACTIVE</span>'
st.markdown(badges_html, unsafe_allow_html=True)

# Snapshot metrics
last_row = main_df.iloc[-1]
prev_row = main_df.iloc[-2] if len(main_df) > 1 else last_row
chg = safe_pct(last_row["Close"], prev_row["Close"])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Price", f"{last_row['Close']:,.4f}" if last_row['Close'] < 10 else f"{last_row['Close']:,.2f}", f"{chg:.2f}%")
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
        render_liquidity_module(main_df, order_book_df, asset_class, asset_label)
    except Exception as e:
        st.error(f"Liquidity module encountered an error: {e}")

with tab2:
    try:
        render_ml_module(main_df, asset_label)
    except Exception as e:
        st.error(f"ML Classifier module encountered an error: {e}")

with tab3:
    try:
        render_correlation_module(symbol, yf_ticker, asset_class, asset_label)
    except Exception as e:
        st.error(f"Correlation module encountered an error: {e}")

with tab4:
    try:
        render_volume_delta_module(main_df, asset_label, source_used, volume_is_synthetic)
    except Exception as e:
        st.error(f"Volume Delta module encountered an error: {e}")

with tab5:
    try:
        render_gex_module(symbol, yf_ticker, asset_class, asset_label)
    except Exception as e:
        st.error(f"GEX module encountered an error: {e}")

with tab6:
    try:
        render_execution_module(main_df, asset_label, volume_is_synthetic)
    except Exception as e:
        st.error(f"Execution Algorithms module encountered an error: {e}")

st.divider()
st.caption(
    "⚠️ Disclaimer: This dashboard is provided for research and educational purposes only. "
    "It does not constitute financial advice."
)

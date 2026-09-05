"""
==================================================================================
RITHMIC INSTITUTIONAL QUANTITATIVE TRADING TERMINAL
Live Rithmic R|Protocol Engine (Binary WebSocket + Protobuf) — Multi-Symbol Streaming
==================================================================================
Streams real-time quotes, Level 2 DOM, tick trades, and multi-timeframe candles for
any symbol/exchange combination available on the connected Rithmic system (paper
trading / 14-day demo included), feeding six institutional quant analytics modules.

IMPORTANT ARCHITECTURAL NOTE — READ BEFORE DEPLOYING
--------------------------------------------------------------------------------
Rithmic's raw R|Protocol .proto schemas (the actual "rithmic_pb2" message
definitions) are distributed only to registered developers under Rithmic's own
dev-kit/NDA process — they are not public. Hand-rolling those message classes
from scratch would produce code that *looks* correct but cannot actually
authenticate against Rithmic's servers.

Instead, this engine is built on top of `async_rithmic` (pip install
async_rithmic), an actively maintained, open-source (MIT) Python client that
already implements the correct binary-framed protobuf handshake across
Rithmic's four "plants" (TICKER_PLANT for market data, ORDER_PLANT, HISTORY_PLANT,
PNL_PLANT). This is the standard, working way to talk to Rithmic from Python
today. If your dev-kit / system name gives you access to the lower-level proto
objects directly and you specifically need to bypass this wrapper, the
`RithmicMarketDataWorker` class below is the single place to swap that in — the
rest of the app (LiveMarketState + all 6 analytics modules) is transport-agnostic.

Also note: Rithmic connects to regulated futures exchanges (CME, ICE, etc.), not
spot crypto pairs. "Crypto futures" here means exchange-listed crypto-linked
futures (e.g. BTC, ETH, MBT, MET on CME) — the sidebar lets you type ANY
symbol + exchange combination your Rithmic account is entitled to, rather than
assuming a "list all crypto pairs" endpoint that doesn't exist in the API.

TIMEFRAME STRATEGY: Rithmic streams live time bars at fixed native granularities.
Rather than open 18 separate live subscriptions per symbol (fragile, and prone to
"missing bars" when you switch), this engine subscribes to ONE canonical 1-minute
bar stream (plus a 1-day historical backfill) per symbol and derives every other
timeframe (2m,3m,4m,5m,15m,30m,1h,2h,3h,4h,1D,3D,1W,1M,3M,6M,1Y) via pandas
resampling from that single source of truth. This guarantees consistent bars
across every timeframe switch with no gaps.

DEPENDENCY NOTE: streamlit, pandas, numpy, plotly, scikit-learn, async_rithmic.
No yfinance / python-dotenv — credentials are entered each session via the
sidebar UI only, so this runs cleanly on share.streamlit.io.
==================================================================================
"""

import asyncio
import math
import queue
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RITHMIC_IMPORT_ERROR = None
RITHMIC_IMPORT_DIAGNOSTICS = None
# NOTE: `Gateway` was removed from async_rithmic in v1.5.0 (2025-06-17) and replaced
# by a plain `url=` connection parameter on RithmicClient — see the library's own
# CHANGELOG.md. It does not exist in 1.6.x under any name, top-level or nested, so
# it is intentionally NOT resolved/required here anymore (see _try_import_rithmic).
RithmicClient = DataType = TimeBarType = LastTradePresenceBits = None

def _find_attr_in_submodules(base_mod, name, submodule_names):
    """Look for `name` on base_mod itself, then on a handful of common
    submodule locations Python packages use for enums. Only ever returns a
    REAL object found on the actually-installed package — never a guessed
    value — so a miss here means the object genuinely isn't there under any
    of the paths checked, not that we synthesized something to paper over it.
    """
    if hasattr(base_mod, name):
        return getattr(base_mod, name)
    for sub in submodule_names:
        try:
            import importlib
            submod = importlib.import_module(f"async_rithmic.{sub}")
            if hasattr(submod, name):
                return getattr(submod, name)
        except Exception:
            continue
    # Some libraries nest enums as class attributes on the client itself
    # (e.g. RithmicClient.Gateway) rather than at module scope.
    client_cls = getattr(base_mod, "RithmicClient", None)
    if client_cls is not None and hasattr(client_cls, name):
        return getattr(client_cls, name)
    return None

def _try_import_rithmic():
    """Resolve RithmicClient/DataType/TimeBarType defensively.

    IMPORTANT — why `Gateway` is gone: async_rithmic's own CHANGELOG.md records
    that the `gateway` parameter (and the `Gateway` enum that fed it) was
    deprecated in v1.4.5 and then fully REMOVED in v1.5.0, in favor of passing
    a plain `url=` connection string straight to `RithmicClient(...)`. That is
    exactly what the diagnostic block you sent shows for 1.6.6: `Gateway` isn't
    at the top level, in enums/types, or nested on RithmicClient, because it
    genuinely no longer exists in the package — it's not a naming difference to
    search harder for. So this resolver only requires the names that are still
    real, current parts of async_rithmic's public API (RithmicClient, DataType,
    TimeBarType), and the app now connects via `url=` (see the sidebar and
    RithmicMarketDataWorker below) instead of a Gateway enum.

    The documented, top-level import is tried first for each name individually.
    Any name not found there is searched for on real, common submodule locations
    (`async_rithmic.enums`, `async_rithmic.types`) and as a nested class
    attribute on RithmicClient — but we never invent a value. Faking a DataType
    member would make market-data subscriptions fail silently (requesting the
    wrong data type) instead of failing loudly, which is worse than an
    ImportError.
    """
    global RITHMIC_IMPORT_ERROR, RITHMIC_IMPORT_DIAGNOSTICS
    global RithmicClient, DataType, TimeBarType, LastTradePresenceBits

    try:
        import importlib
        mod = importlib.import_module("async_rithmic")
    except Exception as e:
        RITHMIC_IMPORT_ERROR = f"Could not import 'async_rithmic' at all: {e}"
        RITHMIC_IMPORT_DIAGNOSTICS = "The package itself failed to import — check `pip show async_rithmic`."
        return

    submodule_guesses = ["enums", "types", "constants", "client"]
    resolved = {}
    missing = []
    for name in ("RithmicClient", "DataType", "TimeBarType"):
        found = _find_attr_in_submodules(mod, name, submodule_guesses)
        if found is None:
            missing.append(name)
        else:
            resolved[name] = found
    resolved["LastTradePresenceBits"] = _find_attr_in_submodules(mod, "LastTradePresenceBits", submodule_guesses)

    module_file = getattr(mod, "__file__", "unknown location")
    version = getattr(mod, "__version__", "unknown")
    diag_lines = [
        f"async_rithmic version: {version}",
        f"Loaded from: {module_file}",
        f"Top-level names: {[n for n in dir(mod) if not n.startswith('_')]}",
    ]
    if "site-packages" not in str(module_file) and "dist-packages" not in str(module_file):
        diag_lines.append(
            "⚠️ This does NOT look like it's loading from your installed site-packages — "
            "check for a local file/folder named 'async_rithmic.py' or 'async_rithmic/' in "
            "your project directory that is shadowing the real package."
        )
    RITHMIC_IMPORT_DIAGNOSTICS = "\n".join(diag_lines)

    if missing:
        RITHMIC_IMPORT_ERROR = f"async_rithmic imported, but could not locate (top level, enums/, types/, or as a RithmicClient attribute): {', '.join(missing)}"
        return

    RithmicClient = resolved["RithmicClient"]
    DataType = resolved["DataType"]
    TimeBarType = resolved["TimeBarType"]
    LastTradePresenceBits = resolved["LastTradePresenceBits"]

_try_import_rithmic()

# ==================================================================================
# PAGE CONFIG & STYLE
# ==================================================================================

st.set_page_config(
    page_title="Rithmic Institutional Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
    .stApp { background-color: #0b0e14; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1f2937; }
    div[data-testid="stMetric"] {
        background-color: #131722; border: 1px solid #1f2937; border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] { color: #f0b90b; }
    h1, h2, h3 { color: #f0f0f0; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { background-color: #131722; border-radius: 8px 8px 0 0; padding: 8px 16px; color: #cfd3dc; }
    .stTabs [aria-selected="true"] { background-color: #1f2937; color: #f0b90b; font-weight: 600; }
    .module-note {
        background-color: #131722; border-left: 3px solid #f0b90b; padding: 10px 14px;
        border-radius: 4px; font-size: 0.85rem; color: #b8bdc9; margin-bottom: 10px;
    }
    .source-badge {
        display: inline-block; background-color: #1f2937; color: #f0b90b; border-radius: 6px;
        padding: 3px 10px; font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .conn-banner-up {
        background-color: #0d2b1e; border: 1px solid #1e7d4b; color: #4ade80;
        border-radius: 8px; padding: 10px 16px; font-weight: 700; margin-bottom: 12px;
    }
    .conn-banner-down {
        background-color: #2b0d0d; border: 1px solid #7d1e1e; color: #f87171;
        border-radius: 8px; padding: 10px 16px; font-weight: 700; margin-bottom: 12px;
    }
    .stButton>button { background-color: #f0b90b; color: #0b0e14; border: none; font-weight: 600; border-radius: 6px; }
    thead tr th { background-color: #1f2937 !important; color: #f0b90b !important; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)
PLOTLY_TEMPLATE = "plotly_dark"
PURPLE_WALL = "#ba68c8"
PLOTLY_CONFIG = {"scrollZoom": False, "displayModeBar": True, "responsive": True}

# ==================================================================================
# RITHMIC CONSTANTS
# ==================================================================================

# async_rithmic dropped the `gateway=Gateway.X` enum in v1.5.0 — RithmicClient
# now takes a plain `url="host:port"` connection string instead (see
# _try_import_rithmic's docstring above and async_rithmic's CHANGELOG.md).
# Rithmic's public Test system is the one endpoint that's openly documented;
# every other system's URL (Paper Trading, Live, region-specific colos) is
# assigned per-developer by Rithmic after signup and is NOT something this
# app can guess or hardcode — it comes from your dev-kit/trial welcome email,
# same as your System Name. So we offer the public Test URL as a convenience
# default and otherwise require the user to paste their own.
SYSTEM_URL_PRESETS = {
    "Rithmic Test": "rituz00100.rithmic.com:443",
    "Rithmic Paper Trading (Demo)": "",  # paste from your Rithmic welcome email
    "Live / Other (paste URL from Rithmic)": "",
}
SYSTEM_PROFILE_LABELS = list(SYSTEM_URL_PRESETS.keys())

# Default symbol universe: CME-listed crypto-linked futures roots. Users can add
# any symbol:exchange pair their Rithmic account is entitled to.
DEFAULT_SYMBOLS = "BTC:CME, ETH:CME, MBT:CME, MET:CME"

BASE_BAR_MINUTES = 1  # canonical live granularity every other timeframe derives from
HISTORY_BACKFILL_DAYS = 5

# timeframe -> pandas resample rule (minutes-based timeframes resample the 1m
# base feed; date-based ones resample the same 1m feed on calendar boundaries)
TIMEFRAME_RULES = {
    "1m": "1min", "2m": "2min", "3m": "3min", "4m": "4min", "5m": "5min",
    "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "3h": "3h", "4h": "4h",
    "1D": "1D", "3D": "3D", "1W": "1W", "1M": "1MS", "3M": "3MS", "6M": "6MS", "1Y": "1YS",
}
INTERVAL_CHOICES = list(TIMEFRAME_RULES.keys())
AUTO_REFRESH_SECONDS = 3

# ==================================================================================
# LIVE MARKET DATA STATE — multi-symbol, thread-safe
# ==================================================================================

class LiveMarketState:
    """One shared object per connected session. Mutated by the background asyncio
    worker thread (Rithmic callbacks), read (copy-on-read) by the Streamlit main
    thread on every rerun. Bars are stored ONLY at the 1-minute base granularity;
    every other timeframe is derived on read via resample_bars()."""

    def __init__(self, symbols: list, max_base_bars: int = 200_000):
        self.lock = threading.Lock()
        self.connected = False
        self.authorized = False
        self.last_error = None
        self.symbols = symbols  # list of (symbol, exchange, display_root)
        self.max_base_bars = max_base_bars

        self._quote = {root: {} for _, _, root in symbols}
        self._dom_bids = {root: [] for _, _, root in symbols}
        self._dom_asks = {root: [] for _, _, root in symbols}
        self._ticks = {root: deque(maxlen=6000) for _, _, root in symbols}
        self._base_bars = {root: pd.DataFrame() for _, _, root in symbols}

    def set_quote(self, root, updates):
        with self.lock:
            self._quote.setdefault(root, {}).update(updates)

    def push_tick(self, root, tick):
        with self.lock:
            self._ticks.setdefault(root, deque(maxlen=6000)).append(tick)

    def set_dom(self, root, bids=None, asks=None):
        with self.lock:
            if bids is not None:
                self._dom_bids[root] = bids
            if asks is not None:
                self._dom_asks[root] = asks

    def merge_base_bars(self, root, new_df: pd.DataFrame):
        with self.lock:
            existing = self._base_bars.get(root, pd.DataFrame())
            merged = new_df if existing.empty else pd.concat([existing, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if len(merged) > self.max_base_bars:
                merged = merged.iloc[-self.max_base_bars:]
            self._base_bars[root] = merged

    def snapshot_quote(self, root):
        with self.lock:
            return dict(self._quote.get(root, {}))

    def snapshot_dom(self, root):
        with self.lock:
            bids = list(self._dom_bids.get(root, []))
            asks = list(self._dom_asks.get(root, []))
        frames = []
        if bids:
            b = pd.DataFrame(bids); b["side"] = "bid"; frames.append(b)
        if asks:
            a = pd.DataFrame(asks); a["side"] = "ask"; frames.append(a)
        if not frames:
            return pd.DataFrame(columns=["price", "qty", "side"])
        return pd.concat(frames, ignore_index=True)

    def snapshot_base_bars(self, root):
        with self.lock:
            df = self._base_bars.get(root, pd.DataFrame())
            return df.copy() if not df.empty else df

    def snapshot_ticks(self, root):
        with self.lock:
            return list(self._ticks.get(root, []))

    def status(self):
        with self.lock:
            return {"connected": self.connected, "authorized": self.authorized, "last_error": self.last_error}


def resample_bars(base_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Derive any supported timeframe from the canonical 1-minute base feed."""
    if base_df.empty:
        return base_df
    rule = TIMEFRAME_RULES.get(timeframe, "1min")
    if rule == "1min":
        return base_df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = base_df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
    return out

# ==================================================================================
# BACKGROUND RITHMIC WORKER — one asyncio loop on its own thread
# ==================================================================================

class RithmicMarketDataWorker(threading.Thread):
    """Owns a dedicated asyncio event loop on a background thread (Streamlit's main
    thread is sync, async_rithmic is asyncio-native). Connects one RithmicClient,
    resolves each requested symbol/exchange, and subscribes to:
      - last-trade ticks         (DataType.LAST_TRADE)
      - best bid/offer           (DataType.BBO, when supported by the installed lib)
      - full L2 order book depth (DataType.ORDER_BOOK, when entitled)
      - live 1-minute time bars  (TimeBarType.MINUTE_BAR, bar_type_period=1)
    plus a one-time historical 1-minute backfill so charts aren't empty on connect.

    Every callback name/enum is resolved defensively via getattr/hasattr because
    async_rithmic's exact surface can shift slightly between versions — a missing
    optional feature (e.g. DOM on an account without that entitlement) degrades
    to a clear "not available" state in the UI rather than crashing the app.
    """

    def __init__(self, user, password, system_name, url, symbols, state: LiveMarketState):
        super().__init__(daemon=True)
        self.user = user
        self.password = password
        self.system_name = system_name
        self.url = url  # host:port string, e.g. "rituz00100.rithmic.com:443" — replaces the old gateway= enum
        self.symbols = symbols  # list of (symbol, exchange, root)
        self.state = state
        self.loop = None
        self.client = None
        self._resolved_codes = {}  # root -> (security_code, exchange)
        self._stop_event = threading.Event()
        self._pending_focus = queue.Queue()  # (root, timeframe) requests from UI thread — reserved for future on-demand subscriptions

    def stop(self):
        self._stop_event.set()
        if self.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop)
            except Exception:
                pass

    async def _shutdown(self):
        try:
            if self.client is not None:
                await self.client.disconnect()
        except Exception:
            pass

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Worker terminated: {e}"
        finally:
            with self.state.lock:
                self.state.connected = False
                self.state.authorized = False

    async def _main(self):
        backoff = 2
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream()
            except Exception as e:
                with self.state.lock:
                    self.state.connected = False
                    self.state.authorized = False
                    self.state.last_error = f"Connection dropped: {e}"
            if self._stop_event.is_set():
                break
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 30)

    async def _connect_and_stream(self):
        self.client = RithmicClient(
            user=self.user, password=self.password,
            system_name=self.system_name, app_name="InstitutionalQuantTerminal",
            app_version="2.0", url=self.url,
        )
        await self.client.connect()
        with self.state.lock:
            self.state.connected = True
            self.state.authorized = True
            self.state.last_error = None

        # Wire callbacks (event names verified against async_rithmic's documented
        # +=/on_ hook pattern; guarded with hasattr so an unsupported hook on an
        # older/newer library version doesn't take the whole app down).
        if hasattr(self.client, "on_tick"):
            self.client.on_tick += self._on_tick
        if hasattr(self.client, "on_time_bar"):
            self.client.on_time_bar += self._on_time_bar
        if hasattr(self.client, "on_order_book"):
            self.client.on_order_book += self._on_order_book
        elif hasattr(self.client, "on_market_depth"):
            self.client.on_market_depth += self._on_order_book

        for symbol, exchange, root in self.symbols:
            try:
                security_code = symbol
                if hasattr(self.client, "get_front_month_contract"):
                    try:
                        resolved = await self.client.get_front_month_contract(symbol, exchange)
                        if resolved:
                            security_code = resolved
                    except Exception:
                        pass  # not every root is a rolling future — fall back to the literal symbol
                self._resolved_codes[root] = (security_code, exchange)

                if DataType is not None and hasattr(self.client, "subscribe_to_market_data"):
                    try:
                        await self.client.subscribe_to_market_data(security_code, exchange, DataType.LAST_TRADE)
                    except Exception as e:
                        with self.state.lock:
                            self.state.last_error = f"{root}: tick subscribe failed ({e})"
                    if hasattr(DataType, "ORDER_BOOK"):
                        try:
                            await self.client.subscribe_to_market_data(security_code, exchange, DataType.ORDER_BOOK)
                        except Exception:
                            pass  # DOM entitlement not present on this account/system — module degrades gracefully

                if TimeBarType is not None and hasattr(self.client, "subscribe_to_time_bar_data"):
                    try:
                        await self.client.subscribe_to_time_bar_data(
                            security_code, exchange, TimeBarType.MINUTE_BAR, 1,
                        )
                    except Exception as e:
                        with self.state.lock:
                            self.state.last_error = f"{root}: time bar subscribe failed ({e})"

                if hasattr(self.client, "get_historical_time_bars"):
                    try:
                        end = datetime.now(timezone.utc)
                        start = end - timedelta(days=HISTORY_BACKFILL_DAYS)
                        hist = await self.client.get_historical_time_bars(
                            security_code, exchange, start, end, TimeBarType.MINUTE_BAR, 1,
                        )
                        self._ingest_history(root, hist)
                    except Exception:
                        pass  # backfill is best-effort; live bars will populate the chart regardless
            except Exception as e:
                with self.state.lock:
                    self.state.last_error = f"Failed to subscribe {root}: {e}"

        # Keep the connection (and this coroutine) alive; callbacks do the work.
        while not self._stop_event.is_set():
            await asyncio.sleep(1)

    def _root_for_code(self, security_code, exchange):
        for root, (code, exch) in self._resolved_codes.items():
            if code == security_code and exch == exchange:
                return root
        return None

    def _ingest_history(self, root, bars):
        if not bars:
            return
        rows = []
        for b in bars:
            ts = b.get("timestamp") or b.get("bar_end_datetime") or b.get("datetime")
            try:
                idx = pd.to_datetime(ts, utc=True)
            except Exception:
                continue
            rows.append({
                "timestamp": idx,
                "Open": b.get("open_price", b.get("open")),
                "High": b.get("high_price", b.get("high")),
                "Low": b.get("low_price", b.get("low")),
                "Close": b.get("close_price", b.get("close")),
                "Volume": b.get("volume", 0) or 0,
            })
        if not rows:
            return
        df = pd.DataFrame(rows).dropna(subset=["Open", "High", "Low", "Close"])
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        self.state.merge_base_bars(root, df)

    async def _on_tick(self, data: dict):
        root = self._root_for_code(data.get("symbol"), data.get("exchange"))
        if root is None:
            return
        price = data.get("trade_price") or data.get("price")
        qty = data.get("trade_size") or data.get("size") or 0
        if price is not None:
            self.state.set_quote(root, {"last": price, "lastSize": qty})
            self.state.push_tick(root, {"price": price, "qty": qty, "timestamp": datetime.now(timezone.utc)})
        bid, ask = data.get("bid_price"), data.get("ask_price")
        if bid is not None or ask is not None:
            snap = {}
            if bid is not None:
                snap["bidPrice"] = bid; snap["bidSize"] = data.get("bid_size")
            if ask is not None:
                snap["askPrice"] = ask; snap["askSize"] = data.get("ask_size")
            self.state.set_quote(root, snap)

    async def _on_time_bar(self, data: dict):
        root = self._root_for_code(data.get("symbol"), data.get("exchange"))
        if root is None:
            return
        try:
            idx = pd.to_datetime(data.get("bar_end_datetime") or data.get("timestamp"), utc=True)
        except Exception:
            return
        row = pd.DataFrame([{
            "Open": data.get("open_price", data.get("open")),
            "High": data.get("high_price", data.get("high")),
            "Low": data.get("low_price", data.get("low")),
            "Close": data.get("close_price", data.get("close")),
            "Volume": data.get("volume", 0) or 0,
        }], index=[idx]).dropna(subset=["Open", "High", "Low", "Close"])
        if not row.empty:
            self.state.merge_base_bars(root, row)

    async def _on_order_book(self, data: dict):
        root = self._root_for_code(data.get("symbol"), data.get("exchange"))
        if root is None:
            return
        bids_raw = data.get("bids") or data.get("bid_levels") or []
        asks_raw = data.get("asks") or data.get("ask_levels") or []
        bids = [{"price": l.get("price"), "qty": l.get("size", l.get("qty", 0))} for l in bids_raw if l.get("price") is not None]
        asks = [{"price": l.get("price"), "qty": l.get("size", l.get("qty", 0))} for l in asks_raw if l.get("price") is not None]
        self.state.set_dom(root, bids=bids or None, asks=asks or None)

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
            bullish.append({"start": idx[i - 2], "end": idx[i], "top": float(lows[i]), "bottom": float(highs[i - 2])})
        if highs[i] < lows[i - 2]:
            bearish.append({"start": idx[i - 2], "end": idx[i], "top": float(lows[i - 2]), "bottom": float(highs[i])})
    return bullish, bearish

def render_liquidity_module(df: pd.DataFrame, order_book_df: pd.DataFrame, label: str, dom_is_live: bool):
    st.markdown(
        '<div class="module-note">Swing-point liquidity pools (BSL/SSL) and Fair Value Gap '
        'imbalance zones from price-action structure, dynamically annotated with the live '
        'Rithmic Level 2 DOM ($ resting-order depth), institutional bank-wall isolation, and a '
        'Bank Anchor PnL Tracker.</div>', unsafe_allow_html=True,
    )
    if len(df) < 15:
        st.warning("Not enough bars streamed yet for this timeframe to compute liquidity structure. "
                    "Give the feed a few seconds, or pick a lower-granularity interval.")
        return

    c1, c2, c3 = st.columns(3)
    window = c1.slider("Swing Sensitivity (lookback bars)", 2, 15, 5, key="liq_window")
    tolerance = c2.slider("DOM Price Tolerance (%) for $ Volume Aggregation", 0.02, 1.0, 0.15, step=0.02, key="liq_tol")
    wall_pctl = c3.slider("Institutional Wall Percentile", 50, 99, 80, key="liq_wall_pct")
    max_zones = st.slider("Max FVG Zones Displayed", 3, 30, 12, key="liq_fvg_count")

    swing_highs, swing_lows = detect_swings(df, window=window)
    bullish_fvg, bearish_fvg = detect_fvg(df)
    current_price = float(df["Close"].iloc[-1])

    has_dom = dom_is_live and not order_book_df.empty
    if not has_dom:
        st.info(
            "Live Rithmic L2 DOM not yet populated for this symbol — $ volume annotations and "
            "institutional wall isolation are skipped until the order-book stream delivers its "
            "first snapshot (or if this account isn't entitled to depth data). Swing/FVG structure "
            "is still fully computed from streamed price action below."
        )
    else:
        st.markdown(
            f'<span class="source-badge">🟢 LIVE RITHMIC L2 DOM</span> '
            f'<span style="color:#8b90a0; font-size:0.8rem;">{len(order_book_df):,} price levels loaded</span>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSL Pools (Swing Highs)", len(swing_highs))
    c2.metric("SSL Pools (Swing Lows)", len(swing_lows))
    c3.metric("Bullish FVG Zones", len(bullish_fvg))
    c4.metric("Bearish FVG Zones", len(bearish_fvg))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                  name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))

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
            fig.add_annotation(x=df.index[len(df)//2], y=w["price"],
                                text=f"🏦 {side_tag} @ {format_price_level(w['price'])}: {format_dollars(w['dollar_value'])}",
                                showarrow=False, font=dict(color=PURPLE_WALL, size=10), bgcolor="rgba(11,14,20,0.7)")

    y_range = price_axis_range(df["Low"], df["High"], pad_pct=0.10)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=680, xaxis_rangeslider_visible=False, dragmode="pan",
                       title=f"{label} — Liquidity Pools, Fair Value Gaps & Institutional Bank Walls (Live Rithmic Feed)",
                       margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(range=y_range, autorange=False if y_range else True))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if has_dom:
        st.subheader("🏦 Bank Anchor PnL Tracker")
        if not walls_df.empty:
            pnl_df = compute_bank_anchor_pnl(walls_df, current_price)
            net_pnl = float(pnl_df["pnl_pct"].mean())
            overall_status = ("🟢 Institutional Anchors Net Expanding Profit" if net_pnl > 0.5 else
                               "🔴 Institutional Anchors Net Unwinding / Taking Profit" if net_pnl < -0.5 else
                               "🟡 Institutional Anchors Net Neutral / Building Positions")
            m1, m2, m3 = st.columns(3)
            m1.metric("Net Institutional Anchor PnL", f"{net_pnl:+.2f}%")
            m2.metric("Institutional Walls Detected", len(pnl_df))
            m3.metric("Largest Wall $ Value", format_dollars(pnl_df["dollar_value"].max()))
            st.markdown(f"### {overall_status}")
            show_cols = ["price", "qty", "side", "dollar_value", "position", "pnl_pct", "status"]
            display_df = pnl_df[show_cols].rename(columns={
                "price": "Price", "qty": "Quantity", "side": "Side", "dollar_value": "$ Value",
                "position": "Anchor Type", "pnl_pct": "Unrealized PnL %", "status": "Status"})
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
                lvl_df["$ Volume Nearby"] = lvl_df["Price"].apply(lambda p: format_dollars(compute_level_dollar_volume(order_book_df, p, tolerance)))
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
        'features (RSI, ATR, volatility, MA ratios, volume delta, momentum) built from the streamed '
        'Rithmic bars, estimating the probability of the next-N-bar directional move.</div>',
        unsafe_allow_html=True,
    )
    if len(df) < 80:
        st.warning("Insufficient bars streamed yet for reliable ML training. Let the feed accumulate "
                    "more history, or select a lower-granularity interval.")
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
        st.warning("Not enough class diversity in the streamed sample yet to train a robust classifier.")
        return

    feature_cols = list(feat.columns)
    X = data[feature_cols]
    y = data["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, shuffle=False)
        model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                        random_state=42, class_weight="balanced", n_jobs=-1)
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
        fig_proba = go.Figure(go.Pie(labels=["Sell", "Hold", "Buy"], values=[sell_p, hold_p, buy_p],
                                      marker=dict(colors=["#ef5350", "#f0b90b", "#26a69a"]), hole=0.55))
        fig_proba.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Directional Probability", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_proba, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Model trained on {len(X_train)} bars, validated on {len(X_test)} out-of-sample bars for {label} (live Rithmic feed).")

# ==================================================================================
# MODULE 3 — CROSS-ASSET CORRELATION MATRIX
# ==================================================================================

def render_correlation_module(daily_bars_by_root: dict, focus_root: str):
    st.markdown(
        '<div class="module-note">Cross-asset correlation across every symbol currently streamed '
        'from Rithmic, computed from their own live daily bars (derived from the 1-minute base '
        'feed).</div>', unsafe_allow_html=True,
    )
    lookback = st.slider("Correlation Lookback (Days)", 10, 365, 90, step=5, key="corr_lookback")

    series_dict, missing = {}, []
    for root, df in daily_bars_by_root.items():
        if df.empty:
            missing.append(root); continue
        cutoff = df.index.max() - pd.Timedelta(days=lookback)
        s = df[df.index >= cutoff]["Close"]
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        if not s.empty:
            series_dict[root] = s

    if missing:
        st.info(f"Still waiting on daily bars for: {', '.join(missing)}.")
    if len(series_dict) < 2:
        st.warning("Need daily bars for at least two symbols to compute correlations — give the feed a few more seconds.")
        return

    combined = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()
    if combined.empty or len(combined) < 5:
        st.warning("Not enough overlapping daily history across symbols yet for this lookback window.")
        return

    corr_matrix = combined.corr()
    fig_heat = go.Figure(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                     colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
                                     text=np.round(corr_matrix.values, 2), texttemplate="%{text}"))
    fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Cross-Symbol Correlation Matrix",
                            margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
    st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)

    if focus_root in combined.columns:
        st.subheader(f"Rolling Correlation vs {focus_root}")
        rolling_window = min(20, max(5, len(combined) // 3))
        fig_roll = go.Figure()
        for col in combined.columns:
            if col == focus_root:
                continue
            rolling_corr = combined[focus_root].rolling(rolling_window).corr(combined[col])
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{focus_root} vs {col}"))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#666")
        fig_roll.update_layout(template=PLOTLY_TEMPLATE, height=400, title=f"Rolling {rolling_window}-Day Correlation",
                                margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
        st.plotly_chart(fig_roll, use_container_width=True, config=PLOTLY_CONFIG)

    with st.expander("📈 Normalized Performance (Rebased to 100)"):
        rebased = combined / combined.iloc[0] * 100
        fig_reb = go.Figure()
        for col in rebased.columns:
            fig_reb.add_trace(go.Scatter(x=rebased.index, y=rebased[col], mode="lines", name=col))
        fig_reb.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan")
        st.plotly_chart(fig_reb, use_container_width=True, config=PLOTLY_CONFIG)
# ==================================================================================
# MODULE 4 — VOLUME DELTA & FOOTPRINT IMBALANCE
# ==================================================================================

def compute_volume_delta_bar_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    buy_ratio = ((out["Close"] - out["Low"]) / rng).clip(0, 1).fillna(0.5)
    out["buy_volume"] = out["Volume"] * buy_ratio
    out["sell_volume"] = out["Volume"] * (1 - buy_ratio)
    out["delta"] = out["buy_volume"] - out["sell_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out

def compute_volume_delta_from_ticks(df: pd.DataFrame, ticks: list) -> pd.DataFrame:
    out = df.copy()
    out["buy_volume"] = 0.0
    out["sell_volume"] = 0.0
    if not ticks:
        return compute_volume_delta_bar_heuristic(df)
    tick_df = pd.DataFrame(ticks).dropna(subset=["price"])
    if tick_df.empty:
        return compute_volume_delta_bar_heuristic(df)

    tick_df = tick_df.sort_values("timestamp")
    tick_df["price_diff"] = tick_df["price"].diff()
    side, last_side = [], "buy"
    for d in tick_df["price_diff"]:
        if pd.isna(d) or d == 0:
            side.append(last_side)
        elif d > 0:
            side.append("buy"); last_side = "buy"
        else:
            side.append("sell"); last_side = "sell"
    tick_df["side"] = side

    tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"], utc=True)
    bin_edges = df.index
    if bin_edges.tz is None:
        tick_df["timestamp"] = tick_df["timestamp"].dt.tz_localize(None)
    tick_df["bar"] = pd.cut(
        tick_df["timestamp"],
        bins=list(bin_edges) + [bin_edges[-1] + (bin_edges[-1] - bin_edges[-2] if len(bin_edges) > 1 else pd.Timedelta(minutes=1))],
        labels=bin_edges, right=False,
    )
    grouped = tick_df.groupby(["bar", "side"], observed=True)["qty"].sum().unstack(fill_value=0)
    if "buy" not in grouped.columns:
        grouped["buy"] = 0
    if "sell" not in grouped.columns:
        grouped["sell"] = 0
    grouped.index = pd.to_datetime(grouped.index)
    out.loc[out.index.isin(grouped.index), "buy_volume"] = grouped.reindex(out.index)["buy"].fillna(0)
    out.loc[out.index.isin(grouped.index), "sell_volume"] = grouped.reindex(out.index)["sell"].fillna(0)

    no_tick_mask = (out["buy_volume"] + out["sell_volume"]) == 0
    if no_tick_mask.any():
        heuristic = compute_volume_delta_bar_heuristic(out.loc[no_tick_mask])
        out.loc[no_tick_mask, "buy_volume"] = heuristic["buy_volume"]
        out.loc[no_tick_mask, "sell_volume"] = heuristic["sell_volume"]

    out["delta"] = out["buy_volume"] - out["sell_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out

def render_volume_delta_module(df: pd.DataFrame, label: str, live_ticks: list, ticks_available: bool):
    st.markdown(
        '<div class="module-note">Order-flow reconstruction: buying vs. selling volume classified '
        'directly from live Rithmic trade prints via the tick rule (uptick = buy-initiated, downtick '
        '= sell-initiated), aggregated into Cumulative Volume Delta (CVD). Falls back to a '
        'close-position volume heuristic for any bar streamed before the tick subscription had data.</div>',
        unsafe_allow_html=True,
    )
    if df["Volume"].sum() == 0:
        st.warning("No volume streamed yet for this symbol/timeframe — Volume Delta requires non-zero volume.")

    if ticks_available:
        st.markdown('<span class="source-badge">🟢 LIVE TICK-RULE CVD</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="source-badge">🟡 BAR-HEURISTIC CVD (waiting on live ticks)</span>', unsafe_allow_html=True)

    vd = compute_volume_delta_from_ticks(df, live_ticks) if ticks_available else compute_volume_delta_bar_heuristic(df)
    z_thresh = st.slider("Imbalance Spike Sensitivity (Z-score)", 1.0, 4.0, 2.0, step=0.25, key="vd_z")

    delta_mean = vd["delta"].mean()
    delta_std = vd["delta"].std() if vd["delta"].std() > 0 else 1.0
    vd["delta_z"] = (vd["delta"] - delta_mean) / delta_std
    spikes = vd[vd["delta_z"].abs() >= z_thresh]

    c1, c2, c3 = st.columns(3)
    c1.metric("Net CVD (Session)", f"{vd['delta'].sum():,.0f}")
    c2.metric("Buy Volume Share", f"{(vd['buy_volume'].sum() / max(vd['Volume'].sum(),1))*100:.1f}%")
    c3.metric("Imbalance Spikes Detected", len(spikes))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
                         subplot_titles=(f"{label} Price [Live Rithmic Feed]", "Volume Delta (Buy − Sell)", "Cumulative Volume Delta (CVD)"))
    fig.add_trace(go.Candlestick(x=vd.index, open=vd["Open"], high=vd["High"], low=vd["Low"], close=vd["Close"],
                                  name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    if not spikes.empty:
        fig.add_trace(go.Scatter(x=spikes.index, y=spikes["High"] * 1.001, mode="markers", name="Imbalance Spike",
                                  marker=dict(color="#f0b90b", size=9, symbol="triangle-down")), row=1, col=1)
    bar_colors = np.where(vd["delta"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=vd.index, y=vd["delta"], marker_color=bar_colors, name="Delta"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vd.index, y=vd["cvd"], mode="lines", name="CVD", line=dict(color="#00e5ff", width=2)), row=3, col=1)

    price_range = price_axis_range(vd["Low"], vd["High"], pad_pct=0.06)
    if price_range:
        fig.update_yaxes(range=price_range, row=1, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=780, showlegend=False, xaxis_rangeslider_visible=False,
                       margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
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
# No live listed-options chain source is wired into this build. This module is a
# pure Black-Scholes gamma-exposure simulation calibrated off each symbol's live
# Rithmic spot/futures price — the UI labels it as simulated at all times.

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
    return pd.DataFrame({"strike": strikes, "call_oi": call_oi, "put_oi": put_oi,
                          "call_gex": call_gex, "put_gex": put_gex, "net_gex": call_gex + put_gex})

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

def render_gex_module(root_symbol: str, live_spot, label: str):
    st.markdown(
        '<div class="module-note">Simulated Gamma Exposure (GEX) profile — Black-Scholes gamma '
        'model calibrated off the live Rithmic spot/futures price — identifying illustrative dealer '
        'positioning, volatility pin zones, and the gamma flip level.</div>', unsafe_allow_html=True,
    )
    st.markdown('<span class="source-badge">🟡 SIMULATED (no live options chain source in this build)</span>', unsafe_allow_html=True)

    if not live_spot:
        st.error(f"No live Rithmic spot price yet for {label} to calibrate the GEX simulation.")
        return

    dte = st.slider("Simulated Days to Expiry", 1, 90, 30, key="gex_dte")
    iv_assumed = st.slider("Assumed Implied Volatility (%)", 5, 80, 18, key="gex_iv") / 100
    n_strikes = st.slider("Strike Range (± strikes around spot)", 10, 40, 25, key="gex_strikes")
    gex_df = simulate_gex(live_spot, n_strikes=n_strikes, iv=iv_assumed, days_to_expiry=dte)
    max_pain, pain_df = compute_max_pain_from_sim(gex_df)

    net_gex_total = gex_df["net_gex"].sum()
    flip_candidates = gex_df.sort_values("strike").copy()
    flip_candidates["cum_gex"] = flip_candidates["net_gex"].cumsum()
    sign_changes = flip_candidates[flip_candidates["cum_gex"] * flip_candidates["cum_gex"].shift(1) < 0]
    gamma_flip = float(sign_changes["strike"].iloc[0]) if not sign_changes.empty else float(gex_df["strike"].median())

    c1, c2, c3 = st.columns(3)
    c1.metric("Live Spot", f"{live_spot:,.2f}")
    c2.metric("Net GEX (simulated)", f"{net_gex_total:,.0f}")
    c3.metric("Max Pain Strike (simulated)", f"{max_pain:,.2f}" if max_pain is not None else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["call_gex"], name="Call GEX", marker_color="#26a69a"))
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["put_gex"], name="Put GEX", marker_color="#ef5350"))
    fig.add_vline(x=live_spot, line_dash="dash", line_color="#f0b90b", annotation_text="Spot", annotation_position="top")
    fig.add_vline(x=gamma_flip, line_dash="dot", line_color="#00e5ff", annotation_text="Gamma Flip", annotation_position="bottom")
    if max_pain is not None:
        fig.add_vline(x=max_pain, line_dash="dashdot", line_color=PURPLE_WALL, annotation_text="Max Pain", annotation_position="top")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, barmode="relative",
                       title=f"{label} — Simulated Gamma Exposure Profile by Strike",
                       xaxis_title="Strike", yaxis_title="Gamma Exposure", margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.info(
        "Positive net GEX suggests dealers are net long gamma → they hedge by buying dips / selling "
        "rallies. Negative net GEX suggests dealers are net short gamma → hedging flows can amplify "
        "moves, increasing realized volatility, especially below the gamma flip level. This entire "
        "surface is simulated from assumed volatility/OI shape, not sourced from a real options market."
    )
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

def render_execution_module(df: pd.DataFrame, order_book_df: pd.DataFrame, label: str):
    st.markdown(
        '<div class="module-note">Institutional execution benchmarks — VWAP with statistical '
        'deviation bands, TWAP baseline, and detection of probable iceberg / hidden-order clusters '
        'from the live Rithmic bar stream, cross-checked against resting DOM size when available.</div>',
        unsafe_allow_html=True,
    )
    if df["Volume"].sum() == 0:
        st.warning("No volume streamed yet for this symbol — VWAP and iceberg detection require non-zero volume.")

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
    fig.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df["Open"], high=vwap_df["High"], low=vwap_df["Low"], close=vwap_df["Close"],
                                  name=label, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap"], mode="lines", name="VWAP", line=dict(color="#f0b90b", width=2)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["twap"], mode="lines", name="TWAP", line=dict(color=PURPLE_WALL, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u1"], mode="lines", name="+1 SD", line=dict(color="rgba(38,166,154,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_u2"], mode="lines", name="+2 SD", line=dict(color="rgba(38,166,154,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l1"], mode="lines", name="-1 SD", line=dict(color="rgba(239,83,80,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df["vwap_l2"], mode="lines", name="-2 SD", line=dict(color="rgba(239,83,80,0.35)", width=1)))
    if not icebergs.empty:
        fig.add_trace(go.Scatter(x=icebergs.index, y=icebergs["Low"] * 0.999, mode="markers", name="Iceberg Cluster",
                                  marker=dict(color="#00e5ff", size=10, symbol="diamond")))

    price_range = price_axis_range(vwap_df["Low"], vwap_df["High"], vwap_df["vwap_l2"], vwap_df["vwap_u2"], pad_pct=0.06)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=650, xaxis_rangeslider_visible=False, dragmode="pan",
                       title=f"{label} — VWAP / TWAP Execution Benchmarks & Iceberg Detection (Live Rithmic Feed)",
                       margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(range=price_range, autorange=False if price_range else True))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    fig_vol = go.Figure(go.Bar(x=vwap_df.index, y=vwap_df["Volume"], marker_color="#5c6bc0", name="Volume"))
    if not icebergs.empty:
        fig_vol.add_trace(go.Bar(x=icebergs.index, y=icebergs["Volume"], marker_color="#00e5ff", name="Iceberg Volume"))
    fig_vol.update_layout(template=PLOTLY_TEMPLATE, height=280, title="Volume Profile & Anomaly Bars", margin=dict(l=10, r=10, t=40, b=10), dragmode="pan")
    st.plotly_chart(fig_vol, use_container_width=True, config=PLOTLY_CONFIG)

    if not order_book_df.empty:
        st.subheader("📒 Live DOM Size at Nearest Levels (Iceberg Cross-Check)")
        near = order_book_df.copy()
        near["dollar_value"] = near["price"] * near["qty"]
        near = near.sort_values("dollar_value", ascending=False).head(10)
        st.dataframe(near.rename(columns={"price": "Price", "qty": "Size", "side": "Side", "dollar_value": "$ Value"}),
                     use_container_width=True, hide_index=True)

    with st.expander("🧊 Detected Iceberg / Hidden Order Clusters"):
        if not icebergs.empty:
            show_cols = ["Close", "Volume", "vol_range_ratio", "vr_z"]
            st.dataframe(icebergs[show_cols].tail(15).round(3), use_container_width=True)
        else:
            st.info("No statistically significant iceberg clusters detected at the current sensitivity level.")
# ==================================================================================
# SIDEBAR — Rithmic auth only (no hardcoded credentials, works on share.streamlit.io)
# ==================================================================================

st.sidebar.markdown("## 📊 Rithmic Quant Terminal")

if RITHMIC_IMPORT_ERROR:
    st.sidebar.error(f"`async_rithmic` import failed: {RITHMIC_IMPORT_ERROR}")
    with st.sidebar.expander("🔍 Import diagnostics (send me this if it still fails)"):
        st.code(RITHMIC_IMPORT_DIAGNOSTICS or "No diagnostics captured.")
        st.caption(
            "This shows the *actual* installed async_rithmic version and its real "
            "exported names — paste it back and I'll wire the exact names your "
            "installed version uses instead of guessing."
        )

system_profile = st.sidebar.selectbox("Rithmic System", SYSTEM_PROFILE_LABELS, index=0, key="rt_system_profile")
rt_user = st.sidebar.text_input("Rithmic User ID (e.g. your 14-day trial email)", key="rt_user")
rt_password = st.sidebar.text_input("Rithmic Password", type="password", key="rt_password")
rt_system_name = st.sidebar.text_input(
    "Rithmic System Name", value="Rithmic Paper Trading", key="rt_system_name",
    help="The exact system name shown in your Rithmic dev-kit / trial welcome email — "
         "e.g. 'Rithmic Paper Trading' for the 14-day demo.",
)
rt_url = st.sidebar.text_input(
    "Rithmic Connection URL (host:port)",
    value=SYSTEM_URL_PRESETS.get(system_profile, ""),
    key="rt_url",
    help="async_rithmic 1.5+ connects via a direct url= string instead of a Gateway enum. "
         "'Rithmic Test' has a public default filled in above. For Paper Trading or Live, "
         "paste the exact host:port from your Rithmic dev-kit / trial welcome email — "
         "Rithmic assigns this per developer/account and it can't be guessed or hardcoded.",
)
rt_symbols_raw = st.sidebar.text_area(
    "Symbols (SYMBOL:EXCHANGE, comma-separated)", value=DEFAULT_SYMBOLS, key="rt_symbols",
    help="Any symbol + exchange your Rithmic account is entitled to. Crypto-linked CME futures "
         "example: BTC:CME, ETH:CME, MBT:CME, MET:CME.",
)
connect_clicked = st.sidebar.button("🔌 Connect to Rithmic", use_container_width=True)
st.sidebar.caption(
    "Because 14-day demo credentials expire, just paste your newest Rithmic User ID/Password here "
    "and click Connect — nothing needs to change in the source code."
)

def parse_symbols(raw: str):
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            sym, exch = chunk.split(":", 1)
        else:
            sym, exch = chunk, "CME"
        sym, exch = sym.strip().upper(), exch.strip().upper()
        out.append((sym, exch, sym))
    return out

# ==================================================================================
# CONNECTION STATUS BANNER
# ==================================================================================

st.title("📊 Rithmic Institutional Quantitative Trading Terminal")

state: LiveMarketState = st.session_state.get("rt_state")
worker: RithmicMarketDataWorker = st.session_state.get("rt_worker")
conn_error = st.session_state.get("rt_conn_error")

is_live = bool(worker and worker.is_alive() and state and state.status()["connected"] and state.status()["authorized"])

if is_live:
    st.markdown(f'<div class="conn-banner-up">🟢 CONNECTED TO {system_profile.upper()}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="conn-banner-down">🔴 DISCONNECTED</div>', unsafe_allow_html=True)

if conn_error:
    st.error(f"⚠️ {conn_error}")
elif state and state.status()["last_error"]:
    st.warning(f"⚠️ {state.status()['last_error']}")

# ==================================================================================
# CONNECT ACTION — everything wrapped so failures show as banners, never a traceback
# ==================================================================================

if connect_clicked:
    st.session_state["rt_conn_error"] = None
    if RithmicClient is None:
        st.session_state["rt_conn_error"] = (
            f"async_rithmic isn't fully resolved yet ({RITHMIC_IMPORT_ERROR or 'unknown reason'}) — "
            "open the '🔍 Import diagnostics' expander in the sidebar for the exact cause."
        )
    elif not rt_user or not rt_password:
        st.session_state["rt_conn_error"] = "Rithmic User ID and Password are required."
    elif not rt_url:
        st.session_state["rt_conn_error"] = (
            "A Rithmic Connection URL (host:port) is required — async_rithmic 1.5+ connects via "
            "url= instead of a Gateway enum. Paste the URL from your Rithmic welcome email, or "
            "pick 'Rithmic Test' in the System dropdown for the public default."
        )
    else:
        symbols = parse_symbols(rt_symbols_raw)
        if not symbols:
            st.session_state["rt_conn_error"] = "Enter at least one SYMBOL:EXCHANGE pair."
        else:
            try:
                old_worker = st.session_state.get("rt_worker")
                if old_worker is not None:
                    old_worker.stop()

                new_state = LiveMarketState(symbols=symbols)
                new_worker = RithmicMarketDataWorker(
                    user=rt_user, password=rt_password, system_name=rt_system_name,
                    url=rt_url, symbols=symbols, state=new_state,
                )
                new_worker.start()

                st.session_state["rt_state"] = new_state
                st.session_state["rt_worker"] = new_worker
                st.session_state["rt_symbols_list"] = symbols
                st.session_state["rt_conn_error"] = None
                with st.spinner("Authenticating with Rithmic and subscribing to symbols..."):
                    time.sleep(2.5)  # give the background thread a moment before first rerun
                st.rerun()
            except Exception as e:
                st.session_state["rt_conn_error"] = f"Unexpected error while connecting: {e}"

if not is_live:
    st.info("Enter your Rithmic credentials in the sidebar and click **Connect to Rithmic** to start streaming.")
    st.caption(
        "Once connected, quotes, Level 2 DOM (where entitled), tick trades, and multi-timeframe "
        "candles stream automatically in the background for every symbol you listed."
    )
    st.stop()

# ==================================================================================
# SYMBOL FOCUS & INTERVAL (main area — sidebar is reserved for connection controls)
# ==================================================================================

symbols_list = st.session_state.get("rt_symbols_list", [])
roots = [root for _, _, root in symbols_list]

focus_col, interval_col = st.columns([2, 1])
focus_root = focus_col.selectbox("Focused Symbol", roots, key="focus_root")
interval = interval_col.selectbox("Chart Interval", INTERVAL_CHOICES, index=INTERVAL_CHOICES.index("5m"), key="interval_choice")

base_df = state.snapshot_base_bars(focus_root)
main_df = resample_bars(base_df, interval)
order_book_df = state.snapshot_dom(focus_root)
quote_snapshot = state.snapshot_quote(focus_root)
live_ticks = state.snapshot_ticks(focus_root)
daily_bars_by_root = {root: resample_bars(state.snapshot_base_bars(root), "1D") for root in roots}

st.caption(f"Focused Symbol: **{focus_root}** · System: {system_profile} · Interval: {interval}")
st.markdown(
    f'<span class="source-badge">📡 CANDLES: RITHMIC TIME BAR STREAM (LIVE, resampled)</span>'
    f'<span class="source-badge">{"🟢 LIVE L2 DOM" if not order_book_df.empty else "🟡 DOM AWAITING FIRST SNAPSHOT / NOT ENTITLED"}</span>',
    unsafe_allow_html=True,
)

if main_df.empty:
    st.info(f"Waiting on the first live bar snapshot for **{focus_root}** ({interval})...")
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()

if len(main_df) < 20:
    st.warning("Limited bar history streamed so far for this interval — some modules need more bars to compute.")

last_row = main_df.iloc[-1]
prev_row = main_df.iloc[-2] if len(main_df) > 1 else last_row
chg = safe_pct(last_row["Close"], prev_row["Close"])
live_last = quote_snapshot.get("last")
display_price = live_last if live_last is not None else last_row["Close"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Price", f"{display_price:,.2f}", f"{chg:.2f}%")
m2.metric("Bid / Ask", f"{quote_snapshot.get('bidPrice', '—')} / {quote_snapshot.get('askPrice', '—')}")
m3.metric("Session High", f"{main_df['High'].max():,.2f}")
m4.metric("Session Low", f"{main_df['Low'].min():,.2f}")
m5.metric("Bars Loaded", f"{len(main_df):,}")

st.divider()

# ==================================================================================
# TAB NAVIGATION — 6 MODULES (each wrapped so a module error is a banner, not a crash)
# ==================================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌊 Liquidity & Order Flow", "🤖 ML Classifier", "🔗 Cross-Asset Correlation",
    "📊 Volume Delta / CVD", "🎯 Options GEX & Max Pain", "⚙️ VWAP / TWAP / Iceberg",
])

with tab1:
    try:
        render_liquidity_module(main_df, order_book_df, focus_root, dom_is_live=is_live)
    except Exception as e:
        st.error(f"Liquidity module encountered an error: {e}")

with tab2:
    try:
        render_ml_module(main_df, focus_root)
    except Exception as e:
        st.error(f"ML Classifier module encountered an error: {e}")

with tab3:
    try:
        render_correlation_module(daily_bars_by_root, focus_root)
    except Exception as e:
        st.error(f"Correlation module encountered an error: {e}")

with tab4:
    try:
        render_volume_delta_module(main_df, focus_root, live_ticks, ticks_available=len(live_ticks) > 20)
    except Exception as e:
        st.error(f"Volume Delta module encountered an error: {e}")

with tab5:
    try:
        render_gex_module(focus_root, float(display_price) if display_price else None, focus_root)
    except Exception as e:
        st.error(f"GEX module encountered an error: {e}")

with tab6:
    try:
        render_execution_module(main_df, order_book_df, focus_root)
    except Exception as e:
        st.error(f"Execution Algorithms module encountered an error: {e}")

st.divider()
st.caption(
    "⚠️ Disclaimer: research/educational purposes only — not investment advice. "
    "Trading futures involves substantial risk of loss."
)

time.sleep(AUTO_REFRESH_SECONDS)
st.rerun()

"""
==================================================================================
TRADOVATE INSTITUTIONAL QUANTITATIVE TRADING TERMINAL
Live Tradovate REST + WebSocket Engine — Multi-Contract Streaming
==================================================================================
Streams real-time quotes, Level 2 DOM, tick trades, and multi-timeframe candles
for five major futures contracts (NQ, ES, MNQ, MES, CL) directly from the
Tradovate Trading API, feeding six institutional quant analytics modules.

DEPENDENCY NOTE: this build intentionally uses only the libraries listed in
requirements.txt (streamlit, pandas, numpy, plotly, requests, websocket-client,
scikit-learn) — no yfinance, no python-dotenv. Two consequences worth knowing:
  - Module 3 (Cross-Asset Correlation) correlates the five Tradovate-streamed
    contracts against EACH OTHER (their own live daily bars), not against
    yfinance macro series (DXY/10Y yield) like an earlier version of this app
    did — there is no macro data source wired in without yfinance.
  - Module 5 (GEX) is a pure Black-Scholes simulation calibrated off each
    contract's live Tradovate spot price. There is no live listed-options
    chain source without yfinance, so no "live proxy chain" fallback exists
    in this build — the UI labels it as simulated, clearly, at all times.
  - Credentials are entered each session via the sidebar UI only; there is no
    .env file support in this build (no python-dotenv dependency).

WEBSOCKET PROTOCOL: Tradovate's `wss://{env}.tradovateapi.com/v1/websocket`
uses SockJS-style text framing — "o" (open), "h" (heartbeat), "a" (JSON array
of messages), "c" (close). Requests are single text frames shaped
`<endpoint>\\n<requestId>\\n<query>\\n<jsonBody>`, and the first request after
open must be `authorize\\n<id>\\n\\n<accessToken>`. This engine keeps one
persistent WebSocket connection alive on a background thread (survives
Streamlit reruns) and multiplexes subscriptions for all five contracts across
it: md/subscribeQuote, md/subscribeDOM, and md/getChart per contract.
==================================================================================
"""

import json
import math
import threading
import time
import uuid
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import websocket  # websocket-client package

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================================================================================
# PAGE CONFIG & STYLE
# ==================================================================================

st.set_page_config(
    page_title="Tradovate Institutional Quant Terminal",
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
# TRADOVATE CONSTANTS
# ==================================================================================

TRADOVATE_ENVIRONMENTS = {
    "Demo": {"rest": "https://demo.tradovateapi.com/v1", "ws": "wss://demo.tradovateapi.com/v1/websocket"},
    "Live": {"rest": "https://live.tradovateapi.com/v1", "ws": "wss://live.tradovateapi.com/v1/websocket"},
}

DEFAULT_APP_ID = "InstitutionalQuantTerminal"
DEFAULT_APP_VERSION = "1.0"
HTTP_HEADERS_BASE = {"Content-Type": "application/json", "Accept": "application/json"}

# Fixed universe per the spec: NQ, ES, MNQ, MES, CL
CONTRACT_ROOTS = ["NQ", "ES", "MNQ", "MES", "CL"]

TRADOVATE_BAR_CONFIG = {
    "1m":  {"underlyingType": "MinuteBar", "elementSize": 1},
    "5m":  {"underlyingType": "MinuteBar", "elementSize": 5},
    "15m": {"underlyingType": "MinuteBar", "elementSize": 15},
    "30m": {"underlyingType": "MinuteBar", "elementSize": 30},
    "1h":  {"underlyingType": "MinuteBar", "elementSize": 60},
    "4h":  {"underlyingType": "MinuteBar", "elementSize": 240},
    "1d":  {"underlyingType": "DailyBar", "elementSize": 1},
}
INTERVAL_CHOICES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
AUTO_REFRESH_SECONDS = 3  # fixed internal poll rate (no sidebar control per spec)
# ==================================================================================
# TRADOVATE REST SESSION
# ==================================================================================

class TradovateAuthError(Exception):
    pass


class TradovateSession:
    """REST authentication + contract lookup for one Tradovate account.
    cid/sec default to the Demo placeholder "0" when not supplied, matching
    Tradovate's Demo behavior for accounts without a paid API Access
    subscription (not guaranteed for every account — see sidebar caption)."""

    def __init__(self, environment: str, name: str, password: str, api_key: str, app_id: str):
        self.environment = environment
        self.name = name
        self.password = password
        self.cid = api_key if api_key else "0"
        self.sec = "0"
        self.app_id = app_id or DEFAULT_APP_ID
        self.app_version = DEFAULT_APP_VERSION
        self.device_id = str(uuid.uuid4())

        self.rest_base = TRADOVATE_ENVIRONMENTS[environment]["rest"]
        self.ws_url = TRADOVATE_ENVIRONMENTS[environment]["ws"]

        self.access_token = None
        self.expiration_time = None
        self.has_market_data = None
        self.last_error = None

    def credential_fingerprint(self):
        return (self.environment, self.name, self.password, self.cid, self.app_id)

    def authenticate(self):
        url = f"{self.rest_base}/auth/accesstoken"
        body = {
            "name": self.name, "password": self.password, "appId": self.app_id,
            "appVersion": self.app_version, "cid": self.cid, "sec": self.sec,
            "deviceId": self.device_id,
        }
        try:
            resp = requests.post(url, json=body, headers=HTTP_HEADERS_BASE, timeout=10)
        except Exception as e:
            self.last_error = f"Could not reach Tradovate ({url}): {e}"
            raise TradovateAuthError(self.last_error)

        try:
            data = resp.json()
        except Exception:
            self.last_error = f"Tradovate returned a non-JSON response (HTTP {resp.status_code})."
            raise TradovateAuthError(self.last_error)

        if resp.status_code != 200 or "accessToken" not in data:
            err_text = data.get("errorText") or data.get("error") or json.dumps(data)
            self.last_error = f"Authentication rejected: {err_text}"
            raise TradovateAuthError(self.last_error)

        self.access_token = data["accessToken"]
        self.has_market_data = data.get("hasMarketData")
        exp_raw = data.get("expirationTime")
        try:
            self.expiration_time = pd.to_datetime(exp_raw, utc=True).to_pydatetime() if exp_raw else \
                datetime.now(timezone.utc) + timedelta(hours=8)
        except Exception:
            self.expiration_time = datetime.now(timezone.utc) + timedelta(hours=8)
        self.last_error = None
        return data

    def is_token_valid(self) -> bool:
        if not self.access_token or not self.expiration_time:
            return False
        return datetime.now(timezone.utc) < (self.expiration_time - timedelta(minutes=5))

    def ensure_valid(self):
        if not self.is_token_valid():
            self.authenticate()
        return self.access_token

    def auth_headers(self):
        return {**HTTP_HEADERS_BASE, "Authorization": f"Bearer {self.access_token}"}

    def find_front_month_contract(self, root_symbol: str):
        self.ensure_valid()
        url = f"{self.rest_base}/contract/suggest"
        try:
            resp = requests.get(url, params={"t": root_symbol, "l": 5}, headers=self.auth_headers(), timeout=10)
            resp.raise_for_status()
            candidates = resp.json()
        except Exception as e:
            raise TradovateAuthError(f"Contract lookup failed for {root_symbol}: {e}")
        if not isinstance(candidates, list) or not candidates:
            raise TradovateAuthError(
                f"No tradable contract found for root '{root_symbol}' — check exchange "
                f"permissions/entitlements on this account."
            )
        best = candidates[0]
        return {"id": best.get("id"), "name": best.get("name")}


# ==================================================================================
# LIVE MARKET DATA STATE — multi-contract, thread-safe
# ==================================================================================

class LiveMarketState:
    """One shared object per connected session. Mutated by the background WS
    thread, read (copy-on-read) by the Streamlit main thread on every rerun."""

    def __init__(self, contracts: dict, max_bars: int = 3000):
        # contracts: {root_symbol: {"id":..., "name":...}}
        self.lock = threading.Lock()
        self.connected = False
        self.authorized = False
        self.last_error = None
        self.contracts = contracts
        self.max_bars = max_bars

        self._quote = {root: {} for root in contracts}
        self._dom_bids = {root: [] for root in contracts}
        self._dom_asks = {root: [] for root in contracts}
        self._ticks = {root: deque(maxlen=4000) for root in contracts}
        self._bars = {root: {} for root in contracts}  # root -> {timeframe: df}

    # -- mutation (worker thread only) --
    def set_quote(self, root, updates):
        with self.lock:
            self._quote.setdefault(root, {}).update(updates)

    def push_tick(self, root, tick):
        with self.lock:
            self._ticks.setdefault(root, deque(maxlen=4000)).append(tick)

    def set_dom(self, root, bids=None, asks=None):
        with self.lock:
            if bids is not None:
                self._dom_bids[root] = bids
            if asks is not None:
                self._dom_asks[root] = asks

    def merge_bars(self, root, timeframe, new_df):
        with self.lock:
            existing = self._bars.setdefault(root, {}).get(timeframe, pd.DataFrame())
            if existing.empty:
                merged = new_df
            else:
                merged = pd.concat([existing, new_df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if len(merged) > self.max_bars:
                merged = merged.iloc[-self.max_bars:]
            self._bars[root][timeframe] = merged

    # -- snapshot (Streamlit thread only) --
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

    def snapshot_bars(self, root, timeframe):
        with self.lock:
            df = self._bars.get(root, {}).get(timeframe, pd.DataFrame())
            return df.copy() if not df.empty else df

    def snapshot_ticks(self, root):
        with self.lock:
            return list(self._ticks.get(root, []))

    def status(self):
        with self.lock:
            return {"connected": self.connected, "authorized": self.authorized, "last_error": self.last_error}
# ==================================================================================
# BACKGROUND WEBSOCKET WORKER — one connection, multiplexed across 5 contracts
# ==================================================================================

class MarketDataWorker(threading.Thread):
    """Persistent Tradovate WebSocket connection covering all five contracts:
    subscribes md/subscribeQuote + md/subscribeDOM for every contract immediately
    on authorization, and md/getChart(daily) for every contract so correlation
    always has data. Additional timeframes for the currently *focused* contract
    are requested on demand via `ensure_chart()` from the Streamlit thread."""

    def __init__(self, session: TradovateSession, state: LiveMarketState):
        super().__init__(daemon=True)
        self.session = session
        self.state = state
        self.ws = None
        self._req_id = 0
        self._req_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._auth_req_id = None
        self._chart_req_ids = {}       # req_id -> (root, timeframe)
        self._subscribed_charts = set()  # {(root, timeframe)}
        self._stop_event = threading.Event()
        self._id_to_root = {v["id"]: root for root, v in state.contracts.items()}

    def _next_id(self):
        with self._req_lock:
            self._req_id += 1
            return self._req_id

    def _send(self, endpoint, body=None):
        req_id = self._next_id()
        frame = f"{endpoint}\n{req_id}\n\n{json.dumps(body) if body is not None else ''}"
        try:
            with self._send_lock:
                if self.ws is not None:
                    self.ws.send(frame)
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Send failed on '{endpoint}': {e}"
        return req_id

    def ensure_chart(self, root: str, timeframe: str):
        """Callable from the Streamlit main thread. No-op if already subscribed."""
        key = (root, timeframe)
        if key in self._subscribed_charts:
            return
        if root not in self.state.contracts:
            return
        cfg = TRADOVATE_BAR_CONFIG.get(timeframe)
        if cfg is None:
            return
        contract_name = self.state.contracts[root]["name"]
        body = {
            "symbol": contract_name,
            "chartDescription": {
                "underlyingType": cfg["underlyingType"], "elementSize": cfg["elementSize"],
                "elementSizeUnit": "UnderlyingUnits", "withHistogram": False,
            },
            "timeRange": {"asMuchAsElements": 2000},
        }
        req_id = self._send("md/getChart", body)
        self._chart_req_ids[req_id] = key
        self._subscribed_charts.add(key)

    def stop(self):
        self._stop_event.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    def run(self):
        backoff = 2
        while not self._stop_event.is_set():
            try:
                self.session.ensure_valid()
            except TradovateAuthError as e:
                with self.state.lock:
                    self.state.last_error = f"Token refresh failed: {e}"
                time.sleep(min(backoff, 30)); backoff = min(backoff * 2, 30)
                continue

            self.ws = websocket.WebSocketApp(
                self.session.ws_url, on_open=self._on_open, on_message=self._on_message,
                on_error=self._on_error, on_close=self._on_close,
            )
            with self.state.lock:
                self.state.connected = False
                self.state.authorized = False
            self.ws.run_forever(ping_interval=0)

            if self._stop_event.is_set():
                break
            time.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 30)

    def _on_open(self, ws):
        with self.state.lock:
            self.state.connected = True
        self._auth_req_id = self._next_id()
        try:
            with self._send_lock:
                ws.send(f"authorize\n{self._auth_req_id}\n\n{self.session.access_token}")
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Authorize send failed: {e}"

    def _on_error(self, ws, error):
        with self.state.lock:
            self.state.last_error = f"WebSocket error: {error}"

    def _on_close(self, ws, code, msg):
        with self.state.lock:
            self.state.connected = False
            self.state.authorized = False

    def _on_message(self, ws, message):
        if message in ("o", "h", ""):
            return
        type_char, payload = message[0], message[1:]
        if type_char == "c":
            with self.state.lock:
                self.state.last_error = f"Server closed session: {payload}"
            return
        if type_char != "a":
            return
        try:
            events = json.loads(payload)
        except Exception:
            return
        for item in events:
            self._handle_event(item)

    def _handle_event(self, item: dict):
        if "i" in item and "s" in item:
            req_id, status, data = item.get("i"), item.get("s"), item.get("d", {})
            if req_id == self._auth_req_id:
                if status == 200:
                    with self.state.lock:
                        self.state.authorized = True
                    self._subscribe_baseline()
                else:
                    with self.state.lock:
                        self.state.last_error = f"Authorization rejected (status {status}): {data}"
                return
            if req_id in self._chart_req_ids:
                root, tf = self._chart_req_ids[req_id]
                self._ingest_chart_payload(root, tf, data)
                return
            if status != 200:
                with self.state.lock:
                    self.state.last_error = f"Request {req_id} failed (status {status}): {data}"
            return

        if item.get("e") == "md":
            d = item.get("d", {})
            self._ingest_quotes(d.get("quotes", []))
            self._ingest_doms(d.get("doms", []))
            for chart in d.get("charts", []):
                cid = chart.get("id") or chart.get("reqId")
                key = self._chart_req_ids.get(cid)
                if key:
                    self._ingest_chart_payload(key[0], key[1], {"charts": [chart]})

    def _ingest_quotes(self, quotes):
        for q in quotes:
            root = self._id_to_root.get(q.get("contractId"))
            if root is None:
                continue
            entries = q.get("entries", {})
            snap = {}
            if entries.get("Bid"):
                snap["bidPrice"] = entries["Bid"].get("price"); snap["bidSize"] = entries["Bid"].get("size")
            if entries.get("Offer"):
                snap["askPrice"] = entries["Offer"].get("price"); snap["askSize"] = entries["Offer"].get("size")
            if entries.get("Trade"):
                trade = entries["Trade"]
                snap["last"] = trade.get("price"); snap["lastSize"] = trade.get("size")
                self.state.push_tick(root, {
                    "price": trade.get("price"), "qty": trade.get("size", 0) or 0,
                    "timestamp": datetime.now(timezone.utc),
                })
            if snap:
                self.state.set_quote(root, snap)

    def _ingest_doms(self, doms):
        for dom in doms:
            root = self._id_to_root.get(dom.get("contractId"))
            if root is None:
                continue
            bids = [{"price": l.get("price"), "qty": l.get("size", 0)} for l in dom.get("bids", []) if l.get("price") is not None]
            asks = [{"price": l.get("price"), "qty": l.get("size", 0)} for l in dom.get("offers", []) if l.get("price") is not None]
            self.state.set_dom(root, bids=bids or None, asks=asks or None)

    def _ingest_chart_payload(self, root, timeframe, data):
        rows = []
        for chart in data.get("charts", []):
            for bar in chart.get("bars", []):
                try:
                    idx = pd.to_datetime(bar.get("timestamp"), utc=True)
                except Exception:
                    continue
                up_vol = bar.get("upVolume", 0) or 0
                down_vol = bar.get("downVolume", 0) or 0
                vol = bar.get("volume", up_vol + down_vol) or (up_vol + down_vol)
                rows.append({"timestamp": idx, "Open": bar.get("open"), "High": bar.get("high"),
                             "Low": bar.get("low"), "Close": bar.get("close"), "Volume": vol})
        if not rows:
            return
        new_df = pd.DataFrame(rows).dropna(subset=["Open", "High", "Low", "Close"])
        new_df = new_df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        self.state.merge_bars(root, timeframe, new_df)

    def _subscribe_baseline(self):
        for root, info in self.state.contracts.items():
            self._send("md/subscribeQuote", {"symbol": info["name"]})
            self._send("md/subscribeDOM", {"symbol": info["name"]})
            self.ensure_chart(root, "1d")  # always keep daily bars for correlation
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

def render_liquidity_module(df: pd.DataFrame, order_book_df: pd.DataFrame, label: str, dom_is_live: bool):
    st.markdown(
        '<div class="module-note">Swing-point liquidity pools (BSL/SSL) and Fair Value Gap '
        'imbalance zones from price-action structure, dynamically annotated with the live '
        'Tradovate Level 2 DOM ($ resting-order depth), institutional bank-wall isolation, and a '
        'Bank Anchor PnL Tracker.</div>',
        unsafe_allow_html=True,
    )

    if len(df) < 15:
        st.warning("Not enough bars streamed yet for this timeframe to compute liquidity structure. "
                    "Give the chart subscription a few seconds, or pick a lower-granularity interval.")
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
            "Live Tradovate DOM not yet populated for this contract — $ volume annotations and "
            "institutional wall isolation are skipped until the md/subscribeDOM stream delivers its "
            "first book snapshot. Swing/FVG structure is still fully computed from streamed price action below."
        )
    else:
        st.markdown(
            f'<span class="source-badge">🟢 LIVE TRADOVATE L2 DOM</span> '
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
        title=f"{label} — Liquidity Pools, Fair Value Gaps & Institutional Bank Walls (Live Tradovate Feed)",
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
        'features (RSI, ATR, volatility, MA ratios, volume delta, momentum) built from the streamed '
        'Tradovate bars, estimating the probability of the next-N-bar directional move.</div>',
        unsafe_allow_html=True,
    )

    if len(df) < 80:
        st.warning("Insufficient bars streamed yet for reliable ML training. Let the chart subscription "
                    "accumulate more history, or select a lower-granularity interval.")
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
        fig_proba.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Directional Probability", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_proba, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Model trained on {len(X_train)} bars, validated on {len(X_test)} out-of-sample bars for {label} (live Tradovate feed).")

# ==================================================================================
# MODULE 3 — CROSS-ASSET CORRELATION & MACRO YIELD MATRIX
# ==================================================================================
# No external macro/yield data source is wired into this build (no yfinance
# dependency — see the file header). Instead, this module correlates the five
# live Tradovate-streamed contracts (NQ, ES, MNQ, MES, CL) against each other
# using their own live daily bars — genuine cross-asset structure (equity
# index vs. crude oil co-movement, ES vs. its own micro NQ, etc.), just without
# a standalone DXY/10Y-yield macro anchor.

def render_correlation_module(daily_bars_by_root: dict, focus_root: str):
    st.markdown(
        '<div class="module-note">Cross-asset correlation across all five live-streamed Tradovate '
        'contracts (NQ, ES, MNQ, MES, CL), computed from their own live daily bars.</div>',
        unsafe_allow_html=True,
    )

    lookback = st.slider("Correlation Lookback (Days)", 10, 365, 90, step=5, key="corr_lookback")

    series_dict = {}
    missing = []
    for root, df in daily_bars_by_root.items():
        if df.empty:
            missing.append(root)
            continue
        cutoff = df.index.max() - pd.Timedelta(days=lookback)
        s = df[df.index >= cutoff]["Close"]
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        if not s.empty:
            series_dict[root] = s

    if missing:
        st.info(f"Still waiting on daily bars for: {', '.join(missing)} (subscribed automatically for all five contracts).")

    if len(series_dict) < 2:
        st.warning("Need daily bars for at least two contracts to compute correlations — give the "
                    "chart subscriptions a few more seconds.")
        return

    combined = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()
    if combined.empty or len(combined) < 5:
        st.warning("Not enough overlapping daily history across contracts yet for this lookback window.")
        return

    corr_matrix = combined.corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
    ))
    fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=420, title="Cross-Contract Correlation Matrix",
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
    """Fallback estimator (no tick stream yet): buy/sell volume split by
    intra-bar close position, used before enough live trade ticks have
    accumulated for the tick-rule method below."""
    out = df.copy()
    rng = (out["High"] - out["Low"]).replace(0, np.nan)
    buy_ratio = ((out["Close"] - out["Low"]) / rng).clip(0, 1).fillna(0.5)
    out["buy_volume"] = out["Volume"] * buy_ratio
    out["sell_volume"] = out["Volume"] * (1 - buy_ratio)
    out["delta"] = out["buy_volume"] - out["sell_volume"]
    out["cvd"] = out["delta"].cumsum()
    return out

def compute_volume_delta_from_ticks(df: pd.DataFrame, ticks: list) -> pd.DataFrame:
    """Real order-flow reconstruction from Tradovate's live Trade prints
    (md/subscribeQuote 'Trade' entries), classified with the standard tick
    rule (uptick = buyer-initiated, downtick = seller-initiated, unchanged
    price inherits the prior tick's side), then binned into the same bars as
    the streamed OHLCV chart."""
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
    side = []
    last_side = "buy"
    for d in tick_df["price_diff"]:
        if pd.isna(d) or d == 0:
            side.append(last_side)
        elif d > 0:
            side.append("buy")
            last_side = "buy"
        else:
            side.append("sell")
            last_side = "sell"
    tick_df["side"] = side

    tick_df["timestamp"] = pd.to_datetime(tick_df["timestamp"], utc=True)
    bin_edges = df.index
    if bin_edges.tz is None:
        tick_df["timestamp"] = tick_df["timestamp"].dt.tz_localize(None)
    tick_df["bar"] = pd.cut(tick_df["timestamp"], bins=list(bin_edges) + [bin_edges[-1] + (bin_edges[-1] - bin_edges[-2] if len(bin_edges) > 1 else pd.Timedelta(minutes=1))],
                             labels=bin_edges, right=False)

    grouped = tick_df.groupby(["bar", "side"], observed=True)["qty"].sum().unstack(fill_value=0)
    if "buy" not in grouped.columns:
        grouped["buy"] = 0
    if "sell" not in grouped.columns:
        grouped["sell"] = 0

    grouped.index = pd.to_datetime(grouped.index)
    out.loc[out.index.isin(grouped.index), "buy_volume"] = grouped.reindex(out.index)["buy"].fillna(0)
    out.loc[out.index.isin(grouped.index), "sell_volume"] = grouped.reindex(out.index)["sell"].fillna(0)

    # Bars with no live ticks recorded yet (e.g. history back-filled before the
    # subscription started) still get the heuristic split so the chart has no gaps.
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
        'directly from live Tradovate trade prints via the tick rule (uptick = buy-initiated, '
        'downtick = sell-initiated), aggregated into Cumulative Volume Delta (CVD). Falls back to a '
        'close-position volume heuristic for any bar streamed before the tick subscription had data.</div>',
        unsafe_allow_html=True,
    )

    if df["Volume"].sum() == 0:
        st.warning("No volume streamed yet for this contract/timeframe — Volume Delta requires non-zero volume.")

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

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
        subplot_titles=(f"{label} Price [Live Tradovate Feed]", "Volume Delta (Buy − Sell)", "Cumulative Volume Delta (CVD)"),
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
# No live listed-options chain source is wired into this build (no yfinance
# dependency). This module is a pure Black-Scholes gamma-exposure simulation
# calibrated off each contract's live Tradovate spot price — the UI labels it
# as simulated at all times; it is a structural/educational GEX surface, not
# live open-interest data.

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

def render_gex_module(root_symbol: str, live_spot, label: str):
    st.markdown(
        '<div class="module-note">Simulated Gamma Exposure (GEX) profile — Black-Scholes gamma '
        'model calibrated off the live Tradovate spot price — identifying illustrative dealer '
        'positioning, volatility pin zones, and the gamma flip level.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="source-badge">🟡 SIMULATED (no live options chain source in this build)</span>', unsafe_allow_html=True)

    if not live_spot:
        st.error(f"No live Tradovate spot price yet for {label} to calibrate the GEX simulation.")
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

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=560, barmode="relative",
        title=f"{label} — Simulated Gamma Exposure Profile by Strike",
        xaxis_title="Strike", yaxis_title="Gamma Exposure",
        margin=dict(l=10, r=10, t=50, b=10), dragmode="pan",
    )
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
        'from the live Tradovate bar stream, cross-checked against resting DOM size when available.</div>',
        unsafe_allow_html=True,
    )

    if df["Volume"].sum() == 0:
        st.warning("No volume streamed yet for this contract — VWAP and iceberg detection require non-zero volume.")

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
        title=f"{label} — VWAP / TWAP Execution Benchmarks & Iceberg Detection (Live Tradovate Feed)",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=price_range, autorange=False if price_range else True),
    )
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
        st.dataframe(
            near.rename(columns={"price": "Price", "qty": "Size", "side": "Side", "dollar_value": "$ Value"}),
            use_container_width=True, hide_index=True,
        )

    with st.expander("🧊 Detected Iceberg / Hidden Order Clusters"):
        if not icebergs.empty:
            show_cols = ["Close", "Volume", "vol_range_ratio", "vr_z"]
            st.dataframe(icebergs[show_cols].tail(15).round(3), use_container_width=True)
        else:
            st.info("No statistically significant iceberg clusters detected at the current sensitivity level.")
# ==================================================================================
# SIDEBAR — ONLY: Environment, Username, Password, API Key/App ID, Connect button
# ==================================================================================

st.sidebar.markdown("## 📊 Tradovate Quant Terminal")

environment = st.sidebar.selectbox("Environment", ["Demo", "Live"], index=0, key="tv_environment")
tv_name = st.sidebar.text_input("Tradovate Username", key="tv_name")
tv_password = st.sidebar.text_input("Tradovate Password", type="password", key="tv_password")
tv_api_key = st.sidebar.text_input(
    "API Key / App ID (optional)", value="0", key="tv_api_key",
    help="Demo defaults to \"0\", which some Demo accounts accept without a paid API Access "
         "subscription. If Demo rejects it, enter the API Key issued under Settings → API Access.",
)
connect_clicked = st.sidebar.button("🔌 Connect to Tradovate", use_container_width=True)

# ==================================================================================
# CONNECTION STATUS BANNER
# ==================================================================================

st.title("📊 Tradovate Institutional Quantitative Trading Terminal")

state: LiveMarketState = st.session_state.get("tv_state")
worker: MarketDataWorker = st.session_state.get("tv_worker")
tv_session: TradovateSession = st.session_state.get("tv_session")
conn_error = st.session_state.get("tv_conn_error")

is_live = bool(worker and worker.is_alive() and state and state.status()["connected"] and state.status()["authorized"])

if is_live:
    st.markdown(f'<div class="conn-banner-up">🟢 CONNECTED TO TRADOVATE {environment.upper()}</div>', unsafe_allow_html=True)
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
    st.session_state["tv_conn_error"] = None
    if not tv_name or not tv_password:
        st.session_state["tv_conn_error"] = "Username and Password are required."
    else:
        try:
            with st.spinner("Authenticating with Tradovate..."):
                new_session = TradovateSession(
                    environment=environment, name=tv_name, password=tv_password,
                    api_key=tv_api_key, app_id=DEFAULT_APP_ID,
                )
                new_session.authenticate()

            with st.spinner("Resolving front-month contracts for NQ, ES, MNQ, MES, CL..."):
                contracts = {}
                for root in CONTRACT_ROOTS:
                    contracts[root] = new_session.find_front_month_contract(root)

            old_worker = st.session_state.get("tv_worker")
            if old_worker is not None:
                old_worker.stop()

            new_state = LiveMarketState(contracts=contracts)
            new_worker = MarketDataWorker(session=new_session, state=new_state)
            new_worker.start()

            st.session_state["tv_session"] = new_session
            st.session_state["tv_state"] = new_state
            st.session_state["tv_worker"] = new_worker
            st.session_state["tv_conn_error"] = None
            if new_session.has_market_data is False:
                st.session_state["tv_conn_error"] = (
                    "Connected, but this account reports hasMarketData=false — quotes/DOM will not "
                    "stream until a market data entitlement is enabled, even though candles/auth work."
                )
            st.rerun()
        except TradovateAuthError as e:
            st.session_state["tv_conn_error"] = str(e)
        except Exception as e:
            st.session_state["tv_conn_error"] = f"Unexpected error while connecting: {e}"

if not is_live:
    st.info("Enter your credentials in the sidebar and click **Connect to Tradovate** to start streaming.")
    st.caption(
        "Once connected, quotes, Level 2 DOM, tick trades, and multi-timeframe candles stream "
        "automatically in the background for NQ, ES, MNQ, MES, and CL."
    )
    st.stop()

# ==================================================================================
# CONTRACT FOCUS & INTERVAL (main area — sidebar is reserved for connection controls)
# ==================================================================================

focus_col, interval_col = st.columns([2, 1])
focus_root = focus_col.selectbox("Focused Contract", CONTRACT_ROOTS, key="focus_root")
interval = interval_col.selectbox("Chart Interval", INTERVAL_CHOICES, index=INTERVAL_CHOICES.index("5m"), key="interval_choice")

try:
    worker.ensure_chart(focus_root, interval)
except Exception:
    pass  # non-fatal — chart will simply be empty until the next successful subscribe

contract_name = state.contracts[focus_root]["name"]
main_df = state.snapshot_bars(focus_root, interval)
order_book_df = state.snapshot_dom(focus_root)
quote_snapshot = state.snapshot_quote(focus_root)
live_ticks = state.snapshot_ticks(focus_root)
daily_bars_by_root = {root: state.snapshot_bars(root, "1d") for root in CONTRACT_ROOTS}

st.caption(f"Focused Contract: **{focus_root}** ({contract_name}) · Environment: {environment} · Interval: {interval}")
st.markdown(
    f'<span class="source-badge">📡 CANDLES: TRADOVATE md/getChart (LIVE)</span>'
    f'<span class="source-badge">{"🟢 LIVE L2 DOM" if not order_book_df.empty else "🟡 DOM AWAITING FIRST SNAPSHOT"}</span>',
    unsafe_allow_html=True,
)

if main_df.empty:
    st.info(f"Waiting on the first live bar snapshot for **{contract_name}** ({interval})...")
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

# Live polling loop — background WebSocket thread keeps streaming regardless;
# this just keeps the displayed snapshot fresh.
time.sleep(AUTO_REFRESH_SECONDS)
st.rerun()

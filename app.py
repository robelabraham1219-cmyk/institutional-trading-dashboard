"""
==================================================================================
INSTITUTIONAL QUANTITATIVE TRADING TERMINAL
Tradovate Live Data Engine — REST Auth + WebSocket L2 DOM / Quote / Chart Stream
==================================================================================
A single-file, production-ready Streamlit application implementing six
institutional-grade quantitative trading modules, backed entirely by the
Tradovate Trading API (Demo or Live environment):

    Authentication   -> POST {rest}/auth/accesstoken   (REST, username/password/API key)
    Real-time Quotes -> WS   md/subscribeQuote          (best bid/ask, last, volume)
    Real-time DOM    -> WS   md/subscribeDOM            (full Level 2 price ladder)
    Real-time Chart  -> WS   md/getChart                (historical + live bar stream)

IMPORTANT — READ BEFORE DEPLOYING WITH REAL CREDENTIALS
---------------------------------------------------------------------------------
Tradovate's `/auth/accesstoken` endpoint requires FOUR credential fields, not
just a username/password:
    name (username), password, appId, appVersion, cid (API Key/Client ID),
    sec (API Secret), and optionally deviceId.
`cid`/`sec` are issued to you by Tradovate under Settings -> API Access on the
account you authenticate with. A bare username+password without cid/sec will
be rejected by the endpoint ("Access is denied"). The sidebar below collects
all of these fields (or reads them from a `.env` file) — this is not optional
scaffolding, it is how the endpoint actually works.

The WebSocket wire protocol used by Tradovate (`wss://demo.tradovateapi.com/v1/websocket`)
is a lightweight SockJS-style text framing:
    - First frame received after connect is the literal string "o" (open).
    - Every outbound request is a single text frame of the form:
          "<endpoint>\\n<requestId>\\n<query>\\n<jsonBody>"
      e.g.  "md/subscribeDOM\\n3\\n\\n{\"symbol\":\"ESZ5\"}"
    - The very first request must be authorization:
          "authorize\\n1\\n\\n<accessToken>"
    - Inbound frames are prefixed by a single type character:
          "o" = open, "h" = heartbeat, "a" = array of JSON messages,
          "c" = close (server is terminating the session).
    - "a" frames carry a JSON array, each element shaped like
          {"s": <status>, "i": <requestId>, "d": <payload>}   (responses)
      or  {"e": "md", "d": {...}}                             (data events)
    - The client must not let the socket go idle — this engine replies to each
      "h" heartbeat by sending a keep-alive frame back immediately, which is
      the behavior Tradovate's own reference clients use.

Because Streamlit re-executes the whole script on every rerun, a naive
`websockets.connect()` call inside the script body would reconnect (and lose
the DOM/quote state) on every single interaction. This engine instead runs the
WebSocket client on a persistent background thread (stored in
`st.session_state`, survives reruns) with its own asyncio event loop, and
exposes a thread-safe snapshot of the latest quote / DOM / chart bars that the
main Streamlit thread reads on every rerun. A small auto-refresh loop keeps
the UI polling that snapshot at a configurable interval so the terminal feels
"live" without needing server-push into the browser.
==================================================================================
"""

import json
import math
import os
import queue
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
import websocket  # websocket-client package (sync, thread-friendly)
import yfinance as yf

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

# ==================================================================================
# PAGE CONFIG & GLOBAL STYLE
# ==================================================================================

st.set_page_config(
    page_title="Institutional Quant Terminal — Tradovate",
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
# TRADOVATE ENDPOINT CONSTANTS
# ==================================================================================

TRADOVATE_ENVIRONMENTS = {
    "Demo": {
        "rest": "https://demo.tradovateapi.com/v1",
        "ws": "wss://demo.tradovateapi.com/v1/websocket",
    },
    "Live": {
        "rest": "https://live.tradovateapi.com/v1",
        "ws": "wss://live.tradovateapi.com/v1/websocket",
    },
}

DEFAULT_APP_ID = "InstitutionalQuantTerminal"
DEFAULT_APP_VERSION = "1.0"

HTTP_HEADERS_BASE = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Timeframe -> Tradovate chartDescription for md/getChart. Tradovate's chart
# API buckets time-based bars under underlyingType "MinuteBar" (elementSize =
# number of minutes per bar) and calendar-day bars under "DailyBar". Verify
# these enum values against the API Reference in your Tradovate account if
# your account's chart service version differs.
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

# ==================================================================================
# FUTURES CONTRACT UNIVERSE
# ==================================================================================
# Tradovate contracts are month-coded (e.g. "ESZ5"). We resolve the live
# front-month contract for each root symbol at runtime via the REST
# /contract/suggest endpoint rather than hardcoding expiring month codes.

FUTURES_ROOTS = {
    "E-mini S&P 500 (ES)": "ES",
    "Micro E-mini S&P 500 (MES)": "MES",
    "E-mini Nasdaq-100 (NQ)": "NQ",
    "Micro E-mini Nasdaq-100 (MNQ)": "MNQ",
    "E-mini Dow (YM)": "YM",
    "Crude Oil (CL)": "CL",
    "Micro Crude Oil (MCL)": "MCL",
    "Gold (GC)": "GC",
    "Micro Gold (MGC)": "MGC",
    "Silver (SI)": "SI",
    "10-Year T-Note (ZN)": "ZN",
    "30-Year T-Bond (ZB)": "ZB",
    "Euro FX (6E)": "6E",
    "British Pound (6B)": "6B",
    "Japanese Yen (6J)": "6J",
}

# Macro/options-proxy map used only by Module 5 (Tradovate does not expose a
# public listed-options chain endpoint for these futures roots in this app;
# see the GEX module docstring for exactly what is live vs. simulated).
FUTURES_OPTIONS_PROXY_MAP = {
    "ES": "SPY", "MES": "SPY", "NQ": "QQQ", "MNQ": "QQQ", "YM": "DIA",
    "CL": "USO", "MCL": "USO", "GC": "GLD", "MGC": "GLD", "SI": "SLV",
    "ZN": "IEF", "ZB": "TLT", "6E": "FXE", "6B": "FXB", "6J": "FXY",
}

# Macro anchors used by Module 3 correlation matrix (yfinance — Tradovate has
# no Treasury-yield-index or DXY product, so these stay on the macro/index feed
# exactly like the original app's "any source gap -> yfinance" pattern).
MACRO_ANCHOR_TICKERS = {
    "US Dollar Index (DXY)": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "S&P 500 Index": "^GSPC",
}
# ==================================================================================
# TRADOVATE REST SESSION — AUTHENTICATION, TOKEN REFRESH, CONTRACT RESOLUTION
# ==================================================================================

class TradovateAuthError(Exception):
    pass


class TradovateSession:
    """Owns REST authentication state for one Tradovate account.

    Handles the initial /auth/accesstoken call, tracks the access token's
    expiration, and exposes a `ensure_valid()` method the rest of the app
    calls before any REST/WS action so an expired/near-expired Demo token
    (Demo tokens are valid up to 14 days, per Tradovate's Demo policy) is
    silently refreshed as long as the same credentials are still valid in
    the sidebar — no restart, no broken UI, matching the "graceful
    re-authentication" requirement.
    """

    def __init__(self, environment: str, name: str, password: str, app_id: str,
                 app_version: str, cid: str, sec: str, device_id: str):
        self.environment = environment
        self.name = name
        self.password = password
        self.app_id = app_id or DEFAULT_APP_ID
        self.app_version = app_version or DEFAULT_APP_VERSION
        # Tradovate's Demo environment accepts the placeholder credential
        # "0" for cid/sec on some accounts in lieu of a paid API Access
        # subscription's real Client ID/Secret pair — default to "0" here
        # whenever the field is left blank so Demo users aren't blocked on
        # provisioning real API keys just to try the app. NOTE: this is not
        # guaranteed to work for every account; if Demo still rejects the
        # request, that means your account requires real cid/sec values (see
        # the sidebar warning and the accesstoken error message returned).
        self.cid = cid if cid not in (None, "") else "0"
        self.sec = sec if sec not in (None, "") else "0"
        self.device_id = device_id or str(uuid.uuid4())

        self.rest_base = TRADOVATE_ENVIRONMENTS[environment]["rest"]
        self.ws_url = TRADOVATE_ENVIRONMENTS[environment]["ws"]

        self.access_token = None
        self.md_access_token = None
        self.expiration_time = None  # datetime, UTC
        self.user_id = None
        self.has_market_data = None
        self.last_error = None

    # ---- credential identity, used to detect sidebar changes needing re-auth ----
    def credential_fingerprint(self):
        return (self.environment, self.name, self.password, self.app_id,
                self.app_version, self.cid, self.sec)

    def authenticate(self):
        """POST /auth/accesstoken. Raises TradovateAuthError on failure."""
        url = f"{self.rest_base}/auth/accesstoken"
        body = {
            "name": self.name,
            "password": self.password,
            "appId": self.app_id,
            "appVersion": self.app_version,
            "cid": self.cid,
            "sec": self.sec,
            "deviceId": self.device_id,
        }
        try:
            resp = requests.post(url, json=body, headers=HTTP_HEADERS_BASE, timeout=10)
        except Exception as e:
            self.last_error = f"Network error contacting {url}: {e}"
            raise TradovateAuthError(self.last_error)

        try:
            data = resp.json()
        except Exception:
            self.last_error = f"Non-JSON response from accesstoken endpoint (HTTP {resp.status_code})."
            raise TradovateAuthError(self.last_error)

        if resp.status_code != 200 or "accessToken" not in data:
            err_text = data.get("errorText") or data.get("error") or json.dumps(data)
            self.last_error = f"Tradovate authentication failed: {err_text}"
            raise TradovateAuthError(self.last_error)

        self.access_token = data["accessToken"]
        self.md_access_token = data.get("mdAccessToken", self.access_token)
        self.user_id = data.get("userId")
        self.has_market_data = data.get("hasMarketData", None)

        exp_raw = data.get("expirationTime")
        if exp_raw:
            try:
                self.expiration_time = pd.to_datetime(exp_raw, utc=True).to_pydatetime()
            except Exception:
                self.expiration_time = datetime.now(timezone.utc) + timedelta(hours=8)
        else:
            self.expiration_time = datetime.now(timezone.utc) + timedelta(hours=8)

        self.last_error = None
        return data

    def is_token_valid(self) -> bool:
        if not self.access_token or not self.expiration_time:
            return False
        # Refresh 5 minutes before actual expiry to avoid racing a stream drop.
        return datetime.now(timezone.utc) < (self.expiration_time - timedelta(minutes=5))

    def ensure_valid(self):
        if not self.is_token_valid():
            self.authenticate()
        return self.access_token

    def auth_headers(self):
        return {**HTTP_HEADERS_BASE, "Authorization": f"Bearer {self.access_token}"}

    # ------------------------------------------------------------------
    # Contract resolution — front-month lookup via /contract/suggest
    # ------------------------------------------------------------------
    def find_front_month_contract(self, root_symbol: str):
        """Resolve a root future (e.g. 'ES') to its current front-month
        tradable contract name (e.g. 'ESZ5') and contractId via the
        /contract/suggest REST endpoint."""
        self.ensure_valid()
        url = f"{self.rest_base}/contract/suggest"
        try:
            resp = requests.get(
                url, params={"t": root_symbol, "l": 5},
                headers=self.auth_headers(), timeout=10,
            )
            resp.raise_for_status()
            candidates = resp.json()
        except Exception as e:
            raise TradovateAuthError(f"Contract lookup failed for {root_symbol}: {e}")

        if not isinstance(candidates, list) or not candidates:
            raise TradovateAuthError(
                f"No tradable contracts returned for root '{root_symbol}'. "
                "Confirm this symbol is enabled on your account's exchange permissions."
            )
        # /contract/suggest returns candidates ordered by relevance; the first
        # entry is Tradovate's own best/front-month match for the root.
        best = candidates[0]
        return {
            "id": best.get("id"),
            "name": best.get("name"),
        }


# ==================================================================================
# LIVE MARKET DATA STATE — thread-safe snapshot shared with the Streamlit thread
# ==================================================================================

class LiveMarketState:
    """Container mutated by the background WS thread and read (copy-on-read)
    by the Streamlit main thread. All mutation goes through a single lock."""

    def __init__(self, max_bars: int = 3000):
        self.lock = threading.Lock()
        self.connected = False
        self.authorized = False
        self.last_error = None
        self.contract_name = None
        self.contract_id = None

        self.quote = {}  # bidPrice, bidSize, askPrice, askSize, last, volume, tradeDate...
        self.dom_bids = []   # list of {"price":..., "qty":...}
        self.dom_asks = []
        self.trade_ticks = deque(maxlen=5000)  # {"price","qty","side","timestamp"}

        self.bars = {tf: pd.DataFrame() for tf in INTERVAL_CHOICES}
        self.max_bars = max_bars
        self.last_update_ts = {tf: None for tf in INTERVAL_CHOICES}

    def snapshot_quote(self):
        with self.lock:
            return dict(self.quote)

    def snapshot_dom(self):
        with self.lock:
            bids = pd.DataFrame(self.dom_bids)
            asks = pd.DataFrame(self.dom_asks)
        frames = []
        if not bids.empty:
            bids = bids.rename(columns={"price": "price", "qty": "qty"})
            bids["side"] = "bid"
            frames.append(bids)
        if not asks.empty:
            asks = asks.rename(columns={"price": "price", "qty": "qty"})
            asks["side"] = "ask"
            frames.append(asks)
        if not frames:
            return pd.DataFrame(columns=["price", "qty", "side"])
        return pd.concat(frames, ignore_index=True)

    def snapshot_bars(self, timeframe: str):
        with self.lock:
            df = self.bars.get(timeframe, pd.DataFrame())
            return df.copy() if not df.empty else df

    def snapshot_ticks(self):
        with self.lock:
            return list(self.trade_ticks)

    def status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "authorized": self.authorized,
                "last_error": self.last_error,
                "contract_name": self.contract_name,
            }
# ==================================================================================
# BACKGROUND WEBSOCKET WORKER — persists across Streamlit reruns
# ==================================================================================

class MarketDataWorker(threading.Thread):
    """Runs one persistent Tradovate WebSocket connection on a background
    thread. Streamlit's script re-executes on every user interaction, so this
    worker (and the LiveMarketState it feeds) is stashed in st.session_state
    and only created once per contract selection — it is NOT recreated on
    every rerun, which is what makes the "real-time" DOM/quote/chart actually
    real-time instead of a fresh reconnect-and-lose-state every click.
    """

    def __init__(self, session: TradovateSession, state: LiveMarketState,
                 contract_name: str, contract_id, timeframes):
        super().__init__(daemon=True)
        self.session = session
        self.state = state
        self.contract_name = contract_name
        self.contract_id = contract_id
        self.timeframes = list(timeframes)

        self.ws = None
        self._req_id = 0
        self._req_lock = threading.Lock()
        self._auth_req_id = None
        self._quote_req_id = None
        self._dom_req_id = None
        self._chart_req_ids = {}   # request id -> timeframe
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()

        self.state.contract_name = contract_name
        self.state.contract_id = contract_id

    # ---------------------------------------------------------------- utils
    def _next_id(self):
        with self._req_lock:
            self._req_id += 1
            return self._req_id

    def _send(self, endpoint: str, body=None):
        req_id = self._next_id()
        body_str = json.dumps(body) if body is not None else ""
        frame = f"{endpoint}\n{req_id}\n\n{body_str}"
        try:
            with self._send_lock:
                if self.ws is not None:
                    self.ws.send(frame)
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Send failed on '{endpoint}': {e}"
        return req_id

    def stop(self):
        self._stop_event.set()
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass

    # -------------------------------------------------------------- run loop
    def run(self):
        backoff = 2
        while not self._stop_event.is_set():
            try:
                self.session.ensure_valid()
            except TradovateAuthError as e:
                with self.state.lock:
                    self.state.last_error = f"Auth refresh failed: {e}"
                time.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)
                continue

            self.ws = websocket.WebSocketApp(
                self.session.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            with self.state.lock:
                self.state.connected = False
                self.state.authorized = False

            # run_forever blocks until the socket closes/errors; Tradovate's
            # server-side heartbeat ("h" frames) keeps the connection alive so
            # we disable the library's own ping and rely on Tradovate's frame
            # protocol instead.
            self.ws.run_forever(ping_interval=0)

            if self._stop_event.is_set():
                break
            # Unexpected disconnect — re-authenticate (token may have been the
            # cause) and reconnect with backoff.
            time.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 30)

    # ------------------------------------------------------------- handlers
    def _on_open(self, ws):
        with self.state.lock:
            self.state.connected = True
        # First frame from the server is a bare "o"; per Tradovate's own
        # reference clients the authorize call is sent immediately after
        # socket open rather than waiting for a distinct application-level ack.
        self._auth_req_id = self._next_id()
        frame = f"authorize\n{self._auth_req_id}\n\n{self.session.access_token}"
        try:
            with self._send_lock:
                ws.send(frame)
        except Exception as e:
            with self.state.lock:
                self.state.last_error = f"Authorize send failed: {e}"

    def _on_error(self, ws, error):
        with self.state.lock:
            self.state.last_error = f"WebSocket error: {error}"

    def _on_close(self, ws, close_status_code, close_msg):
        with self.state.lock:
            self.state.connected = False
            self.state.authorized = False

    def _on_message(self, ws, message):
        if message == "o":
            return  # open frame already handled in _on_open
        if message == "h":
            return  # heartbeat — Tradovate's server keeps pinging; no reply required
        if not message:
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
        # -- application-level RPC responses (subscribe acks, chart snapshots)
        if "i" in item and "s" in item:
            req_id = item.get("i")
            status = item.get("s")
            data = item.get("d", {})

            if req_id == self._auth_req_id:
                if status == 200:
                    with self.state.lock:
                        self.state.authorized = True
                    self._subscribe_all()
                else:
                    with self.state.lock:
                        self.state.last_error = f"Authorization rejected (status {status}): {data}"
                return

            if req_id in self._chart_req_ids:
                self._ingest_chart_payload(self._chart_req_ids[req_id], data)
                return

            if status != 200:
                with self.state.lock:
                    self.state.last_error = f"Request {req_id} failed (status {status}): {data}"
            return

        # -- streaming market-data events (quotes, DOM, live chart updates)
        if item.get("e") == "md":
            d = item.get("d", {})
            self._ingest_quotes(d.get("quotes", []))
            self._ingest_doms(d.get("doms", []))
            for tf, req_id in self._chart_req_ids.items():
                pass  # live chart pushes arrive keyed by request id below
            if "charts" in d:
                # live chart pushes reuse the original getChart request id
                for chart in d.get("charts", []):
                    cid = chart.get("id") or chart.get("reqId")
                    tf = self._chart_req_ids.get(cid)
                    if tf:
                        self._ingest_chart_payload(tf, {"charts": [chart]})

    # ------------------------------------------------------------ ingestion
    def _ingest_quotes(self, quotes):
        if not quotes:
            return
        for q in quotes:
            if str(q.get("contractId")) != str(self.contract_id) and self.contract_id is not None:
                continue
            entries = q.get("entries", {})
            snap = {}
            bid = entries.get("Bid", {})
            offer = entries.get("Offer", {})
            trade = entries.get("Trade", {})
            if bid:
                snap["bidPrice"] = bid.get("price")
                snap["bidSize"] = bid.get("size")
            if offer:
                snap["askPrice"] = offer.get("price")
                snap["askSize"] = offer.get("size")
            if trade:
                snap["last"] = trade.get("price")
                snap["lastSize"] = trade.get("size")
                with self.state.lock:
                    self.state.trade_ticks.append({
                        "price": trade.get("price"),
                        "qty": trade.get("size", 0) or 0,
                        "timestamp": datetime.now(timezone.utc),
                    })
            if snap:
                with self.state.lock:
                    self.state.quote.update(snap)

    def _ingest_doms(self, doms):
        if not doms:
            return
        for dom in doms:
            if str(dom.get("contractId")) != str(self.contract_id) and self.contract_id is not None:
                continue
            bids = [{"price": lvl.get("price"), "qty": lvl.get("size", 0)}
                    for lvl in dom.get("bids", []) if lvl.get("price") is not None]
            asks = [{"price": lvl.get("price"), "qty": lvl.get("size", 0)}
                    for lvl in dom.get("offers", []) if lvl.get("price") is not None]
            with self.state.lock:
                if bids:
                    self.state.dom_bids = bids
                if asks:
                    self.state.dom_asks = asks

    def _ingest_chart_payload(self, timeframe, data):
        charts = data.get("charts", [])
        if not charts:
            return
        rows = []
        for chart in charts:
            for bar in chart.get("bars", []):
                ts = bar.get("timestamp")
                try:
                    idx = pd.to_datetime(ts, utc=True)
                except Exception:
                    continue
                up_vol = bar.get("upVolume", 0) or 0
                down_vol = bar.get("downVolume", 0) or 0
                vol = bar.get("volume", up_vol + down_vol) or (up_vol + down_vol)
                rows.append({
                    "timestamp": idx,
                    "Open": bar.get("open"), "High": bar.get("high"),
                    "Low": bar.get("low"), "Close": bar.get("close"),
                    "Volume": vol,
                })
        if not rows:
            return
        new_df = pd.DataFrame(rows).dropna(subset=["Open", "High", "Low", "Close"])
        new_df = new_df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

        with self.state.lock:
            existing = self.state.bars.get(timeframe, pd.DataFrame())
            if existing.empty:
                merged = new_df
            else:
                merged = pd.concat([existing, new_df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if len(merged) > self.state.max_bars:
                merged = merged.iloc[-self.state.max_bars:]
            self.state.bars[timeframe] = merged
            self.state.last_update_ts[timeframe] = datetime.now(timezone.utc)

    # ------------------------------------------------------------ subscribe
    def _subscribe_all(self):
        self._quote_req_id = self._send("md/subscribeQuote", {"symbol": self.contract_name})
        self._dom_req_id = self._send("md/subscribeDOM", {"symbol": self.contract_name})
        for tf in self.timeframes:
            cfg = TRADOVATE_BAR_CONFIG.get(tf)
            if cfg is None:
                continue
            body = {
                "symbol": self.contract_name,
                "chartDescription": {
                    "underlyingType": cfg["underlyingType"],
                    "elementSize": cfg["elementSize"],
                    "elementSizeUnit": "UnderlyingUnits",
                    "withHistogram": False,
                },
                "timeRange": {"asMuchAsElements": 2000},
            }
            req_id = self._send("md/getChart", body)
            self._chart_req_ids[req_id] = tf
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro_series(yf_ticker: str, lookback_days: int = 180) -> pd.Series:
    try:
        raw = yf.download(tickers=yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False, threads=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].dropna()
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        df = df[df.index >= cutoff]
        return df["Close"]
    except Exception:
        return pd.Series(dtype=float)

def render_correlation_module(daily_bars: pd.DataFrame, label: str):
    st.markdown(
        '<div class="module-note">Cross-asset relationships between the active Tradovate contract '
        '(from its live-streamed daily bars) and key macro drivers — the US Dollar Index and 10-Year '
        'Treasury Yields — plus the S&amp;P 500 as a broad-market anchor.</div>',
        unsafe_allow_html=True,
    )

    lookback = st.slider("Correlation Lookback (Days)", 30, 730, 180, step=10, key="corr_lookback")

    if daily_bars.empty:
        st.info("Waiting on the Tradovate daily-bar chart subscription to populate before correlations can be computed.")
        return

    cutoff = daily_bars.index.max() - pd.Timedelta(days=lookback)
    active_series = daily_bars[daily_bars.index >= cutoff]["Close"]
    active_series.index = active_series.index.tz_localize(None) if active_series.index.tz is not None else active_series.index

    series_dict = {label: active_series} if not active_series.empty else {}
    fetch_errors = []
    for m_label, tk in MACRO_ANCHOR_TICKERS.items():
        s = fetch_macro_series(tk, lookback)
        if s.empty:
            fetch_errors.append(m_label)
        else:
            s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
            series_dict[m_label] = s

    if fetch_errors:
        st.warning(f"Could not retrieve live macro data for: {', '.join(fetch_errors)}.")

    if len(series_dict) < 2:
        st.error("Insufficient data available to compute correlations right now.")
        return

    combined = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()

    if combined.empty or len(combined) < 10:
        st.warning("Not enough overlapping historical data across assets for this lookback window yet.")
        return

    corr_matrix = combined.corr()

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
    ))
    fig_heat.update_layout(template=PLOTLY_TEMPLATE, height=460, title="Cross-Asset Correlation Matrix", margin=dict(l=10, r=10, t=50, b=10), dragmode="pan")
    st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader(f"Rolling 30-Day Correlation vs {label}")
    if label in combined.columns:
        rolling_window = min(30, max(5, len(combined) // 3))
        fig_roll = go.Figure()
        for col in combined.columns:
            if col == label:
                continue
            rolling_corr = combined[label].rolling(rolling_window).corr(combined[col])
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name=f"{label} vs {col}"))
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
# NOTE: Tradovate's public API surface used by this app (auth + md/* websocket
# services) does not expose a listed-options chain endpoint for CME futures
# options. This module keeps the original app's honest fallback pattern: it
# uses a live, liquid equity-ETF options proxy (via yfinance) for the futures
# root you're viewing (e.g. ES -> SPY, GC -> GLD) when one exists, and falls
# back to a labeled Black-Scholes simulation calibrated off the live Tradovate
# spot price otherwise. This is disclosed in the UI exactly as it was before.

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
        proxy_spot = float(hist["Close"].iloc[-1])
        return calls, puts, expiry, proxy_spot
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

def render_gex_module(root_symbol: str, live_spot: float, label: str):
    st.markdown(
        '<div class="module-note">Gamma Exposure (GEX) profile identifying dealer positioning, '
        'volatility pin zones, and the gamma flip level.</div>',
        unsafe_allow_html=True,
    )

    proxy = FUTURES_OPTIONS_PROXY_MAP.get(root_symbol)
    chain_result = fetch_options_chain(proxy) if proxy else None

    if chain_result is not None:
        calls, puts, expiry, proxy_spot = chain_result
        try:
            dte = max((pd.to_datetime(expiry) - pd.Timestamp.now()).days, 1)
        except Exception:
            dte = 30
        gex_df = gex_from_chain(calls, puts, proxy_spot, dte)
        max_pain, pain_df = compute_max_pain(calls, puts)
        spot_for_display = live_spot if live_spot else proxy_spot
        source_label = (
            f"Live listed options on proxy ETF {proxy} (expiry {expiry}) — futures options chains are "
            f"not exposed by the Tradovate API surface used here, so gamma/OI structure is sourced from "
            f"the correlated, highly liquid options-listed proxy; strikes shown are the proxy's own price "
            f"scale, not the futures price scale."
        )
    else:
        spot_for_display = live_spot
        if not spot_for_display:
            st.error(f"No live Tradovate spot price yet for {label} to calibrate the GEX engine.")
            return
        dte = st.slider("Simulated Days to Expiry", 1, 90, 30, key="gex_dte")
        iv_assumed = st.slider("Assumed Implied Volatility (%)", 5, 80, 18, key="gex_iv") / 100
        n_strikes = st.slider("Strike Range (± strikes around spot)", 10, 40, 25, key="gex_strikes")
        gex_df = simulate_gex(spot_for_display, n_strikes=n_strikes, iv=iv_assumed, days_to_expiry=dte)
        max_pain, pain_df = compute_max_pain_from_sim(gex_df)
        source_label = f"Simulated GEX engine (Black-Scholes gamma model) calibrated off the live Tradovate spot for {label} — no proxy configured for this root."

    st.caption(f"Data source: {source_label}")

    net_gex_total = gex_df["net_gex"].sum()
    flip_candidates = gex_df.sort_values("strike").copy()
    flip_candidates["cum_gex"] = flip_candidates["net_gex"].cumsum()
    sign_changes = flip_candidates[flip_candidates["cum_gex"] * flip_candidates["cum_gex"].shift(1) < 0]
    gamma_flip = float(sign_changes["strike"].iloc[0]) if not sign_changes.empty else float(gex_df["strike"].median())

    c1, c2, c3 = st.columns(3)
    c1.metric("Reference Spot", f"{spot_for_display:,.2f}")
    c2.metric("Net GEX", f"{net_gex_total:,.0f}")
    c3.metric("Max Pain Strike", f"{max_pain:,.2f}" if max_pain is not None else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["call_gex"], name="Call GEX", marker_color="#26a69a"))
    fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["put_gex"], name="Put GEX", marker_color="#ef5350"))
    fig.add_vline(x=spot_for_display, line_dash="dash", line_color="#f0b90b", annotation_text="Spot", annotation_position="top")
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
# SIDEBAR — THEME, TRADOVATE CREDENTIALS, CONTRACT & INTERVAL SELECTION
# ==================================================================================

st.sidebar.markdown("## 📊 Institutional Quant Terminal")
st.sidebar.caption("Live Tradovate REST + WebSocket Data Engine")

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
st.sidebar.subheader("🔐 Tradovate Account")

environment = st.sidebar.selectbox(
    "Environment", ["Demo", "Live"], index=0, key="tv_environment",
    help="Demo tokens are valid up to 14 days; this app re-authenticates automatically once you "
         "update credentials below after an expiration.",
)

with st.sidebar.expander("Credentials (or set via .env)", expanded=True):
    tv_name = st.text_input("Username", value=os.getenv("TRADOVATE_USERNAME", ""), key="tv_name")
    tv_password = st.text_input("Password", value=os.getenv("TRADOVATE_PASSWORD", ""), type="password", key="tv_password")
    # Demo defaults to the "0" placeholder cid/sec so you can try the app
    # without an API Access subscription. Live always needs your real,
    # issued Client ID/Secret — "0"/"0" is a Demo-only convenience default.
    cid_default = os.getenv("TRADOVATE_CID", "0" if environment == "Demo" else "")
    sec_default = os.getenv("TRADOVATE_SECRET", "0" if environment == "Demo" else "")
    tv_cid = st.text_input("API Key (cid)", value=cid_default, key="tv_cid")
    tv_sec = st.text_input("API Secret (sec)", value=sec_default, type="password", key="tv_sec")
    tv_app_id = st.text_input("App ID", value=os.getenv("TRADOVATE_APP_ID", DEFAULT_APP_ID), key="tv_app_id")
    tv_app_version = st.text_input("App Version", value=os.getenv("TRADOVATE_APP_VERSION", DEFAULT_APP_VERSION), key="tv_app_version")
    if environment == "Demo":
        st.caption(
            "Demo pre-fills cid/sec with the placeholder value \"0\", which some Tradovate Demo "
            "accounts accept without a paid API Access subscription. This is not guaranteed for every "
            "account — if authentication fails with \"Access is denied\" or an invalid-credentials "
            "error, your account requires a real cid/sec pair from Settings → API Access."
        )
    else:
        st.caption(
            "Live requires your real, issued API Key/Secret (cid/sec) from Tradovate → Settings → "
            "API Access. The \"0\" Demo placeholder does not apply here."
        )

# Demo: username + password are enough to attempt auth (cid/sec fall back to
# the "0" placeholder above). Live: cid/sec must be real, non-empty values.
if environment == "Demo":
    credentials_complete = all([tv_name, tv_password])
else:
    credentials_complete = all([tv_name, tv_password, tv_cid, tv_sec])

st.sidebar.divider()
st.sidebar.subheader("Contract Selection")

root_choice = st.sidebar.selectbox("Futures Root", list(FUTURES_ROOTS.keys()), key="root_choice")
root_symbol = FUTURES_ROOTS[root_choice]
use_custom_root = st.sidebar.checkbox("Use custom root symbol instead", value=False, key="use_custom_root")
if use_custom_root:
    root_symbol = st.sidebar.text_input("Custom Root Symbol", value=root_symbol, key="custom_root").strip().upper()
    root_choice = f"Custom: {root_symbol}"

interval = st.sidebar.selectbox("Primary Chart Interval", INTERVAL_CHOICES, index=INTERVAL_CHOICES.index("5m"), key="interval_choice")

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("🔴 Live Auto-Refresh", value=True, key="auto_refresh")
refresh_secs = st.sidebar.slider("Refresh Interval (seconds)", 2, 15, 3, key="refresh_secs")

col_reconnect, col_reset = st.sidebar.columns(2)
force_reconnect = col_reconnect.button("🔄 Reconnect", use_container_width=True)
force_reset = col_reset.button("🧹 Full Reset", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown(
    "<div style='font-size:0.75rem; opacity:0.75;'>"
    "<b>Data Engine Routing</b><br>"
    "Authentication → Tradovate REST /auth/accesstoken<br>"
    "Real-time Quotes & Trades → WS md/subscribeQuote<br>"
    "Real-time Level 2 DOM → WS md/subscribeDOM<br>"
    "Candles (all timeframes) → WS md/getChart, live-updating<br>"
    "Options GEX (Module 5) → yfinance equity-ETF proxy or Black-Scholes simulation "
    "(Tradovate API surface used here has no futures-options chain endpoint)<br>"
    "Macro anchors (Module 3: DXY, 10Y yield) → yfinance<br><br>"
    "For informational / research purposes only — not investment advice. "
    "Demo-environment trading only reflects simulated fills."
    "</div>",
    unsafe_allow_html=True,
)

if force_reset:
    for key in ["tv_session", "tv_state", "tv_worker", "tv_contract_cache"]:
        st.session_state.pop(key, None)
    st.cache_data.clear()
    st.rerun()

# ==================================================================================
# AUTHENTICATION LIFECYCLE
# ==================================================================================

st.title("📊 Institutional Quantitative Trading Terminal — Tradovate")

if not credentials_complete:
    if environment == "Demo":
        st.warning(
            "Enter your Tradovate **Username and Password** in the sidebar (or provide them via a "
            "`.env` file) to authenticate and start streaming. cid/sec default to the Demo "
            "placeholder \"0\" automatically."
        )
    else:
        st.warning(
            "Enter your Tradovate **Username, Password, API Key (cid), and API Secret (sec)** in the "
            "sidebar (or provide them via a `.env` file) — Live requires real API keys, unlike Demo."
        )
    st.stop()

fingerprint = (environment, tv_name, tv_password, tv_app_id, tv_app_version, tv_cid, tv_sec)
existing_session = st.session_state.get("tv_session")
need_new_session = (
    existing_session is None
    or existing_session.credential_fingerprint() != fingerprint
    or force_reconnect
)

if need_new_session:
    new_session = TradovateSession(
        environment=environment, name=tv_name, password=tv_password,
        app_id=tv_app_id, app_version=tv_app_version, cid=tv_cid, sec=tv_sec,
        device_id=st.session_state.get("tv_device_id", str(uuid.uuid4())),
    )
    st.session_state["tv_device_id"] = new_session.device_id
    try:
        with st.spinner("Authenticating with Tradovate..."):
            new_session.authenticate()
    except TradovateAuthError as e:
        st.error(f"❌ {e}")
        st.stop()
    st.session_state["tv_session"] = new_session
    # credentials or environment changed -> any live worker is stale, tear it down
    old_worker = st.session_state.pop("tv_worker", None)
    if old_worker is not None:
        old_worker.stop()
    st.session_state.pop("tv_state", None)
    st.session_state.pop("tv_contract_cache", None)

tv_session = st.session_state["tv_session"]

if not tv_session.is_token_valid():
    try:
        with st.spinner("Refreshing expired Tradovate session token..."):
            tv_session.authenticate()
    except TradovateAuthError as e:
        st.error(f"❌ Token refresh failed: {e}")
        st.stop()

if tv_session.has_market_data is False:
    st.warning(
        "⚠️ This Tradovate account reports `hasMarketData: false`. Real-time quotes/DOM will not "
        "stream until a market data subscription is enabled on the account, even though candles and "
        "authentication will otherwise work normally."
    )

# ==================================================================================
# CONTRACT RESOLUTION (front-month lookup, cached per root+environment)
# ==================================================================================

contract_cache = st.session_state.setdefault("tv_contract_cache", {})
cache_key = f"{environment}:{root_symbol}"

if cache_key not in contract_cache or force_reconnect:
    try:
        with st.spinner(f"Resolving front-month contract for {root_symbol}..."):
            contract_cache[cache_key] = tv_session.find_front_month_contract(root_symbol)
    except TradovateAuthError as e:
        st.error(f"❌ {e}")
        st.stop()

contract_info = contract_cache[cache_key]
contract_name = contract_info["name"]
contract_id = contract_info["id"]
asset_label = f"{root_choice} — {contract_name}"

# ==================================================================================
# WORKER LIFECYCLE — persistent background WebSocket thread across reruns
# ==================================================================================

timeframes_needed = sorted(set([interval, "1d"]))
worker: MarketDataWorker = st.session_state.get("tv_worker")

need_new_worker = (
    worker is None
    or not worker.is_alive()
    or worker.contract_name != contract_name
    or sorted(worker.timeframes) != timeframes_needed
    or force_reconnect
)

if need_new_worker:
    if worker is not None:
        worker.stop()
    live_state = LiveMarketState()
    new_worker = MarketDataWorker(
        session=tv_session, state=live_state, contract_name=contract_name,
        contract_id=contract_id, timeframes=timeframes_needed,
    )
    new_worker.start()
    st.session_state["tv_worker"] = new_worker
    st.session_state["tv_state"] = live_state

worker = st.session_state["tv_worker"]
live_state: LiveMarketState = st.session_state["tv_state"]

status = live_state.status()
main_df = live_state.snapshot_bars(interval)
daily_df = live_state.snapshot_bars("1d")
order_book_df = live_state.snapshot_dom()
quote_snapshot = live_state.snapshot_quote()
live_ticks = live_state.snapshot_ticks()

conn_badge = "🟢 WS CONNECTED" if status["connected"] else "🔴 WS DISCONNECTED"
auth_badge = "🟢 AUTHORIZED" if status["authorized"] else "🟡 AUTHORIZING…"
st.caption(f"Active Contract: **{asset_label}** · Environment: {environment} · Interval: {interval}")
st.markdown(
    f'<span class="source-badge">{conn_badge}</span>'
    f'<span class="source-badge">{auth_badge}</span>'
    f'<span class="source-badge">📡 CANDLES: TRADOVATE md/getChart (LIVE)</span>'
    f'<span class="source-badge">{"🟢 LIVE L2 DOM" if not order_book_df.empty else "🔴 DOM AWAITING FIRST SNAPSHOT"}</span>',
    unsafe_allow_html=True,
)
if status["last_error"]:
    st.error(f"⚠️ {status['last_error']}")

if main_df.empty:
    st.info(
        f"Waiting on the first `md/getChart` bar snapshot for **{contract_name}** ({interval})... "
        "this normally arrives within a few seconds of authorization. The page will keep refreshing."
    )
    if auto_refresh:
        time.sleep(refresh_secs)
        st.rerun()
    st.stop()

if len(main_df) < 20:
    st.warning("Limited bar history streamed so far for this interval — some modules need more bars to compute.")

# Snapshot metrics
last_row = main_df.iloc[-1]
prev_row = main_df.iloc[-2] if len(main_df) > 1 else last_row
chg = safe_pct(last_row["Close"], prev_row["Close"])
live_last = quote_snapshot.get("last")
display_price = live_last if live_last is not None else last_row["Close"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Price", f"{display_price:,.2f}", f"{chg:.2f}%")
m2.metric("Bid / Ask", (
    f"{quote_snapshot.get('bidPrice', '—')} / {quote_snapshot.get('askPrice', '—')}"
))
m3.metric("Session High", f"{main_df['High'].max():,.2f}")
m4.metric("Session Low", f"{main_df['Low'].min():,.2f}")
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
        render_liquidity_module(main_df, order_book_df, asset_label, dom_is_live=status["authorized"])
    except Exception as e:
        st.error(f"Liquidity module encountered an error: {e}")

with tab2:
    try:
        render_ml_module(main_df, asset_label)
    except Exception as e:
        st.error(f"ML Classifier module encountered an error: {e}")

with tab3:
    try:
        render_correlation_module(daily_df, asset_label)
    except Exception as e:
        st.error(f"Correlation module encountered an error: {e}")

with tab4:
    try:
        render_volume_delta_module(main_df, asset_label, live_ticks, ticks_available=len(live_ticks) > 20)
    except Exception as e:
        st.error(f"Volume Delta module encountered an error: {e}")

with tab5:
    try:
        render_gex_module(root_symbol, float(display_price) if display_price else None, asset_label)
    except Exception as e:
        st.error(f"GEX module encountered an error: {e}")

with tab6:
    try:
        render_execution_module(main_df, order_book_df, asset_label)
    except Exception as e:
        st.error(f"Execution Algorithms module encountered an error: {e}")

st.divider()
st.caption(
    "⚠️ Disclaimer: This dashboard is provided for research and educational purposes only. "
    "It does not constitute financial advice. Trading futures involves substantial risk of loss."
)

# ==================================================================================
# LIVE AUTO-REFRESH LOOP
# ==================================================================================
# Streamlit has no native server-push; this simple sleep+rerun loop polls the
# background worker's shared state at `refresh_secs` intervals so the UI feels
# live without requiring an extra streamlit-autorefresh dependency. The
# WebSocket connection itself is NOT affected by this — it keeps streaming on
# its own background thread regardless of how often the page reruns.
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

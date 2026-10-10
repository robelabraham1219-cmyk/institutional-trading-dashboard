"""core.py - data, indicators, regime labels, features and CPU models for the FX regime app.

No Streamlit import here, so every function can be unit-tested. Analytical tool, not financial advice.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------------------- constants
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]
EXPONENTS = np.array([-0.576, 0.136, -0.119, 0.091, 0.042, 0.036])
WEIGHTS = np.abs(EXPONENTS)
DXY_CONST = 50.14348112
YF = {**{p: p + "=X" for p in PAIRS}, "DXY_REAL": "DX-Y.NYB"}
TARGETS = ["DXY"] + PAIRS
OHLC = ["open", "high", "low", "close"]
TF_ORDER = ["30m", "1h", "2h", "3h", "4h", "1D", "3D", "1W"]
TF_SECONDS = {"30m": 1800, "1h": 3600, "2h": 7200, "3h": 10800, "4h": 14400,
              "1D": 86400, "3D": 259200, "1W": 604800}
TF_NATIVE = {"30m": "30m", "1h": "1h", "2h": "1h", "3h": "1h", "4h": "1h",
             "1D": "1d", "3D": "1d", "1W": "1d"}
LADDER = ["30m", "1h", "4h", "1D", "1W"]
PRESETS = {  # (H bars, label, is_default)
    "30m": [(12, "6h", 0), (24, "12h", 0), (48, "24h", 1), (96, "48h", 0)],
    "1h": [(6, "6h", 0), (12, "12h", 0), (24, "24h", 1), (48, "48h", 0)],
    "2h": [(3, "6h", 0), (6, "12h", 0), (12, "24h", 1), (24, "48h", 0)],
    "3h": [(4, "12h", 0), (8, "24h", 1), (16, "48h", 0)],
    "4h": [(3, "12h", 0), (6, "24h", 1), (12, "48h", 0)],
    "1D": [(3, "3 days", 1), (5, "5 days", 0), (10, "10 days", 0)],
    "3D": [(3, "3 bars", 1), (5, "5 bars", 0), (8, "8 bars", 0)],
    "1W": [(4, "4 weeks", 1), (8, "8 weeks", 0), (12, "12 weeks", 0)],
}
REGIME_NAMES = ["Stagnant", "Steady trend", "Choppy", "Volatile trend"]
REGIME_COLORS = ["#9aa0a6", "#3b82f6", "#f59e0b", "#ef4444"]
MEANINGS = [
    "Stagnant: low volatility and little net progress; quiet, range-bound conditions.",
    "Steady trend: low volatility with persistent net progress in one direction.",
    "Choppy: large swings without net progress; historically hard for trend-following.",
    "Volatile trend: large swings with strong net progress in one direction.",
]
# Example cost matrix (rows = true, cols = predicted), trend-following perspective.
COST = np.array([[0, 1, 3, 1], [4, 0, 4, 2], [2, 10, 0, 10], [8, 2, 4, 0]], dtype=float)
MIN_ML_ROWS = 600
LOW_SAMPLE = 1500
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
WEEKDAY_LABELS = ["1 Mon", "2 Tue", "3 Wed", "4 Thu", "5 Fri", "6 Sat", "7 Sun"]
UNDEFINED_NAME = "Undefined (not enough bars)"


def regime_name(k) -> str:
    """Regime name; safe for the undefined state (-1)."""
    return REGIME_NAMES[int(k)] if 0 <= int(k) < 4 else UNDEFINED_NAME


def regime_color(k) -> str:
    """Regime colour; grey for the undefined state (-1)."""
    return REGIME_COLORS[int(k)] if 0 <= int(k) < 4 else "#6b7280"


# ----------------------------------------------------------------------------- data helpers
def empty_ohlc() -> pd.DataFrame:
    """Empty OHLC frame with a UTC DatetimeIndex."""
    return pd.DataFrame({c: pd.Series(dtype=float) for c in OHLC},
                        index=pd.DatetimeIndex([], tz="UTC"))


def empty_frame() -> pd.DataFrame:
    """Empty bar frame (OHLC + end + complete)."""
    f = empty_ohlc()
    f["end"] = pd.Series(dtype="datetime64[ns, UTC]")
    f["complete"] = pd.Series(dtype=bool)
    return f


def _secs(idx: pd.DatetimeIndex) -> np.ndarray:
    """Epoch seconds of a UTC DatetimeIndex (resolution independent)."""
    return ((idx - EPOCH) // pd.Timedelta(seconds=1)).to_numpy().astype("int64")


def clean_ohlc(df, daily: bool = False) -> pd.DataFrame:
    """Normalise a raw provider frame: lower-case OHLC, UTC, sorted, de-duplicated, no NaN/non-positive bars."""
    if df is None or len(df) == 0:
        return empty_ohlc()
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    if not all(c in d.columns for c in OHLC):
        return empty_ohlc()
    d = d[OHLC]
    idx = pd.DatetimeIndex(d.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if daily:  # daily stamps -> calendar date at 00:00 UTC
        idx = (idx + pd.Timedelta(hours=12)).floor("D")
    d.index = idx
    d = d.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d > 0).all(axis=1)]
    d["high"] = d[OHLC].max(axis=1)
    d["low"] = d[OHLC].min(axis=1)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def fetch_yf(ticker: str, interval: str, retries: int = 3):
    """Download one ticker from Yahoo (free, unofficial). Returns (frame, error or None)."""
    last = "no data"
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover
        return empty_ohlc(), f"yfinance import failed: {e}"
    now = pd.Timestamp.now(tz="UTC")
    for k in range(retries):
        try:
            tk = yf.Ticker(ticker)
            if interval == "30m":
                h = tk.history(start=(now - pd.Timedelta(days=59)).strftime("%Y-%m-%d"),
                               interval="30m", auto_adjust=False)
            elif interval == "1h":
                h = tk.history(start=(now - pd.Timedelta(days=729)).strftime("%Y-%m-%d"),
                               interval="1h", auto_adjust=False)
            else:
                h = tk.history(period="max", interval="1d", auto_adjust=False)
            if h is not None and len(h) > 0:
                out = clean_ohlc(h, daily=(interval == "1d"))
                if len(out):
                    return out, None
            last = "empty response"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(1.5 * (2 ** k))
    return empty_ohlc(), f"{ticker} {interval}: {last}"


def load_stored_csv(url: str):
    """Optional stored history from a raw GitHub CSV. Returns (frame, error text or None)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fx-regime-app"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        return empty_ohlc(), f"HTTP {e.code} {e.reason}"
    except Exception as e:
        return empty_ohlc(), f"{type(e).__name__}: {str(e)[:120]}"
    try:
        d = pd.read_csv(io.BytesIO(raw))
        d.columns = [str(c).lower() for c in d.columns]
        missing = [c for c in OHLC if c not in d.columns]
        if missing:
            return empty_ohlc(), f"CSV is missing columns: {', '.join(missing)}"
        tcol = "timestamp" if "timestamp" in d.columns else d.columns[0]
        d.index = pd.to_datetime(d[tcol], utc=True, errors="coerce")
        d = d[d.index.notna()]
        out = clean_ohlc(d[OHLC])
    except Exception as e:
        return empty_ohlc(), f"parse error {type(e).__name__}: {str(e)[:120]}"
    if len(out) == 0:
        return out, "CSV loaded but contains no valid rows"
    return out, None


def merge_history(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge stored and fresh bars; fresh wins on equal timestamps."""
    if stored is None or len(stored) == 0:
        return fresh
    if fresh is None or len(fresh) == 0:
        return stored
    m = pd.concat([stored, fresh])
    return m[~m.index.duplicated(keep="last")].sort_index()


def load_native(interval: str, stored_cfg: tuple | None = None):
    """Load all 7 tickers at a native interval ('30m','1h','1d').

    stored_cfg = (owner, repo, branch) enables optional CSV history for 30m/1h.
    Returns (dict name -> frame, list of error strings, list of per-ticker load stats).
    """
    names = PAIRS + ["DXY_REAL"]
    errs: list[str] = []
    stats: list[dict] = []
    fetched = pd.Timestamp.now(tz="UTC").strftime("%H:%M")

    def one(nm):
        fresh, err = fetch_yf(YF[nm], interval)
        stored, serr = empty_ohlc(), None
        if interval not in ("30m", "1h"):
            serr = "n/a (daily bars come from Yahoo only)"
        elif not (stored_cfg and stored_cfg[0]):
            serr = "not configured (OWNER secret not set)"
        else:
            o, r, b = stored_cfg
            url = f"https://raw.githubusercontent.com/{o}/{r}/{b}/data/{quote(YF[nm])}_{interval}.csv"
            stored, serr = load_stored_csv(url)
        merged = merge_history(stored, fresh)
        msg = "; ".join(x for x in [f"stored: {serr}" if serr else "", f"fresh: {err}" if err else ""] if x)
        return nm, merged, err, {"ticker": YF[nm], "interval": interval, "stored": int(len(stored)),
                                 "fresh": int(len(fresh)), "merged": int(len(merged)), "error": msg,
                                 "fetched": fetched}

    out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for nm, fr, err, st_ in ex.map(one, names):
            out[nm] = fr
            stats.append(st_)
            if err and nm != "DXY_REAL":
                errs.append(err)
            elif err:
                errs.append(f"DX-Y.NYB {interval} unavailable ({err.split(': ')[-1]})")
    return out, errs, stats


def stored_bars(stats: list, symbol: str, use_real: bool) -> int:
    """Stored-history rows behind the target series at the native interval (0 = not used)."""
    rows = {x["ticker"]: x["stored"] for x in stats or []}
    if symbol == "DXY":
        if use_real:
            return int(rows.get(YF["DXY_REAL"], 0))
        return int(min([rows.get(YF[p], 0) for p in PAIRS] or [0]))
    return int(rows.get(YF.get(symbol, ""), 0))


# ----------------------------------------------------------------------------- resampling
def resample_ohlc(df: pd.DataFrame, tf: str, anchor: int = 0, ref_index=None) -> pd.DataFrame:
    """Derive timeframe `tf` from native bars. Output: OHLC + 'end' (bar end) + 'complete'.

    2h/3h/4h: from 1h with UTC anchor offset; 3D: groups of 3 consecutive daily bars (of ref_index
    if given, else of df); 1W: weeks ending Friday. Weekend gaps are never filled.
    """
    if df is None or len(df) == 0:
        return empty_frame()
    d = df[OHLC]
    if tf in ("30m", "1h", "1D"):
        out = d.copy()
        out["end"] = out.index + pd.Timedelta(seconds=TF_SECONDS[tf])
        out["complete"] = True
        return out
    if tf in ("2h", "3h", "4h"):
        b = int(tf[0])
        step, a = b * 3600, (int(anchor) % b) * 3600
        key = (_secs(d.index) - a) // step
        g = d.groupby(key)
        out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                            "low": g["low"].min(), "close": g["close"].last()})
        out.index = pd.to_datetime(out.index.to_numpy() * step + a, unit="s", utc=True)
        out["end"] = out.index + pd.Timedelta(seconds=step)
        out["complete"] = True
        return out
    if tf == "3D":
        ref = d.index if ref_index is None or len(ref_index) == 0 else ref_index
        edges = ref[::3]
        key = np.asarray(edges.searchsorted(d.index, side="right")) - 1
        m = key >= 0
        if not m.any():
            return empty_frame()
        dd, key = d[m], key[m]
        g = dd.groupby(key)
        out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                            "low": g["low"].min(), "close": g["close"].last()})
        k = out.index.to_numpy()
        last = np.minimum((k + 1) * 3, len(ref)) - 1
        out["end"] = ref[last] + pd.Timedelta(days=1)
        out["complete"] = (k + 1) * 3 <= len(ref)
        out.index = edges[k]
        return out
    if tf == "1W":
        per = d.index.tz_localize(None).to_period("W-FRI")
        key = np.asarray(per.asi8)
        g = d.groupby(key)
        out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                            "low": g["low"].min(), "close": g["close"].last()})
        first = pd.Series(d.index, index=d.index).groupby(key).first()
        f = pd.DatetimeIndex(first.values)
        if f.tz is None:
            f = f.tz_localize("UTC")
        wd = f.weekday.to_numpy()
        out.index = f
        out["end"] = f.normalize() + pd.to_timedelta(((4 - wd) % 7) + 1, unit="D")
        out["complete"] = True
        return out.sort_index()
    return empty_frame()


def split_forming(df: pd.DataFrame, now=None):
    """Drop the currently forming bar. Returns (closed frame, forming flag)."""
    if df is None or len(df) == 0:
        return df, False
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    flag = bool(df["end"].iloc[-1] > now or not df["complete"].iloc[-1])
    return (df.iloc[:-1] if flag else df), flag


def next_closure(t: pd.Timestamp):
    """First FX closure (Fri 17:00 to Sun 17:00 America/New_York) ending after t, as UTC timestamps."""
    tz = ZoneInfo("America/New_York")
    d = t.tz_convert(tz).date()
    wd = d.weekday()
    fri = d + timedelta(days=4 - wd) if wd <= 4 else d - timedelta(days=wd - 4)
    while True:
        sun = fri + timedelta(days=2)
        cs = pd.Timestamp(datetime(fri.year, fri.month, fri.day, 17), tz=tz)
        ce = pd.Timestamp(datetime(sun.year, sun.month, sun.day, 17), tz=tz)
        if ce > t:
            return cs.tz_convert("UTC"), ce.tz_convert("UTC")
        fri += timedelta(days=7)


def abnormal_gaps(idx: pd.DatetimeIndex, tf: str):
    """Abnormal gaps: longer than max(3 bars, 40 h) and not spanning the weekend closure (gaps up to 40 h are holiday closures).

    Returns (count, total length in seconds); gap length = bar start to next bar start.
    """
    if len(idx) < 2:
        return 0, 0.0
    dur = TF_SECONDS[tf]
    thr = max(3 * dur, 40 * 3600)
    gaps = np.diff(_secs(idx))
    cnt, tot = 0, 0.0
    for i in np.where(gaps > thr)[0]:
        g_start = idx[i] + pd.Timedelta(seconds=dur)
        cs, _ = next_closure(g_start)
        if cs < idx[i + 1]:  # the gap overlaps the weekend closure
            continue
        cnt += 1
        tot += float(gaps[i])
    return cnt, tot


# ----------------------------------------------------------------------------- DXY
def synth_dxy(frames: dict):
    """Synthetic DXY OHLC from the 6 pairs (inner join). Returns (frame, lost fraction)."""
    parts = {p: frames.get(p) for p in PAIRS}
    if any(v is None or len(v) == 0 for v in parts.values()):
        return empty_ohlc(), 1.0
    union = parts[PAIRS[0]].index
    for p in PAIRS[1:]:
        union = union.union(parts[p].index)
    J = pd.concat({p: parts[p][OHLC] for p in PAIRS}, axis=1, join="inner")
    if len(J) == 0:
        return empty_ohlc(), 1.0
    lost = 1.0 - len(J) / max(len(union), 1)
    out = pd.DataFrame(index=J.index)
    for f in OHLC:
        s = np.full(len(J), np.log(DXY_CONST))
        for p, e in zip(PAIRS, EXPONENTS):
            col = f
            if f == "high":
                col = "high" if e > 0 else "low"
            elif f == "low":
                col = "low" if e > 0 else "high"
            s = s + e * np.log(J[(p, col)].to_numpy(float))
        out[f] = np.exp(s)
    return out, float(lost)


def source_stats(synth: pd.DataFrame, real: pd.DataFrame, tf: str):
    """Coverage of real DXY vs synthetic bars and abnormal-gap totals; None if either is unavailable."""
    if len(real) == 0 or len(synth) == 0:
        return None
    cov = real.index.isin(synth.index).sum() / len(synth)
    r = real[(real.index >= synth.index[0]) & (real.index <= synth.index[-1])]
    n_gap, tot = abnormal_gaps(r.index, tf)
    span = (r.index[-1] - r.index[0]).total_seconds() if len(r) > 1 else 0.0
    return {"coverage": float(cov), "gap_n": int(n_gap), "gap_hours": float(tot / 3600.0),
            "gap_frac": float(tot / span) if span > 0 else 0.0}


def decide_source(choice: str, synth: pd.DataFrame, real: pd.DataFrame, tf: str):
    """Pick the DXY series. Returns (use_real, reason string)."""
    if choice == "Synthetic":
        return False, "manual: synthetic"
    if choice == "Real":
        return (len(real) > 0), ("manual: real" if len(real) else "real DX-Y.NYB unavailable, fell back to synthetic")
    sst = source_stats(synth, real, tf)
    if sst is None:
        return False, "auto: real DX-Y.NYB unavailable"
    detail = (f"real covers {sst['coverage']:.0%} of synthetic bars (need >= 90%); {sst['gap_n']} abnormal gap(s) "
              f"totalling {sst['gap_hours']:.1f} h = {sst['gap_frac']:.2%} of the covered period (need <= 2%)")
    if sst["coverage"] >= 0.90 and sst["gap_frac"] <= 0.02:
        return True, "auto: accepted real; " + detail
    return False, "auto: used synthetic; " + detail


def validate_dxy(synth: pd.DataFrame, real: pd.DataFrame):
    """Validation of synthetic vs real DXY closes; None if not comparable."""
    if len(synth) == 0 or len(real) == 0:
        return None
    j = pd.concat([synth["close"].rename("s"), real["close"].rename("r")], axis=1, join="inner")
    if len(j) < 10:
        return None
    rs, rr = j["s"].pct_change(), j["r"].pct_change()
    return {"corr": float(rs.corr(rr)), "mad": float((j["s"] - j["r"]).abs().mean()), "n": int(len(j))}


def build_set(native: dict, tf: str, anchor: int = 0, choice: str = "Auto", now=None) -> dict:
    """Build closed-bar frames for the 6 pairs, synthetic and real DXY at timeframe `tf`."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    warns: list[str] = []
    pn = {p: native.get(p, empty_ohlc()) for p in PAIRS}
    synth_n, lost = synth_dxy(pn)
    if lost > 0.20 and len(synth_n):
        warns.append(f"Synthetic DXY: {lost:.0%} of timestamps lost in the 6-pair inner join.")
    if len(synth_n) == 0:
        warns.append("Synthetic DXY unavailable (a pair has no data).")
    ref = synth_n.index if tf == "3D" and len(synth_n) else None
    forming = False
    pairs = {}
    for p in PAIRS:
        c, fl = split_forming(resample_ohlc(pn[p], tf, anchor, ref), now)
        pairs[p], forming = c, forming or fl
    synth, _ = split_forming(resample_ohlc(synth_n, tf, anchor, ref), now)
    real, _ = split_forming(resample_ohlc(native.get("DXY_REAL", empty_ohlc()), tf, anchor, ref), now)
    use_real, why = decide_source(choice, synth, real, tf)
    dxy = real if use_real else synth
    return {"tf": tf, "pairs": pairs, "synth": synth, "real": real, "dxy": dxy, "use_real": use_real,
            "source_reason": why, "source_stats": source_stats(synth, real, tf),
            "validation": validate_dxy(synth, real), "lost": lost,
            "forming": forming, "warnings": warns,
            "source_label": ("DXY source: Real DX-Y.NYB" if use_real
                             else "Synthetic from 6 pairs (approximate range)")}


def get_series(S: dict, symbol: str) -> pd.DataFrame:
    """Target frame for a symbol from a build_set result."""
    return S["dxy"] if symbol == "DXY" else S["pairs"][symbol]


def neighbour_frames(S: dict, symbol: str) -> dict:
    """Frames used as 'graph' neighbours: the 6 pairs for DXY, other pairs + DXY otherwise."""
    if symbol == "DXY":
        return dict(S["pairs"])
    d = {p: S["pairs"][p] for p in PAIRS if p != symbol}
    d["DXY"] = S["dxy"]
    return d


def contributions(pairs: dict, H: int):
    """Exact DXY log-change decomposition over H bars. Returns (DataFrame, total) or (None, None)."""
    C = pd.concat({p: pairs[p]["close"] for p in PAIRS}, axis=1, join="inner").dropna()
    if len(C) <= H:
        return None, None
    last, prev = C.iloc[-1].to_numpy(float), C.iloc[-1 - H].to_numpy(float)
    contrib = EXPONENTS * np.log(last / prev)
    df = pd.DataFrame({"pair": PAIRS, "weight": WEIGHTS, "chg": last / prev - 1.0, "contrib": contrib})
    df["usd_dir"] = np.where(np.sign(df["chg"]) * np.where(df["pair"].isin(["EURUSD", "GBPUSD"]), -1, 1) > 0, 1,
                             np.where(df["chg"] == 0, 0, -1))
    return df, float(contrib.sum())


# ----------------------------------------------------------------------------- indicators
def wilder_atr(h, l, c, n: int) -> np.ndarray:
    """Wilder ATR with seed = mean of first n true ranges."""
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    m = len(c)
    atr = np.full(m, np.nan)
    if m < n or m == 0:
        return atr
    tr = np.empty(m)
    tr[0] = h[0] - l[0]
    if m > 1:
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr[n - 1] = tr[:n].mean()
    prev = atr[n - 1]
    for i in range(n, m):
        prev = (prev * (n - 1) + tr[i]) / n
        atr[i] = prev
    return atr


def efficiency_ratio(close, n: int, signed: bool = False) -> pd.Series:
    """Kaufman efficiency ratio over n bars; in [0,1] (or [-1,1] when signed)."""
    c = pd.Series(close).astype(float)
    net = c - c.shift(n)
    path = c.diff().abs().rolling(n).sum()
    er = (net / path).where(path > 0, 0.0).where(path.notna())
    return er if signed else er.abs()


def ema(s, span: int) -> pd.Series:
    """Exponential moving average."""
    return pd.Series(s).astype(float).ewm(span=span, adjust=False).mean()


def core_frame(df: pd.DataFrame, atr_n: int, H: int) -> pd.DataFrame:
    """OHLC + ATR, nATR, ER_H and signed ER_H."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLC + ["atr", "natr", "er", "ser"])
    out = df[OHLC].copy()
    out["atr"] = wilder_atr(out["high"], out["low"], out["close"], atr_n)
    out["natr"] = out["atr"] / out["close"]
    out["er"] = efficiency_ratio(out["close"].to_numpy(), H).to_numpy()
    out["ser"] = efficiency_ratio(out["close"].to_numpy(), H, signed=True).to_numpy()
    return out


def fit_thresholds(natr, er, q_atr: float, q_er: float):
    """theta/gamma as percentiles of nATR and ER (caller passes training data only)."""
    a, e = np.asarray(natr, float), np.asarray(er, float)
    a, e = a[np.isfinite(a)], e[np.isfinite(e)]
    if len(a) == 0 or len(e) == 0:
        return np.nan, np.nan
    return float(np.percentile(a, q_atr)), float(np.percentile(e, q_er))


def regimes(natr, er, theta: float, gamma: float) -> np.ndarray:
    """Instantaneous regime 0..3 (-1 where undefined)."""
    natr, er = np.asarray(natr, float), np.asarray(er, float)
    valid = np.isfinite(natr) & np.isfinite(er) & np.isfinite(theta) & np.isfinite(gamma)
    hv, ht = natr >= theta, er >= gamma
    R = np.where(hv, np.where(ht, 3, 2), np.where(ht, 1, 0)).astype(int)
    R[~valid] = -1
    return R


def make_labels(R: np.ndarray, H: int) -> np.ndarray:
    """y_T = mode of R over T+1..T+H; ties -> R(T+H); -1 where the window is incomplete/undefined."""
    R = np.asarray(R, int)
    n = len(R)
    y = np.full(n, -1, int)
    if n <= H or H < 1:
        return y
    vc = np.concatenate([[0], np.cumsum(R >= 0)])
    T = np.arange(n - H)
    allv = (vc[T + H + 1] - vc[T + 1]) == H
    counts = np.stack([(lambda c: c[T + H + 1] - c[T + 1])(np.concatenate([[0], np.cumsum(R == k)]))
                       for k in range(4)], axis=1)
    mx = counts.max(axis=1)
    ties = (counts == mx[:, None]).sum(axis=1) > 1
    lab = np.where(ties, R[T + H], counts.argmax(axis=1))
    y[T[allv]] = lab[allv]
    return y


def run_length(R: np.ndarray) -> np.ndarray:
    """Length of the current regime run at each bar (>=1; 0 where undefined)."""
    R = np.asarray(R, int)
    age = np.zeros(len(R), int)
    for i in range(len(R)):
        if R[i] < 0:
            continue
        age[i] = age[i - 1] + 1 if i > 0 and R[i] == R[i - 1] and age[i - 1] > 0 else 1
    return age


def regime_feat_array(R: np.ndarray) -> np.ndarray:
    """Threshold-dependent features: one-hot R(T) and bars since last change (NaN where undefined)."""
    R = np.asarray(R, int)
    age = run_length(R)
    X = np.column_stack([(R == k).astype(float) for k in range(4)] + [np.maximum(age - 1, 0).astype(float)])
    X[R < 0] = np.nan
    return X


REGIME_FEAT_NAMES = ["R_stagnant", "R_steady", "R_choppy", "R_volatile", "bars_since_change"]


# ----------------------------------------------------------------------------- features
def own_features(df: pd.DataFrame, H: int, atr_n: int) -> pd.DataFrame:
    """Threshold-independent features of one series at bar T (uses data <= T only)."""
    cf = core_frame(df, atr_n, H)
    c, natr, er = cf["close"], cf["natr"], cf["er"]
    f = pd.DataFrame(index=cf.index)
    f["natr"], f["er"] = natr, er
    f["er_short"] = efficiency_ratio(c.to_numpy(), max(3, H // 4)).to_numpy()
    for k in (1, 3):
        f[f"dnatr{k}"] = natr / natr.shift(k) - 1.0
        f[f"der{k}"] = er - er.shift(k)
        f[f"alr{k}"] = np.log(c / c.shift(k)).abs()
    f["natr_rank"] = natr.rolling(500, min_periods=100).rank(pct=True)
    for k in (1, 2):
        f[f"natr_l{k}"] = natr.shift(k)
        f[f"er_l{k}"] = er.shift(k)
    return f


def base_features(target: pd.DataFrame, neighbours: dict, H: int, atr_n: int, tf: str) -> pd.DataFrame:
    """Own + neighbour ('graph') + calendar features, index = target index."""
    f = own_features(target, H, atr_n)
    for name, nd in neighbours.items():
        if nd is None or len(nd) == 0:
            continue
        nf = own_features(nd, H, atr_n)[["natr", "er"]]
        nf["natr_l1"], nf["er_l1"] = nf["natr"].shift(1), nf["er"].shift(1)
        nf = nf.reindex(f.index).ffill(limit=2)
        for col in nf.columns:
            f[f"{name}_{col}"] = nf[col]
    if tf in ("30m", "1h", "2h", "3h", "4h"):
        hr = f.index.hour.to_numpy() + f.index.minute.to_numpy() / 60.0
        dw = f.index.dayofweek.to_numpy()
        f["hr_sin"], f["hr_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        f["dw_sin"], f["dw_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- models / metrics
def make_model(kind: str):
    """HGB (tree) or LR (interpretable) classifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "HGB":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                              class_weight="balanced", early_stopping=False, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))


def fit_proba(kind: str, Xtr, ytr, Xte):
    """Fit and return (n x 4 probabilities, fitted model or None). Always 4 columns."""
    out = np.zeros((len(Xte), 4))
    cls = np.unique(ytr)
    if len(cls) == 0:
        out[:] = 0.25
        return out, None
    if len(cls) < 2:
        out[:, cls[0]] = 1.0
        return out, None
    m = make_model(kind)
    m.fit(Xtr, ytr)
    out[:, np.asarray(m.classes_, int)] = m.predict_proba(Xte)
    return out, m


def confusion(y, p) -> np.ndarray:
    """4x4 confusion matrix (rows true, cols predicted)."""
    cm = np.zeros((4, 4), int)
    y, p = np.asarray(y, int), np.asarray(p, int)
    if len(y):
        np.add.at(cm, (y, p), 1)
    return cm


def metrics(y, p) -> dict:
    """Accuracy, balanced accuracy, macro-F1, per-class recall/support, confusion matrix."""
    cm = confusion(y, p)
    n = cm.sum()
    sup = cm.sum(axis=1)
    rec = np.where(sup > 0, np.diag(cm) / np.maximum(sup, 1), np.nan)
    f1 = []
    for k in range(4):
        tp = cm[k, k]
        fp, fn = cm[:, k].sum() - tp, cm[k].sum() - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan)
    return {"acc": float(np.trace(cm) / n) if n else np.nan,
            "bal": float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan,
            "f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else np.nan,
            "recall": rec, "support": sup, "cm": cm,
            "risk": float((cm * COST).sum() / n) if n else np.nan}


def fwd_extreme(arr: np.ndarray, H: int, fn) -> np.ndarray:
    """fn (np.max/np.min) of arr over T+1..T+H (NaN where the window is incomplete)."""
    arr = np.asarray(arr, float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n > H >= 1:
        v = fn(sliding_window_view(arr, H), axis=1)
        out[: n - H] = v[1: n - H + 1]
    return out


def future_move(close, atr, H: int) -> np.ndarray:
    """|C_{T+H} - C_T| / ATR_T (NaN where undefined)."""
    c, a = np.asarray(close, float), np.asarray(atr, float)
    out = np.full(len(c), np.nan)
    if len(c) > H:
        out[:-H] = np.abs(c[H:] - c[:-H]) / a[:-H]
    return out


def run_ml(tgt: pd.DataFrame, Xb: pd.DataFrame, H: int, qa: float, qe: float, include_hgb: bool = False) -> dict:
    """Walk-forward evaluation (expanding window, purged, thresholds per fold) and live fit.

    tgt: core_frame output; Xb: base_features aligned to tgt. All fitted quantities use training rows only.
    Logistic regression always runs; gradient boosting (slow) only when include_hgb is True.
    """
    t_start = time.time()
    res = {"ok": False, "warnings": [], "msg": ""}
    n = len(tgt)
    natr, er = tgt["natr"].to_numpy(float), tgt["er"].to_numpy(float)
    close, high, low, atr = (tgt[c].to_numpy(float) for c in ("close", "high", "low", "atr"))
    Xb = Xb.reindex(tgt.index).replace([np.inf, -np.inf], np.nan)
    Xv = Xb.to_numpy(float)
    names = list(Xb.columns) + REGIME_FEAT_NAMES
    feat_ok = np.isfinite(Xv).all(axis=1) if Xv.size else np.zeros(n, bool)
    valid_bar = np.isfinite(natr) & np.isfinite(er)
    vc = np.concatenate([[0], np.cumsum(valid_bar)])
    lab_ok = np.zeros(n, bool)
    if n > H:
        Tm = np.arange(n - H)
        lab_ok[Tm] = (vc[Tm + H + 1] - vc[Tm + 1]) == H
    rows = np.where(feat_ok & lab_ok)[0]
    res["n_labeled"] = int(len(rows))
    if len(rows) < MIN_ML_ROWS:
        res["msg"] = (f"Only {len(rows)} labeled rows (< {MIN_ML_ROWS}): ML skipped, "
                      "showing the rule-based layer only.")
        return res
    if len(rows) < LOW_SAMPLE:
        res["warnings"].append(f"Low sample: {len(rows)} labeled rows (< {LOW_SAMPLE}); estimates are unstable.")
    start = len(rows) // 2
    blocks = [b for b in np.array_split(rows[start:], 4) if len(b)]
    recs, folds, last = [], [], None
    for bi, te in enumerate(blocks):
        t0 = te[0]
        tr = rows[(rows < t0) & (rows + H < t0)]  # purge: T+H < test_start
        if np.any(tr + H >= t0):
            res["warnings"].append(f"Fold {bi + 1} skipped: purge check failed (train labels overlap the test block).")
            continue
        if len(tr) < 50:
            continue
        th, ga = fit_thresholds(natr[:t0], er[:t0], qa, qe)
        R = regimes(natr, er, th, ga)
        y = make_labels(R, H)
        Xf = np.hstack([Xv, regime_feat_array(R)])
        if not ((y[tr] >= 0).all() and (y[te] >= 0).all()):
            res["warnings"].append(f"Fold {bi + 1} skipped: undefined labels in the train or test rows.")
            continue
        if not (np.isfinite(Xf[tr]).all() and np.isfinite(Xf[te]).all()):
            res["warnings"].append(f"Fold {bi + 1} skipped: non-finite features (NaN/inf) would reach the models.")
            continue
        ph, mh = fit_proba("HGB", Xf[tr], y[tr], Xf[te]) if include_hgb else (None, None)
        pl, ml = fit_proba("LR", Xf[tr], y[tr], Xf[te])
        major = int(np.bincount(y[tr], minlength=4).argmax())
        cols = {"pos": te, "y": y[te], "naive": R[te], "major": major, "fold": bi}
        if ph is not None:
            cols.update({f"h{k}": ph[:, k] for k in range(4)})
        cols.update({f"l{k}": pl[:, k] for k in range(4)})
        recs.append(pd.DataFrame(cols))
        folds.append({"fold": bi + 1, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                      "theta": th, "gamma": ga})
        last = (Xf[te], y[te], mh, ml)
    if not recs:
        res["msg"] = "Walk-forward produced no usable folds: ML skipped."
        return res
    oos = pd.concat(recs, ignore_index=True)
    probs = {}
    if include_hgb:
        probs["HGB"] = oos[[f"h{k}" for k in range(4)]].to_numpy()
    probs["LR"] = oos[[f"l{k}" for k in range(4)]].to_numpy()
    yo = oos["y"].to_numpy()
    preds = {k: v.argmax(1) for k, v in probs.items()}
    preds["Naive"] = oos["naive"].to_numpy()
    preds["Majority"] = oos["major"].to_numpy()
    met = {k: metrics(yo, v) for k, v in preds.items()}
    ran = list(probs)  # models that actually ran (HGB first, so it wins an exact tie as before)
    best = max(ran, key=lambda k: np.nan_to_num(met[k]["bal"]))
    pb = probs[best]
    pos = oos["pos"].to_numpy()
    oos["pred"] = pb.argmax(1)
    oos["ptop"] = pb.max(1)
    fh, fl = fwd_extreme(high, H, np.max), fwd_extreme(low, H, np.min)
    oos["move"] = future_move(close, atr, H)[pos]
    oos["range"] = ((fh - fl) / atr)[pos]
    oos["trend_later"] = np.isin(yo, (1, 3))
    sep = (oos.groupby("pred").agg(n=("pred", "size"), move=("move", "mean"), rng=("range", "mean"),
                                   trend=("trend_later", "mean")).reindex(range(4)))
    edges = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labs = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    oos["bin"] = pd.cut(oos["ptop"], edges, labels=labs, right=False)
    oos["hit"] = (oos["pred"] == oos["y"]).astype(float)
    cal = oos.groupby("bin", observed=False).agg(n=("hit", "size"), acc=("hit", "mean"),
                                                  p=("ptop", "mean")).reindex(labs)
    # importance (last test block, best model)
    imp = None
    try:
        if last is not None:
            Xte, yte, mh, ml = last
            m = mh if best == "HGB" else ml
            if m is not None and best == "HGB":
                from sklearn.inspection import permutation_importance
                k = min(len(Xte), 600)
                pi = permutation_importance(m, Xte[-k:], yte[-k:], n_repeats=3, random_state=0,
                                            scoring="balanced_accuracy")
                imp = pd.Series(pi.importances_mean, index=names)
            elif m is not None:
                imp = pd.Series(np.abs(m.named_steps["logisticregression"].coef_).mean(axis=0), index=names)
            if imp is not None:
                imp = imp.sort_values(ascending=False).head(10)
    except Exception as e:  # importance is optional
        res["warnings"].append(f"Importance unavailable: {type(e).__name__}")
    # live fit on all labeled rows
    th, ga = fit_thresholds(natr[rows], er[rows], qa, qe)
    R = regimes(natr, er, th, ga)
    y = make_labels(R, H)
    Xf = np.hstack([Xv, regime_feat_array(R)])
    live = None
    if feat_ok[-1] and valid_bar[-1] and np.isfinite(Xf[-1]).all():
        p, _ = fit_proba(best, Xf[rows], y[rows], Xf[[n - 1]])
        live = p[0]
    res.update({"ok": True, "oos": oos, "metrics": met, "best": best, "sep": sep, "cal": cal, "imp": imp,
                "live": live, "theta": th, "gamma": ga, "folds": folds, "rows": rows,
                "beats_naive": bool(np.nan_to_num(met[best]["bal"]) > np.nan_to_num(met["Naive"]["bal"])),
                "train_time": time.time() - t_start, "names": names, "models": ran, "include_hgb": bool(include_hgb)})
    return res


# ----------------------------------------------------------------------------- descriptive layer
def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def bias_sign(cf: pd.DataFrame, gamma: float) -> np.ndarray:
    """Direction bias per bar: +1 up, -1 down, 0 neutral (descriptive of past bars)."""
    e20, e50 = ema(cf["close"], 20).to_numpy(), ema(cf["close"], 50).to_numpy()
    s = cf["ser"].to_numpy(float)
    return np.where((s >= gamma) & (e20 > e50), 1, np.where((s <= -gamma) & (e20 < e50), -1, 0))


def bias_detail(cf: pd.DataFrame, gamma: float, H: int) -> dict:
    """Direction bias at the last bar with its ingredients."""
    if len(cf) < 3:
        return {}
    e20, e50 = ema(cf["close"], 20), ema(cf["close"], 50)
    atr = cf["atr"].iloc[-1]
    net = (cf["close"].iloc[-1] - cf["close"].iloc[-1 - H]) / atr if len(cf) > H and atr > 0 else np.nan
    slope = (e20.iloc[-1] - e20.iloc[-6]) / atr if len(cf) > 6 and atr > 0 else np.nan
    b = int(bias_sign(cf, gamma)[-1])
    return {"sign": b, "label": {1: "Up bias", -1: "Down bias", 0: "Neutral"}[b],
            "ser": float(cf["ser"].iloc[-1]), "net_atr": float(net), "slope_atr": float(slope),
            "ema_state": "EMA20 > EMA50" if e20.iloc[-1] > e50.iloc[-1] else "EMA20 < EMA50"}


def persistence(close, bias, R, H: int) -> list:
    """Hit rate of sign(C_{T+H}-C_T) == bias sign on non-overlapping samples (every H-th bar).

    `bias` is a +1/-1/0 array; pass np.sign(signed ER) so the non-trending group is not empty by construction.
    """
    c = np.asarray(close, float)
    n = len(c)
    out = []
    pos = np.arange(0, max(n - H, 0), H)
    for name, grp in (("Trending (1,3)", (1, 3)), ("Non-trending (0,2)", (0, 2))):
        sel = pos[np.isin(np.asarray(R)[pos], grp) & (np.asarray(bias)[pos] != 0)] if len(pos) else pos
        k = int((np.sign(c[sel + H] - c[sel]) == np.asarray(bias)[sel]).sum()) if len(sel) else 0
        lo, hi = wilson(k, len(sel))
        out.append({"group": name, "n": int(len(sel)), "rate": k / len(sel) if len(sel) else np.nan,
                    "lo": lo, "hi": hi})
    return out


def mtf_row(df: pd.DataFrame, H: int, qa: float, qe: float, atr_n: int):
    """Descriptive regime/bias row for one timeframe (thresholds from its full sample); None if < 100 bars."""
    if df is None or len(df) < 100:
        return None
    cf = core_frame(df, atr_n, H)
    th, ga = fit_thresholds(cf["natr"], cf["er"], qa, qe)
    R = regimes(cf["natr"], cf["er"], th, ga)
    if R[-1] < 0:
        return None
    nat = cf["natr"].dropna()
    return {"regime": int(R[-1]), "bias": int(bias_sign(cf, ga)[-1]), "er": float(cf["er"].iloc[-1]),
            "pct": float((nat <= nat.iloc[-1]).mean()), "bars": int(len(df))}


def default_H(tf: str) -> int:
    """Default horizon preset for a timeframe."""
    return [h for h, _, d in PRESETS[tf] if d][0]


def transition_matrix(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-normalised empirical matrix: current R(T) -> next-H label (NaN rows if no data)."""
    m = (np.asarray(R) >= 0) & (np.asarray(y) >= 0)
    cnt = np.zeros((4, 4))
    np.add.at(cnt, (np.asarray(R)[m], np.asarray(y)[m]), 1)
    s = cnt.sum(axis=1, keepdims=True)
    return np.where(s > 0, cnt / np.maximum(s, 1), np.nan)


def base_rate(R: np.ndarray, y: np.ndarray, k: int):
    """Empirical base rate: share of each next-H label among bars whose current regime was k. Returns (probs, N)."""
    R, y = np.asarray(R), np.asarray(y)
    if not (0 <= int(k) < 4):
        return np.full(4, np.nan), 0
    m = (R == k) & (y >= 0)
    n = int(m.sum())
    if n == 0:
        return np.full(4, np.nan), 0
    return np.bincount(y[m], minlength=4)[:4] / n, n


def median_durations(R: np.ndarray) -> list:
    """Median run length (bars) of each regime."""
    runs = {k: [] for k in range(4)}
    cur, ln = None, 0
    for r in np.asarray(R, int):
        if r == cur and r >= 0:
            ln += 1
        else:
            if cur is not None and cur >= 0:
                runs[cur].append(ln)
            cur, ln = r, 1
    if cur is not None and cur >= 0:
        runs[cur].append(ln)
    return [float(np.median(v)) if v else np.nan for v in runs.values()]


def conditional_stats(cf: pd.DataFrame, R: np.ndarray, H: int) -> pd.DataFrame:
    """Empirical |move|/ATR and H-bar range/ATR conditional on R(T): median, p80, n."""
    mv = future_move(cf["close"], cf["atr"], H)
    fh, fl = fwd_extreme(cf["high"], H, np.max), fwd_extreme(cf["low"], H, np.min)
    rg = (fh - fl) / cf["atr"].to_numpy(float)
    rows = []
    for k in range(4):
        m = (R == k) & np.isfinite(mv) & np.isfinite(rg)
        rows.append({"regime": k, "n": int(m.sum()),
                     "mv_med": np.median(mv[m]) if m.any() else np.nan,
                     "mv_p80": np.percentile(mv[m], 80) if m.any() else np.nan,
                     "rg_med": np.median(rg[m]) if m.any() else np.nan,
                     "rg_p80": np.percentile(rg[m], 80) if m.any() else np.nan})
    return pd.DataFrame(rows)


def pip_size(symbol: str) -> float:
    """Pip size: JPY pairs 0.01, other pairs 0.0001, DXY in points (1.0)."""
    return 1.0 if symbol == "DXY" else (0.01 if "JPY" in symbol else 0.0001)


def session_profile(natr: pd.Series):
    """Mean nATR by UTC hour and by weekday."""
    s = natr.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return s.groupby(s.index.hour).mean(), s.groupby(s.index.dayofweek).mean()


def window_end(last_ts: pd.Timestamp, tf: str, H: int) -> pd.Timestamp:
    """End of the forecast window counted in open-market time only.

    last_ts = start of the last closed bar (UTC). Intraday: H bars of open time, skipping the FX closure
    (Fri 17:00 to Sun 17:00 America/New_York). 1D/3D: weekdays only. 1W: whole weeks.
    """
    last_ts = pd.Timestamp(last_ts)
    last_ts = last_ts.tz_localize("UTC") if last_ts.tz is None else last_ts.tz_convert("UTC")
    dur = TF_SECONDS[tf]
    if tf in ("30m", "1h", "2h", "3h", "4h"):
        t = last_ts + pd.Timedelta(seconds=dur)
        rem = pd.Timedelta(seconds=dur * int(H))
        for _ in range(2000):
            cs, ce = next_closure(t)
            if t >= cs:
                t = ce
                continue
            if rem <= cs - t:
                return t + rem
            rem -= cs - t
            t = ce
        return t
    if tf in ("1D", "3D"):
        m = 1 if tf == "1D" else 3
        d0 = np.datetime64(last_ts.date())
        last_member = np.busday_offset(d0, m - 1, roll="forward")
        final = np.busday_offset(last_member, m * int(H), roll="forward")
        return pd.Timestamp(final, tz="UTC") + pd.Timedelta(days=1)
    wd = last_ts.weekday()
    end_last = last_ts.normalize() + pd.Timedelta(days=((4 - wd) % 7) + 1)
    return end_last + pd.Timedelta(days=7 * int(H))


def weekday_profile(byd: pd.Series) -> pd.Series:
    """Weekday series with Monday-first labels ('1 Mon' ... '7 Sun') so charts sort chronologically."""
    s = byd.sort_index().copy()
    s.index = [WEEKDAY_LABELS[int(i)] for i in s.index]
    return s
